"""Provider-neutral, durable inventory boundary for TESS sector products."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

SCHEMA_VERSION = "1"
SELECTION_ALGORITHM_VERSION = "spoc-cadence-preference-v1"
PREFERRED_SPOC_CADENCE_SECONDS = 120.0


@dataclass(frozen=True)
class TessArchiveProduct:
    sector: int | None
    tic_id: int | None
    target_name: str | None
    observation_id: str | None = None
    mast_observation_id: str | None = None
    data_uri: str | None = None
    product_uri: str | None = None
    product_filename: str | None = None
    author: str | None = None
    cadence_seconds: float | None = None
    data_rights: str | None = None
    source_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TessSectorInventoryEntry:
    product: TessArchiveProduct


@dataclass(frozen=True)
class TessSkippedProduct:
    product: TessArchiveProduct
    reason: str


@dataclass(frozen=True)
class TessSectorInventory:
    schema_version: str
    sector: int
    provider_id: str
    provider_version: str
    selection_algorithm_version: str
    entries: tuple[TessSectorInventoryEntry, ...]
    skipped: tuple[TessSkippedProduct, ...]


class TessSectorArchiveProvider(Protocol):
    id: str
    version: str

    def inventory_sector(self, sector: int) -> Sequence[TessArchiveProduct]: ...

    def download_light_curve(self, product: TessArchiveProduct, destination: Path) -> Path: ...


def _key(product: TessArchiveProduct):
    return (
        product.cadence_seconds if product.cadence_seconds is not None else float("inf"),
        product.product_filename or "", product.product_uri or "", product.observation_id or "",
    )


def _priority(product: TessArchiveProduct):
    author = (product.author or "").upper()
    cadence = product.cadence_seconds
    if author == "SPOC" and cadence is not None and abs(cadence - PREFERRED_SPOC_CADENCE_SECONDS) < 0.5:
        return (0, *_key(product))
    if author == "SPOC":
        return (1, *_key(product))
    return (2, *_key(product))


def select_sector_products(sector: int, products: Sequence[TessArchiveProduct]):
    admitted: dict[int, list[TessArchiveProduct]] = {}
    skipped: list[TessSkippedProduct] = []
    for product in products:
        reason = None
        if product.sector != sector:
            reason = "INVALID_SECTOR"
        elif product.tic_id is None:
            reason = "NO_PARSEABLE_TIC_ID"
        elif (product.author or "").upper() not in {"SPOC", "TESS-SPOC"}:
            reason = "UNSUPPORTED_AUTHOR"
        elif (product.data_rights or "PUBLIC").upper() not in {"PUBLIC", ""}:
            reason = "NONPUBLIC_PRODUCT"
        if reason:
            skipped.append(TessSkippedProduct(product, reason))
        else:
            admitted.setdefault(int(product.tic_id), []).append(product)
    entries = []
    for tic_id in sorted(admitted):
        candidates = sorted(admitted[tic_id], key=_priority)
        entries.append(TessSectorInventoryEntry(candidates[0]))
        skipped.extend(TessSkippedProduct(item, "DUPLICATE_LOWER_PRIORITY_PRODUCT") for item in candidates[1:])
    skipped.sort(key=lambda item: (item.reason, item.product.tic_id or -1, _key(item.product)))
    return tuple(entries), tuple(skipped)


class TessSectorInventoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def create_or_load(self, sector: int, provider: TessSectorArchiveProvider) -> TessSectorInventory:
        if self.path.exists():
            inventory = self.load()
            expected = (SCHEMA_VERSION, sector, provider.id, provider.version)
            actual = (inventory.schema_version, inventory.sector, inventory.provider_id, inventory.provider_version)
            if actual != expected:
                raise RuntimeError(f"Incompatible persisted TESS sector inventory: expected {expected}, found {actual}")
            return inventory
        entries, skipped = select_sector_products(sector, provider.inventory_sector(sector))
        inventory = TessSectorInventory(SCHEMA_VERSION, sector, provider.id, provider.version,
                                        SELECTION_ALGORITHM_VERSION, entries, skipped)
        self._write(inventory)
        return inventory

    def load(self) -> TessSectorInventory:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        def product(value): return TessArchiveProduct(**value)
        return TessSectorInventory(
            raw["schema_version"], int(raw["sector"]), raw["provider_id"], raw["provider_version"],
            raw["selection_algorithm_version"],
            tuple(TessSectorInventoryEntry(product(x["product"])) for x in raw["entries"]),
            tuple(TessSkippedProduct(product(x["product"]), x["reason"]) for x in raw["skipped"]),
        )

    def _write(self, inventory: TessSectorInventory):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(asdict(inventory), handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
            if self.path.exists():
                raise FileExistsError(f"Inventory appeared while being created: {self.path}")
            os.replace(temporary, self.path); temporary = ""
        finally:
            if temporary and os.path.exists(temporary): os.unlink(temporary)


class MastTessSectorArchiveProvider:
    """MAST interpretation is deliberately confined to this adapter."""
    id = "mast-tess-sector-archive"
    version = "1"

    @staticmethod
    def _tic(value: Any) -> int | None:
        match = re.search(r"(?:TIC\s*)?(\d+)", str(value or ""), re.I)
        return int(match.group(1)) if match else None

    def inventory_sector(self, sector: int) -> Sequence[TessArchiveProduct]:
        try:
            from astroquery.mast import Observations
        except ImportError as error:
            raise RuntimeError("astroquery is required for MAST inventory") from error
        observations = Observations.query_criteria(obs_collection="TESS", sequence_number=sector,
                                                   dataproduct_type="timeseries")
        products = Observations.get_product_list(observations)
        observation_fields = {}
        for row in observations:
            fields = self._row_fields(row)
            observation_fields[str(fields.get("obsid") or "")] = fields
        result = []
        for row in products:
            product_fields = self._row_fields(row)
            parent = str(product_fields.get("parent_obsid") or "")
            observation = observation_fields.get(parent, {})
            fields = {**observation, **product_fields}
            filename = str(product_fields.get("productFilename") or "")
            if "lc.fits" not in filename.lower():
                continue
            obs_id = observation.get("obs_id") or fields.get("obs_id") or fields.get("parent_obsid")
            target = fields.get("target_name") or fields.get("targetName") or obs_id
            result.append(TessArchiveProduct(
                sector=sector, tic_id=self._tic(target) or self._tic(filename), target_name=str(target),
                observation_id=str(obs_id) if obs_id is not None else None,
                mast_observation_id=str(fields.get("obsid") or fields.get("parent_obsid") or "") or None,
                data_uri=fields.get("dataURI"), product_uri=fields.get("dataURI"), product_filename=filename,
                author=str(observation.get("provenance_name") or fields.get("provenance_name") or ""),
                cadence_seconds=float(fields["t_exptime"]) if fields.get("t_exptime") is not None else None,
                data_rights=str(fields.get("dataRights") or "PUBLIC"), source_fields=fields))
        return result

    @staticmethod
    def _row_fields(row) -> dict[str, Any]:
        fields = {}
        for name in row.colnames:
            value = row[name]
            if str(value) == "--":
                value = None
            elif hasattr(value, "item"):
                value = value.item()
            if value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
                value = str(value)
            fields[name] = value
        return fields

    def download_light_curve(self, product: TessArchiveProduct, destination: Path) -> Path:
        from astroquery.mast import Observations
        uri = product.product_uri or product.data_uri
        if not uri: raise RuntimeError("Selected MAST product has no data URI.")
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / (product.product_filename or "tess-light-curve.fits")
        result = Observations.download_file(uri, local_path=str(path), cache=True)
        status = result[0] if isinstance(result, (tuple, list)) and result else result
        if str(status).upper() != "COMPLETE" or not path.exists():
            raise RuntimeError(f"MAST download failed for {uri}")
        return path
