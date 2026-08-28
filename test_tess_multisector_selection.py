import unittest

from workflows.tess.tess_multisector import (
    MAX_INDEPENDENT_SECTORS,
    _rank_independent_sectors,
)


class TessIndependentSectorSelectionTests(unittest.TestCase):
    def test_bounded_selection_balances_distant_and_nearby_repeats(self):
        ranked = _rank_independent_sectors(
            [21, 22, 40, 60, 80, 81],
            primary_sector=20,
        )

        self.assertEqual([81, 21, 80, 22, 60, 40], ranked)
        selected = ranked[:MAX_INDEPENDENT_SECTORS]
        self.assertEqual([81, 21, 80, 22], selected)
        self.assertEqual(2, sum(sector <= 22 for sector in selected))
        self.assertEqual(2, sum(sector >= 80 for sector in selected))

    def test_selection_is_deduplicated_and_deterministic(self):
        self.assertEqual(
            [60, 21, 40, 22],
            _rank_independent_sectors([40, 21, 60, 22, 40], 20),
        )

    def test_unknown_primary_retains_canonical_sector_order(self):
        self.assertEqual(
            [21, 22, 40],
            _rank_independent_sectors([40, 21, 22, 21], None),
        )


if __name__ == "__main__":
    unittest.main()
