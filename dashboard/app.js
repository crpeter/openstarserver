const $ = selector => document.querySelector(selector);
const fmt = value => value == null ? "—" : Intl.NumberFormat(undefined, {notation: "compact", maximumFractionDigits: 1}).format(value);
const count = value => value == null ? "—" : Intl.NumberFormat().format(value);
const duration = seconds => seconds == null ? "—" : seconds < 60 ? `${seconds.toFixed(1)} s` : seconds < 3600 ? `${(seconds / 60).toFixed(1)} min` : `${(seconds / 3600).toFixed(1)} h`;
const relative = timestamp => {
  if (!timestamp) return "Never";
  const seconds = Math.max(0, Date.now() / 1000 - timestamp);
  return seconds < 60 ? `${Math.floor(seconds)}s ago` : seconds < 3600 ? `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s ago` : seconds < 86400 ? `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m ago` : `${Math.floor(seconds / 86400)}d ago`;
};
const throughput = value => value == null ? "—" : `${fmt(value)} eval/s`;
const advertisedBackends = capabilities => {
  const value = capabilities && (capabilities.computeBackends ?? capabilities.supportedComputeBackends ?? capabilities.backends ?? capabilities.computeBackend ?? capabilities.backend);
  const backendID = entry => typeof entry === "string" ? entry : entry && typeof entry === "object" && typeof entry.id === "string" ? entry.id : typeof entry === "number" ? String(entry) : null;
  if (Array.isArray(value)) {
    const ids = value.map(backendID).filter(id => id && id.trim());
    return ids.length ? ids.join(", ") : null;
  }
  const id = backendID(value);
  return id && id.trim() ? id : null;
};
function element(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text != null) node.textContent = String(options.text);
  if (options.title) node.title = options.title;
  for (const child of children) node.append(child);
  return node;
}
function replace(node, children) { node.replaceChildren(...children); }
// Preserve existing DOM (including focus, scroll and open controls) when a
// polling response has not changed this section.
function refreshSection(node, value, children) {
  const signature = JSON.stringify(value);
  if (node.dataset.signature === signature) return;
  node.dataset.signature = signature;
  node.replaceChildren(...children);
}
let workers = [];

function reconcileKeyed(container, items, key, render) {
  const existing = new Map([...container.children].map(node => [node.dataset.key, node]));
  for (const item of items) {
    const id = String(key(item));
    const node = existing.get(id) || render(item);
    node.dataset.key = id;
    if (existing.has(id)) render(item, node);
    container.append(node);
    existing.delete(id);
  }
  for (const node of existing.values()) node.remove();
}

function renderWorkers() {
  const query = $("#filter").value.toLowerCase();
  const visible = workers.filter(worker => JSON.stringify(worker).toLowerCase().includes(query));
  reconcileKeyed($("#workers"), visible, worker => worker.id, (worker, current) => {
    const identity = element("td", {className: "device"}, [
      element("b", {text: worker.name || worker.id}),
      element("small", {text: [worker.hardwareModel, worker.platform, worker.osVersion].filter(Boolean).join(" · ") || "Telemetry unavailable"})
    ]);
    const connectionLabel = worker.connectionState === "connected" ? "ONLINE" : worker.connectionState === "recently_disconnected" ? "RECENTLY DISCONNECTED" : "OFFLINE";
    const badge = element("span", {className: `badge ${worker.connectionState}`, text: connectionLabel});
    const assignment = element("td");
    if (worker.currentAssignments.length) {
      assignment.append(element("b", {text: `${worker.currentAssignments.length} × ${worker.currentAssignments[0].workloadID || "work unit"}`}), element("br"), element("small", {text: worker.currentAssignments[0].projectID || "Unknown project"}));
    } else assignment.textContent = "—";
    const seen = element("td", {text: relative(worker.lastSeenAt), title: worker.lastSeenAt ? new Date(worker.lastSeenAt * 1000).toISOString() : "Unavailable"});
    const row = current || element("tr");
    row.replaceChildren(identity, element("td", {}, [badge]), assignment, element("td", {text: count(worker.completedWorkUnits)}), element("td", {text: duration(worker.cumulativeRuntimeSeconds)}), element("td", {text: throughput(worker.measuredThroughput)}), element("td", {text: throughput(worker.metalThroughput)}), seen);
    if (!current) row.addEventListener("click", () => openDetail(worker.id));
    return row;
  });
}

function labelledRows(fields) {
  return Object.entries(fields).map(([label, value]) => element("div", {className: "row"}, [element("span", {text: label}), element("b", {text: value ?? "Unavailable"})]));
}
function jsonBlock(value) { return element("pre", {text: JSON.stringify(value, null, 2)}); }
function renderSectors(sweeps) {
  const panel = $("#sectorPanel");
  panel.hidden = !sweeps.length;
  if (!sweeps.length) { replace($("#sectors"), []); return; }
  refreshSection($("#sectors"), sweeps, sweeps.map(sweep => {
    const percent = Math.max(0, Math.min(1, sweep.progress || 0));
    const metrics = [["Remaining", sweep.remaining], ["Runnable", sweep.runnable], ["In flight or recovery", sweep.inFlightOrRecovery], ["Admitted", sweep.admitted], ["Inventory", sweep.inventory]];
    return element("article", {className: "sector"}, [
      element("div", {className: "sector-heading"}, [element("h3", {text: `TESS Sector ${sweep.sector}`}), element("span", {className: `badge ${sweep.status === "COMPLETE" ? "active" : "idle"}`, text: sweep.status})]),
      element("div", {className: "sector-total"}, [element("b", {text: `${count(sweep.complete)} / ${count(sweep.inventory)}`}), element("span", {text: " targets complete"}), element("strong", {text: `${(percent * 100).toFixed(1)}%`})]),
      element("div", {className: "sector-bar", title: `${(percent * 100).toFixed(1)}% complete`}, [Object.assign(element("i"), {style: `width:${percent * 100}%`})]),
      element("div", {className: "sector-metrics"}, metrics.map(([label, value]) => element("div", {}, [element("span", {text: label}), element("b", {text: count(value)})])))
    ]);
  }));
}
async function openDetail(id) {
  const dialog = $("#detail");
  const body = $("#detailBody");
  replace(body, [element("p", {text: "Loading worker telemetry…"})]);
  dialog.showModal();
  try {
    const response = await fetch(`/api/dashboard/workers/${encodeURIComponent(id)}`);
    if (!response.ok) throw new Error(`Worker request failed (${response.status})`);
    const worker = await response.json();
    const fields = {Status: `${worker.connectionState} / ${worker.computeState}`, Platform: worker.platform, "Hardware model": worker.hardwareModel, "OS version": worker.osVersion, "Worker version": worker.workerVersion, GPU: worker.gpuName, "Advertised compute backend(s)": advertisedBackends(worker.capabilities), "Worker throughput": throughput(worker.measuredThroughput), "Accelerator throughput": throughput(worker.metalThroughput), "CPU cores": worker.processorCount, Memory: worker.memoryGB && `${worker.memoryGB} GB`, Battery: worker.batteryLevel, Power: worker.powerState, Thermal: worker.thermalState, "Low power": worker.lowPowerMode, Network: worker.network, Completed: worker.completedWorkUnits, Failed: worker.failedWorkUnits, "Cumulative runtime": duration(worker.cumulativeRuntimeSeconds), "Metal runtime": duration(worker.metalSeconds), "Last seen": worker.lastSeenAt && new Date(worker.lastSeenAt * 1000).toISOString(), "Latest error": worker.latestError};
    replace(body, [element("h2", {text: worker.name || worker.id}), element("p", {text: worker.id}), element("h3", {text: "Telemetry"}), element("div", {className: "detailgrid"}, labelledRows(fields)), element("h3", {text: `Current assignments (${worker.currentAssignments.length})`}), jsonBlock(worker.currentAssignments), element("h3", {text: "Recent completed work"}), jsonBlock(worker.recentCompleted.length ? worker.recentCompleted : "No retained completed work"), element("h3", {text: "Recent errors"}), jsonBlock(worker.recentFailures.length ? worker.recentFailures : "No retained errors"), element("h3", {text: "Advertised capabilities"}), jsonBlock(worker.capabilities)]);
  } catch (error) { replace(body, [element("h2", {text: "Worker unavailable"}), element("p", {text: error.message})]); }
}

async function refreshFleet() {
  try {
    const [snapshot, history] = await Promise.all(["/api/dashboard/summary", "/api/dashboard/history"].map(url => fetch(url).then(response => response.json())));
    workers = snapshot.workers.slice().sort((a, b) => (b.measuredThroughput ?? -1) - (a.measuredThroughput ?? -1) || a.id.localeCompare(b.id));
    const cards = [["Known workers", count(snapshot.summary.knownWorkers)], ["Connected", count(snapshot.summary.connectedWorkers)], ["Computing", count(snapshot.summary.activeWorkers)], ["Running units", count(snapshot.summary.runningWorkUnits)], ["Completed", count(snapshot.summary.completedWorkUnits)], ["Compute time", duration(snapshot.summary.workerComputeSeconds)], ["Fleet worker throughput", throughput(snapshot.summary.measuredThroughput), "Total sample-frequency evaluations divided by total worker compute seconds; intended for backend-neutral comparison."]];
    if (snapshot.summary.metalThroughput != null) cards.push(["Fleet Metal throughput", throughput(snapshot.summary.metalThroughput), "Metal execution throughput only; excludes broader worker overhead."]);
    refreshSection($("#stats"), cards, cards.map(([label, value, title]) => element("div", {className: "stat", title}, [element("span", {text: label}), element("b", {text: value})])));
    renderWorkers();
    $("#updated").textContent = `${snapshot.summary.health.toUpperCase()} · updated ${relative(snapshot.summary.updatedAt)}`;
    const contributions = Object.entries(history.completedByDeviceModel || {}).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    const contributionRows = contributions.length ? contributions.map(([model, acceptedWorkUnits]) => element("div", {className: "row"}, [element("span", {text: model}), element("b", {text: `${count(acceptedWorkUnits)} units`})])) : [element("p", {text: "Contributions will appear after accepted work."})];
    refreshSection($("#contribution"), contributions, [element("div", {className: "rows"}, contributionRows)]);
  } catch (_) { $("#updated").textContent = "CONNECTION LOST"; }
}

async function refreshScience() {
  try {
    const activity = await fetch("/api/dashboard/activity").then(response => response.json());
    renderSectors(activity.sectorSweeps || []);
    const projects = activity.projects.map(project => element("div", {className: "row"}, [element("div", {}, [element("b", {text: project.projectID || "Project"}), element("small", {text: ` · ${project.workloadID || "No workload"}`}), element("div", {className: "bar"}, [Object.assign(element("i"), {style: `width:${100 * (project.projectProgress || 0)}%`})])]), element("span", {text: `${project.projectCompletedWorkUnits || 0} / ${project.projectTotalWorkUnits || 0}`})]));
    const runs = (activity.scienceRuns || []).map(run => element("div", {className: "row"}, [element("span", {text: run.metadata.sector ? `${run.kind} · sector ${run.metadata.sector}` : run.kind}), element("b", {text: run.condition === "degraded" ? "DEGRADED" : run.status})]));
    refreshSection($("#activity"), activity.projects, [element("div", {className: "rows"}, projects.length ? projects : [element("p", {text: "No active projects"})])]);
    refreshSection($("#scienceRuns"), activity.scienceRuns, [element("div", {className: "rows"}, runs.length ? runs : [element("p", {text: "No cataloged science runs"})])]);
  } catch (_) { /* Fleet health remains independently visible. */ }
}
$("#filter").addEventListener("input", renderWorkers);
refreshFleet();
refreshScience();
setInterval(refreshFleet, 10000);
setInterval(refreshScience, 3000);

let targetPage = 1, targetPages = 1, targetController = null, targetQueryTimer = null;
const targetFilters = () => new URLSearchParams({page: targetPage, pageSize: 24,
  q: $("#targetSearch").value, status: $("#targetStatus").value,
  sector: $("#targetSector").value, health: $("#targetHealth").value,
  sort: $("#targetSort").value});
function targetStatusClass(status) {
  return ["FAILED", "BLOCKED", "RECOVERY_REQUIRED"].includes(status) ? "error" :
    ["COMPLETE", "COMPLETED", "FINISHED"].includes(status) ? "active" : "idle";
}
function renderTargetCard(target) {
  const card = element("article", {className: "target-card"}); card.tabIndex = 0;
  const progress = target.stageCounts.total ? 100 * target.stageCounts.completed / target.stageCounts.total : 0;
  card.replaceChildren(
    element("div", {className: "target-top"}, [element("span", {className: `badge ${targetStatusClass(target.status)}`, text: target.status}), element("span", {className: "identity", text: target.classification || "Classification pending"})]),
    element("h3", {text: target.targetName}), element("div", {className: "identity", text: [target.ticID && `TIC ${target.ticID}`, target.gaiaID && `Gaia ${target.gaiaID}`].filter(Boolean).join(" · ") || target.investigationID}),
    element("p", {className: "target-conclusion", text: target.currentClaim || "No scientific conclusion recorded yet."}),
    element("div", {className: "target-measures"}, [element("div", {}, [element("span", {text: target.resolvedPhysicalPeriod != null ? "Physical period" : "Detected period"}), element("b", {text: `${target.resolvedPhysicalPeriod ?? target.detectedPeriod ?? "Not recorded"}${target.resolvedPhysicalPeriod != null || target.detectedPeriod != null ? " d" : ""}`})]), element("div", {}, [element("span", {text: "Sectors"}), element("b", {text: [...target.primarySectors, ...target.independentSectors].join(", ") || "Not recorded"})])]),
    element("div", {className: "evidence", title: `${target.stageCounts.completed} of ${target.stageCounts.total} stages complete`}, [Object.assign(element("i"), {style: `width:${progress}%`})]),
    element("div", {className: "target-next"}, [element("span", {text: "Recommended next test"}), element("b", {text: target.recommendedNextTest || "No recommendation recorded"})]),
    element("div", {className: "target-foot"}, [element("span", {text: `${target.runCount} preserved run${target.runCount === 1 ? "" : "s"}`}), element("span", {text: relative(target.updatedAt)})]));
  const open = () => openTarget(target.targetID); card.addEventListener("click", open);
  card.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); }});
  return card;
}
async function loadTargets() {
  const gallery = $("#targetGallery");
  gallery.replaceChildren(...Array.from({length: 6}, () => element("div", {className: "skeleton"})));
  try {
    const response = await fetch(`/api/dashboard/targets?${targetFilters()}`);
    if (!response.ok) throw new Error(`Targets request failed (${response.status})`);
    const data = await response.json();
    const labels = [["Distinct targets", "totalTargets"], ["Active", "activeInvestigations"], ["Complete", "completedInvestigations"], ["Unresolved", "unresolvedTargets"], ["Source localized", "sourceLocalizedTargets"], ["Companion resolved", "companionNatureResolvedTargets"], ["Degraded", "degradedTargets"]];
    replace($("#targetStats"), labels.map(([label, key]) => element("div", {className: "stat"}, [element("span", {text: label}), element("b", {text: count(data.stats[key] || 0)})])));
    replace(gallery, data.targets.length ? data.targets.map(renderTargetCard) : [element("section", {className: "panel stat"}, [element("h3", {text: "No cataloged targets"}), element("p", {text: "Standalone investigation records will appear here when durable science history is cataloged."})])]);
    targetPages = Math.max(1, Math.ceil(data.total / data.pageSize)); targetPage = data.page;
    $("#targetPage").textContent = `Page ${targetPage} of ${targetPages}`; $("#targetPrev").disabled = targetPage <= 1; $("#targetNext").disabled = targetPage >= targetPages;
  } catch (error) { replace(gallery, [element("section", {className: "panel stat"}, [element("h3", {text: "Targets unavailable"}), element("p", {text: error.message}), Object.assign(element("button", {text: "Retry"}), {onclick: loadTargets})])]); }
}
function timeline(stages) { return element("div", {className: "timeline"}, stages.map(stage => element("article", {}, [element("b", {text: `${stage.id} · ${stage.status}`}), element("div", {className: "identity", text: stage.handler})]))); }
async function openTarget(id) {
  if (targetController) targetController.abort(); targetController = new AbortController();
  const signal = targetController.signal, dialog = $("#detail"), body = $("#detailBody");
  replace(body, [element("div", {className: "skeleton"})]); if (!dialog.open) dialog.showModal();
  try {
    const response = await fetch(`/api/dashboard/targets/${encodeURIComponent(id)}`, {signal});
    if (!response.ok) throw new Error(`Target request failed (${response.status})`); const target = await response.json();
    const latest = target.runs[0];
    const sections = [element("div", {className: "target-detail-head"}, [element("p", {className: "eyebrow", text: "TARGET DOSSIER"}), element("h2", {text: target.targetName}), element("p", {className: "identity", text: [target.ticID && `TIC ${target.ticID}`, target.gaiaID && `Gaia ${target.gaiaID}`, target.coordinates.ra != null && `RA ${target.coordinates.ra}`, target.coordinates.dec != null && `Dec ${target.coordinates.dec}`].filter(Boolean).join(" · ")})])];
    if (target.answerKeyUsed) sections.push(element("div", {className: "warning-callout", text: "Answer-key or external known-object evidence was used in this history. This result must not be interpreted as blind."}));
    sections.push(element("section", {className: "detail-section"}, [element("h3", {text: "What OpenStar knows"}), element("p", {text: target.currentClaim || "No authoritative scientific claim was recorded."}), element("div", {className: "detailgrid"}, labelledRows({Classification: target.classification, "Detected period": target.detectedPeriod, "Physical period": target.resolvedPhysicalPeriod, "Source attribution": typeof target.sourceAttribution === "string" ? target.sourceAttribution : target.sourceAttribution && JSON.stringify(target.sourceAttribution), "Physical mechanism": typeof target.physicalMechanism === "string" ? target.physicalMechanism : target.physicalMechanism && JSON.stringify(target.physicalMechanism), "Companion nature": typeof target.companionNature === "string" ? target.companionNature : target.companionNature && JSON.stringify(target.companionNature)}))]), element("section", {className: "detail-section"}, [element("h3", {text: "Evidence chain"}), timeline(latest.stages)]), element("section", {className: "detail-section"}, [element("h3", {text: "Recommended next test"}), element("p", {text: target.recommendedNextTest || "No authoritative recommendation recorded."})]), element("section", {className: "detail-section"}, [element("h3", {text: `Preserved run history (${target.runCount})`}), ...target.runs.map(run => element("div", {className: "row"}, [element("span", {text: run.investigationID}), element("b", {text: `${run.status} · ${run.stageCounts.completed}/${run.stageCounts.total} stages`})]))]), element("details", {}, [element("summary", {text: "Provenance and artifact metadata"}), jsonBlock({workflow: target.workflow, projectID: target.projectID, datasetID: target.datasetID, hashes: target.provenanceHashes, artifacts: target.artifacts})]), element("section", {className: "detail-section", text: "Loading persisted visual evidence…"}));
    replace(body, sections);
    fetch(`/api/dashboard/targets/${encodeURIComponent(id)}/visuals`, {signal}).then(r => {if (!r.ok) throw new Error(); return r.json();}).then(visual => { const section = body.lastElementChild; section.replaceChildren(element("h3", {text: "Evidence overview"}), timeline(visual.stageTimeline), element("p", {text: visual.message})); }).catch(error => {if (error.name !== "AbortError") body.lastElementChild.replaceChildren(element("p", {text: "Visualization evidence unavailable."}), Object.assign(element("button", {text: "Retry"}), {onclick: () => openTarget(id)}));});
  } catch (error) { if (error.name !== "AbortError") replace(body, [element("h2", {text: "Target unavailable"}), element("p", {text: error.message}), Object.assign(element("button", {text: "Retry"}), {onclick: () => openTarget(id)})]); }
}
for (const button of document.querySelectorAll(".nav")) button.addEventListener("click", () => {
  document.querySelectorAll(".nav").forEach(node => node.classList.toggle("active", node === button));
  const targets = button.dataset.view === "targets"; $("#fleetView").hidden = targets; $("#targetsView").hidden = !targets; if (targets) loadTargets();
});
for (const control of [$("#targetSearch"), $("#targetStatus"), $("#targetSector"), $("#targetHealth"), $("#targetSort")]) control.addEventListener("input", () => { clearTimeout(targetQueryTimer); targetQueryTimer = setTimeout(() => {targetPage = 1; loadTargets();}, 180); });
$("#targetPrev").addEventListener("click", () => {if (targetPage > 1) {targetPage--; loadTargets(); window.scrollTo({top: 0});}}); $("#targetNext").addEventListener("click", () => {if (targetPage < targetPages) {targetPage++; loadTargets(); window.scrollTo({top: 0});}});
$("#detail").addEventListener("close", () => {if (targetController) targetController.abort();});
