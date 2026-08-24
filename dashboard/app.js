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
