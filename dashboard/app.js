const $ = selector => document.querySelector(selector);
const compactFmt = value => value == null ? "—" : Intl.NumberFormat(undefined, {notation: "compact", maximumFractionDigits: 1}).format(value);
const exactFmt = value => value == null ? "—" : Intl.NumberFormat().format(value);
const duration = seconds => seconds == null ? "—" : seconds < 60 ? `${seconds.toFixed(1)} s` : seconds < 3600 ? `${(seconds / 60).toFixed(1)} min` : `${(seconds / 3600).toFixed(1)} h`;
const relative = timestamp => {
  if (!timestamp) return "Never";
  const seconds = Math.max(0, Date.now() / 1000 - timestamp);
  return seconds < 60 ? "just now" : seconds < 3600 ? `${Math.floor(seconds / 60)}m ago` : seconds < 86400 ? `${Math.floor(seconds / 3600)}h ago` : `${Math.floor(seconds / 86400)}d ago`;
};
const relativeISO = value => value && !Number.isNaN(Date.parse(value)) ? relative(Date.parse(value) / 1000) : "Never";

function element(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text != null) node.textContent = String(options.text);
  if (options.title) node.title = options.title;
  for (const child of children) node.append(child);
  return node;
}
function setText(node, value) {
  const text = value == null ? "" : String(value);
  if (node.textContent !== text) node.textContent = text;
}
function setClass(node, className) {
  if (node.className !== className) node.className = className;
}
function setTitle(node, title) {
  if (node.title !== title) node.title = title;
}
function replace(node, children) { node.replaceChildren(...children); }
function rowsContainer(root) {
  let rows = root.firstElementChild;
  if (!rows || !rows.classList.contains("rows")) {
    rows = element("div", {className: "rows"});
    root.replaceChildren(rows);
  }
  return rows;
}
function reconcile(container, items, keyFor, createNode, updateNode) {
  const existing = new Map(Array.from(container.children).map(node => [node.dataset.key, node]));
  items.forEach((item, index) => {
    const key = String(keyFor(item));
    let node = existing.get(key);
    if (!node) {
      node = createNode(item);
      node.dataset.key = key;
    }
    existing.delete(key);
    updateNode(node, item);
    const current = container.children[index];
    if (current !== node) container.insertBefore(node, current || null);
  });
  for (const node of existing.values()) node.remove();
}

let workers = [];

function createWorkerRow(worker) {
  const name = element("b");
  const info = element("small");
  const badge = element("span", {className: "badge idle"});
  const assignmentMain = element("b");
  const assignmentSub = element("small");
  const assignment = element("td", {}, [assignmentMain, element("br"), assignmentSub]);
  const completed = element("td");
  const compute = element("td");
  const throughput = element("td");
  const seen = element("td");
  const row = element("tr", {}, [
    element("td", {className: "device"}, [name, info]),
    element("td", {}, [badge]),
    assignment,
    completed,
    compute,
    throughput,
    seen,
  ]);
  row._refs = {name, info, badge, assignmentMain, assignmentSub, completed, compute, throughput, seen};
  row.addEventListener("click", () => openDetail(worker.id));
  return row;
}
function updateWorkerRow(row, worker) {
  const refs = row._refs;
  setText(refs.name, worker.name || worker.id);
  setText(refs.info, [worker.hardwareModel, worker.platform, worker.osVersion].filter(Boolean).join(" · ") || "Telemetry unavailable");
  setClass(refs.badge, `badge ${worker.computeState}`);
  setText(refs.badge, worker.computeState.toUpperCase());
  if (worker.currentAssignments.length) {
    setText(refs.assignmentMain, `${worker.currentAssignments.length} × ${worker.currentAssignments[0].workloadID || "work unit"}`);
    setText(refs.assignmentSub, worker.currentAssignments[0].projectID || "Unknown project");
  } else {
    setText(refs.assignmentMain, "—");
    setText(refs.assignmentSub, "");
  }
  setText(refs.completed, exactFmt(worker.completedWorkUnits));
  setText(refs.compute, duration(worker.cumulativeRuntimeSeconds));
  setText(refs.throughput, worker.measuredThroughput == null ? "—" : `${compactFmt(worker.measuredThroughput)} eval/s`);
  setText(refs.seen, relative(worker.lastSeenAt));
  setTitle(refs.seen, worker.lastSeenAt ? new Date(worker.lastSeenAt * 1000).toISOString() : "Unavailable");
  const query = $("#filter").value.toLowerCase();
  row.hidden = !!query && !JSON.stringify(worker).toLowerCase().includes(query);
}
function renderWorkers() {
  reconcile($("#workers"), workers, worker => worker.id, createWorkerRow, updateWorkerRow);
}

function createStat([label]) {
  const labelNode = element("span", {text: label});
  const valueNode = element("b");
  const node = element("div", {className: "stat"}, [labelNode, valueNode]);
  node._refs = {labelNode, valueNode};
  return node;
}
function renderStats(summary) {
  const cards = [
    ["Known workers", exactFmt(summary.knownWorkers)],
    ["Connected", exactFmt(summary.connectedWorkers)],
    ["Computing", exactFmt(summary.activeWorkers)],
    ["Running units", exactFmt(summary.runningWorkUnits)],
    ["Completed", exactFmt(summary.completedWorkUnits)],
    ["Compute time", duration(summary.workerComputeSeconds)],
  ];
  reconcile($("#stats"), cards, item => item[0], createStat, (node, item) => {
    setText(node._refs.labelNode, item[0]);
    setText(node._refs.valueNode, item[1]);
  });
}

function labelledRows(fields) {
  return Object.entries(fields).map(([label, value]) => element("div", {className: "row"}, [element("span", {text: label}), element("b", {text: value ?? "Unavailable"})]));
}
function jsonBlock(value) { return element("pre", {text: JSON.stringify(value, null, 2)}); }

function createScienceRunRow() {
  const name = element("b");
  const kind = element("small");
  const badge = element("span", {className: "badge idle"});
  const updated = element("small");
  const node = element("div", {className: "row"}, [
    element("div", {}, [name, kind]),
    element("div", {}, [badge, updated]),
  ]);
  node._refs = {name, kind, badge, updated};
  return node;
}
function updateScienceRunRow(node, run) {
  const refs = node._refs;
  const badgeClass = run.status === "RUNNING" ? "active" : run.status === "FAILED" ? "error" : "idle";
  setText(refs.name, run.displayName || run.id);
  setText(refs.kind, ` · ${run.kind || "science"}`);
  setClass(refs.badge, `badge ${badgeClass}`);
  setText(refs.badge, run.status || "UNKNOWN");
  setText(refs.updated, ` · ${relativeISO(run.updatedAt)}`);
}
function renderScienceRuns(runs) {
  const panel = $("#scienceRunsPanel");
  panel.hidden = !runs.length;
  const rows = rowsContainer($("#scienceRuns"));
  reconcile(rows, runs, run => run.id, createScienceRunRow, updateScienceRunRow);
}

const sectorMetricNames = ["Remaining", "Runnable", "In flight or recovery", "Admitted", "Inventory"];
function createSectorCard() {
  const heading = element("h3");
  const badge = element("span", {className: "badge idle"});
  const total = element("b");
  const percent = element("strong");
  const barFill = element("i");
  const bar = element("div", {className: "sector-bar"}, [barFill]);
  const metrics = new Map();
  const metricNodes = sectorMetricNames.map(label => {
    const value = element("b");
    metrics.set(label, value);
    return element("div", {}, [element("span", {text: label}), value]);
  });
  const node = element("article", {className: "sector"}, [
    element("div", {className: "sector-heading"}, [heading, badge]),
    element("div", {className: "sector-total"}, [total, element("span", {text: " targets complete"}), percent]),
    bar,
    element("div", {className: "sector-metrics"}, metricNodes),
  ]);
  node._refs = {heading, badge, total, percent, bar, barFill, metrics};
  return node;
}
function updateSectorCard(node, sweep) {
  const refs = node._refs;
  const progress = Math.max(0, Math.min(1, sweep.progress || 0));
  const percentText = `${(progress * 100).toFixed(2)}%`;
  const status = sweep.status === "COMPLETE" ? "COMPLETE" : (sweep.runStatus || sweep.status);
  setText(refs.heading, `TESS Sector ${sweep.sector}`);
  setClass(refs.badge, `badge ${status === "COMPLETE" ? "active" : "idle"}`);
  setText(refs.badge, status);
  setText(refs.total, `${exactFmt(sweep.complete)} / ${exactFmt(sweep.inventory)}`);
  setText(refs.percent, percentText);
  setTitle(refs.bar, `${percentText} complete`);
  const width = `${progress * 100}%`;
  if (refs.barFill.style.width !== width) refs.barFill.style.width = width;
  const values = new Map([
    ["Remaining", sweep.remaining],
    ["Runnable", sweep.runnable],
    ["In flight or recovery", sweep.inFlightOrRecovery],
    ["Admitted", sweep.admitted],
    ["Inventory", sweep.inventory],
  ]);
  for (const [label, valueNode] of refs.metrics) setText(valueNode, exactFmt(values.get(label)));
}
function renderSectors(sweeps) {
  const panel = $("#sectorPanel");
  panel.hidden = !sweeps.length;
  reconcile($("#sectors"), sweeps, sweep => sweep.runID || `sector-${sweep.sector}`, createSectorCard, updateSectorCard);
}

function createProjectRow() {
  const name = element("b");
  const workload = element("small");
  const barFill = element("i");
  const completed = element("span");
  const node = element("div", {className: "row"}, [
    element("div", {}, [name, workload, element("div", {className: "bar"}, [barFill])]),
    completed,
  ]);
  node._refs = {name, workload, barFill, completed};
  return node;
}
function updateProjectRow(node, project) {
  setText(node._refs.name, project.projectID || "Project");
  setText(node._refs.workload, ` · ${project.workloadID || "No workload"}`);
  const width = `${100 * (project.projectProgress || 0)}%`;
  if (node._refs.barFill.style.width !== width) node._refs.barFill.style.width = width;
  setText(node._refs.completed, `${exactFmt(project.projectCompletedWorkUnits || 0)} / ${exactFmt(project.projectTotalWorkUnits || 0)}`);
}
function renderProjects(projects) {
  const root = $("#activity");
  const rows = rowsContainer(root);
  if (!projects.length) {
    reconcile(rows, [{projectID: "__empty__"}], item => item.projectID, () => element("p", {text: "No active projects"}), () => {});
    return;
  }
  reconcile(rows, projects, project => project.projectID || project.workloadID || "project", createProjectRow, updateProjectRow);
}

function createContributionRow() {
  const id = element("span");
  const units = element("b");
  const node = element("div", {className: "row"}, [id, units]);
  node._refs = {id, units};
  return node;
}
function updateContributionRow(node, contribution) {
  setText(node._refs.id, contribution.nodeID);
  setText(node._refs.units, `${exactFmt(contribution.acceptedWorkUnits)} units`);
}
function renderContribution(history) {
  const rows = rowsContainer($("#contribution"));
  const items = history.contributionByWorker || [];
  if (!items.length) {
    reconcile(rows, [{nodeID: "__empty__"}], item => item.nodeID, () => element("p", {text: "Contributions will appear after accepted work."}), () => {});
    return;
  }
  reconcile(rows, items.slice(0, 8), item => item.nodeID, createContributionRow, updateContributionRow);
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
    const fields = {Status: `${worker.connectionState} / ${worker.computeState}`, Platform: worker.platform, "Hardware model": worker.hardwareModel, "OS version": worker.osVersion, "Worker version": worker.workerVersion, GPU: worker.gpuName, "CPU cores": worker.processorCount, Memory: worker.memoryGB && `${worker.memoryGB} GB`, Battery: worker.batteryLevel, Power: worker.powerState, Thermal: worker.thermalState, "Low power": worker.lowPowerMode, Network: worker.network, Completed: worker.completedWorkUnits, Failed: worker.failedWorkUnits, "Cumulative runtime": duration(worker.cumulativeRuntimeSeconds), "Metal runtime": duration(worker.metalSeconds), "Last seen": worker.lastSeenAt && new Date(worker.lastSeenAt * 1000).toISOString(), "Latest error": worker.latestError};
    replace(body, [element("h2", {text: worker.name || worker.id}), element("p", {text: worker.id}), element("h3", {text: "Telemetry"}), element("div", {className: "detailgrid"}, labelledRows(fields)), element("h3", {text: `Current assignments (${worker.currentAssignments.length})`}), jsonBlock(worker.currentAssignments), element("h3", {text: "Recent completed work"}), jsonBlock(worker.recentCompleted.length ? worker.recentCompleted : "No retained completed work"), element("h3", {text: "Recent errors"}), jsonBlock(worker.recentFailures.length ? worker.recentFailures : "No retained errors"), element("h3", {text: "Advertised capabilities"}), jsonBlock(worker.capabilities)]);
  } catch (error) { replace(body, [element("h2", {text: "Worker unavailable"}), element("p", {text: error.message})]); }
}

let activityRefreshRunning = false;
let fleetRefreshRunning = false;

async function refreshActivity() {
  if (activityRefreshRunning) return;
  activityRefreshRunning = true;
  try {
    const response = await fetch("/api/dashboard/activity", {cache: "no-store"});
    if (!response.ok) throw new Error(`Activity request failed (${response.status})`);
    const activity = await response.json();
    renderScienceRuns(activity.scienceRuns || []);
    renderSectors(activity.sectorSweeps || []);
    renderProjects(activity.projects || []);
  } catch (_) {
    $("#updated").textContent = "CONNECTION LOST";
  } finally {
    activityRefreshRunning = false;
  }
}

async function refreshFleet() {
  if (fleetRefreshRunning) return;
  fleetRefreshRunning = true;
  try {
    const [snapshotResponse, historyResponse] = await Promise.all([
      fetch("/api/dashboard/summary", {cache: "no-store"}),
      fetch("/api/dashboard/history", {cache: "no-store"}),
    ]);
    if (!snapshotResponse.ok || !historyResponse.ok) throw new Error("Dashboard request failed");
    const [snapshot, history] = await Promise.all([snapshotResponse.json(), historyResponse.json()]);
    workers = snapshot.workers;
    renderStats(snapshot.summary);
    renderWorkers();
    renderContribution(history);
    setText($("#updated"), `${snapshot.summary.health.toUpperCase()} · updated ${relative(snapshot.summary.updatedAt)}`);
  } catch (_) {
    $("#updated").textContent = "CONNECTION LOST";
  } finally {
    fleetRefreshRunning = false;
  }
}

async function refresh() {
  await Promise.all([refreshFleet(), refreshActivity()]);
}

$("#filter").addEventListener("input", renderWorkers);
refresh();
setInterval(refreshActivity, 2000);
setInterval(refreshFleet, 10000);
