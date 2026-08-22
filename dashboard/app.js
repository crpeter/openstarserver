const $ = selector => document.querySelector(selector);
const fmt = value => value == null ? "—" : Intl.NumberFormat(undefined, {notation: "compact", maximumFractionDigits: 1}).format(value);
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
function replace(node, children) { node.replaceChildren(...children); }
let workers = [];

function renderWorkers() {
  const query = $("#filter").value.toLowerCase();
  const rows = workers.filter(worker => JSON.stringify(worker).toLowerCase().includes(query)).map(worker => {
    const identity = element("td", {className: "device"}, [
      element("b", {text: worker.name || worker.id}),
      element("small", {text: [worker.hardwareModel, worker.platform, worker.osVersion].filter(Boolean).join(" · ") || "Telemetry unavailable"})
    ]);
    const badge = element("span", {className: `badge ${worker.computeState}`, text: worker.computeState.toUpperCase()});
    const assignment = element("td");
    if (worker.currentAssignments.length) {
      assignment.append(element("b", {text: `${worker.currentAssignments.length} × ${worker.currentAssignments[0].workloadID || "work unit"}`}), element("br"), element("small", {text: worker.currentAssignments[0].projectID || "Unknown project"}));
    } else assignment.textContent = "—";
    const seen = element("td", {text: relative(worker.lastSeenAt), title: worker.lastSeenAt ? new Date(worker.lastSeenAt * 1000).toISOString() : "Unavailable"});
    const row = element("tr", {}, [identity, element("td", {}, [badge]), assignment, element("td", {text: fmt(worker.completedWorkUnits)}), element("td", {text: duration(worker.cumulativeRuntimeSeconds)}), element("td", {text: worker.measuredThroughput == null ? "—" : `${fmt(worker.measuredThroughput)} eval/s`}), seen]);
    row.addEventListener("click", () => openDetail(worker.id));
    return row;
  });
  replace($("#workers"), rows);
}

function labelledRows(fields) {
  return Object.entries(fields).map(([label, value]) => element("div", {className: "row"}, [element("span", {text: label}), element("b", {text: value ?? "Unavailable"})]));
}
function jsonBlock(value) { return element("pre", {text: JSON.stringify(value, null, 2)}); }
function renderScienceRuns(runs) {
  const panel = $("#scienceRunsPanel");
  panel.hidden = !runs.length;
  if (!runs.length) { replace($("#scienceRuns"), []); return; }
  replace($("#scienceRuns"), [element("div", {className: "rows"}, runs.map(run => {
    const badgeClass = run.status === "RUNNING" ? "active" : run.status === "FAILED" ? "error" : "idle";
    return element("div", {className: "row"}, [
      element("div", {}, [element("b", {text: run.displayName || run.id}), element("small", {text: ` · ${run.kind || "science"}`})]),
      element("div", {}, [element("span", {className: `badge ${badgeClass}`, text: run.status || "UNKNOWN"}), element("small", {text: ` · ${relativeISO(run.updatedAt)}`})])
    ]);
  }))]);
}
function renderSectors(sweeps) {
  const panel = $("#sectorPanel");
  panel.hidden = !sweeps.length;
  if (!sweeps.length) { replace($("#sectors"), []); return; }
  replace($("#sectors"), sweeps.map(sweep => {
    const percent = Math.max(0, Math.min(1, sweep.progress || 0));
    const metrics = [["Remaining", sweep.remaining], ["Runnable", sweep.runnable], ["In flight or recovery", sweep.inFlightOrRecovery], ["Admitted", sweep.admitted], ["Inventory", sweep.inventory]];
    const status = sweep.runStatus === "RUNNING" ? "RUNNING" : sweep.status;
    return element("article", {className: "sector"}, [
      element("div", {className: "sector-heading"}, [element("h3", {text: `TESS Sector ${sweep.sector}`}), element("span", {className: `badge ${status === "COMPLETE" ? "active" : "idle"}`, text: status})]),
      element("div", {className: "sector-total"}, [element("b", {text: `${fmt(sweep.complete)} / ${fmt(sweep.inventory)}`}), element("span", {text: " targets complete"}), element("strong", {text: `${(percent * 100).toFixed(1)}%`})]),
      element("div", {className: "sector-bar", title: `${(percent * 100).toFixed(1)}% complete`}, [Object.assign(element("i"), {style: `width:${percent * 100}%`})]),
      element("div", {className: "sector-metrics"}, metrics.map(([label, value]) => element("div", {}, [element("span", {text: label}), element("b", {text: fmt(value)})])))
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
    const fields = {Status: `${worker.connectionState} / ${worker.computeState}`, Platform: worker.platform, "Hardware model": worker.hardwareModel, "OS version": worker.osVersion, "Worker version": worker.workerVersion, GPU: worker.gpuName, "CPU cores": worker.processorCount, Memory: worker.memoryGB && `${worker.memoryGB} GB`, Battery: worker.batteryLevel, Power: worker.powerState, Thermal: worker.thermalState, "Low power": worker.lowPowerMode, Network: worker.network, Completed: worker.completedWorkUnits, Failed: worker.failedWorkUnits, "Cumulative runtime": duration(worker.cumulativeRuntimeSeconds), "Metal runtime": duration(worker.metalSeconds), "Last seen": worker.lastSeenAt && new Date(worker.lastSeenAt * 1000).toISOString(), "Latest error": worker.latestError};
    replace(body, [element("h2", {text: worker.name || worker.id}), element("p", {text: worker.id}), element("h3", {text: "Telemetry"}), element("div", {className: "detailgrid"}, labelledRows(fields)), element("h3", {text: `Current assignments (${worker.currentAssignments.length})`}), jsonBlock(worker.currentAssignments), element("h3", {text: "Recent completed work"}), jsonBlock(worker.recentCompleted.length ? worker.recentCompleted : "No retained completed work"), element("h3", {text: "Recent errors"}), jsonBlock(worker.recentFailures.length ? worker.recentFailures : "No retained errors"), element("h3", {text: "Advertised capabilities"}), jsonBlock(worker.capabilities)]);
  } catch (error) { replace(body, [element("h2", {text: "Worker unavailable"}), element("p", {text: error.message})]); }
}

async function refresh() {
  try {
    const [snapshot, activity, history] = await Promise.all(["/api/dashboard/summary", "/api/dashboard/activity", "/api/dashboard/history"].map(url => fetch(url).then(response => response.json())));
    workers = snapshot.workers;
    const cards = [["Known workers", snapshot.summary.knownWorkers], ["Connected", snapshot.summary.connectedWorkers], ["Computing", snapshot.summary.activeWorkers], ["Running units", snapshot.summary.runningWorkUnits], ["Completed", snapshot.summary.completedWorkUnits], ["Compute time", duration(snapshot.summary.workerComputeSeconds)]];
    replace($("#stats"), cards.map(([label, value]) => element("div", {className: "stat"}, [element("span", {text: label}), element("b", {text: value})])));
    renderWorkers();
    renderScienceRuns(activity.scienceRuns || []);
    renderSectors(activity.sectorSweeps || []);
    $("#updated").textContent = `${snapshot.summary.health.toUpperCase()} · updated ${relative(snapshot.summary.updatedAt)}`;
    replace($("#activity"), [element("div", {className: "rows"}, activity.projects.length ? activity.projects.map(project => element("div", {className: "row"}, [element("div", {}, [element("b", {text: project.projectID || "Project"}), element("small", {text: ` · ${project.workloadID || "No workload"}`}), element("div", {className: "bar"}, [Object.assign(element("i"), {style: `width:${100 * (project.projectProgress || 0)}%`})])]), element("span", {text: `${project.projectCompletedWorkUnits || 0} / ${project.projectTotalWorkUnits || 0}`})])) : [element("p", {text: "No active projects"})])]);
    replace($("#contribution"), [element("div", {className: "rows"}, history.contributionByWorker.length ? history.contributionByWorker.slice(0, 8).map(node => element("div", {className: "row"}, [element("span", {text: node.nodeID}), element("b", {text: `${fmt(node.acceptedWorkUnits)} units`})])) : [element("p", {text: "Contributions will appear after accepted work."})])]);
  } catch (_) { $("#updated").textContent = "CONNECTION LOST"; }
}
$("#filter").addEventListener("input", renderWorkers);
refresh();
setInterval(refresh, 10000);
