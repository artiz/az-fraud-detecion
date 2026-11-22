/* Elliptic graph explorer: timeline + per-step force-directed graph on canvas (d3 v7). */
(() => {
  const COLORS = { illicit: "#d62728", licit: "#1f77b4", unknown: "#b8b8b8" };
  const state = { meta: null, step: 1, mode: "label", graph: null, sim: null, transform: d3.zoomIdentity,
                  hover: null, selected: null, showUnknown: true, showEdges: true, nodeById: new Map() };

  const canvas = document.getElementById("graph");
  const ctx = canvas.getContext("2d");
  const tooltip = document.getElementById("tooltip");
  const status = document.getElementById("status");
  const modeSel = document.getElementById("mode");
  const stepInput = document.getElementById("step");
  const stepLabel = document.getElementById("step-label");

  const api = (path) => fetch(path).then((r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); });

  // ---------------------------------------------------------------- colouring
  function labelName(l) { return l === 1 ? "illicit" : l === 0 ? "licit" : "unknown"; }
  function nodeClass(n) {
    if (state.mode === "label") return labelName(n.label);
    const s = n[`s_${state.mode}`];
    if (s === undefined) return "unknown";
    return s >= state.meta.models[state.mode].threshold ? "illicit" : "licit";
  }
  function nodeVisible(n) { return state.showUnknown || n.label !== -1; }

  // ---------------------------------------------------------------- timeline
  function drawTimeline() {
    const svg = d3.select("#timeline-svg");
    const W = svg.node().clientWidth, H = svg.node().clientHeight;
    const m = { top: 14, right: 10, bottom: 18, left: 36 };
    const rows = state.meta.timesteps;
    const x = d3.scaleBand().domain(rows.map((r) => r.timestep)).range([m.left, W - m.right]).padding(0.15);
    const y = d3.scaleSqrt().domain([0, d3.max(rows, (r) => r.licit + r.illicit)]).range([H - m.bottom, m.top]);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    // train / val / test shading
    const splits = d3.groups(rows, (r) => r.split);
    for (const [name, rs] of splits) {
      const x0 = x(rs[0].timestep), x1 = x(rs[rs.length - 1].timestep) + x.bandwidth();
      g.append("rect").attr("x", x0).attr("y", m.top - 12).attr("width", x1 - x0).attr("height", H - m.top - m.bottom + 12)
        .attr("fill", name === "test" ? "#fff3f3" : name === "val" ? "#f3f7ff" : "#f5f5f5");
      g.append("text").attr("class", "split-label").attr("x", (x0 + x1) / 2).attr("y", m.top - 3).attr("text-anchor", "middle").text(name);
    }
    g.selectAll(".lic").data(rows).join("rect").attr("class", "timeline-bar lic")
      .attr("x", (r) => x(r.timestep)).attr("width", x.bandwidth())
      .attr("y", (r) => y(r.licit + r.illicit)).attr("height", (r) => y(0) - y(r.licit + r.illicit))
      .attr("fill", COLORS.licit).attr("opacity", 0.45);
    g.selectAll(".ill").data(rows).join("rect").attr("class", (r) => "timeline-bar ill" + (r.timestep === state.step ? " selected" : ""))
      .attr("x", (r) => x(r.timestep)).attr("width", x.bandwidth())
      .attr("y", (r) => y(r.illicit)).attr("height", (r) => y(0) - y(r.illicit))
      .attr("fill", COLORS.illicit);
    g.selectAll(".hit").data(rows).join("rect").attr("class", "timeline-bar hit")
      .attr("x", (r) => x(r.timestep)).attr("width", x.bandwidth()).attr("y", m.top - 12).attr("height", H - m.top - m.bottom + 12)
      .attr("fill", "transparent")
      .on("click", (_, r) => setStep(r.timestep))
      .append("title").text((r) => `step ${r.timestep}: ${r.illicit} illicit, ${r.licit} licit, ${r.unknown} unknown, ${r.edges} edges`);
    // dark-market shutdown marker (Elliptic paper: after step 43)
    const xs = x(43) + x.bandwidth();
    g.append("line").attr("x1", xs).attr("x2", xs).attr("y1", m.top - 12).attr("y2", H - m.bottom).attr("stroke", "#000").attr("stroke-dasharray", "3,3");
    g.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).tickValues(rows.map((r) => r.timestep).filter((t) => t % 4 === 1 || t === 49)).tickSize(3));
    g.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(4).tickSize(3));
  }

  // ---------------------------------------------------------------- graph
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function radius(n) { return 2 + Math.min(5, Math.sqrt(n.in + n.out)); }

  function draw() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    ctx.translate(state.transform.x, state.transform.y);
    ctx.scale(state.transform.k, state.transform.k);
    const g = state.graph;
    if (!g) { ctx.restore(); return; }
    const focus = state.selected || state.hover;
    const neigh = focus ? focus._neighbors : null;
    if (state.showEdges) {
      ctx.lineWidth = 0.6 / state.transform.k;
      ctx.strokeStyle = "rgba(90,90,90,0.18)";
      ctx.beginPath();
      for (const e of g.links) {
        if (!nodeVisible(e.source) || !nodeVisible(e.target)) continue;
        if (focus && e.source !== focus && e.target !== focus) { ctx.moveTo(e.source.x, e.source.y); ctx.lineTo(e.target.x, e.target.y); }
        else if (!focus) { ctx.moveTo(e.source.x, e.source.y); ctx.lineTo(e.target.x, e.target.y); }
      }
      ctx.stroke();
      if (focus) {
        ctx.lineWidth = 1.6 / state.transform.k; ctx.strokeStyle = "#333"; ctx.beginPath();
        for (const e of g.links) if (e.source === focus || e.target === focus) { ctx.moveTo(e.source.x, e.source.y); ctx.lineTo(e.target.x, e.target.y); }
        ctx.stroke();
      }
    }
    for (const cls of ["unknown", "licit", "illicit"]) {  // draw illicit last so they stay on top
      ctx.fillStyle = COLORS[cls];
      ctx.beginPath();
      for (const n of g.nodes) {
        if (!nodeVisible(n) || nodeClass(n) !== cls) continue;
        const r = radius(n) / Math.sqrt(state.transform.k);
        ctx.moveTo(n.x + r, n.y); ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
      }
      ctx.globalAlpha = focus && neigh ? 0.35 : 1;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (focus && neigh) {
        ctx.beginPath();
        for (const n of g.nodes) {
          if (!nodeVisible(n) || nodeClass(n) !== cls || !(n === focus || neigh.has(n))) continue;
          const r = (radius(n) + 1) / Math.sqrt(state.transform.k);
          ctx.moveTo(n.x + r, n.y); ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
        }
        ctx.fill();
      }
    }
    if (focus) {
      ctx.lineWidth = 2 / state.transform.k; ctx.strokeStyle = "#000";
      ctx.beginPath(); ctx.arc(focus.x, focus.y, (radius(focus) + 2) / Math.sqrt(state.transform.k), 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.restore();
  }

  function layout(graph) {
    if (state.sim) state.sim.stop();
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const n = graph.nodes.length;
    const sim = d3.forceSimulation(graph.nodes)
      .force("link", d3.forceLink(graph.links).id((d) => d.id).distance(12).strength(0.6))
      .force("charge", d3.forceManyBody().strength(n > 3000 ? -6 : -15).theta(0.9).distanceMax(250))
      .force("center", d3.forceCenter(w / 2, h / 2))
      .force("collide", d3.forceCollide(3))
      .alphaDecay(n > 3000 ? 0.05 : 0.03)
      .stop();
    state.sim = sim;
    let ticks = 0;
    const maxTicks = n > 5000 ? 120 : 250;
    function frame() {
      if (sim.alpha() < 0.01 || ticks >= maxTicks) { status.textContent = statusText(); draw(); return; }
      for (let i = 0; i < 4 && ticks < maxTicks; i++, ticks++) sim.tick();
      status.textContent = `${statusText()} · layout ${Math.round(100 * ticks / maxTicks)}%`;
      draw();
      requestAnimationFrame(frame);
    }
    frame();
  }

  function statusText() {
    const g = state.graph; if (!g) return "";
    const c = { illicit: 0, licit: 0, unknown: 0 };
    for (const n of g.nodes) c[labelName(n.label)]++;
    return `step ${state.step}: ${g.nodes.length} nodes (${c.illicit} illicit, ${c.licit} licit, ${c.unknown} unknown), ${g.links.length} edges`;
  }

  async function loadStep(step) {
    status.textContent = `loading step ${step}…`;
    const data = await api(`/api/graph/${step}`);
    const nodes = data.nodes.map((n) => ({ ...n, _neighbors: new Set() }));
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const links = data.edges.map(([s, t]) => ({ source: byId.get(s), target: byId.get(t) }));
    for (const l of links) { l.source._neighbors.add(l.target); l.target._neighbors.add(l.source); }
    state.graph = { nodes, links }; state.nodeById = byId; state.hover = null;
    if (state.selected) state.selected = byId.get(state.selected.id) || null;
    state.transform = d3.zoomIdentity; d3.select(canvas).call(zoom.transform, d3.zoomIdentity);
    layout(state.graph);
  }

  function setStep(step) {
    step = Math.max(1, Math.min(49, step));
    state.step = step; stepInput.value = step;
    const r = state.meta.timesteps.find((t) => t.timestep === step);
    stepLabel.textContent = r ? `step ${step} (${r.split}) · ${r.illicit} illicit · ${r.licit} licit · ${r.unknown} unknown · ${r.edges} edges` : `step ${step}`;
    drawTimeline();
    loadStep(step);
  }

  // ---------------------------------------------------------------- interaction
  const zoom = d3.zoom().scaleExtent([0.2, 20]).on("zoom", (ev) => { state.transform = ev.transform; draw(); });
  d3.select(canvas).call(zoom);

  function pick(ev) {
    if (!state.graph) return null;
    const [mx, my] = d3.pointer(ev, canvas);
    const [x, y] = state.transform.invert([mx, my]);
    let best = null, bd = (8 / state.transform.k) ** 2;
    for (const n of state.graph.nodes) {
      if (!nodeVisible(n)) continue;
      const d = (n.x - x) ** 2 + (n.y - y) ** 2;
      if (d < bd) { bd = d; best = n; }
    }
    return best;
  }
  canvas.addEventListener("mousemove", (ev) => {
    const n = pick(ev);
    if (n !== state.hover) { state.hover = n; draw(); }
    if (n) {
      const scores = Object.keys(state.meta.models).map((m) => `${m}: ${n[`s_${m}`]?.toFixed(3)}`).join(" · ");
      tooltip.textContent = `tx ${n.id} · ${labelName(n.label)} · in ${n.in} / out ${n.out}${scores ? " · " + scores : ""}`;
      tooltip.style.left = `${ev.offsetX + 12}px`; tooltip.style.top = `${ev.offsetY + 12}px`; tooltip.hidden = false;
    } else tooltip.hidden = true;
  });
  canvas.addEventListener("mouseleave", () => { tooltip.hidden = true; state.hover = null; draw(); });
  canvas.addEventListener("click", (ev) => { const n = pick(ev); state.selected = n; draw(); if (n) showNode(n.id); });

  async function showNode(txId) {
    const d = await api(`/api/node/${txId}`);
    const panel = document.getElementById("node-details");
    document.querySelector("#panel .hint").hidden = true;
    const models = Object.keys(d.scores);
    const scoreRows = models.map((m) => {
      const s = d.scores[m], thr = state.meta.models[m].threshold;
      return `<tr><td>${m}</td><td><div class="bar ${d.flags[m] ? "pos" : ""}" style="width:${Math.round(100 * s)}%"></div>${s.toFixed(3)} (threshold ${thr.toFixed(2)}) → ${d.flags[m] ? "flag" : "ok"}</td></tr>`;
    }).join("");
    const nb = (ids) => ids.length ? `<div class="nb-list">${ids.map((i) => { const n = state.nodeById.get(i); return `<span class="nb ${n ? labelName(n.label) : ""}" data-id="${i}">${i}</span>`; }).join(" ")}</div>` : "<span class='hint'>none</span>";
    const feats = Object.entries(d.features);
    const topFeats = feats.map(([k, v]) => [k, v]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 12);
    panel.innerHTML = `
      <h2>tx ${d.tx_id} <span class="tag ${d.label_name}">${d.label_name}</span></h2>
      <div class="hint">time step ${d.timestep} (${d.split})</div>
      <h3>Model scores</h3>
      ${models.length ? `<table class="kv">${scoreRows}</table>` : "<span class='hint'>no models loaded (pass --artifacts)</span>"}
      ${models.length ? `<div style="margin-top:6px">${models.map((m) => `<button data-explain="${m}">explain with ${m}</button>`).join(" ")}</div><div id="explanation"></div>` : ""}
      <h3>Inputs (${d.in_neighbors.length})</h3><div>${nb(d.in_neighbors)}</div>
      <h3>Outputs (${d.out_neighbors.length})</h3><div>${nb(d.out_neighbors)}</div>
      <h3>Largest features (of ${feats.length})</h3>
      <table class="kv">${topFeats.map(([k, v]) => `<tr><td>${k}</td><td>${v.toFixed(3)}</td></tr>`).join("")}</table>`;
    panel.hidden = false;
    panel.querySelectorAll(".nb").forEach((el) => el.addEventListener("click", () => {
      const n = state.nodeById.get(+el.dataset.id);
      if (n) { state.selected = n; draw(); showNode(n.id); }
    }));
    panel.querySelectorAll("[data-explain]").forEach((btn) => btn.addEventListener("click", () => explain(d.tx_id, btn.dataset.explain)));
  }

  async function explain(txId, model) {
    const box = document.getElementById("explanation");
    box.innerHTML = `<span class="hint">running ${model === "graphsage" ? "GNNExplainer" : "SHAP"}…</span>`;
    try {
      const e = await api(`/api/explain/${txId}?model=${model}`);
      if (model === "xgboost") {
        box.innerHTML = `<h3>SHAP (log-odds), p=${e.probability.toFixed(3)}</h3><table class="kv">${e.top_features.map((f) =>
          `<tr><td>${f.feature}</td><td>${f.shap >= 0 ? "+" : ""}${f.shap.toFixed(3)} <span class="hint">(value ${f.value.toFixed(2)})</span></td></tr>`).join("")}</table>`;
      } else {
        box.innerHTML = `<h3>GNNExplainer, p=${e.illicit_probability.toFixed(3)}</h3>
          <div class="hint">${e.subgraph.num_nodes}-node computation subgraph</div>
          <table class="kv">${e.top_features.slice(0, 8).map((f) => `<tr><td>${f.feature}</td><td>mask ${f.mask.toFixed(2)}</td></tr>`).join("")}</table>
          <h3>Influential edges</h3>
          <table class="kv">${e.top_edges.filter((x) => !x.self_loop).slice(0, 8).map((x) =>
            `<tr><td>${x.weight.toFixed(2)}</td><td><span class="nb ${labelName(x.src_label)}" data-id="${x.src_tx_id}">${x.src_tx_id}</span> → <span class="nb ${labelName(x.dst_label)}" data-id="${x.dst_tx_id}">${x.dst_tx_id}</span></td></tr>`).join("")}</table>`;
        box.querySelectorAll(".nb").forEach((el) => el.addEventListener("click", () => {
          const n = state.nodeById.get(+el.dataset.id); if (n) { state.selected = n; draw(); showNode(n.id); }
        }));
      }
    } catch (err) { box.innerHTML = `<span class="hint">failed: ${err.message}</span>`; }
  }

  // ---------------------------------------------------------------- wiring
  stepInput.addEventListener("input", () => { stepLabel.textContent = `step ${stepInput.value}`; });
  stepInput.addEventListener("change", () => setStep(+stepInput.value));
  document.getElementById("prev").addEventListener("click", () => setStep(state.step - 1));
  document.getElementById("next").addEventListener("click", () => setStep(state.step + 1));
  document.addEventListener("keydown", (ev) => { if (ev.key === "ArrowLeft") setStep(state.step - 1); if (ev.key === "ArrowRight") setStep(state.step + 1); });
  modeSel.addEventListener("change", () => { state.mode = modeSel.value; draw(); });
  document.getElementById("show-unknown").addEventListener("change", (ev) => { state.showUnknown = ev.target.checked; draw(); });
  document.getElementById("show-edges").addEventListener("change", (ev) => { state.showEdges = ev.target.checked; draw(); });
  window.addEventListener("resize", () => { resizeCanvas(); if (state.meta) drawTimeline(); });

  api("/api/meta").then((meta) => {
    state.meta = meta;
    for (const m of Object.keys(meta.models)) {
      const o = document.createElement("option"); o.value = m; o.textContent = `${m} prediction`; modeSel.appendChild(o);
    }
    resizeCanvas();
    setStep(+(new URLSearchParams(location.search).get("step") || 1));
  }).catch((err) => { status.textContent = `failed to load /api/meta: ${err.message}`; });
})();
