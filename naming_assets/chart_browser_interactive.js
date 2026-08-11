
(function () {
  const P = { model:0, sid:1, scale:2, comp:3, token:4, code:5, z1:6, z2:7, q1:8, q2:9, pooled:10 };
  const W = { model:0, sid:1, scale:2, token:3, progress:4, anchor:5, phase:6, sessionPhase:7, mutual:8, nearest:9, behaviors:10 };
  const C = { model:0, scale:1, comp:2, code:3, active:4, z1:5, z2:6, pooled:7 };
  const R = { model:0, scale:1, comp:2, code:3, count:4, means:5, modalityMeans:6, zMeans:7, modalityZ:8 };
    const IMPORTANT_Z = 1.0;
    const IMPORTANT_RANK_MIN_Z = 1e-6;
  const palette = ["#d73027","#fdae61","#fee08b","#1a9850","#4575b4","#984ea3","#ff7f00","#4daf4a","#377eb8","#e41a1c","#a65628","#f781bf","#999999","#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494"];
  const modColors = ["#d73027","#fdae61","#fee08b","#1a9850","#4575b4","#984ea3"];
  function $(id) { return document.getElementById(id); }
  function fmt(v, n = 2) { return Number.isFinite(Number(v)) ? Number(v).toFixed(n) : "n/a"; }
  function key(...parts) { return parts.join("|"); }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function init() {
    const data = window.INTERACTIVE_EMBEDDINGS;
    if (!data) return;
    const assets = window.ASSETS || (() => {
      try { return JSON.parse(document.getElementById("assets-json")?.textContent || "[]"); }
      catch (_) { return []; }
    })();
    const canvas = $("embedCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const controls = {
      model: $("embedModel"),
      preset: $("embedPreset"),
      scale: $("embedScale"),
      xComp: $("embedXComponent"),
      yComp: $("embedYComponent"),
      topologyMode: $("embedTopologyMode"),
      topologyContext: $("embedTopologyContext"),
      topologyContextReadout: $("embedTopologyContextReadout"),
      codeUsageContext: $("embedCodeUsageContext"),
      codeUsageContextReadout: $("embedCodeUsageContextReadout"),
      codeUsageStack: $("embedCodeUsageStack"),
      transitionMode: $("embedTransitionMode"),
      transitionContext: $("embedTransitionContext"),
      transitionContextReadout: $("embedTransitionContextReadout"),
      color: $("embedColor"),
      showBg: $("embedShowBg"),
      latentPositions: $("embedLatentPositions"),
      showCodes: $("embedShowCodes") || { checked:true, addEventListener:() => {} },
      showHeatmap: $("embedShowHeatmap"),
      showBarChart: $("embedShowBarChart"),
      showTransitions: $("embedShowTransitions"),
      showTopology: $("embedShowTopology"),
      showNamingStars: $("embedShowNamingStars"),
      showKeypoints: $("embedShowKeypoints"),
      showTrails: $("embedShowTrails"),
      tailLength: $("embedTailLength"),
      windowsPerSecond: $("embedWindowsPerSecond"),
      zoomIn: $("embedZoomIn"),
      zoomOut: $("embedZoomOut"),
      tailReadout: $("embedTailReadout"),
      selectedSessionsActive: $("embedSelectedSessionsActive"),
      resetSelectedSessions: $("embedResetSelectedSessions"),
      filterSessions: $("embedFilterSessions"),
      highlightSessions: $("embedHighlightSessions"),
      progress: $("embedProgress"),
      play: $("embedPlay"),
      export: $("embedExport"),
      reset: $("embedResetView"),
    };
    const status = $("embedStatus");
    const detail = $("embedDetail");
    const bars = $("embedBars");
    const sessionList = $("embedSessionList");
    const codeProfile = $("embedCodeProfile");
    const tooltip = $("embedTooltip");
    const hoverPanel = $("embedHoverPanel");
    const pending = $("embedPending");
    const legend = $("embedLegendOverlay");
    const relatedPanel = $("relatedPanel");
    const relatedList = $("relatedList");
    const relatedCount = $("relatedCount");
    const relatedToggle = $("relatedToggle");
    const relatedSearch = $("relatedSearch");
    const relatedSimilar = $("relatedSimilar");
    const relatedAll = $("relatedAll");
    const relatedTitleText = relatedPanel?.querySelector(".relatedTitle b");
    const relatedControls = relatedPanel?.querySelector(".relatedControls");
    const sessionByKey = new Map(data.sessions.map(s => [key(s.model, s.sid), s]));
    const pointByExact = new Map();
    const pointsByMSC = new Map();
    const windowsByExact = new Map();
    const windowsBySession = new Map();
    const codebooksByMSC = new Map();
    const profileByMSCCode = new Map();
    const profileByMCCode = new Map();
    let currentDrawn = [];
    let currentTailHits = [];
    let currentCodeMarks = [];
    let currentTransitionHits = [];
    let currentCanvasMeta = "";
    let hoveredSessionKey = null;
    let viewKey = "";
    let viewEx = null;
    let playing = false;
    let timer = null;
    let lastFrameMs = 0;
    let playProgress = null;
    let dragging = false;
    let lastDrag = null;
    let sessionRows = [];
    let relatedMode = "similar";
    let bootingUrlState = true;
    let pendingUrlSync = null;
    let lastUrlSyncMs = 0;

    for (const p of data.points) {
      pointByExact.set(key(p[P.model], p[P.sid], p[P.scale], p[P.comp], p[P.token]), p);
      const k = key(p[P.model], p[P.scale], p[P.comp]);
      if (!pointsByMSC.has(k)) pointsByMSC.set(k, []);
      pointsByMSC.get(k).push(p);
    }
    for (const w of data.windows) {
      windowsByExact.set(key(w[W.model], w[W.sid], w[W.scale], w[W.token]), w);
      const k = key(w[W.model], w[W.sid], w[W.scale]);
      if (!windowsBySession.has(k)) windowsBySession.set(k, []);
      windowsBySession.get(k).push(w);
    }
    for (const arr of windowsBySession.values()) arr.sort((a,b) => a[W.progress] - b[W.progress]);
    for (const c of data.codebooks) {
      const k = key(c[C.model], c[C.scale], c[C.comp]);
      if (!codebooksByMSC.has(k)) codebooksByMSC.set(k, []);
      codebooksByMSC.get(k).push(c);
    }
    function combineProfiles(rows) {
      if (!rows.length) return null;
      const nBeh = rows[0][R.means].length, nMod = rows[0][R.modalityMeans].length;
      const beh = Array(nBeh).fill(0), mods = Array(nMod).fill(0), z = Array(nBeh).fill(0), modZ = Array(nMod).fill(0);
      let total = 0;
      for (const r of rows) {
        const w = Math.max(0, Number(r[R.count]) || 0);
        total += w;
        for (let i = 0; i < nBeh; i++) beh[i] += (Number(r[R.means][i]) || 0) * w;
        for (let i = 0; i < nMod; i++) mods[i] += (Number(r[R.modalityMeans][i]) || 0) * w;
        for (let i = 0; i < nBeh; i++) z[i] += (Number((r[R.zMeans] || [])[i]) || 0) * w;
        for (let i = 0; i < nMod; i++) modZ[i] += (Number((r[R.modalityZ] || [])[i]) || 0) * w;
      }
      const denom = total || rows.length;
      return [rows[0][R.model], -1, rows[0][R.comp], rows[0][R.code], total, beh.map(v => v / denom), mods.map(v => v / denom), z.map(v => v / denom), modZ.map(v => v / denom)];
    }
    for (const r of data.codeProfiles) {
      profileByMSCCode.set(key(r[R.model], r[R.scale], r[R.comp], r[R.code]), r);
      const aggKey = key(r[R.model], r[R.comp], r[R.code]);
      if (!profileByMCCode.has(aggKey)) profileByMCCode.set(aggKey, []);
      profileByMCCode.get(aggKey).push(r);
    }
    for (const [aggKey, rows] of [...profileByMCCode.entries()]) profileByMCCode.set(aggKey, combineProfiles(rows));

    function options(vals, labels) {
      return vals.map((v, i) => `<option value="${esc(v)}">${esc(labels ? labels[i] : v)}</option>`).join("");
    }
    function selectedMulti(select) {
      return new Set([...select.selectedOptions].map(o => o.value));
    }
    function checkboxValues(id) {
      return new Set([...document.querySelectorAll(`#${id} input[type=checkbox]:checked`)].map(x => x.value));
    }
    function checkboxAllValues(id) {
      return new Set([...document.querySelectorAll(`#${id} input[type=checkbox]`)].map(x => x.value));
    }
    function checkboxState(id) {
      return [...document.querySelectorAll(`#${id} input[type=checkbox]:checked`)].map(x => x.value);
    }
    function setCheckboxState(id, values) {
      if (!Array.isArray(values)) return;
      const wanted = new Set(values.map(String));
      document.querySelectorAll(`#${id} input[type=checkbox]`).forEach(x => { x.checked = wanted.has(String(x.value)); });
    }
    function setIfOption(select, value) {
      if (!select || value === undefined || value === null) return false;
      const v = String(value);
      if (![...select.options].some(o => o.value === v)) return false;
      select.value = v;
      return true;
    }
    function fillChecks(id, values, labelMap = {}) {
      const node = $(id);
      node.innerHTML = values.map(v => `<label><input type="checkbox" value="${esc(v)}" checked> ${esc(labelMap[v] || v)}</label>`).join("");
      node.querySelectorAll("input").forEach(x => x.addEventListener("change", () => { refreshSessionPickers(); draw(); }));
    }
    function fillCompareChecks(id, values, labelMap = {}, selected = null) {
      const node = $(id);
      if (!node) return;
      const wanted = selected ? new Set(selected) : null;
      node.innerHTML = values.map(v => {
        const checked = wanted ? wanted.has(v) : true;
        return `<label><input type="checkbox" value="${esc(v)}" ${checked ? "checked" : ""}> ${esc(labelMap[v] || v)}</label>`;
      }).join("");
      node.querySelectorAll("input").forEach(x => x.addEventListener("change", draw));
    }
    function fillSessionChecks(id, rows, defaultChecked = true) {
      const node = $(id);
      if (!node) return;
      const existing = [...node.querySelectorAll("input[type=checkbox]")];
      const hadExisting = existing.length > 0;
      const selected = new Set(existing.filter(x => x.checked).map(x => x.value));
      node.innerHTML = rows.map(s => {
        const v = sessionBaseKey(s);
        const checked = hadExisting ? selected.has(v) : defaultChecked;
        const label = `${s.participant} ${s.session} ${s.language} ${s.namingGroup}`;
        return `<label><input type="checkbox" value="${esc(v)}" ${checked ? "checked" : ""}> ${esc(label)}</label>`;
      }).join("");
      node.querySelectorAll("input").forEach(x => x.addEventListener("change", draw));
    }
    function compareFilters(prefix) {
      return {
        languages: checkboxValues(`${prefix}_language`),
        hearings: checkboxValues(`${prefix}_hearing`),
        sessions: checkboxValues(`${prefix}_session`),
        naming: checkboxValues(`${prefix}_naming`),
        phases: checkboxValues(`${prefix}_proximity`),
        sessionPhases: checkboxValues(`${prefix}_sessionPhase`),
        mutualAttention: checkboxValues(`${prefix}_mutualAttention`),
      };
    }
    function compareState(prefix) {
      return {
        l: checkboxState(`${prefix}_language`),
        a: checkboxState(`${prefix}_hearing`),
        s: checkboxState(`${prefix}_session`),
        n: checkboxState(`${prefix}_naming`),
        p: checkboxState(`${prefix}_proximity`),
        ph: checkboxState(`${prefix}_sessionPhase`),
        ma: checkboxState(`${prefix}_mutualAttention`),
      };
    }
    function applyCompareState(prefix, st) {
      if (!st) return;
      setCheckboxState(`${prefix}_language`, st.l);
      setCheckboxState(`${prefix}_hearing`, st.a);
      setCheckboxState(`${prefix}_session`, st.s);
      setCheckboxState(`${prefix}_naming`, st.n);
      setCheckboxState(`${prefix}_proximity`, st.p);
      setCheckboxState(`${prefix}_sessionPhase`, st.ph);
      setCheckboxState(`${prefix}_mutualAttention`, st.ma);
    }
    function encodeUrlState(obj) {
      const bytes = new TextEncoder().encode(JSON.stringify(obj));
      let bin = "";
      bytes.forEach(b => { bin += String.fromCharCode(b); });
      return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
    }
    function decodeUrlState(raw) {
      try {
        let txt = String(raw || "").replace(/-/g, "+").replace(/_/g, "/");
        while (txt.length % 4) txt += "=";
        const bin = atob(txt);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return JSON.parse(new TextDecoder("utf-8").decode(bytes));
      } catch (_) {
        return null;
      }
    }
    function collectUrlState() {
      const st = {
        m: controls.model.value,
        sc: controls.scale.value,
        x: controls.xComp.value,
        y: controls.yComp.value,
        pr: controls.progress.value,
        bg: controls.showBg.checked ? 1 : 0,
        lp: controls.latentPositions.checked ? 1 : 0,
        dots: controls.showKeypoints.checked ? 1 : 0,
        trails: controls.showTrails.checked ? 1 : 0,
        stars: controls.showNamingStars.checked ? 1 : 0,
        topo: controls.showTopology.checked ? 1 : 0,
        trans: controls.showTransitions.checked ? 1 : 0,
        color: controls.color.value,
        tail: controls.tailLength.value,
        cuCtx: controls.codeUsageContext ? controls.codeUsageContext.value : "1000",
        cuStack: controls.codeUsageStack ? controls.codeUsageStack.value : "none",
        topCtx: controls.topologyContext.value,
        topMode: controls.topologyMode.value,
        trCtx: controls.transitionContext.value,
        trMode: controls.transitionMode.value,
        selActive: controls.selectedSessionsActive?.checked ? 1 : 0,
        l: checkboxState("embedLanguageChecks"),
        a: checkboxState("embedHearingChecks"),
        se: checkboxState("embedSessionChecks"),
        n: checkboxState("embedNamingChecks"),
        prox: checkboxState("embedProximityChecks"),
        ph: checkboxState("embedSessionPhaseChecks"),
        ma: checkboxState("embedMutualAttentionChecks"),
        tA: compareState("topoA"),
        tB: compareState("topoB"),
        rA: compareState("transA"),
        rB: compareState("transB"),
      };
      if (controls.selectedSessionsActive?.checked) {
        st.fs = checkboxState("embedFilterSessions");
        st.hs = checkboxState("embedHighlightSessions");
      }
      return st;
    }
    function syncUrlStateNow() {
      if (bootingUrlState) return;
      const url = new URL(window.location.href);
      url.searchParams.set("state", encodeUrlState(collectUrlState()));
      window.history.replaceState(null, "", url.pathname + url.search + url.hash);
      lastUrlSyncMs = Date.now();
    }
    function scheduleUrlStateSync() {
      if (bootingUrlState || !window.history || !window.URL) return;
      const elapsed = Date.now() - lastUrlSyncMs;
      if (elapsed > 500) {
        syncUrlStateNow();
        return;
      }
      if (pendingUrlSync) return;
      pendingUrlSync = setTimeout(() => {
        pendingUrlSync = null;
        syncUrlStateNow();
      }, 520 - elapsed);
    }
    function applyUrlState() {
      const st = decodeUrlState(new URLSearchParams(window.location.search).get("state"));
      if (!st) return false;
      if (setIfOption(controls.model, st.m)) fillScaleControls();
      if (setIfOption(controls.scale, st.sc)) fillAxisControls();
      setIfOption(controls.xComp, st.x);
      setIfOption(controls.yComp, st.y);
      if (st.pr !== undefined) controls.progress.value = String(st.pr);
      if (st.bg !== undefined) controls.showBg.checked = !!Number(st.bg);
      if (st.lp !== undefined) controls.latentPositions.checked = !!Number(st.lp);
      if (st.dots !== undefined) controls.showKeypoints.checked = !!Number(st.dots);
      if (st.trails !== undefined) controls.showTrails.checked = !!Number(st.trails);
      if (st.stars !== undefined) controls.showNamingStars.checked = !!Number(st.stars);
      if (st.topo !== undefined) controls.showTopology.checked = !!Number(st.topo);
      if (st.trans !== undefined) controls.showTransitions.checked = !!Number(st.trans);
      setIfOption(controls.color, st.color);
      if (st.tail !== undefined) controls.tailLength.value = String(st.tail);
      if (controls.codeUsageContext && st.cuCtx !== undefined) controls.codeUsageContext.value = String(st.cuCtx);
      setIfOption(controls.codeUsageStack, st.cuStack);
      if (st.topCtx !== undefined) controls.topologyContext.value = String(st.topCtx);
      setIfOption(controls.topologyMode, st.topMode);
      if (st.trCtx !== undefined) controls.transitionContext.value = String(st.trCtx);
      setIfOption(controls.transitionMode, st.trMode);
      if (controls.selectedSessionsActive && st.selActive !== undefined) controls.selectedSessionsActive.checked = !!Number(st.selActive);
      setCheckboxState("embedLanguageChecks", st.l);
      setCheckboxState("embedHearingChecks", st.a);
      setCheckboxState("embedSessionChecks", st.se);
      setCheckboxState("embedNamingChecks", st.n);
      setCheckboxState("embedProximityChecks", st.prox);
      setCheckboxState("embedSessionPhaseChecks", st.ph);
      setCheckboxState("embedMutualAttentionChecks", st.ma);
      applyCompareState("topoA", st.tA);
      applyCompareState("topoB", st.tB);
      applyCompareState("transA", st.rA);
      applyCompareState("transB", st.rB);
      refreshSessionPickers();
      setCheckboxState("embedFilterSessions", st.fs);
      setCheckboxState("embedHighlightSessions", st.hs);
      return true;
    }
    function inferMapMode(f) {
      if (f.showCodeUsage) return "code_usage_chart";
      if (f.showHeatmap) return "heatmap_chart";
      if (f.showBarChart) return "code_decomp";
      if (!f.latentPositions) {
        if (!f.showTransitions && !f.showKeypoints && !f.showTrails && !f.showTopology && !f.showNamingStars) return "code_decomp";
        return "code_layout";
      }
      return "latent";
    }
    function inferBgMode(f) {
      if (f.mapMode === "code_decomp" || f.mapMode === "code_layout" || f.mapMode === "heatmap_chart" || f.mapMode === "code_usage_chart") return "none";
      return f.showBg ? "code" : "none";
    }
    function tailFractionFromSlider() {
      const raw = clamp(Number(controls.tailLength.value || 0), 0, 1000);
      if (raw <= 800) return raw / 800 * 0.10;
      return 0.10 + ((raw - 800) / 200) ** 2 * 0.40;
    }
    function contextFractionFromSlider(control, readout) {
      const raw = clamp(Number(control?.value || 0), 0, 1000);
      const frac = raw / 1000;
      if (readout) {
        if (frac <= 0.001) readout.textContent = "current frame";
        else if (frac >= 0.999) readout.textContent = "full session";
        else readout.textContent = `${fmt(frac * 100, 1)}% session`;
      }
      return frac;
    }
    function modelScales(modelIdx) {
      const m = data.models.find(x => x.idx === modelIdx);
      return m ? m.scales.map(s => data.scales.indexOf(s)).filter(i => i >= 0) : [];
    }
    function availableComponents(modelIdx, scaleIdx) {
      const comps = new Set();
      for (const p of data.points) if (p[P.model] === modelIdx && p[P.scale] === scaleIdx) comps.add(p[P.comp]);
      return [...comps].sort((a,b) => data.components[a] === "joint" ? -1 : data.components[b] === "joint" ? 1 : data.components[a].localeCompare(data.components[b]));
    }
    function fillModelControls() {
      controls.model.innerHTML = options(data.models.map(m => m.idx), data.models.map(m => m.label));
      if (data.pendingModels && data.pendingModels.length) {
        pending.innerHTML = data.pendingModels.map(m => `<b>${esc(m.label)}</b> pending: ${esc(m.reason)} (${esc(m.path)})`).join("<br>");
      } else {
        pending.innerHTML = "";
      }
      fillScaleControls();
    }
    function fillPresetControls() {
      if (!controls.preset) return;
      const presets = (data.highLowPresets || []).filter(p => Number(p.model ?? 0) === Number(controls.model.value || 0)).slice(0, 9);
      controls.preset.innerHTML = `<option value="">Alignment preset</option>` + presets.map(p => `<option value="${esc(p.rank)}">${esc(p.label || `Preset ${p.rank}`)}</option>`).join("");
    }
    function fillScaleControls() {
      const modelIdx = Number(controls.model.value || 0);
      const oldScale = controls.scale.value;
      const scales = modelScales(modelIdx);
      controls.scale.innerHTML = options(scales, scales.map(i => data.scaleLabels[i]));
      if (oldScale !== "" && scales.includes(Number(oldScale))) controls.scale.value = oldScale;
      else if (scales.length) {
        const s5 = scales.find(i => data.scales[i] === "s5");
        controls.scale.value = String(s5 !== undefined ? s5 : scales[0]);
      }
      fillAxisControls();
      fillPresetControls();
    }
    function fillAxisControls() {
      const modelIdx = Number(controls.model.value || 0);
      const scaleIdx = Number(controls.scale.value || 0);
      const oldX = controls.xComp.value, oldY = controls.yComp.value;
      const comps = availableComponents(modelIdx, scaleIdx);
      controls.xComp.innerHTML = options(comps, comps.map(i => data.components[i]));
      controls.yComp.innerHTML = options(comps, comps.map(i => data.components[i]));
      if (oldX !== "" && comps.includes(Number(oldX))) controls.xComp.value = oldX;
      else {
        const joint = comps.find(i => data.components[i] === "joint");
        controls.xComp.value = String(joint !== undefined ? joint : comps[0]);
      }
      if (oldY !== "" && comps.includes(Number(oldY))) controls.yComp.value = oldY;
      else {
        const joint = comps.find(i => data.components[i] === "joint");
        controls.yComp.value = String(joint !== undefined ? joint : comps[0]);
      }
      refreshSessionPickers();
      draw();
    }
    function sessionBaseKey(s) {
      return `${s.participant}|${s.session}|${s.language}`;
    }
    function fillSessionControls() {
      const seenRows = new Map();
      for (const s of data.sessions) {
        const uniqueKey = sessionBaseKey(s);
        if (!seenRows.has(uniqueKey)) seenRows.set(uniqueKey, s);
      }
      sessionRows = [...seenRows.values()].sort((a,b) => `${a.participant} ${a.session}`.localeCompare(`${b.participant} ${b.session}`));
      fillChecks("embedLanguageChecks", [...new Set(sessionRows.map(s => s.language).filter(Boolean))].sort());
      fillChecks("embedHearingChecks", [...new Set(sessionRows.map(s => s.hearing).filter(Boolean))].sort(), {"Hearing":"Hearing","Deaf":"Deaf","Comparison":"Comparison","Unknown":"Unknown"});
      fillChecks("embedSessionChecks", [...new Set(sessionRows.map(s => s.session).filter(Boolean))].sort());
      fillChecks("embedNamingChecks", [...new Set(sessionRows.map(s => s.namingGroup).filter(Boolean))].sort(), {low:"low naming", high:"high naming"});
      fillChecks("embedProximityChecks", data.phases, {far:"far", before:"before", during:"during", after:"after"});
      setCheckGroupExact("embedProximityChecks", ["during"]);
      fillChecks("embedSessionPhaseChecks", data.sessionPhases || ["early", "middle", "late"], {early:"early", middle:"middle", late:"late"});
      fillChecks("embedMutualAttentionChecks", data.mutualAttentionLevels || [], {
        "none":"none",
        "object-aligned":"object-aligned",
        "person-aligned":"person-aligned",
        "coordinated joint attention":"coordinated JA",
        "naming-aligned joint attention":"naming-aligned JA",
      });
      const langs = [...new Set(sessionRows.map(s => s.language).filter(Boolean))].sort();
      const hear = [...new Set(sessionRows.map(s => s.hearing).filter(Boolean))].sort();
      const sess = [...new Set(sessionRows.map(s => s.session).filter(Boolean))].sort();
      const naming = [...new Set(sessionRows.map(s => s.namingGroup).filter(Boolean))].sort();
      for (const prefix of ["topoA", "transA"]) {
        fillCompareChecks(`${prefix}_language`, langs);
        fillCompareChecks(`${prefix}_hearing`, hear, {"Hearing":"Hearing","Deaf":"Deaf","Comparison":"Comparison","Unknown":"Unknown"});
        fillCompareChecks(`${prefix}_session`, sess);
        fillCompareChecks(`${prefix}_naming`, naming, {low:"low naming", high:"high naming"}, ["low"]);
        fillCompareChecks(`${prefix}_proximity`, data.phases, {far:"far", before:"before", during:"during", after:"after"}, ["during"]);
        fillCompareChecks(`${prefix}_sessionPhase`, data.sessionPhases || ["early", "middle", "late"], {early:"early", middle:"middle", late:"late"});
        fillCompareChecks(`${prefix}_mutualAttention`, data.mutualAttentionLevels || [], {
          "none":"none",
          "object-aligned":"object",
          "person-aligned":"person",
          "coordinated joint attention":"coordinated JA",
          "naming-aligned joint attention":"naming JA",
        });
      }
      for (const prefix of ["topoB", "transB"]) {
        fillCompareChecks(`${prefix}_language`, langs);
        fillCompareChecks(`${prefix}_hearing`, hear, {"Hearing":"Hearing","Deaf":"Deaf","Comparison":"Comparison","Unknown":"Unknown"});
        fillCompareChecks(`${prefix}_session`, sess);
        fillCompareChecks(`${prefix}_naming`, naming, {low:"low naming", high:"high naming"}, ["high"]);
        fillCompareChecks(`${prefix}_proximity`, data.phases, {far:"far", before:"before", during:"during", after:"after"}, ["during"]);
        fillCompareChecks(`${prefix}_sessionPhase`, data.sessionPhases || ["early", "middle", "late"], {early:"early", middle:"middle", late:"late"});
        fillCompareChecks(`${prefix}_mutualAttention`, data.mutualAttentionLevels || [], {
          "none":"none",
          "object-aligned":"object",
          "person-aligned":"person",
          "coordinated joint attention":"coordinated JA",
          "naming-aligned joint attention":"naming JA",
        });
      }
      refreshSessionPickers();
      refreshOpenAccordions();
    }
    function filters() {
      const tail = tailFractionFromSlider();
      if (controls.tailReadout) controls.tailReadout.textContent = `${fmt(tail * 100, 2)}% session`;
      const spatialLayerActive = controls.showTransitions.checked || controls.showKeypoints.checked || controls.showTrails.checked || controls.showTopology.checked || controls.showNamingStars.checked;
      const latentOn = controls.latentPositions ? controls.latentPositions.checked : true;
      const bgOn = controls.showBg ? controls.showBg.checked : true;
      const inferredCodeUsage = !spatialLayerActive && bgOn && !latentOn;
      const inferredBarChart = !spatialLayerActive && !bgOn && !latentOn;
      const out = {
        model: Number(controls.model.value || 0),
        scale: Number(controls.scale.value || 0),
        xComp: Number(controls.xComp.value || 0),
        yComp: Number(controls.yComp.value || 0),
        topologyMode: controls.topologyMode.value,
        topologyContext: contextFractionFromSlider(controls.topologyContext, controls.topologyContextReadout),
        codeUsageContext: contextFractionFromSlider(controls.codeUsageContext, controls.codeUsageContextReadout),
        codeUsageStack: controls.codeUsageStack ? controls.codeUsageStack.value : "none",
        transitionMode: controls.transitionMode.value,
        transitionContext: contextFractionFromSlider(controls.transitionContext, controls.transitionContextReadout),
        color: controls.color.value,
        showBg: bgOn,
        showCodes: true,
        latentPositions: latentOn,
        showHeatmap: false,
        showCodeUsage: inferredCodeUsage,
        showBarChart: inferredBarChart,
        showTransitions: controls.showTransitions.checked,
        showTopology: controls.showTopology.checked,
        showNamingStars: controls.showNamingStars.checked,
        showKeypoints: controls.showKeypoints.checked,
        showTrails: controls.showTrails.checked,
        tail,
        windowsPerSecond: Math.max(0.1, Number(controls.windowsPerSecond.value || 1)),
        progress: Number(controls.progress.value || 0) / 10000,
        selectedSessionsActive: controls.selectedSessionsActive ? controls.selectedSessionsActive.checked : false,
        filterSessions: checkboxValues("embedFilterSessions"),
        filterSessionUniverse: checkboxAllValues("embedFilterSessions"),
        highlightSessions: checkboxValues("embedHighlightSessions"),
        languages: checkboxValues("embedLanguageChecks"),
        hearings: checkboxValues("embedHearingChecks"),
        sessions: checkboxValues("embedSessionChecks"),
        naming: checkboxValues("embedNamingChecks"),
        phases: checkboxValues("embedProximityChecks"),
        sessionPhases: checkboxValues("embedSessionPhaseChecks"),
        mutualAttention: checkboxValues("embedMutualAttentionChecks"),
        topologyA: compareFilters("topoA"),
        topologyB: compareFilters("topoB"),
        transitionA: compareFilters("transA"),
        transitionB: compareFilters("transB"),
      };
      out.mapMode = inferMapMode(out);
      out.bgMode = inferBgMode(out);
      return out;
    }
    function sessionPass(s, f) {
      if (!s || s.model !== f.model) return false;
      if (f.selectedSessionsActive && f.filterSessionUniverse.size && !f.filterSessions.has(sessionBaseKey(s))) return false;
      if (f.languages.size && !f.languages.has(s.language)) return false;
      if (f.hearings.size && !f.hearings.has(s.hearing)) return false;
      if (f.sessions.size && !f.sessions.has(s.session)) return false;
      if (f.naming.size && !f.naming.has(s.namingGroup)) return false;
      return true;
    }
    function sessionMetaPass(s, f) {
      if (!s || s.model !== f.model) return false;
      if (f.languages.size && !f.languages.has(s.language)) return false;
      if (f.hearings.size && !f.hearings.has(s.hearing)) return false;
      if (f.sessions.size && !f.sessions.has(s.session)) return false;
      if (f.naming.size && !f.naming.has(s.namingGroup)) return false;
      return true;
    }
    function comparePointPass(p, cmp) {
      const s = sessionByKey.get(key(p.model, p.sid));
      if (!s) return false;
      if (cmp.languages.size && !cmp.languages.has(s.language)) return false;
      if (cmp.hearings.size && !cmp.hearings.has(s.hearing)) return false;
      if (cmp.sessions.size && !cmp.sessions.has(s.session)) return false;
      if (cmp.naming.size && !cmp.naming.has(s.namingGroup)) return false;
      if (cmp.phases.size && p.window && !cmp.phases.has(data.phases[p.window[W.phase]])) return false;
      if (cmp.sessionPhases.size && p.window && !cmp.sessionPhases.has(data.sessionPhases[p.window[W.sessionPhase]])) return false;
      if (cmp.mutualAttention.size && p.window && !cmp.mutualAttention.has(data.mutualAttentionLevels[p.window[W.mutual]])) return false;
      return true;
    }
    function refreshSessionPickers() {
      if (!controls.filterSessions || !controls.highlightSessions) return;
      const f = filters();
      const rows = sessionRows.filter(s => sessionMetaPass(s, f));
      fillSessionChecks("embedFilterSessions", rows, true);
      fillSessionChecks("embedHighlightSessions", rows, false);
    }
    function resetSelectedSessionsToGlobal() {
      const f = filters();
      const rows = sessionRows.filter(s => sessionMetaPass(s, f));
      fillSessionChecks("embedFilterSessions", rows, true);
      fillSessionChecks("embedHighlightSessions", rows, false);
      if (controls.selectedSessionsActive) controls.selectedSessionsActive.checked = false;
      draw();
    }
    function updateMovementControls() {
      document.querySelectorAll(".movementOnly").forEach(node => { node.style.display = ""; });
      controls.play.innerHTML = playing ? "&#10074;&#10074;" : "&#9658;";
      controls.play.setAttribute("aria-label", playing ? "Pause" : "Play");
      if (!controls.showKeypoints.checked) hoveredSessionKey = null;
    }
    function updateArrangementControls() {
      const codeUsage = controls.latentPositions && !controls.latentPositions.checked;
      document.querySelectorAll(".codeUsageOnly").forEach(node => { node.style.display = codeUsage ? "" : "none"; });
      refreshOpenAccordions();
    }
    function setVisualizationSpeed(v) {
      controls.windowsPerSecond.value = String(v);
      document.querySelectorAll(".speedBtn").forEach(btn => {
        const active = Math.abs(Number(btn.dataset.speed) - Number(v)) < 1e-9;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-pressed", active ? "true" : "false");
      });
      draw();
    }
    function visualizationSpeedLabel(value = null) {
      const v = Number(value ?? controls.windowsPerSecond.value ?? 0.5);
      const btn = [...document.querySelectorAll(".speedBtn")].find(b => Math.abs(Number(b.dataset.speed) - v) < 1e-9);
      return btn ? btn.textContent.trim() : `${fmt(v, 1)}x`;
    }
    function windowPass(w, f) {
      if (!w || w[W.model] !== f.model || w[W.scale] !== f.scale) return false;
      if (f.phases.size && !f.phases.has(data.phases[w[W.phase]])) return false;
      if (f.sessionPhases.size && !f.sessionPhases.has(data.sessionPhases[w[W.sessionPhase]])) return false;
      if (f.mutualAttention.size && !f.mutualAttention.has(data.mutualAttentionLevels[w[W.mutual]])) return false;
      return sessionPass(sessionByKey.get(key(w[W.model], w[W.sid])), f);
    }
    function makeSeriesForSession(s, f) {
      const xPts = (pointsByMSC.get(key(f.model, f.scale, f.xComp)) || []).filter(p => p[P.sid] === s.sid);
      const rows = [];
      for (const px of xPts) {
        const py = f.xComp === f.yComp ? px : pointByExact.get(key(px[P.model], px[P.sid], px[P.scale], f.yComp, px[P.token]));
        const w = windowsByExact.get(key(px[P.model], px[P.sid], px[P.scale], px[P.token]));
        if (!py || !windowPass(w, f)) continue;
        const same = f.xComp === f.yComp;
        rows.push({
          model: px[P.model], sid: px[P.sid], scale: px[P.scale], token: px[P.token],
          progress: w[W.progress], anchor: w[W.anchor], window: w,
          x: same ? px[P.z1] : px[P.pooled],
          y: same ? px[P.z2] : py[P.pooled],
          code: same ? px[P.code] : ((px[P.code] * 17 + py[P.code] * 7) % palette.length),
          comp: same ? f.xComp : -1,
          xCode: px[P.code], yCode: py[P.code],
          xComp: f.xComp, yComp: f.yComp, pair: !same,
        });
      }
      rows.sort((a,b) => a.progress - b.progress);
      return rows;
    }
    function allSeries(f) {
      const rows = [];
      for (const s of data.sessions) {
        if (!sessionPass(s, f)) continue;
        const series = makeSeriesForSession(s, f);
        if (series.length) rows.push({ session: s, series });
      }
      return rows;
    }
    function median(vals) {
      if (!vals.length) return 0;
      const s = vals.slice().sort((a, b) => a - b);
      const mid = Math.floor(s.length / 2);
      return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
    }
    function typicalWindowCount(f) {
      const counts = [];
      for (const s of data.sessions) {
        if (!sessionPass(s, f)) continue;
        const arr = windowsBySession.get(key(f.model, s.sid, f.scale));
        if (arr && arr.length) counts.push(arr.length);
      }
      return Math.max(1, median(counts) || 1);
    }
    function interp(series, progress) {
      if (!series.length) return null;
      if (progress <= series[0].progress) return {...series[0]};
      if (progress >= series[series.length - 1].progress) return {...series[series.length - 1]};
      let lo = 0, hi = series.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (series[mid].progress <= progress) lo = mid; else hi = mid;
      }
      const a = series[lo], b = series[hi];
      const t = (progress - a.progress) / Math.max(1e-9, b.progress - a.progress);
      const nearest = t < 0.5 ? a : b;
      return {...nearest, x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, progress};
    }
    function pathSlice(series, start, end) {
      if (!series.length) return [];
      const pts = [];
      const a = interp(series, start), b = interp(series, end);
      if (a) pts.push(a);
      for (const p of series) if (p.progress > start && p.progress < end) pts.push(p);
      if (b) pts.push(b);
      return pts.sort((x,y) => x.progress - y.progress);
    }
    function resampledPath(series, start, end) {
      if (!series.length || end < start) return [];
      const span = Math.max(0, end - start);
      const n = Math.round(clamp(span * Math.max(2, series.length) * 4.0, 12, 360));
      const pts = [];
      for (let i = 0; i < n; i++) {
        const t = n === 1 ? end : start + (end - start) * i / (n - 1);
        const p = interp(series, t);
        if (p) pts.push(p);
      }
      return pts;
    }
    function contextSlice(series, progress, frac, minPadSteps = 1) {
      if (!series.length) return [];
      if (frac >= 0.999) return series;
      const step = 1 / Math.max(1, series.length - 1);
      const half = frac <= 0.001 ? step * minPadSteps : frac / 2;
      const start = Math.max(0, progress - half);
      const end = Math.min(1, progress + half);
      const pts = pathSlice(series, start, end);
      return pts.length ? pts : [interp(series, progress)].filter(Boolean);
    }
    function codeUsageContextPoints(series, progress, frac) {
      if (!series.length) return [];
      if (frac >= 0.999) return series;
      const step = 1 / Math.max(1, series.length - 1);
      const half = frac <= 0.001 ? step : frac / 2;
      const start = Math.max(0, progress - half);
      const end = Math.min(1, progress + half);
      const pts = series.filter(p => p.progress >= start && p.progress <= end);
      if (pts.length) return pts;
      let best = series[0], bestD = Math.abs(series[0].progress - progress);
      for (const p of series.slice(1)) {
        const d = Math.abs(p.progress - progress);
        if (d < bestD) { best = p; bestD = d; }
      }
      return [best];
    }
    function currentPoints(seriesRows, f) {
      return seriesRows.map(r => interp(r.series, f.progress)).filter(Boolean);
    }
    function baseExtent(seriesRows, f, codebook) {
      const xs = [], ys = [];
      for (const r of seriesRows) for (const p of r.series) { xs.push(p.x); ys.push(p.y); }
      if (!xs.length || !ys.length) return { xmin:-1, xmax:1, ymin:-1, ymax:1 };
      let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
      const dx = Math.max(0.05, (xmax - xmin) * 0.08), dy = Math.max(0.05, (ymax - ymin) * 0.08);
      return { xmin:xmin-dx, xmax:xmax+dx, ymin:ymin-dy, ymax:ymax+dy };
    }
    function codebookFor(f) {
      if (f.xComp === f.yComp) return { same:true, items: codebooksByMSC.get(key(f.model, f.scale, f.xComp)) || [] };
      return {
        same:false,
        xItems: codebooksByMSC.get(key(f.model, f.scale, f.xComp)) || [],
        yItems: codebooksByMSC.get(key(f.model, f.scale, f.yComp)) || [],
      };
    }
    function codeRingLayout(seriesRows, f, cb) {
      const states = new Map();
      const addState = st => {
        if (states.has(st.key)) return;
        states.set(st.key, {
          key:st.key,
          label:st.label,
          color:st.color,
          pair:st.pair,
          code:st.code,
          xCode:st.xCode,
          yCode:st.yCode,
          xComp:f.xComp,
          yComp:f.yComp,
        });
      };
      if (cb.same) {
        for (const c of cb.items) addState({ key:String(c[C.code]), code:c[C.code], pair:false, label:`C${c[C.code]}`, color:palette[c[C.code] % palette.length] });
      }
      for (const row of seriesRows) for (const p of row.series) addState(transitionState(p));
      const ordered = [...states.values()].sort((a, b) => {
        const ak = String(a.key).split(",").map(Number), bk = String(b.key).split(",").map(Number);
        for (let i = 0; i < Math.max(ak.length, bk.length); i++) {
          const d = (ak[i] || 0) - (bk[i] || 0);
          if (d) return d;
        }
        return String(a.key).localeCompare(String(b.key));
      });
      const centers = new Map();
      const n = ordered.length;
      ordered.forEach((st, i) => {
        const angle = n <= 1 ? -Math.PI / 2 : -Math.PI / 2 + i / n * Math.PI * 2;
        centers.set(st.key, {
          ...st,
          x:n <= 1 ? 0 : Math.cos(angle),
          y:n <= 1 ? 0 : Math.sin(angle),
        });
      });
      return { centers, ordered };
    }
    function projectSeriesRows(seriesRows, f, ring) {
      if (f.mapMode !== "code_layout") return seriesRows;
      return seriesRows.map(row => ({
        session: row.session,
        series: row.series.map(p => {
          const st = transitionState(p);
          const c = ring.centers.get(st.key);
          return c ? {...p, latentX:p.x, latentY:p.y, x:c.x, y:c.y, ringState:st.key} : {...p};
        }),
      }));
    }
    function ringExtent() {
      return { xmin:-1.28, xmax:1.28, ymin:-1.28, ymax:1.28 };
    }
    function resetView(f, ex) {
      viewKey = key(f.model, f.scale, f.xComp, f.yComp, f.mapMode);
      viewEx = {...ex};
    }
    function ensureView(f, ex) {
      const k = key(f.model, f.scale, f.xComp, f.yComp, f.mapMode);
      if (k !== viewKey || !viewEx) resetView(f, ex);
      return viewEx;
    }
    function mapPt(p, ex, rect) {
      return {
        x: rect.x + (p.x - ex.xmin) / Math.max(1e-9, ex.xmax - ex.xmin) * rect.w,
        y: rect.y + (1 - (p.y - ex.ymin) / Math.max(1e-9, ex.ymax - ex.ymin)) * rect.h,
      };
    }
    function unmap(x, y, ex, rect) {
      return {
        x: ex.xmin + (x - rect.x) / rect.w * (ex.xmax - ex.xmin),
        y: ex.ymax - (y - rect.y) / rect.h * (ex.ymax - ex.ymin),
      };
    }
    const namingStatsCache = new Map();
    function namingStatsForModel(modelIdx) {
      if (namingStatsCache.has(modelIdx)) return namingStatsCache.get(modelIdx);
      const rates = data.sessions.filter(s => s.model === modelIdx).map(s => Number(s.eventsPerMin) || 0).sort((a,b) => a-b);
      const q = p => rates.length ? rates[Math.min(rates.length - 1, Math.max(0, Math.floor(p * (rates.length - 1))))] : 0;
      const stats = { min:rates[0] || 0, max:rates[rates.length - 1] || 1, q33:q(1/3), q67:q(2/3) };
      namingStatsCache.set(modelIdx, stats);
      return stats;
    }
    function namingTertile(s) {
      const stats = namingStatsForModel(s.model);
      const rate = Number(s.eventsPerMin) || 0;
      if (rate <= stats.q33) return "low";
      if (rate <= stats.q67) return "mid";
      return "high";
    }
    function lerp(a, b, t) { return Math.round(a + (b - a) * clamp(t, 0, 1)); }
    function mixHex(a, b, t) {
      const ca = hexToRgb(a), cb = hexToRgb(b);
      return `rgb(${lerp(ca[0], cb[0], t)},${lerp(ca[1], cb[1], t)},${lerp(ca[2], cb[2], t)})`;
    }
    function namingRateColor(s) {
      const stats = namingStatsForModel(s.model);
      const t = clamp(((Number(s.eventsPerMin) || 0) - stats.min) / Math.max(1e-9, stats.max - stats.min), 0, 1);
      return mixHex("#d9d9d9", "#111111", t);
    }
    function colorForPoint(p, f) {
      const s = sessionByKey.get(key(p.model, p.sid));
      if (f.color === "code") return palette[p.code % palette.length];
      if (f.color === "language") return s.language === "NGT" ? "#b2182b" : s.language === "NL" ? "#2166ac" : "#777";
      if (f.color === "aud") return s.hearing === "Deaf" ? "#d73027" : s.hearing === "Hearing" ? "#1a9850" : s.hearing === "Comparison" ? "#984ea3" : "#777";
      if (f.color === "naming_binary") return s.namingGroup === "high" ? "#111" : "#bdbdbd";
      if (f.color === "naming_tertile") {
        const t = namingTertile(s);
        return t === "high" ? "#b2182b" : t === "mid" ? "#fdd863" : "#2166ac";
      }
      if (f.color === "naming_rate") return namingRateColor(s);
      if (f.color === "session") return s.session === "S1" ? "#e41a1c" : s.session === "S2" ? "#377eb8" : s.session === "S3" ? "#4daf4a" : "#777";
      if (f.color === "phase") return p.window ? palette[p.window[W.phase] % palette.length] : "#777";
      if (f.color === "session_phase") return p.window ? ["#8dd3c7", "#fdb462", "#bebada"][p.window[W.sessionPhase] % 3] : "#777";
      if (f.color === "mutual_attention") {
        const colors = ["#bdbdbd", "#80b1d3", "#fb8072", "#b3de69", "#fdb462"];
        return p.window ? colors[p.window[W.mutual] % colors.length] : "#777";
      }
      return "#222";
    }
    function stackOptionsForMode(mode) {
      if (mode === "language") return [
        {key:"NGT", label:"NGT", color:"#b2182b"},
        {key:"NL", label:"NL", color:"#2166ac"},
        {key:"other", label:"other", color:"#777"},
      ];
      if (mode === "aud") return [
        {key:"Deaf", label:"Deaf", color:"#d73027"},
        {key:"Hearing", label:"Hearing", color:"#1a9850"},
        {key:"Comparison", label:"Comparison", color:"#984ea3"},
        {key:"Unknown", label:"Unknown", color:"#777"},
      ];
      if (mode === "naming_binary") return [
        {key:"low", label:"low naming", color:"#bdbdbd"},
        {key:"high", label:"high naming", color:"#111111"},
      ];
      if (mode === "naming_tertile") return [
        {key:"low", label:"low", color:"#2166ac"},
        {key:"mid", label:"mid", color:"#fdd863"},
        {key:"high", label:"high", color:"#b2182b"},
      ];
      if (mode === "session") return [
        {key:"S1", label:"S1", color:"#e41a1c"},
        {key:"S2", label:"S2", color:"#377eb8"},
        {key:"S3", label:"S3", color:"#4daf4a"},
      ];
      if (mode === "phase") return data.phases.map((p,i) => ({key:p, label:p, color:palette[i % palette.length]}));
      if (mode === "session_phase") return [
        {key:"early", label:"early", color:"#8dd3c7"},
        {key:"middle", label:"middle", color:"#fdb462"},
        {key:"late", label:"late", color:"#bebada"},
      ];
      if (mode === "mutual_attention") {
        const labels = data.mutualAttentionLevels || [];
        const colors = ["#bdbdbd", "#80b1d3", "#fb8072", "#b3de69", "#fdb462"];
        return labels.map((x,i) => ({key:x, label:x === "coordinated joint attention" ? "coordinated JA" : x === "naming-aligned joint attention" ? "naming JA" : x, color:colors[i % colors.length]}));
      }
      return [];
    }
    function stackInfoForPoint(p, mode) {
      const s = sessionByKey.get(key(p.model, p.sid));
      if (!s) return null;
      if (mode === "language") return stackOptionsForMode(mode).find(x => x.key === (s.language || "other")) || stackOptionsForMode(mode).find(x => x.key === "other");
      if (mode === "aud") return stackOptionsForMode(mode).find(x => x.key === (s.hearing || "Unknown")) || stackOptionsForMode(mode).find(x => x.key === "Unknown");
      if (mode === "naming_binary") return stackOptionsForMode(mode).find(x => x.key === s.namingGroup) || null;
      if (mode === "naming_tertile") return stackOptionsForMode(mode).find(x => x.key === namingTertile(s)) || null;
      if (mode === "session") return stackOptionsForMode(mode).find(x => x.key === s.session) || null;
      if (mode === "phase" && p.window) return stackOptionsForMode(mode).find(x => x.key === data.phases[p.window[W.phase]]) || null;
      if (mode === "session_phase" && p.window) return stackOptionsForMode(mode).find(x => x.key === data.sessionPhases[p.window[W.sessionPhase]]) || null;
      if (mode === "mutual_attention" && p.window) return stackOptionsForMode(mode).find(x => x.key === data.mutualAttentionLevels[p.window[W.mutual]]) || null;
      return null;
    }
    function drawAxes(rect, ex, f) {
      ctx.strokeStyle = "#222"; ctx.lineWidth = 1; ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      if (f.mapMode === "code_layout") {
        const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2;
        const rr = Math.min(rect.w, rect.h) * 0.39;
        ctx.strokeStyle = "rgba(0,0,0,.16)";
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = "#222"; ctx.font = "15px Segoe UI, Arial";
        const label = f.xComp === f.yComp
          ? `${data.components[f.xComp]} code positions`
          : `${data.components[f.xComp]} x ${data.components[f.yComp]} code-pair positions`;
        ctx.fillText(label, rect.x + rect.w / 2 - ctx.measureText(label).width / 2, rect.y + rect.h + 34);
        return;
      }
      ctx.strokeStyle = "rgba(0,0,0,.11)";
      for (let i = 1; i < 4; i++) {
        const x = rect.x + rect.w * i / 4, y = rect.y + rect.h * i / 4;
        ctx.beginPath(); ctx.moveTo(x, rect.y); ctx.lineTo(x, rect.y + rect.h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(rect.x, y); ctx.lineTo(rect.x + rect.w, y); ctx.stroke();
      }
      ctx.fillStyle = "#222"; ctx.font = "15px Segoe UI, Arial";
      const same = f.xComp === f.yComp;
      const xLabel = same ? `${data.components[f.xComp]} latent x` : `${data.components[f.xComp]} pooled score`;
      const yLabel = same ? `${data.components[f.yComp]} latent y` : `${data.components[f.yComp]} pooled score`;
      ctx.fillText(xLabel, rect.x + rect.w / 2 - ctx.measureText(xLabel).width / 2, rect.y + rect.h + 34);
      ctx.save(); ctx.translate(rect.x - 50, rect.y + rect.h / 2 + ctx.measureText(yLabel).width / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
    }
    function hexToRgb(hex) {
      const h = hex.replace("#", "");
      return [parseInt(h.slice(0,2), 16), parseInt(h.slice(2,4), 16), parseInt(h.slice(4,6), 16)];
    }
    function bgColorForCodes(a, b = null) {
      const ca = hexToRgb(palette[a % palette.length]);
      if (b === null || b === a) return `rgba(${ca[0]},${ca[1]},${ca[2]},0.32)`;
      const cb = hexToRgb(palette[b % palette.length]);
      const mix = ca.map((v, i) => Math.round((v + cb[i]) / 2));
      return `rgba(${mix[0]},${mix[1]},${mix[2]},0.34)`;
    }
    function drawBackground(rect, ex, cb, f) {
      if (f.bgMode === "none" || f.mapMode === "code_layout" || f.mapMode === "code_decomp") return;
      const step = 7;
      if (cb.same) {
        if (!cb.items.length) return;
        for (let px = rect.x; px < rect.x + rect.w; px += step) for (let py = rect.y; py < rect.y + rect.h; py += step) {
          const d = unmap(px, py, ex, rect);
          let best = cb.items[0], bd = Infinity;
          for (const c of cb.items) {
            const dist = (d.x - c[C.z1]) ** 2 + (d.y - c[C.z2]) ** 2;
            if (dist < bd) { bd = dist; best = c; }
          }
          ctx.fillStyle = bgColorForCodes(best[C.code]);
          ctx.fillRect(px, py, step + 1, step + 1);
        }
      } else {
        if (!cb.xItems.length || !cb.yItems.length) return;
        for (let px = rect.x; px < rect.x + rect.w; px += step) for (let py = rect.y; py < rect.y + rect.h; py += step) {
          const d = unmap(px, py, ex, rect);
          let bx = cb.xItems[0], by = cb.yItems[0], dx = Infinity, dy = Infinity;
          for (const c of cb.xItems) { const dist = (d.x - c[C.pooled]) ** 2; if (dist < dx) { dx = dist; bx = c; } }
          for (const c of cb.yItems) { const dist = (d.y - c[C.pooled]) ** 2; if (dist < dy) { dy = dist; by = c; } }
          ctx.fillStyle = bgColorForCodes(bx[C.code], by[C.code]);
          ctx.fillRect(px, py, step + 1, step + 1);
        }
      }
    }
    function drawDiamond(x, y, r, fill, stroke = "#111", lineWidth = 1) {
      ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth;
      ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    function drawCodeLabel(text, x, y, anchor = "left") {
      ctx.save();
      ctx.font = "13px Segoe UI, Arial";
      ctx.textBaseline = "middle";
      ctx.textAlign = anchor;
      const padX = 4, padY = 2;
      const w = ctx.measureText(text).width + padX * 2;
      const h = 16 + padY * 2;
      let bx = anchor === "center" ? x - w / 2 : anchor === "right" ? x - w : x;
      let by = y - h / 2;
      ctx.fillStyle = "rgba(255,255,255,.92)";
      ctx.strokeStyle = "rgba(0,0,0,.42)";
      ctx.lineWidth = 1;
      ctx.fillRect(bx, by, w, h);
      ctx.strokeRect(bx, by, w, h);
      ctx.fillStyle = "#111";
      ctx.fillText(text, x, y + 0.5);
      ctx.restore();
    }
    function overlayBoxesInCanvas() {
      const canvasBox = canvas.getBoundingClientRect();
      if (!canvasBox.width || !canvasBox.height) return [];
      const nodes = [...document.querySelectorAll(".explorerControls,.legendOverlay,.hoverInspector,.relatedPanel")];
      return nodes.map(node => {
        const style = window.getComputedStyle(node);
        if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity || 1) <= 0.01) return null;
        const box = node.getBoundingClientRect();
        if (box.width < 8 || box.height < 8) return null;
        const x = (box.left - canvasBox.left) / canvasBox.width * canvas.width;
        const y = (box.top - canvasBox.top) / canvasBox.height * canvas.height;
        const w = box.width / canvasBox.width * canvas.width;
        const h = box.height / canvasBox.height * canvas.height;
        if (x > canvas.width || x + w < 0 || y > canvas.height || y + h < 0) return null;
        return { x, y, w, h };
      }).filter(Boolean);
    }
    function yAxisCodePosition(y, rect, boxes, labelText) {
      const pad = 20;
      const labelW = Math.max(30, ctx.measureText(labelText).width + 12);
      let intervals = [{ lo:rect.x + 28, hi:rect.x + rect.w - 28 }];
      for (const b of boxes) {
        if (y < b.y - pad || y > b.y + b.h + pad) continue;
        const block = { lo:b.x - pad, hi:b.x + b.w + pad };
        const next = [];
        for (const it of intervals) {
          if (block.hi <= it.lo || block.lo >= it.hi) {
            next.push(it);
          } else {
            if (block.lo - it.lo > 44) next.push({ lo:it.lo, hi:block.lo });
            if (it.hi - block.hi > 44) next.push({ lo:block.hi, hi:it.hi });
          }
        }
        intervals = next.length ? next : intervals;
      }
      const minWidth = labelW + 48;
      const usable = intervals.filter(it => it.hi - it.lo >= minWidth);
      const chosen = (usable.length ? usable : intervals)
        .slice()
        .sort((a,b) => (a.lo - b.lo) || ((b.hi - b.lo) - (a.hi - a.lo)))[0] || { lo:rect.x + 28, hi:rect.x + rect.w - 28 };
      const preferred = chosen.lo + 24;
      let x = clamp(preferred, chosen.lo + 18, chosen.hi - 18);
      let anchor = "left";
      let labelX = x + 16;
      if (labelX + labelW > chosen.hi) {
        x = chosen.hi - 18;
        anchor = "right";
        labelX = x - 16;
      }
      return { x, labelX, anchor };
    }
    function drawCodeMarks(rect, ex, cb, f) {
      currentCodeMarks = [];
      ctx.font = "13px Segoe UI, Arial";
      if (cb.same) {
        for (const c of cb.items) {
          const m = mapPt({x:c[C.z1], y:c[C.z2]}, ex, rect);
          if (m.x < rect.x - 10 || m.x > rect.x + rect.w + 10 || m.y < rect.y - 10 || m.y > rect.y + rect.h + 10) continue;
          drawDiamond(m.x, m.y, 8.2, palette[c[C.code] % palette.length], "#111", 1.2);
          currentCodeMarks.push({x:m.x, y:m.y, r:16, model:f.model, scale:f.scale, comp:f.xComp, code:c[C.code], label:`${data.components[f.xComp]} code ${c[C.code]}`});
          drawCodeLabel(String(c[C.code]), m.x + 13, m.y, "left");
        }
      } else {
        for (const c of cb.xItems) {
          const x = rect.x + (c[C.pooled] - ex.xmin) / Math.max(1e-9, ex.xmax - ex.xmin) * rect.w;
          if (x < rect.x || x > rect.x + rect.w) continue;
          const y = rect.y + rect.h - 19;
          drawDiamond(x, y, 7.6, palette[c[C.code] % palette.length], "#111", 1.2);
          currentCodeMarks.push({x, y, r:16, model:f.model, scale:f.scale, comp:f.xComp, code:c[C.code], label:`${data.components[f.xComp]} code ${c[C.code]}`});
          drawCodeLabel(`x${c[C.code]}`, x, y - 18, "center");
        }
        const overlayBoxes = overlayBoxesInCanvas();
        for (const c of cb.yItems) {
          const y = rect.y + (1 - (c[C.pooled] - ex.ymin) / Math.max(1e-9, ex.ymax - ex.ymin)) * rect.h;
          if (y < rect.y || y > rect.y + rect.h) continue;
          const label = `y${c[C.code]}`;
          const pos = yAxisCodePosition(y, rect, overlayBoxes, label);
          const x = pos.x;
          ctx.save();
          ctx.setLineDash([4, 5]);
          ctx.strokeStyle = "rgba(0,0,0,.34)";
          ctx.lineWidth = 1.1;
          ctx.beginPath();
          ctx.moveTo(x + 13, y);
          ctx.lineTo(Math.min(rect.x + rect.w - 5, x + 96), y);
          ctx.stroke();
          ctx.restore();
          drawDiamond(x, y, 9.2, palette[c[C.code] % palette.length], "#111", 1.6);
          currentCodeMarks.push({x, y, r:16, model:f.model, scale:f.scale, comp:f.yComp, code:c[C.code], label:`${data.components[f.yComp]} code ${c[C.code]}`});
          drawCodeLabel(label, pos.labelX, y, pos.anchor);
        }
      }
    }
    function drawCodeRingNodes(rect, ex, ring, f) {
      if (f.mapMode !== "code_layout" || !ring) return;
      ctx.font = "11px Segoe UI, Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const st of ring.ordered) {
        const c = ring.centers.get(st.key);
        if (!c) continue;
        const m = mapPt(c, ex, rect);
        const fill = c.color || palette[(c.code ?? c.xCode ?? 0) % palette.length];
        const label = c.pair ? c.label : `C${c.code}`;
        ctx.save();
        drawDiamond(m.x, m.y, 13.5, fill, "#111", 1.4);
        ctx.fillStyle = "#111";
        ctx.font = label.length > 3 ? "10.5px Segoe UI, Arial" : "12px Segoe UI, Arial";
        const w = ctx.measureText(label).width + 8;
        ctx.fillStyle = "rgba(255,255,255,.88)";
        ctx.strokeStyle = "rgba(0,0,0,.30)";
        ctx.lineWidth = 1;
        ctx.fillRect(m.x - w / 2, m.y + 16, w, 18);
        ctx.strokeRect(m.x - w / 2, m.y + 16, w, 18);
        ctx.fillStyle = "#111";
        ctx.fillText(label, m.x, m.y + 25);
        ctx.restore();
        currentCodeMarks.push({
          x:m.x, y:m.y, r:18, model:f.model, scale:f.scale,
          comp:c.pair ? -1 : f.xComp,
          code:c.code, pair:c.pair,
          xComp:f.xComp, yComp:f.yComp, xCode:c.xCode, yCode:c.yCode,
          label:c.pair ? `${data.components[f.xComp]} C${c.xCode} / ${data.components[f.yComp]} C${c.yCode}` : `${data.components[f.xComp]} code ${c.code}`,
        });
      }
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }
    function pieSvg(values, colors = modColors) {
      const total = values.reduce((a,b) => a + Math.abs(Number(b) || 0), 0) || 1;
      let acc = 0, paths = "";
      values.forEach((v, i) => {
        const frac = Math.abs(Number(v) || 0) / total;
        const a0 = acc * 2 * Math.PI - Math.PI / 2, a1 = (acc + frac) * 2 * Math.PI - Math.PI / 2;
        acc += frac;
        const x0 = 50 + 42 * Math.cos(a0), y0 = 50 + 42 * Math.sin(a0);
        const x1 = 50 + 42 * Math.cos(a1), y1 = 50 + 42 * Math.sin(a1);
        const large = frac > 0.5 ? 1 : 0;
        paths += `<path d="M50 50 L${x0.toFixed(2)} ${y0.toFixed(2)} A42 42 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)} Z" fill="${colors[i % colors.length]}"></path>`;
      });
      return `<svg viewBox="0 0 100 100" class="profilePie">${paths}<circle cx="50" cy="50" r="18" fill="#fbfaf7"></circle></svg>`;
    }
    function sourceBehaviorIndices(compIdx) {
      const comp = data.components[compIdx];
      if (comp && comp !== "joint" && data.modalityChannels[comp]) return data.modalityChannels[comp];
      return data.channels.map((_, i) => i);
    }
    function sourceModalityIndices(compIdx) {
      const comp = data.components[compIdx];
      if (comp && comp !== "joint") {
        const idx = data.modalities.indexOf(comp);
        return idx >= 0 ? [idx] : data.modalities.map((_, i) => i);
      }
      return data.modalities.map((_, i) => i);
    }
    const channelNormCache = new Map();
    function medianOf(vals) {
      if (!vals.length) return 0;
      const s = vals.slice().sort((a,b) => a - b);
      const mid = Math.floor(s.length / 2);
      return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
    }
    function channelNormDenoms(model, scale, comp) {
      const k = key(model, scale ?? "all", comp);
      if (channelNormCache.has(k)) return channelNormCache.get(k);
      const profileRows = data.profile || data.profiles || [];
      let rows = profileRows.filter(r => r[R.model] === model && r[R.comp] === comp && (scale === null || scale === undefined || r[R.scale] === scale));
      if (!rows.length && scale !== null && scale !== undefined) rows = profileRows.filter(r => r[R.model] === model && r[R.comp] === comp);
      const n = data.channels.length;
      const denoms = Array(n).fill(1);
      for (let i = 0; i < n; i++) {
        const vals = rows.map(r => Math.abs(Number((r[R.zMeans] || [])[i]) || 0)).filter(Number.isFinite);
        const med = medianOf(vals);
        const mean = vals.reduce((a,b) => a + b, 0) / Math.max(1, vals.length);
        denoms[i] = Math.max(0.20, med || mean || 1);
      }
      channelNormCache.set(k, denoms);
      return denoms;
    }
    function normalizedBehaviorZ(prof, model, scale, comp) {
      const raw = prof[R.zMeans] || prof[R.means] || [];
      const denoms = channelNormDenoms(model, scale, comp);
      return raw.map((v, i) => (Number(v) || 0) / Math.max(0.20, denoms[i] || 1));
    }
    function zBarRows(zVals, limit, mode = "tip", indices = null, rawVals = null) {
      const idxs = (indices || zVals.map((_, i) => i)).slice(0, limit);
      const rows = idxs.map(i => [i, data.channels[i], Number(zVals[i]) || 0]);
      const maxAbs = Math.max(1.5, ...rows.map(x => Math.abs(x[2])));
      const rankLimit = rows.length > 8 ? 3 : Math.min(2, rows.length);
      const rankImportant = new Set(
        rows
          .map((r, pos) => ({ pos, abs: Math.abs(r[2]) }))
          .filter(r => r.abs >= IMPORTANT_RANK_MIN_Z)
          .sort((a, b) => b.abs - a.abs)
          .slice(0, rankLimit)
          .map(r => r.pos)
      );
      return rows.map((row, pos) => {
        const [, ch, z] = row;
        const pct = Math.min(50, Math.abs(z) / maxAbs * 50);
        const cls = z >= 0 ? "pos" : "neg";
        const left = z >= 0 ? 50 : 50 - pct;
        const raw = rawVals ? Number(rawVals[row[0]]) || 0 : z;
        const label = rawVals
          ? `${z >= 0 ? "+" : ""}${fmt(z, 2)} rel`
          : `${z >= 0 ? "+" : ""}${fmt(z, 2)}z`;
        const title = rawVals ? ` title="raw z: ${raw >= 0 ? "+" : ""}${fmt(raw, 2)}"` : "";
        const thresholdHit = Math.abs(z) >= IMPORTANT_Z;
        const isImportant = thresholdHit || rankImportant.has(pos);
        const rowCls = `${mode === "main" ? "behaviorRow zRow" : "tipBar zTipBar"}${isImportant ? " sig" : ""}${isImportant ? (thresholdHit ? " thresholdSig" : " rankSig") : ""}`;
        const badge = isImportant ? `<small>${thresholdHit ? ">=1z" : "top"}</small>` : "";
        return `<div class="${rowCls}"${title}><span>${esc(ch)}</span><i><b class="${cls}" style="left:${left}%;width:${pct}%"></b></i><em>${label}${badge}</em></div>`;
      }).join("");
    }
    function profilePieParts(mark, prof, behaviorIdxs) {
      const comp = data.components[mark.comp];
      if (comp && comp !== "joint") {
        const vals = behaviorIdxs.map(i => normalizedBehaviorZ(prof, mark.model, mark.scale, mark.comp)[i] || 0);
        const colors = behaviorIdxs.map((_, i) => palette[i % palette.length]);
        const legend = behaviorIdxs.map((i, j) => `<span><i style="background:${colors[j]}"></i>${esc(data.channels[i])}</span>`).join("");
        return { vals, colors, legend };
      }
      const modalityIdxs = sourceModalityIndices(mark.comp);
      const vals = modalityIdxs.map(i => (prof[R.modalityZ] || prof[R.modalityMeans])[i] || 0);
      const colors = modalityIdxs.map(i => modColors[i % modColors.length]);
      const legend = modalityIdxs.map((i, j) => `<span><i style="background:${colors[j]}"></i>${esc(data.modalities[i])}</span>`).join("");
      return { vals, colors, legend };
    }
    function renderCodeProfile(mark) {
      if (!mark) {
        codeProfile.innerHTML = "<b>Code composition</b><br>Hover over a code diamond to see the aggregate behavior profile for windows assigned to that code.";
        return;
      }
      const exact = profileByMSCCode.get(key(mark.model, mark.scale, mark.comp, mark.code));
      const prof = exact || profileByMCCode.get(key(mark.model, mark.comp, mark.code));
      if (!prof) { codeProfile.innerHTML = `<b>${esc(mark.label)}</b><br><span class="muted">Unused codebook entry: no exported windows were assigned to this code, so there is no behavior decomposition to highlight.</span>`; return; }
      const scope = exact ? data.scaleLabels[mark.scale] : "all time windows";
      const behaviorIdxs = sourceBehaviorIndices(mark.comp);
      const rawZ = prof[R.zMeans] || prof[R.means];
      const displayZ = normalizedBehaviorZ(prof, mark.model, exact ? mark.scale : null, mark.comp);
      const barsHtml = zBarRows(displayZ, behaviorIdxs.length, "main", behaviorIdxs, rawZ);
      const pie = profilePieParts(mark, prof, behaviorIdxs);
      codeProfile.innerHTML = `<b>${esc(mark.label)} - Which behaviors are present when this code is used?</b><br><span class="muted">Scope: ${esc(scope)}; windows assigned: ${prof[R.count]}; bars are channel-normalized z-scores, so each behavior is judged relative to how much that behavior varies across codes. Hover a row for raw z.</span><div class="profileGrid">${pieSvg(pie.vals, pie.colors)}<div>${barsHtml}</div></div><div class="pieLegend">${pie.legend}</div>`;
    }
    function codeProfileTooltip(mark) {
      const exact = profileByMSCCode.get(key(mark.model, mark.scale, mark.comp, mark.code));
      const prof = exact || profileByMCCode.get(key(mark.model, mark.comp, mark.code));
      if (!prof) return `<b>${esc(mark.label)}</b><span class="tipMuted">Unused codebook entry: no exported windows were assigned to this code, so there is no behavior decomposition to highlight.</span>`;
      const scope = exact ? data.scaleLabels[mark.scale] : "all time windows";
      const behaviorIdxs = sourceBehaviorIndices(mark.comp);
      const rawZ = prof[R.zMeans] || prof[R.means];
      const displayZ = normalizedBehaviorZ(prof, mark.model, exact ? mark.scale : null, mark.comp);
      const barsHtml = zBarRows(displayZ, behaviorIdxs.length, "tip", behaviorIdxs, rawZ);
      const pie = profilePieParts(mark, prof, behaviorIdxs);
      return `<b>${esc(mark.label)} - Which behaviors are present when this code is used?</b><span class="tipMuted">Scope: ${esc(scope)}; windows assigned: ${prof[R.count]}; bars are channel-normalized; hover rows for raw z.</span><div class="tipProfile"><div class="tipPiePane">${pieSvg(pie.vals, pie.colors)}<div class="tipLegend">${pie.legend}</div></div><div class="tipZPane">${barsHtml}</div></div>`;
    }
    function codeMarkTooltip(mark) {
      if (!mark.pair) return codeProfileTooltip(mark);
      const xMark = { model:mark.model, scale:mark.scale, comp:mark.xComp, code:mark.xCode, label:`X ${data.components[mark.xComp]} C${mark.xCode}` };
      const yMark = { model:mark.model, scale:mark.scale, comp:mark.yComp, code:mark.yCode, label:`Y ${data.components[mark.yComp]} C${mark.yCode}` };
      return `<b>${esc(mark.label)}</b><span class="tipMuted">This node is a displayed code pair, not an extra learned codebook. The panels below show each component code's behavior profile.</span><div class="pairTip">${codeProfileTooltip(xMark)}${codeProfileTooltip(yMark)}</div>`;
    }
    function profileFor(model, scale, comp, code) {
      return profileByMSCCode.get(key(model, scale, comp, code)) || profileByMCCode.get(key(model, comp, code));
    }
    function itemsForComponent(cb, f, compIdx) {
      if (cb.same && compIdx === f.xComp) return cb.items || [];
      if (!cb.same && compIdx === f.xComp) return cb.xItems || [];
      if (!cb.same && compIdx === f.yComp) return cb.yItems || [];
      return codebooksByMSC.get(key(f.model, f.scale, compIdx)) || [];
    }
    function drawCodeDecompBlock(x, y, w, h, compIdx, items, f) {
      const title = `${data.components[compIdx]} codes: behavior z-score profile`;
      const profiles = items
        .slice()
        .sort((a,b) => a[C.code] - b[C.code])
        .map(c => ({ c, prof:profileFor(f.model, f.scale, compIdx, c[C.code]) }))
        .filter(x => x.prof);
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,.76)";
      ctx.strokeStyle = "rgba(0,0,0,.14)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, y, w, h, 8);
      else ctx.rect(x, y, w, h);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#191919";
      ctx.font = "700 16px Segoe UI, Arial";
      ctx.fillText(title, x + 14, y + 24);
      ctx.font = "12px Segoe UI, Arial";
      ctx.fillStyle = "#555";
      ctx.fillText("Bars show channel-normalized deviations; orange is above baseline, blue is below.", x + 14, y + 43);
      if (!profiles.length) {
        ctx.fillStyle = "#777";
        ctx.fillText("No assigned windows exported for these codes.", x + 14, y + 72);
        ctx.restore();
        return;
      }
      const behaviorIdxs = sourceBehaviorIndices(compIdx);
      profiles.forEach(row => { row.displayZ = normalizedBehaviorZ(row.prof, f.model, f.scale, compIdx); });
      const featureScores = behaviorIdxs.map(i => {
        const score = profiles.reduce((acc, row) => acc + Math.abs(Number((row.displayZ || [])[i]) || 0), 0) / Math.max(1, profiles.length);
        return { i, score };
      }).sort((a,b) => b.score - a.score).slice(0, Math.min(9, behaviorIdxs.length));
      const left = x + 58, top = y + 68, bottom = y + h - 22;
      const chartW = Math.max(80, w - 82), chartH = Math.max(60, bottom - top);
      const rowH = chartH / Math.max(1, profiles.length);
      const cellW = chartW / Math.max(1, featureScores.length);
      const maxAbs = Math.max(1.5, ...profiles.flatMap(row => featureScores.map(fs => Math.abs(Number((row.displayZ || [])[fs.i]) || 0))));
      ctx.font = "10.5px Segoe UI, Arial";
      ctx.fillStyle = "#333";
      featureScores.forEach((fs, j) => {
        const label = cleanBehaviorName(data.channels[fs.i]).slice(0, 18);
        ctx.save();
        ctx.translate(left + j * cellW + cellW / 2 - 2, top - 8);
        ctx.rotate(-Math.PI / 5);
        ctx.textAlign = "right";
        ctx.fillText(label, 0, 0);
        ctx.restore();
      });
      profiles.forEach((row, r) => {
        const cy = top + r * rowH + rowH / 2;
        const code = row.c[C.code];
        ctx.textAlign = "left";
        ctx.fillStyle = palette[code % palette.length];
        drawDiamond(x + 20, cy, 6, palette[code % palette.length]);
        ctx.fillStyle = "#111";
        ctx.font = "700 11px Segoe UI, Arial";
        ctx.fillText(String(code), x + 31, cy + 4);
        currentCodeMarks.push({x:x + 20, y:cy, r:12, model:f.model, scale:f.scale, comp:compIdx, code, label:`${data.components[compIdx]} code ${code}`});
        featureScores.forEach((fs, j) => {
          const v = Number((row.displayZ || [])[fs.i]) || 0;
          const cx = left + j * cellW + cellW / 2;
          const bw = Math.max(2, (cellW - 7) * Math.min(1, Math.abs(v) / maxAbs));
          const bh = Math.max(5, rowH * 0.46);
          ctx.fillStyle = "rgba(0,0,0,.08)";
          ctx.fillRect(left + j * cellW + 3, cy - bh / 2, cellW - 6, bh);
          ctx.fillStyle = v >= 0 ? "#f28e2b" : "#4e79a7";
          ctx.fillRect(cx - bw / 2, cy - bh / 2, bw, bh);
        });
      });
      ctx.textAlign = "start";
      ctx.restore();
    }
    function drawCodeDecompositionView(f, cb) {
      currentCodeMarks = [];
      const comps = f.xComp === f.yComp ? [f.xComp] : [f.xComp, f.yComp];
      const margin = 34;
      ctx.fillStyle = "#faf8f2";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#191919";
      ctx.font = "700 20px Segoe UI, Arial";
      ctx.fillText("Code Decomposition", margin, 34);
      ctx.font = "13px Segoe UI, Arial";
      ctx.fillStyle = "#555";
      ctx.fillText(`${data.scaleLabels[f.scale]} | ${comps.map(c => data.components[c]).join(" x ")}`, margin, 55);
      const gap = 18;
      const blockH = (canvas.height - 92 - gap * (comps.length - 1)) / comps.length;
      comps.forEach((compIdx, i) => {
        drawCodeDecompBlock(margin, 74 + i * (blockH + gap), canvas.width - margin * 2, blockH, compIdx, itemsForComponent(cb, f, compIdx), f);
      });
    }
    function heatmapColor(t) {
      t = clamp(t, 0, 1);
      const stops = [[255,255,255], [253,219,149], [244,109,67], [165,0,38]];
      const scaled = t * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(scaled));
      const u = scaled - i;
      const a = stops[i], b = stops[i + 1];
      return `rgb(${lerp(a[0], b[0], u)},${lerp(a[1], b[1], u)},${lerp(a[2], b[2], u)})`;
    }
    function drawBehaviorHeatmapView(f, seriesRows) {
      const seen = new Set();
      const windows = [];
      for (const row of seriesRows) {
        for (const p of row.series) {
          if (!p.window) continue;
          const k = `${p.sid}|${p.token}`;
          if (seen.has(k)) continue;
          seen.add(k);
          windows.push(p.window);
        }
      }
      currentCodeMarks = [];
      currentDrawn = [];
      currentTailHits = [];
      currentTransitionHits = [];
      const n = data.channels.length;
      const mat = Array.from({length:n}, () => Array(n).fill(0));
      for (const w of windows) {
        const vals = w[W.behaviors].map(v => clamp(Number(v) || 0, 0, 1));
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) mat[i][j] += i === j ? vals[i] : vals[i] * vals[j];
        }
      }
      const denom = Math.max(1, windows.length);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) mat[i][j] /= denom;
      const maxV = Math.max(0.001, ...mat.flat());
      const rect = plotRect(f);
      ctx.fillStyle = "#faf8f2";
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      ctx.strokeStyle = "rgba(0,0,0,.10)";
      ctx.lineWidth = 1;
      ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      const labelPad = Math.min(210, Math.max(150, rect.w * 0.24));
      const bottomPad = Math.min(150, Math.max(96, rect.h * 0.24));
      const size = Math.max(220, Math.min(rect.w - labelPad - 58, rect.h - 102 - bottomPad));
      const x0 = rect.x + labelPad;
      const y0 = rect.y + 82;
      const cell = size / Math.max(1, n);
      ctx.fillStyle = "#191919";
      ctx.font = "700 20px Segoe UI, Arial";
      ctx.fillText("Behavior Co-occurrence Heatmap", rect.x, rect.y + 24);
      ctx.font = "13px Segoe UI, Arial";
      ctx.fillStyle = "#555";
      ctx.fillText(`${data.scaleLabels[f.scale]} | visible windows: ${windows.length}`, rect.x, rect.y + 45);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          ctx.fillStyle = heatmapColor(mat[i][j] / maxV);
          ctx.fillRect(x0 + j * cell, y0 + i * cell, Math.ceil(cell) + 0.5, Math.ceil(cell) + 0.5);
        }
      }
      ctx.strokeStyle = "rgba(0,0,0,.18)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, size, size);
      ctx.font = "10.5px Segoe UI, Arial";
      ctx.fillStyle = "#222";
      for (let i = 0; i < n; i++) {
        const label = cleanBehaviorName(data.channels[i]).slice(0, 26);
        ctx.textAlign = "right";
        ctx.fillText(label, x0 - 8, y0 + i * cell + cell * 0.66);
        ctx.save();
        ctx.translate(x0 + i * cell + cell * 0.55, y0 + size + 8);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = "right";
        ctx.fillText(label, 0, 0);
        ctx.restore();
      }
      ctx.textAlign = "start";
      const lx = x0 + size + 18, ly = y0, lw = 16, lh = Math.min(180, size);
      for (let i = 0; i < lh; i++) {
        ctx.fillStyle = heatmapColor(1 - i / Math.max(1, lh - 1));
        ctx.fillRect(lx, ly + i, lw, 1);
      }
      ctx.strokeStyle = "rgba(0,0,0,.25)";
      ctx.strokeRect(lx, ly, lw, lh);
      ctx.fillStyle = "#333";
      ctx.font = "11px Segoe UI, Arial";
      ctx.fillText(fmt(maxV, 2), lx + 22, ly + 8);
      ctx.fillText("0", lx + 22, ly + lh);
      return windows.length;
    }
    function drawCodeUsageHistogramView(f, seriesRows, cb) {
      currentCodeMarks = [];
      currentDrawn = [];
      currentTailHits = [];
      currentTransitionHits = [];
      const rect = plotRect(f);
      const comps = f.xComp === f.yComp ? [f.xComp] : [f.xComp, f.yComp];
      const allPts = [];
      for (const row of seriesRows) {
        const pts = codeUsageContextPoints(row.series, f.progress, f.codeUsageContext);
        for (const p of pts) allPts.push(p);
      }
      ctx.save();
      ctx.fillStyle = "#faf8f2";
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      ctx.strokeStyle = "rgba(0,0,0,.10)";
      ctx.lineWidth = 1;
      ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      ctx.fillStyle = "#191919";
      ctx.font = "700 20px Segoe UI, Arial";
      ctx.fillText("Code Usage", rect.x, rect.y + 24);
      ctx.font = "13px Segoe UI, Arial";
      ctx.fillStyle = "#555";
      const contextText = f.codeUsageContext >= 0.999 ? "full session" : f.codeUsageContext <= 0.001 ? "current frame" : `${fmt(f.codeUsageContext * 100, 1)}% session`;
      ctx.fillText(`${data.scaleLabels[f.scale]} | counted windows: ${allPts.length} | context: ${contextText}`, rect.x, rect.y + 45);
      if (!allPts.length) {
        ctx.fillStyle = "#666";
        ctx.font = "14px Segoe UI, Arial";
        ctx.fillText("No windows match the current filters.", rect.x, rect.y + 82);
        ctx.restore();
        return 0;
      }
      const gap = comps.length > 1 ? 30 : 0;
      const availableH = Math.max(120, rect.h - 94 - gap * (comps.length - 1));
      const panelH = availableH / comps.length;
      comps.forEach((compIdx, i) => {
        const y = rect.y + 74 + i * (panelH + gap);
        drawCodeUsageBlock(rect.x, y, rect.w, panelH, compIdx, allPts, cb, f);
      });
      ctx.restore();
      return allPts.length;
    }
    function drawCodeUsageBlock(x, y, w, h, compIdx, pts, cb, f) {
      const items = itemsForComponent(cb, f, compIdx).slice().sort((a,b) => a[C.code] - b[C.code]);
      const counts = new Map(items.map(c => [c[C.code], 0]));
      const stackMode = f.codeUsageStack || "none";
      let stackMetas = stackMode === "none" ? [] : stackOptionsForMode(stackMode);
      const stackMetaByKey = new Map(stackMetas.map(m => [m.key, m]));
      const stacks = new Map(items.map(c => [c[C.code], new Map()]));
      for (const p of pts) {
        const code = compIdx === f.yComp && f.xComp !== f.yComp ? p.yCode : p.xCode;
        counts.set(code, (counts.get(code) || 0) + 1);
        if (stackMode !== "none") {
          let info = stackInfoForPoint(p, stackMode);
          if (!info) info = {key:"other", label:"other", color:"#999999"};
          if (!stackMetaByKey.has(info.key)) {
            stackMetaByKey.set(info.key, info);
            stackMetas.push(info);
          }
          const byStack = stacks.get(code) || new Map();
          byStack.set(info.key, (byStack.get(info.key) || 0) + 1);
          stacks.set(code, byStack);
        }
      }
      const total = [...counts.values()].reduce((a,b) => a + b, 0);
      ctx.save();
      ctx.textAlign = "start";
      ctx.fillStyle = "#191919";
      ctx.font = "800 15px Segoe UI, Arial";
      ctx.fillText(`${data.components[compIdx]} codebook`, x + 14, y + 25);
      ctx.font = "12px Segoe UI, Arial";
      ctx.fillStyle = "#555";
      ctx.fillText(`${total} filtered windows`, x + 14, y + 43);
      if (!items.length || total <= 0) {
        ctx.fillStyle = "#777";
        ctx.fillText("No assigned windows for this codebook under the current filters.", x + 14, y + 75);
        ctx.restore();
        return;
      }
      const left = x + 62, right = x + w - 24, top = y + 62, bottom = y + h - 48;
      const chartW = Math.max(100, right - left);
      const chartH = Math.max(60, bottom - top);
      const maxCount = Math.max(1, ...items.map(c => counts.get(c[C.code]) || 0));
      const slotW = chartW / Math.max(1, items.length);
      ctx.strokeStyle = "rgba(0,0,0,.34)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, bottom);
      ctx.moveTo(left, bottom);
      ctx.lineTo(right, bottom);
      ctx.stroke();
      ctx.save();
      ctx.translate(x + 16, top + chartH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillStyle = "#333";
      ctx.font = "700 11px Segoe UI, Arial";
      ctx.fillText("count", 0, 0);
      ctx.restore();
      ctx.textAlign = "center";
      ctx.fillStyle = "#333";
      ctx.font = "700 11px Segoe UI, Arial";
      ctx.fillText("code usage", left + chartW / 2, y + h - 8);
      ctx.textAlign = "right";
      ctx.font = "10.5px Segoe UI, Arial";
      ctx.fillStyle = "#555";
      ctx.fillText(String(maxCount), left - 7, top + 4);
      ctx.fillText("0", left - 7, bottom + 3);
      items.forEach((c, j) => {
        const code = c[C.code];
        const count = counts.get(code) || 0;
        const share = count / Math.max(1, total);
        const barH = chartH * count / maxCount;
        const cx = left + j * slotW + slotW / 2;
        const bw = Math.max(10, Math.min(48, slotW * 0.62));
        if (stackMode === "none") {
          ctx.fillStyle = palette[code % palette.length];
          ctx.globalAlpha = count ? 0.88 : 0.20;
          ctx.fillRect(cx - bw / 2, bottom - barH, bw, barH);
          ctx.globalAlpha = 1;
        } else {
          let yCursor = bottom;
          const byStack = stacks.get(code) || new Map();
          stackMetas.forEach(meta => {
            const v = byStack.get(meta.key) || 0;
            if (!v) return;
            const segH = chartH * v / maxCount;
            yCursor -= segH;
            ctx.fillStyle = meta.color;
            ctx.globalAlpha = 0.90;
            ctx.fillRect(cx - bw / 2, yCursor, bw, segH);
          });
          ctx.globalAlpha = count ? 1 : 0.18;
        }
        ctx.strokeStyle = "rgba(0,0,0,.25)";
        ctx.strokeRect(cx - bw / 2, bottom - barH, bw, barH);
        ctx.globalAlpha = 1;
        currentCodeMarks.push({x:cx, y:bottom - Math.max(6, barH) / 2, r:Math.max(11, bw / 2), model:f.model, scale:f.scale, comp:compIdx, code, label:`${data.components[compIdx]} code ${code}`});
        ctx.fillStyle = "#191919";
        ctx.font = "700 11px Segoe UI, Arial";
        ctx.textAlign = "center";
        ctx.fillText(String(code), cx, bottom + 16);
        if (share >= 0.08) {
          ctx.fillStyle = "#333";
          ctx.font = "11px Segoe UI, Arial";
          ctx.fillText(`${Math.round(share * 100)}%`, cx, Math.max(top + 12, bottom - barH - 7));
        }
      });
      ctx.textAlign = "start";
      ctx.restore();
    }
    function scaleSeconds(scaleIdx) {
      const raw = data.scales[scaleIdx] || data.scaleLabels[scaleIdx] || "";
      const m = String(raw).match(/(\d+(?:\.\d+)?)/);
      if (m) return Number(m[1]);
      const label = data.scaleLabels[scaleIdx] || "";
      const n = String(label).match(/(\d+(?:\.\d+)?)/);
      return n ? Number(n[1]) : 1;
    }
    function cleanBehaviorName(ch) {
      return String(ch)
        .replace(/^CG_/, "CG ")
        .replace(/^Child_/, "Child ")
        .replace(/^dyadic_/, "Dyadic ")
        .replace(/_/g, " ");
    }
    function activityRows(pt) {
      const sec = scaleSeconds(pt.scale);
      const vals = pt.window[W.behaviors];
      const rows = [];
      for (let mi = 0; mi < data.modalities.length; mi++) {
        const mod = data.modalities[mi];
        const idxs = data.modalityChannels[mod] || [];
        const active = idxs
          .map(i => ({ name: cleanBehaviorName(data.channels[i]), seconds: vals[i] * sec, frac: clamp(vals[i], 0, 1), channel:i }))
          .filter(x => x.seconds >= Math.max(0.15, sec * 0.08))
          .slice(0, 5);
        rows.push({ mod, color:modColors[mi % modColors.length], active });
      }
      return rows;
    }
    function sessionTooltip(pt) {
      const s = sessionByKey.get(key(pt.model, pt.sid));
      const w = pt.window;
      const sec = scaleSeconds(pt.scale);
      const codeText = pt.pair ? `${data.components[pt.xComp]} C${pt.xCode} x ${data.components[pt.yComp]} C${pt.yCode}` : `${data.components[pt.xComp]} C${pt.xCode}`;
      const rows = activityRows(pt).map(r => {
        const bars = r.active.length
          ? r.active.map(x => `<div class="sessionTipBar"><span>${esc(x.name)}</span><i><b style="width:${Math.round(x.frac * 100)}%;background:${r.color}"></b></i><em>${fmt(x.seconds, 1)}s</em></div>`).join("")
          : `<div class="sessionTipEmpty">nothing sustained in this window</div>`;
        return `<div class="sessionTipMod" style="--mod:${r.color}"><b><i></i>${esc(r.mod)}</b><div>${bars}</div></div>`;
      }).join("");
      return `<b>${esc(s.participant)} ${esc(s.session)} ${esc(s.language)}</b><span class="tipMuted">${esc(data.scaleLabels[pt.scale])} window centered at ${fmt(w[W.anchor], 1)}s; ${esc(data.phases[w[W.phase]])}; ${esc(data.sessionPhases[w[W.sessionPhase]])}; ${esc(data.mutualAttentionLevels[w[W.mutual]])}; nearest naming ${fmt(w[W.nearest], 1)}s; ${esc(codeText)}</span><div class="sessionTipRows">${rows}</div>`;
    }
    function renderWindowDetail(pt) {
      if (!pt) {
        detail.innerHTML = "Hover over a session point to inspect the participant and selected window.";
        bars.innerHTML = "<b>Selected window behavior composition</b><br><span class='muted'>Bars show the mean value of each binary behavior feature inside the selected temporal window. 0 = absent, 1 = present throughout the window.</span>";
        return;
      }
      const s = sessionByKey.get(key(pt.model, pt.sid));
      const w = pt.window;
      const codeText = pt.pair ? `${data.components[pt.xComp]} C${pt.xCode} x ${data.components[pt.yComp]} C${pt.yCode}` : `${data.components[pt.xComp]} C${pt.xCode}`;
      detail.innerHTML = `<b>${esc(s.participant)} ${esc(s.session)} ${esc(s.language)}</b><br>aud. status: ${esc(s.hearing)}; group: ${esc(s.studyGroup)}; age: ${s.ageMonths ?? "n/a"} months<br>naming: ${esc(s.namingGroup)}; ${s.namingEvents} events; ${s.eventsPerMin} events/min<br>${esc(data.scaleLabels[pt.scale])}; ${esc(codeText)}<br>session progress: ${fmt(pt.progress * 100, 2)}%; time: ${fmt(w[W.anchor], 1)} sec; proximity: ${esc(data.phases[w[W.phase]])}; session phase: ${esc(data.sessionPhases[w[W.sessionPhase]])}; mutual attention: ${esc(data.mutualAttentionLevels[w[W.mutual]])}; nearest naming: ${fmt(w[W.nearest], 1)} sec`;
      const rows = data.channels.map((ch, i) => {
        const v = w[W.behaviors][i];
        return `<div class="behaviorRow"><span>${esc(ch)}</span><div><i style="width:${Math.round(clamp(v,0,1)*100)}%"></i></div><b>${fmt(v,2)}</b></div>`;
      }).join("");
      bars.innerHTML = `<b>Selected window behavior composition</b><br><span class="muted">Mean value in the selected temporal window.</span>${rows}`;
    }
    function renderSessionList(points, f) {
      const highlights = f.highlightSessions;
      const sorted = points.slice().sort((a,b) => {
        const sa = sessionByKey.get(key(a.model, a.sid));
        const sb = sessionByKey.get(key(b.model, b.sid));
        const ak = sessionBaseKey(sa), bk = sessionBaseKey(sb);
        const ah = highlights.has(ak) ? 0 : 1, bh = highlights.has(bk) ? 0 : 1;
        if (ah !== bh) return ah - bh;
        return `${sa.participant} ${sa.session}`.localeCompare(`${sb.participant} ${sb.session}`);
      });
      const rows = sorted.slice(0, 80).map(pt => {
        const s = sessionByKey.get(key(pt.model, pt.sid));
        const vals = pt.window[W.behaviors].map((v,i) => [data.channels[i], v]).sort((a,b) => b[1] - a[1]).slice(0, 5).map(([ch,v]) => `${esc(ch)}=${fmt(v,2)}`).join(", ");
        const cls = highlights.has(sessionBaseKey(s)) ? " sessionRow highlight" : "sessionRow";
        return `<div class="${cls}"><b>${esc(s.participant)} ${esc(s.session)}</b><span>${esc(s.language)}; ${esc(s.namingGroup)}; ${esc(data.phases[pt.window[W.phase]])}; ${esc(data.sessionPhases[pt.window[W.sessionPhase]])}; ${esc(data.mutualAttentionLevels[pt.window[W.mutual]])}</span><em>${vals}</em></div>`;
      }).join("");
      sessionList.innerHTML = `<b>Displayed participants at selected time</b><span class="muted">Highlighted sessions are listed first. Values are behavior means in the selected window.</span>${rows || "<p class='muted'>No sessions match the current filters.</p>"}`;
    }
    function filterSummary(f) {
      const model = data.models.find(m => m.idx === f.model)?.label || "model";
      return [
        `model: ${model}`,
        `scale: ${data.scaleLabels[f.scale]}`,
        `x: ${data.components[f.xComp]}`,
        `y: ${data.components[f.yComp]}`,
        `map: ${f.mapMode}`,
        `background: ${f.bgMode}`,
        `latent positions: ${f.latentPositions ? "on" : "off"}`,
        `code usage: ${f.showCodeUsage ? "on" : "off"}`,
        `heatmap: ${f.showHeatmap ? "on" : "off"}`,
        `bar chart: ${f.showBarChart ? "on" : "off"}`,
        `topology mode: ${f.topologyMode}`,
        `topology context: ${fmt(f.topologyContext * 100, 1)}%`,
        `transition mode: ${f.transitionMode}`,
        `transition context: ${fmt(f.transitionContext * 100, 1)}%`,
        `tail length: ${fmt(f.tail * 100, 1)}%`,
        `speed: ${visualizationSpeedLabel(f.windowsPerSecond)}`,
        `codes: ${f.showCodes ? "on" : "off"}`,
        `session points: ${f.showKeypoints ? "on" : "off"}`,
        `session trails: ${f.showTrails ? "on" : "off"}`,
        `naming stars: ${f.showNamingStars ? "on" : "off"}`,
        `topology: ${f.showTopology ? "on" : "off"}`,
        `transition arrows: ${f.showTransitions ? "on" : "off"}`,
        `time: ${fmt(f.progress * 100, 2)}%`,
        `color: ${f.color}`,
        `languages: ${[...f.languages].join("/") || "none"}`,
        `aud. status: ${[...f.hearings].join("/") || "none"}`,
        `sessions: ${[...f.sessions].join("/") || "none"}`,
        `naming: ${[...f.naming].join("/") || "none"}`,
        `proximity: ${[...f.phases].join("/") || "none"}`,
        `session phase: ${[...f.sessionPhases].join("/") || "none"}`,
        `mutual attention: ${[...f.mutualAttention].join("/") || "none"}`,
        `selected session filter: ${f.selectedSessionsActive ? "on" : "off"}`,
        `filtered sessions: ${f.selectedSessionsActive ? f.filterSessions.size : "inactive"}`,
        `highlighted sessions: ${f.highlightSessions.size || "none"}`,
      ].join(" | ");
    }
    function distToSegment(px, py, ax, ay, bx, by) {
      const dx = bx - ax, dy = by - ay;
      if (Math.abs(dx) + Math.abs(dy) < 1e-9) return Math.hypot(px - ax, py - ay);
      const t = clamp(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0, 1);
      return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    }
    function qPoint(a, c, b, t) {
      const u = 1 - t;
      return {
        x: u * u * a.x + 2 * u * t * c.x + t * t * b.x,
        y: u * u * a.y + 2 * u * t * c.y + t * t * b.y,
      };
    }
    function distToQuadratic(px, py, a, c, b) {
      let best = Infinity;
      let prev = qPoint(a, c, b, 0);
      for (let i = 1; i <= 18; i++) {
        const cur = qPoint(a, c, b, i / 18);
        best = Math.min(best, distToSegment(px, py, prev.x, prev.y, cur.x, cur.y));
        prev = cur;
      }
      return best;
    }
    function smoothPolyline(points, passes = 2) {
      let out = points.slice();
      for (let pass = 0; pass < passes; pass++) {
        if (out.length < 3) break;
        const next = [out[0]];
        for (let i = 0; i < out.length - 1; i++) {
          const a = out[i], b = out[i + 1];
          next.push({ x: a.x * 0.75 + b.x * 0.25, y: a.y * 0.75 + b.y * 0.25 });
          next.push({ x: a.x * 0.25 + b.x * 0.75, y: a.y * 0.25 + b.y * 0.75 });
        }
        next.push(out[out.length - 1]);
        out = next;
      }
      return out;
    }
    function drawPaths(seriesRows, f, ex, rect) {
      const tail = clamp(f.tail, 0, 1);
      currentTailHits = [];
      if (!tail || !f.showTrails) return;
      for (const row of seriesRows) {
        const start = Math.max(0, f.progress - tail);
        const end = f.progress;
        const pts = pathSlice(row.series, start, end);
        if (pts.length < 2) continue;
        const sKey = sessionBaseKey(row.session);
        const highlighted = f.highlightSessions.has(sKey);
        const hovered = hoveredSessionKey === sKey;
        const mapped = pts.map(p => mapPt(p, ex, rect));
        const smooth = smoothPolyline(mapped, pts.length >= 5 ? 2 : 1);
        const baseAlpha = hovered ? 0.92 : highlighted ? 0.68 : 0.30;
        const lineWidth = hovered ? 4.2 : highlighted ? 3.0 : 1.7;
        const stroke = hovered || highlighted ? "#000" : "rgba(0,0,0,.58)";
        for (let i = 0; i < mapped.length - 1; i++) {
          currentTailHits.push({ sessionKey:sKey, pt:pts[i + 1], x1:mapped[i].x, y1:mapped[i].y, x2:mapped[i + 1].x, y2:mapped[i + 1].y });
        }
        for (let i = 0; i < smooth.length - 1; i++) {
          const fade = 0.25 + 0.75 * (i + 1) / Math.max(1, smooth.length - 1);
          ctx.save();
          ctx.globalAlpha = baseAlpha * fade;
          ctx.strokeStyle = stroke;
          ctx.lineWidth = lineWidth;
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.beginPath();
          ctx.moveTo(smooth[i].x, smooth[i].y);
          ctx.lineTo(smooth[i + 1].x, smooth[i + 1].y);
          ctx.stroke();
          ctx.restore();
        }
      }
    }
    function transitionState(pt) {
      if (pt.pair) return { key:`${pt.xCode},${pt.yCode}`, xCode:pt.xCode, yCode:pt.yCode, pair:true, label:`${pt.xCode}/${pt.yCode}`, color:bgColorForCodes(pt.xCode, pt.yCode).replace(/0\.\d+\)$/, "0.88)") };
      return { key:String(pt.code), code:pt.code, pair:false, label:`C${pt.code}`, color:palette[pt.code % palette.length] };
    }
    function stateCenters(cb, f) {
      const centers = new Map();
      if (cb.same) {
        for (const c of cb.items) {
          centers.set(String(c[C.code]), {
            x:c[C.z1], y:c[C.z2],
            color:palette[c[C.code] % palette.length],
            label:`C${c[C.code]}`,
          });
        }
      } else {
        const xs = new Map(cb.xItems.map(c => [c[C.code], c[C.pooled]]));
        const ys = new Map(cb.yItems.map(c => [c[C.code], c[C.pooled]]));
        for (const [xc, xv] of xs.entries()) {
          for (const [yc, yv] of ys.entries()) {
            centers.set(`${xc},${yc}`, {
              x:xv, y:yv,
              color:bgColorForCodes(xc, yc).replace(/0\.\d+\)$/, "0.88)"),
              label:`${xc}/${yc}`,
            });
          }
        }
      }
      return centers;
    }
    function transitionSlice(seq, f) {
      return contextSlice(seq, f.progress, f.transitionContext, 2);
    }
    function transitionEdges(seriesRows, f, cb, cmp = null, layoutCenters = null) {
      const counts = new Map();
      const totals = new Map();
      const seededCenters = layoutCenters || stateCenters(cb, f);
      const empiricalCenters = new Map();
      for (const row of seriesRows) {
        const seq = transitionSlice(row.series, f);
        for (const p of seq) {
          if (cmp && !comparePointPass(p, cmp)) continue;
          const st = transitionState(p);
          if (seededCenters.has(st.key)) continue;
          const old = empiricalCenters.get(st.key) || { x:0, y:0, n:0, color:st.color, label:st.label };
          old.x += p.x; old.y += p.y; old.n += 1;
          empiricalCenters.set(st.key, old);
        }
        for (let i = 0; i < seq.length - 1; i++) {
          if (cmp && (!comparePointPass(seq[i], cmp) || !comparePointPass(seq[i + 1], cmp))) continue;
          const a = transitionState(seq[i]);
          const b = transitionState(seq[i + 1]);
          totals.set(a.key, (totals.get(a.key) || 0) + 1);
          if (a.key === b.key) continue;
          const k = `${a.key}>${b.key}`;
          counts.set(k, (counts.get(k) || 0) + 1);
        }
      }
      for (const c of empiricalCenters.values()) {
        c.x /= Math.max(1, c.n);
        c.y /= Math.max(1, c.n);
      }
      const centers = new Map([...seededCenters.entries(), ...empiricalCenters.entries()]);
      const edges = [];
      for (const [k, count] of counts.entries()) {
        const [from, to] = k.split(">");
        const total = totals.get(from) || 1;
        const prob = count / total;
        if (count < 1 || prob < 0.025) continue;
        const a = centers.get(from), b = centers.get(to);
        if (!a || !b) continue;
        edges.push({ from, to, count, prob, a, b });
      }
      edges.sort((a,b) => (b.prob - a.prob) || (b.count - a.count));
      return edges.slice(0, 120);
    }
    function drawArrowBetween(a, b, prob, color, ex, rect, curve) {
      const p0 = mapPt(a, ex, rect), p1 = mapPt(b, ex, rect);
      const dx = p1.x - p0.x, dy = p1.y - p0.y;
      const len = Math.hypot(dx, dy);
      if (len < 14) return null;
      const ux = dx / len, uy = dy / len;
      const start = { x:p0.x + ux * 11, y:p0.y + uy * 11 };
      const end = { x:p1.x - ux * 13, y:p1.y - uy * 13 };
      const mx = (start.x + end.x) / 2, my = (start.y + end.y) / 2;
      const cx = mx - uy * curve, cy = my + ux * curve;
      const alpha = clamp(0.22 + prob * 1.35, 0.22, 0.9);
      const width = clamp(0.9 + prob * 9.0, 0.9, 7.5);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.quadraticCurveTo(cx, cy, end.x, end.y);
      ctx.stroke();
      const tx = end.x - cx, ty = end.y - cy;
      const tlen = Math.hypot(tx, ty) || 1;
      const ax = tx / tlen, ay = ty / tlen;
      const head = 6 + width * 0.9;
      ctx.beginPath();
      ctx.moveTo(end.x, end.y);
      ctx.lineTo(end.x - ax * head - ay * head * 0.45, end.y - ay * head + ax * head * 0.45);
      ctx.lineTo(end.x - ax * head + ay * head * 0.45, end.y - ay * head - ax * head * 0.45);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      return { start, ctrl:{x:cx, y:cy}, end };
    }
    function drawTransitions(seriesRows, f, ex, rect, cb, ring = null) {
      if (!f.showTransitions) return;
      currentTransitionHits = [];
      const layoutCenters = f.mapMode === "code_layout" && ring ? ring.centers : null;
      if (f.transitionMode === "compare") {
        const aEdges = transitionEdges(seriesRows, f, cb, f.transitionA, layoutCenters);
        const bEdges = transitionEdges(seriesRows, f, cb, f.transitionB, layoutCenters);
        const byKeyA = new Map(aEdges.map(e => [`${e.from}>${e.to}`, e]));
        const byKeyB = new Map(bEdges.map(e => [`${e.from}>${e.to}`, e]));
        const allKeys = new Set([...byKeyA.keys(), ...byKeyB.keys()]);
        const diffEdges = [];
        for (const k of allKeys) {
          const ea = byKeyA.get(k), eb = byKeyB.get(k);
          const base = ea || eb;
          const pa = ea ? ea.prob : 0;
          const pb = eb ? eb.prob : 0;
          const diff = pa - pb;
          if (Math.abs(diff) < 0.015) continue;
          diffEdges.push({
            from:base.from, to:base.to, a:base.a, b:base.b,
            prob:Math.abs(diff), diff, probA:pa, probB:pb,
            countA:ea ? ea.count : 0, countB:eb ? eb.count : 0,
          });
        }
        diffEdges.sort((a,b) => Math.abs(b.diff) - Math.abs(a.diff));
        const pairCounts = new Map();
        for (const e of diffEdges) {
          const pk = [e.from, e.to].sort().join("<>");
          pairCounts.set(pk, (pairCounts.get(pk) || 0) + 1);
        }
        for (const e of diffEdges.slice(0, 120)) {
          const pk = [e.from, e.to].sort().join("<>");
          const paired = (pairCounts.get(pk) || 0) > 1;
          const curve = paired ? (String(e.from) < String(e.to) ? 24 : -24) : 10;
          const color = e.diff >= 0 ? "#2166ac" : "#b2182b";
          const hit = drawArrowBetween(e.a, e.b, e.prob, color, ex, rect, curve);
          if (hit) currentTransitionHits.push({ ...hit, from:e.a.label, to:e.b.label, prob:e.prob, diff:e.diff, probA:e.probA, probB:e.probB, countA:e.countA, countB:e.countB });
        }
        return;
      }
      const edges = transitionEdges(seriesRows, f, cb, null, layoutCenters);
      const pairCounts = new Map();
      for (const e of edges) {
        const pk = [e.from, e.to].sort().join("<>");
        pairCounts.set(pk, (pairCounts.get(pk) || 0) + 1);
      }
      for (const e of edges) {
        const pk = [e.from, e.to].sort().join("<>");
        const paired = (pairCounts.get(pk) || 0) > 1;
        const curve = paired ? (String(e.from) < String(e.to) ? 24 : -24) : 10;
        const hit = drawArrowBetween(e.a, e.b, e.prob, e.a.color, ex, rect, curve);
        if (hit) currentTransitionHits.push({ ...hit, from:e.a.label, to:e.b.label, prob:e.prob, count:e.count });
      }
    }
    function legendChip(c, compIdx, prefix = "") {
      const label = prefix ? `${prefix}${c[C.code]}` : `C${c[C.code]}`;
      return `<span class="codeChip" data-model="${c[C.model]}" data-scale="${c[C.scale]}" data-comp="${compIdx}" data-code="${c[C.code]}"><i style="background:${palette[c[C.code] % palette.length]}"></i>${esc(label)}</span>`;
    }
    function markFromCodeChip(chip) {
      if (!chip) return null;
      return {
        model:Number(chip.dataset.model),
        scale:Number(chip.dataset.scale),
        comp:Number(chip.dataset.comp),
        code:Number(chip.dataset.code),
        label:chip.textContent.trim(),
      };
    }
    function showCodeChipProfile(chip) {
      const mark = markFromCodeChip(chip);
      if (!mark) return;
      renderCodeProfile(mark);
      setHoverInspector(codeProfileTooltip(mark), "profile");
    }
    function colorLegendHtml(f) {
      const sw = (color, label) => `<span><i style="background:${color}"></i>${esc(label)}</span>`;
      if (f.color === "language") return `<div class="legendColor"><b>Point color:</b>${sw("#b2182b","NGT")}${sw("#2166ac","NL")}</div>`;
      if (f.color === "aud") return `<div class="legendColor"><b>Point color:</b>${sw("#d73027","Deaf")}${sw("#1a9850","Hearing")}${sw("#984ea3","Comparison")}</div>`;
      if (f.color === "naming_binary") return `<div class="legendColor"><b>Point color:</b>${sw("#bdbdbd","low naming")}${sw("#111111","high naming")}</div>`;
      if (f.color === "naming_tertile") return `<div class="legendColor"><b>Point color:</b>${sw("#2166ac","low")}${sw("#fdd863","mid")}${sw("#b2182b","high")}</div>`;
      if (f.color === "naming_rate") return `<div class="legendColor"><b>Point color:</b><span class="gradientChip"></span><em>naming events/min: low to high</em></div>`;
      if (f.color === "session") return `<div class="legendColor"><b>Point color:</b>${sw("#e41a1c","S1")}${sw("#377eb8","S2")}${sw("#4daf4a","S3")}</div>`;
      if (f.color === "phase") return `<div class="legendColor"><b>Point color:</b>${data.phases.map((p,i) => sw(palette[i % palette.length], p)).join("")}</div>`;
      if (f.color === "session_phase") return `<div class="legendColor"><b>Point color:</b>${sw("#8dd3c7","early")}${sw("#fdb462","middle")}${sw("#bebada","late")}</div>`;
      if (f.color === "mutual_attention") return `<div class="legendColor"><b>Point color:</b>${sw("#bdbdbd","none")}${sw("#80b1d3","object")}${sw("#fb8072","person")}${sw("#b3de69","coordinated JA")}${sw("#fdb462","naming JA")}</div>`;
      if (f.color === "code") return `<div class="legendColor"><b>Point color:</b><em>same code colors as below</em></div>`;
      return "";
    }
    function compareSetLabel(set, labelMap = {}) {
      const vals = [...(set || new Set())].sort();
      if (!vals.length) return "all";
      return vals.map(v => labelMap[v] || v).join("/");
    }
    function compareDescriptor(a, b) {
      const vars = [
        ["Language", "languages", {}],
        ["Aud. status", "hearings", {"Hearing":"Hearing","Deaf":"Deaf","Comparison":"Comparison","Unknown":"Unknown"}],
        ["Session", "sessions", {}],
        ["Naming", "naming", {low:"low naming", high:"high naming"}],
        ["Near naming", "phases", {far:"far", before:"before", during:"during", after:"after"}],
        ["Session phase", "sessionPhases", {early:"early", middle:"middle", late:"late"}],
        ["Mutual attention", "mutualAttention", {
          "none":"none",
          "object-aligned":"object-aligned",
          "person-aligned":"person-aligned",
          "coordinated joint attention":"coordinated JA",
          "naming-aligned joint attention":"naming-aligned JA",
        }],
      ];
      const diffs = vars.filter(([, k]) => [...(a[k] || new Set())].sort().join("|") !== [...(b[k] || new Set())].sort().join("|"));
      if (!diffs.length) return { name:"Selected filters", a:"same", b:"same" };
      if (diffs.length === 1) {
        const [name, k, map] = diffs[0];
        return { name, a:compareSetLabel(a[k], map), b:compareSetLabel(b[k], map) };
      }
      return {
        name:"Selected filters",
        a:diffs.map(([name, k, map]) => `${name}: ${compareSetLabel(a[k], map)}`).join("; "),
        b:diffs.map(([name, k, map]) => `${name}: ${compareSetLabel(b[k], map)}`).join("; "),
      };
    }
    function legendSwatch(color, label, line = false) {
      const style = line ? `background:${color};border-radius:2px;height:3px` : `background:${color}`;
      return `<span><i style="${style}"></i>${esc(label)}</span>`;
    }
    function pointLegendBody(f) {
      if (!f.showKeypoints && !f.showTrails && !f.showNamingStars) return "";
      if (f.color === "language") return `${legendSwatch("#b2182b","NGT")}${legendSwatch("#2166ac","NL")}`;
      if (f.color === "aud") return `${legendSwatch("#d73027","Deaf")}${legendSwatch("#1a9850","Hearing")}${legendSwatch("#984ea3","Comparison")}`;
      if (f.color === "naming_binary") return `${legendSwatch("#bdbdbd","low naming")}${legendSwatch("#111111","high naming")}`;
      if (f.color === "naming_tertile") return `${legendSwatch("#2166ac","low")}${legendSwatch("#fdd863","mid")}${legendSwatch("#b2182b","high")}`;
      if (f.color === "naming_rate") return `<span class="gradientChip"></span><em>naming events/min: low to high</em>`;
      if (f.color === "session") return `${legendSwatch("#e41a1c","S1")}${legendSwatch("#377eb8","S2")}${legendSwatch("#4daf4a","S3")}`;
      if (f.color === "phase") return data.phases.map((p,i) => legendSwatch(palette[i % palette.length], p)).join("");
      if (f.color === "session_phase") return `${legendSwatch("#8dd3c7","early")}${legendSwatch("#fdb462","middle")}${legendSwatch("#bebada","late")}`;
      if (f.color === "mutual_attention") return `${legendSwatch("#bdbdbd","none")}${legendSwatch("#80b1d3","object")}${legendSwatch("#fb8072","person")}${legendSwatch("#b3de69","coordinated JA")}${legendSwatch("#fdb462","naming JA")}`;
      if (f.color === "code") return `<em>same code colors as below</em>`;
      return "";
    }
    function topologyLegendBody(f) {
      if (f.mapMode === "code_usage_chart") return "";
      if (f.mapMode === "heatmap_chart") return `${legendSwatch("#fff","low")}${legendSwatch("#f46d43","higher co-occurrence")}`;
      if (!f.showTopology && !f.showHeatmap) return "";
      if (f.topologyMode === "compare") {
        const cmp = compareDescriptor(f.topologyA, f.topologyB);
        if (f.bgMode === "code" && f.mapMode === "latent") {
          return `${legendSwatch("#111", `${cmp.name}: ${cmp.a}`, true)}${legendSwatch("#fff", `${cmp.name}: ${cmp.b}`, true)}`;
        }
        return `${legendSwatch("#2166ac", `${cmp.name}: ${cmp.a}`, true)}${legendSwatch("#b2182b", `${cmp.name}: ${cmp.b}`, true)}`;
      }
      return legendSwatch("#111", "density", true);
    }
    function transitionLegendBody(f) {
      if (!f.showTransitions) return "";
      if (f.transitionMode === "compare") {
        const cmp = compareDescriptor(f.transitionA, f.transitionB);
        return `${legendSwatch("#2166ac", `${cmp.name}: ${cmp.a}`, true)}${legendSwatch("#b2182b", `${cmp.name}: ${cmp.b}`, true)}`;
      }
      return legendSwatch("#333", "observed probability", true);
    }
    function codeUsageLegendBody(f) {
      if (f.mapMode !== "code_usage_chart") return "";
      if (!f.codeUsageStack || f.codeUsageStack === "none") return `<em>bars use code colors</em>`;
      return stackOptionsForMode(f.codeUsageStack).map(x => legendSwatch(x.color, x.label)).join("");
    }
    function legendLayerRowHtml(f) {
      const items = [
        ["Code usage", codeUsageLegendBody(f)],
        ["Point color", pointLegendBody(f)],
        [f.showHeatmap && f.showTopology ? "Topology + heatmap" : f.showHeatmap ? "Heatmap" : "Topology", topologyLegendBody(f)],
        ["Transitions", transitionLegendBody(f)],
      ].filter(([, body]) => body);
      if (!items.length) return "";
      return `<div class="legendLayerRow">${items.map(([title, body]) => `<div class="legendLayerItem"><b>${esc(title)}:</b>${body}</div>`).join("")}</div>`;
    }
    function topologyLegendHtml(f) {
      if (f.mapMode === "code_usage_chart") return `<div class="legendColor"><b>Code usage:</b><em>Bars count filtered windows assigned to each code.</em></div>`;
      if (f.mapMode === "heatmap_chart") {
        const sw = (color, label) => `<span><i style="background:${color};border-radius:2px"></i>${esc(label)}</span>`;
        return `<div class="legendColor"><b>Heatmap:</b>${sw("#fff","low")}${sw("#f46d43","higher co-occurrence")}</div>`;
      }
      if (!f.showTopology && !f.showHeatmap) return "";
      const sw = (color, label) => `<span><i style="background:${color};border-radius:2px;height:3px"></i>${esc(label)}</span>`;
      if (f.topologyMode === "compare") {
        if (f.bgMode === "code" && f.mapMode === "latent") {
          return `<div class="legendColor"><b>${f.showHeatmap && f.showTopology ? "Topology + heatmap" : f.showHeatmap ? "Heatmap" : "Topology"}:</b>${sw("#111","A > B")}${sw("#fff","B > A")}</div>`;
        }
        return `<div class="legendColor"><b>${f.showHeatmap && f.showTopology ? "Topology + heatmap" : f.showHeatmap ? "Heatmap" : "Topology"}:</b>${sw("#2166ac","A > B")}${sw("#b2182b","B > A")}</div>`;
      }
      return `<div class="legendColor"><b>${f.showHeatmap && f.showTopology ? "Topology + heatmap" : f.showHeatmap ? "Heatmap" : "Topology"}:</b>${sw("#111","density")}</div>`;
    }
    function transitionLegendHtml(f) {
      if (!f.showTransitions) return "";
      const sw = (color, label) => `<span><i style="background:${color};border-radius:2px;height:3px"></i>${esc(label)}</span>`;
      if (f.transitionMode === "compare") {
        return `<div class="legendColor"><b>Transitions:</b>${sw("#2166ac","A > B")}${sw("#b2182b","B > A")}</div>`;
      }
      return `<div class="legendColor"><b>Transitions:</b>${sw("#333","observed probability")}</div>`;
    }
    function renderLegend(f, cb, windows, pointCount) {
      if (!legend) return;
      const codeGroups = [];
      if (cb.same) {
        const chips = cb.items.slice().sort((a,b) => a[C.code] - b[C.code]).map(c => legendChip(c, f.xComp)).join("");
        codeGroups.push(`<div class="legendCodeGroup"><b>${esc(data.components[f.xComp])}</b><div class="legendCodes">${chips}</div></div>`);
      } else {
        const xCodes = cb.xItems.slice().sort((a,b) => a[C.code] - b[C.code]).map(c => legendChip(c, f.xComp, "x")).join("");
        const yCodes = cb.yItems.slice().sort((a,b) => a[C.code] - b[C.code]).map(c => legendChip(c, f.yComp, "y")).join("");
        codeGroups.push(`<div class="legendCodeGroup"><b>X: ${esc(data.components[f.xComp])}</b><div class="legendCodes axisCodes">${xCodes}</div></div>`);
        codeGroups.push(`<div class="legendCodeGroup"><b>Y: ${esc(data.components[f.yComp])}</b><div class="legendCodes axisCodes">${yCodes}</div></div>`);
      }
      const countLine = `${pointCount} sessions at frame; ${windows} windows visible`;
      legend.innerHTML = `
        <div class="legendTitle">Legend</div>
        <div class="legendSub">${esc(countLine)}</div>
        ${legendLayerRowHtml(f)}
        ${codeGroups.join("")}
      `;
      legend.querySelectorAll(".codeChip").forEach(chip => {
        const showChipProfile = () => showCodeChipProfile(chip);
        chip.addEventListener("mouseenter", showChipProfile);
        chip.addEventListener("pointerenter", showChipProfile);
        chip.addEventListener("mousemove", showChipProfile);
        chip.addEventListener("click", showChipProfile);
      });
      layoutRightRail();
    }
    function arr(v) { return Array.isArray(v) ? v : (v ? [v] : []); }
    function hasAny(values, wanted) {
      if (!wanted || !wanted.size) return false;
      return arr(values).some(v => wanted.has(String(v)));
    }
    function componentRelevant(asset, f) {
      const x = data.components[f.xComp], y = data.components[f.yComp];
      const mods = arr(asset.modalities).map(String);
      const comp = String(asset.component || "");
      const scope = String(asset.codeScope || "");
      if (comp === x || comp === y) return true;
      if (mods.includes(x) || mods.includes(y)) return true;
      if ((x === "joint" || y === "joint") && (scope.includes("multimodal") || comp === "joint")) return true;
      if (scope === "modality_pair" && mods.includes(x) && mods.includes(y)) return true;
      return false;
    }
    function relatedScore(asset, f) {
      let score = 0;
      const scale = data.scales[f.scale];
      if (asset.scale === scale) score += 38;
      else if (asset.scale === "all") score += 8;
      if (componentRelevant(asset, f)) score += 34;
      if (hasAny(asset.language, f.languages) || arr(asset.language).includes("all")) score += 10;
      if (hasAny(asset.hearing, f.hearings) || arr(asset.hearing).includes("all")) score += 6;
      if (hasAny(asset.session, f.sessions) || arr(asset.session).includes("all")) score += 6;
      if (hasAny(asset.naming, f.naming) || arr(asset.naming).includes("all")) score += 8;
      if (hasAny(asset.proximity, f.phases) || arr(asset.proximity).includes("all")) score += 8;
      if (f.showTopology && String(asset.title || asset.name).toLowerCase().includes("topology")) score += 14;
      if (f.showHeatmap && `${asset.title || ""} ${asset.name || ""}`.toLowerCase().match(/heatmap|density/)) score += 22;
      if (f.showTransitions && String(asset.title || asset.name).toLowerCase().includes("transition")) score += 14;
      if (f.showCodes && String(asset.title || asset.name).toLowerCase().includes("code")) score += 8;
      if (f.showTransitions && String(asset.title || asset.name).toLowerCase().includes("transition")) score += 24;
      if (f.topologyMode === "compare" && asset.differential) score += 18;
      if (f.transitionMode === "compare" && asset.differential && String(asset.title || asset.name).toLowerCase().includes("transition")) score += 18;
      if (f.showNamingStars && String(asset.title || asset.name).toLowerCase().includes("naming")) score += 8;
      if (asset.type === "video" && (f.showKeypoints || f.showTrails)) score += 8;
      score += Math.min(8, Number(asset.separability || 0) * 8);
      return score;
    }
    function relatedThumb(asset) {
      if (asset.type === "image") return `<img loading="lazy" src="${esc(asset.path)}" alt="">`;
      if (asset.type === "video") return `<video muted preload="metadata" src="${esc(asset.path)}"></video>`;
      return `<div class="relatedFile">${esc(String(asset.type || "file")).toUpperCase()}</div>`;
    }
    function setCheckGroup(id, values) {
      const wanted = new Set(arr(values).map(String));
      const all = !wanted.size || wanted.has("all");
      document.querySelectorAll(`#${id} input[type=checkbox]`).forEach(box => {
        box.checked = all || wanted.has(box.value);
      });
    }
    function checkedArray(id) {
      return [...document.querySelectorAll(`#${id} input[type=checkbox]:checked`)].map(x => x.value);
    }
    function setCheckGroupExact(id, values) {
      const wanted = new Set(arr(values).map(String));
      document.querySelectorAll(`#${id} input[type=checkbox]`).forEach(box => {
        box.checked = wanted.has(box.value);
      });
    }
    const compareFieldIds = [
      ["language", "embedLanguageChecks"],
      ["hearing", "embedHearingChecks"],
      ["session", "embedSessionChecks"],
      ["naming", "embedNamingChecks"],
      ["proximity", "embedProximityChecks"],
      ["sessionPhase", "embedSessionPhaseChecks"],
      ["mutualAttention", "embedMutualAttentionChecks"],
    ];
    function copyGlobalToComparePrefix(prefix) {
      for (const [field, globalId] of compareFieldIds) setCheckGroupExact(`${prefix}_${field}`, checkedArray(globalId));
    }
    function copyComparePrefix(fromPrefix, toPrefix) {
      for (const [field] of compareFieldIds) setCheckGroupExact(`${toPrefix}_${field}`, checkedArray(`${fromPrefix}_${field}`));
    }
    function swapComparePrefixes(aPrefix, bPrefix) {
      for (const [field] of compareFieldIds) {
        const a = checkedArray(`${aPrefix}_${field}`);
        const b = checkedArray(`${bPrefix}_${field}`);
        setCheckGroupExact(`${aPrefix}_${field}`, b);
        setCheckGroupExact(`${bPrefix}_${field}`, a);
      }
    }
    function setSelectIfPossible(select, value) {
      if (!select || value === undefined || value === null || value === "all") return false;
      const v = String(value);
      const opt = [...select.options].find(o => o.value === v || o.textContent === v);
      if (!opt) return false;
      select.value = opt.value;
      return true;
    }
    function applyAssetSettings(asset) {
      const lower = `${asset.name || ""} ${asset.title || ""} ${asset.description || ""} ${asset.section || ""} ${asset.codeScope || ""}`.toLowerCase();
      const vals = field => arr(asset[field]).map(String).filter(v => v && v !== "all");
      const hasVals = (field, wanted) => wanted.every(v => vals(field).includes(v));
      const compName = String(asset.component || "");
      const mods = vals("modalities").filter(v => data.components.includes(v));
      const pairFromComponent = compName.includes("_x_") ? compName.split("_x_").filter(v => data.components.includes(v)) : [];
      const scope = String(asset.codeScope || "");
      const section = String(asset.section || "");
      const isTransition = lower.includes("transition");
      const isTopology = lower.includes("topology") || section === "A_holistic_state_maps" || section === "B_modality_pair_topology";
      const isPath = lower.includes("path") || section === "I_videos" || asset.type === "video";
      const isHeatmap = lower.includes("heatmap") || lower.includes("density") || lower.includes("cooccurrence") || lower.includes("co-occurrence");
      const isCodeProfile = section === "H_behavior_composition" || lower.includes("composition") || lower.includes("code_share") || lower.includes("code share") || lower.includes("occupancy") || lower.includes("bar");

      function compIndex(name) {
        const idx = data.components.indexOf(String(name || ""));
        return idx >= 0 ? idx : null;
      }
      function setAxisNames(xName, yName = xName) {
        const xIdx = compIndex(xName), yIdx = compIndex(yName);
        if (xIdx !== null) controls.xComp.value = String(xIdx);
        if (yIdx !== null) controls.yComp.value = String(yIdx);
      }
      function setAssetAxes() {
        if (scope === "modality_pair" || pairFromComponent.length >= 2) {
          const pair = pairFromComponent.length >= 2 ? pairFromComponent : mods.slice(0, 2);
          if (pair.length >= 2) { setAxisNames(pair[0], pair[1]); return; }
        }
        if (compName && compName !== "all" && data.components.includes(compName)) {
          setAxisNames(compName);
          return;
        }
        if (mods.length === 1) {
          setAxisNames(mods[0]);
          return;
        }
        if (mods.includes("joint") || lower.includes("c0")) {
          setAxisNames("joint");
        }
      }
      function setSpatial({ bg, latent, keypoints, trails, stars, topology, transitions }) {
        if (controls.showBg) controls.showBg.checked = !!bg;
        if (controls.latentPositions) controls.latentPositions.checked = !!latent;
        if (controls.showKeypoints) controls.showKeypoints.checked = !!keypoints;
        if (controls.showTrails) controls.showTrails.checked = !!trails;
        if (controls.showNamingStars) controls.showNamingStars.checked = !!stars;
        if (controls.showTopology) controls.showTopology.checked = !!topology;
        if (controls.showTransitions) controls.showTransitions.checked = !!transitions;
        if (controls.showHeatmap) controls.showHeatmap.checked = false;
        if (controls.showBarChart) controls.showBarChart.checked = false;
      }
      function applyGlobalFields() {
        setCheckGroup("embedLanguageChecks", asset.language);
        setCheckGroup("embedHearingChecks", asset.hearing);
        setCheckGroup("embedSessionChecks", asset.session);
        setCheckGroup("embedNamingChecks", asset.naming);
        setCheckGroup("embedProximityChecks", asset.proximity);
      }
      function seedCompare(prefix) {
        setCheckGroup(`${prefix}_language`, asset.language);
        setCheckGroup(`${prefix}_hearing`, asset.hearing);
        setCheckGroup(`${prefix}_session`, asset.session);
        setCheckGroup(`${prefix}_naming`, asset.naming);
        setCheckGroup(`${prefix}_proximity`, asset.proximity);
      }
      function compareKind() {
        if (lower.match(/s1[_ -]?vs[_ -]?s2/) || lower.match(/s1[_ -]?s2/) || lower.includes("session 1 vs 2")) return "session";
        if (lower.includes("ngt_nl") || lower.includes("ngt vs nl") || lower.includes("language contrast") || (asset.differential && hasVals("language", ["NGT", "NL"]))) return "language";
        if (lower.includes("hearing_vs_deaf") || lower.includes("hearing vs deaf") || (asset.differential && hasVals("hearing", ["Hearing", "Deaf"]))) return "hearing";
        if (lower.includes("high_low") || lower.includes("highlow") || lower.includes("low vs high") || lower.includes("high vs low") || (asset.differential && hasVals("naming", ["low", "high"]))) return "naming";
        if (lower.includes("far_before_during_after") || lower.includes("before_during_after") || lower.includes("proximity")) return "proximity";
        return "";
      }
      function applyCompare(prefixA, prefixB, kind) {
        seedCompare(prefixA);
        seedCompare(prefixB);
        if (kind === "session") {
          setCheckGroup(`${prefixA}_session`, ["S1"]);
          setCheckGroup(`${prefixB}_session`, ["S2"]);
        } else if (kind === "language") {
          setCheckGroup(`${prefixA}_language`, ["NGT"]);
          setCheckGroup(`${prefixB}_language`, ["NL"]);
        } else if (kind === "hearing") {
          setCheckGroup(`${prefixA}_hearing`, ["Hearing"]);
          setCheckGroup(`${prefixB}_hearing`, ["Deaf"]);
        } else if (kind === "naming") {
          setCheckGroup(`${prefixA}_naming`, ["low"]);
          setCheckGroup(`${prefixB}_naming`, ["high"]);
        } else if (kind === "proximity") {
          setCheckGroup(`${prefixA}_proximity`, ["far", "before"]);
          setCheckGroup(`${prefixB}_proximity`, ["during", "after"]);
        }
      }
      function setContextFull(control) {
        if (control) control.value = "1000";
      }

      if (String(asset.scale || "all") !== "all") {
        const scaleIdx = data.scales.indexOf(String(asset.scale));
        if (scaleIdx >= 0) {
          controls.scale.value = String(scaleIdx);
          fillAxisControls();
        }
      }
      setAssetAxes();
      applyGlobalFields();
      const kind = compareKind();
      const useCompare = !!asset.differential || (isTransition && ["session", "language", "hearing", "naming"].includes(kind));

      if (isTransition) {
        setSpatial({ bg:false, latent:false, keypoints:false, trails:false, stars:false, topology:false, transitions:true });
        controls.topologyMode.value = "density";
        controls.transitionMode.value = useCompare ? "compare" : "observed";
        setContextFull(controls.transitionContext);
        if (useCompare) applyCompare("transA", "transB", kind);
        openLayer("transitions");
      } else if (isTopology) {
        setSpatial({ bg:true, latent:true, keypoints:false, trails:false, stars:false, topology:true, transitions:false });
        controls.transitionMode.value = "observed";
        controls.topologyMode.value = useCompare && kind !== "proximity" ? "compare" : "density";
        setContextFull(controls.topologyContext);
        if (controls.topologyMode.value === "compare") applyCompare("topoA", "topoB", kind);
        openLayer("topology");
      } else if (isPath) {
        setSpatial({ bg:true, latent:true, keypoints:true, trails:true, stars:lower.includes("naming") || lower.includes("event"), topology:false, transitions:false });
        controls.topologyMode.value = "density";
        controls.transitionMode.value = "observed";
        openLayer("dots");
      } else if (isHeatmap) {
        setSpatial({ bg:false, latent:true, keypoints:false, trails:false, stars:false, topology:false, transitions:false });
        controls.topologyMode.value = "density";
        controls.transitionMode.value = "observed";
        openLayer("background");
      } else if (isCodeProfile) {
        setSpatial({ bg:false, latent:false, keypoints:false, trails:false, stars:false, topology:false, transitions:false });
        controls.topologyMode.value = "density";
        controls.transitionMode.value = "observed";
        openLayer("background");
      } else {
        setSpatial({ bg:true, latent:true, keypoints:true, trails:false, stars:false, topology:false, transitions:false });
        controls.topologyMode.value = "density";
        controls.transitionMode.value = "observed";
        openLayer("dots");
      }

      if (kind === "session") controls.color.value = "session";
      else if (kind === "language") controls.color.value = "language";
      else if (kind === "hearing") controls.color.value = "aud";
      else if (kind === "naming") controls.color.value = "naming_binary";
      else if (kind === "proximity") controls.color.value = "phase";
      else if (hasVals("naming", ["low", "high"])) controls.color.value = "naming_binary";
      else if (vals("language").length) controls.color.value = "language";
      else if (vals("hearing").length) controls.color.value = "aud";
      else if (hasVals("session", ["S1", "S2"])) controls.color.value = "session";
      else if (vals("proximity").length) controls.color.value = "phase";
      else if (scope.includes("code")) controls.color.value = "code";

      refreshSessionPickers();
      updateMovementControls();
      syncCompareAccordions();
      viewEx = null; viewKey = "";
      draw();
    }
    function applyHighLowPreset(preset) {
      const modelIdx = Number(preset.model ?? controls.model.value ?? 0);
      if (Number(controls.model.value || 0) !== modelIdx) {
        controls.model.value = String(modelIdx);
        fillScaleControls();
      }
      const scaleIdx = data.scales.indexOf(String(preset.scale));
      if (scaleIdx >= 0) {
        controls.scale.value = String(scaleIdx);
        fillAxisControls();
      }
      const xIdx = data.components.indexOf(String(preset.x));
      const yIdx = data.components.indexOf(String(preset.y));
      if (xIdx >= 0) controls.xComp.value = String(xIdx);
      if (yIdx >= 0) controls.yComp.value = String(yIdx);
      const pf = preset.filters || {};
      if (controls.preset) controls.preset.value = String(preset.rank || "");
      setCheckGroup("embedLanguageChecks", pf.language || ["all"]);
      setCheckGroup("embedHearingChecks", pf.hearing || ["all"]);
      setCheckGroup("embedSessionChecks", pf.session || ["all"]);
      setCheckGroup("embedNamingChecks", ["low", "high"]);
      setCheckGroup("embedProximityChecks", pf.proximity || ["during"]);
      setCheckGroup("embedSessionPhaseChecks", pf.sessionPhase || ["all"]);
      setCheckGroup("embedMutualAttentionChecks", pf.mutualAttention || ["all"]);
      if (controls.selectedSessionsActive) controls.selectedSessionsActive.checked = false;
      if (controls.showBg) controls.showBg.checked = true;
      if (controls.latentPositions) controls.latentPositions.checked = true;
      controls.showKeypoints.checked = true;
      controls.showTrails.checked = false;
      controls.showNamingStars.checked = false;
      controls.showTopology.checked = true;
      controls.topologyMode.value = "compare";
      controls.topologyContext.value = "1000";
      controls.showTransitions.checked = false;
      controls.transitionMode.value = "observed";
      controls.color.value = "naming_binary";
      ["language", "hearing", "session", "proximity", "sessionPhase", "mutualAttention"].forEach(field => {
        const fallback = field === "proximity" ? ["during"] : ["all"];
        setCheckGroup(`topoA_${field}`, pf[field] || fallback);
        setCheckGroup(`topoB_${field}`, pf[field] || fallback);
      });
      setCheckGroup("topoA_naming", ["low", "high"]);
      setCheckGroup("topoB_naming", ["low", "high"]);
      const compareField = String(preset.compareField || "naming");
      const compareA = preset.compareA || ["low"];
      const compareB = preset.compareB || ["high"];
      if (compareField === "naming") {
        setCheckGroup("topoA_naming", compareA);
        setCheckGroup("topoB_naming", compareB);
      } else if (compareField === "session") {
        setCheckGroup("topoA_session", compareA);
        setCheckGroup("topoB_session", compareB);
        controls.color.value = "session";
      } else if (compareField === "language") {
        setCheckGroup("topoA_language", compareA);
        setCheckGroup("topoB_language", compareB);
        controls.color.value = "language";
      } else if (compareField === "hearing") {
        setCheckGroup("topoA_hearing", compareA);
        setCheckGroup("topoB_hearing", compareB);
        controls.color.value = "aud";
      }
      openLayer("topology");
      refreshSessionPickers();
      updateMovementControls();
      syncCompareAccordions();
      viewEx = null; viewKey = "";
      draw();
    }
    function renderHighLowPresets(f) {
      if (!relatedPanel || !relatedList) return false;
      const presets = (data.highLowPresets || []).filter(p => Number(p.model ?? 0) === f.model).slice(0, 9);
      if (!presets.length) return false;
      if (relatedTitleText) relatedTitleText.textContent = "High/low presets";
      if (relatedControls) relatedControls.style.display = "none";
      if (relatedCount) relatedCount.textContent = `${presets.length}`;
      const meta = relatedPanel.querySelector(".relatedMeta");
      if (meta) meta.textContent = "Click a preset to align the live embedding view to a high-vs-low naming contrast.";
      const activeScale = data.scales[f.scale];
      const activeX = data.components[f.xComp], activeY = data.components[f.yComp];
      relatedList.classList.add("presetGrid");
      relatedList.innerHTML = presets.map(p => {
        const active = String(p.scale) === activeScale && String(p.x) === activeX && String(p.y) === activeY;
        const xy = String(p.x) === String(p.y) ? String(p.x) : `${p.x} x ${p.y}`;
        return `<button type="button" class="presetButton${active ? " active" : ""}" data-rank="${p.rank}">
          <b>${esc(p.label || `View ${p.rank}`)}</b>
          <span>${esc(p.scaleLabel || p.scale)} | ${esc(xy)}</span>
          <em>sep ${fmt(Number(p.auc || 0), 3)} | ${Number(p.nLow || 0)}L/${Number(p.nHigh || 0)}H</em>
        </button>`;
      }).join("");
      relatedList.querySelectorAll(".presetButton").forEach(btn => {
        btn.addEventListener("click", () => {
          const preset = presets.find(p => String(p.rank) === String(btn.dataset.rank));
          if (preset) applyHighLowPreset(preset);
        });
      });
      layoutRightRail();
      return true;
    }
    function renderRelatedFigures(f) {
      if (!relatedPanel || !relatedList) return;
      if (relatedTitleText) relatedTitleText.textContent = "Related figures";
      if (relatedControls) relatedControls.style.display = "";
      relatedList.classList.remove("presetGrid");
      const q = (relatedSearch?.value || "").trim().toLowerCase();
      const textMatch = asset => {
        if (!q) return true;
        return [asset.title, asset.name, asset.path, asset.description, asset.section, asset.component, ...(asset.modalities || [])]
          .join(" ").toLowerCase().includes(q);
      };
      const selected = assets
        .filter(a => a.type !== "video")
        .filter(textMatch)
        .map(a => ({ asset:a, score:relatedScore(a, f) }))
        .filter(x => relatedMode === "all" || x.score >= 36)
        .sort((a,b) => b.score - a.score || (a.asset.storyOrder || 999) - (b.asset.storyOrder || 999) || String(a.asset.name).localeCompare(String(b.asset.name)))
        .slice(0, relatedMode === "all" ? 80 : 12);
      if (relatedCount) relatedCount.textContent = `${selected.length}`;
      const scale = data.scaleLabels[f.scale];
      const x = data.components[f.xComp], y = data.components[f.yComp];
      const active = [
        scale,
        x === y ? x : `${x} x ${y}`,
        f.mapMode === "code_usage_chart" ? "code usage" : f.mapMode === "heatmap_chart" ? "heatmap" : f.mapMode === "code_decomp" ? "code profiles" : f.mapMode === "code_layout" ? "code positions" : "latent",
        f.bgMode === "none" ? "no bg" : "code bg",
        f.showHeatmap ? "heatmap" : null,
        f.showTopology ? `topology: ${f.topologyMode}` : null,
        f.showTransitions ? `transitions: ${f.transitionMode}` : null,
        f.showTrails ? "trails" : "no trails",
        f.color.replace("naming_", "naming "),
      ].filter(Boolean);
      const meta = relatedPanel.querySelector(".relatedMeta");
      if (meta) meta.textContent = active.join(" | ");
      relatedList.innerHTML = selected.length ? selected.map(({asset, score}) => `
        <div class="relatedCard" title="${esc(asset.description || asset.title || asset.name)}">
          <a class="relatedMedia" href="${esc(asset.path)}" target="_blank">${relatedThumb(asset)}</a>
          <a class="relatedText" href="${esc(asset.path)}" target="_blank">
            <b>${esc(asset.title || asset.name)}</b>
            <span>${esc(asset.scaleLabel || asset.scale || "all")} | ${esc(asset.section || "")}</span>
          </a>
          <div class="relatedActions">
            <button type="button" class="relatedApply" data-id="${esc(asset.id)}">Apply</button>
            <em>${Math.round(score)}</em>
          </div>
        </div>
      `).join("") : `<div class="relatedEmpty">No image figures match this view.</div>`;
      relatedList.querySelectorAll(".relatedApply").forEach(btn => {
        btn.addEventListener("click", ev => {
          ev.preventDefault();
          const asset = assets.find(a => String(a.id) === String(btn.dataset.id));
          if (asset) applyAssetSettings(asset);
        });
      });
      layoutRightRail();
    }
    function contourPoints(seriesRows, pts, f) {
      const out = [];
      if (f.topologyContext >= 0.999) {
        for (const row of seriesRows) for (const p of row.series) out.push(p);
        return out;
      }
      for (const row of seriesRows) {
        for (const p of contextSlice(row.series, f.progress, f.topologyContext, 1)) out.push(p);
      }
      return out.length ? out : pts;
    }
    function edgeInterp(a, b, va, vb, t) {
      const den = vb - va;
      const u = Math.abs(den) < 1e-9 ? 0.5 : clamp((t - va) / den, 0, 1);
      return { x:a.x + (b.x - a.x) * u, y:a.y + (b.y - a.y) * u };
    }
    function drawContourSet(src, ex, rect, strokes, halo = null) {
      src = src.filter(Boolean);
      if (src.length < 4) return;
      const cap = 2500;
      if (src.length > cap) {
        const step = Math.ceil(src.length / cap);
        src = src.filter((_, i) => i % step === 0);
      }
      const nx = 58, ny = 58;
      const grid = Array.from({length: ny}, () => Array(nx).fill(0));
      const sigma = 1.45, radius = 4;
      for (const p of src) {
        const m = mapPt(p, ex, rect);
        const gx = (m.x - rect.x) / rect.w * (nx - 1);
        const gy = (m.y - rect.y) / rect.h * (ny - 1);
        const ix = Math.round(gx), iy = Math.round(gy);
        for (let yy = Math.max(0, iy - radius); yy <= Math.min(ny - 1, iy + radius); yy++) {
          for (let xx = Math.max(0, ix - radius); xx <= Math.min(nx - 1, ix + radius); xx++) {
            const d2 = (xx - gx) ** 2 + (yy - gy) ** 2;
            grid[yy][xx] += Math.exp(-d2 / (2 * sigma * sigma));
          }
        }
      }
      let maxV = 0;
      for (const row of grid) for (const v of row) if (v > maxV) maxV = v;
      if (maxV <= 0) return;
      const thresholds = [0.22, 0.42, 0.62].map(x => x * maxV);
      const cellPt = (x, y) => ({ x:rect.x + x / (nx - 1) * rect.w, y:rect.y + y / (ny - 1) * rect.h });
      const cases = {
        1:[[3,0]], 2:[[0,1]], 3:[[3,1]], 4:[[1,2]], 5:[[3,2],[0,1]], 6:[[0,2]], 7:[[3,2]],
        8:[[2,3]], 9:[[0,2]], 10:[[0,3],[1,2]], 11:[[1,2]], 12:[[3,1]], 13:[[0,1]], 14:[[3,0]]
      };
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      thresholds.forEach((t, ti) => {
        const lineW = ti === 0 ? 0.9 : ti === 1 ? 1.2 : 1.55;
        if (halo) {
          ctx.strokeStyle = halo;
          ctx.lineWidth = lineW + 2.6;
          ctx.beginPath();
          for (let y = 0; y < ny - 1; y++) {
            for (let x = 0; x < nx - 1; x++) {
              const tl = grid[y][x], tr = grid[y][x + 1], br = grid[y + 1][x + 1], bl = grid[y + 1][x];
              const mask = (tl > t ? 1 : 0) | (tr > t ? 2 : 0) | (br > t ? 4 : 0) | (bl > t ? 8 : 0);
              const segs = cases[mask];
              if (!segs) continue;
              const ptl = cellPt(x, y), ptr = cellPt(x + 1, y), pbr = cellPt(x + 1, y + 1), pbl = cellPt(x, y + 1);
              const edges = [
                edgeInterp(ptl, ptr, tl, tr, t),
                edgeInterp(ptr, pbr, tr, br, t),
                edgeInterp(pbl, pbr, bl, br, t),
                edgeInterp(ptl, pbl, tl, bl, t),
              ];
              for (const [a, b] of segs) {
                ctx.moveTo(edges[a].x, edges[a].y);
                ctx.lineTo(edges[b].x, edges[b].y);
              }
            }
          }
          ctx.stroke();
        }
        ctx.strokeStyle = strokes[ti] || strokes[strokes.length - 1] || "rgba(0,0,0,.52)";
        ctx.lineWidth = ti === 0 ? 0.9 : ti === 1 ? 1.2 : 1.55;
        ctx.beginPath();
        for (let y = 0; y < ny - 1; y++) {
          for (let x = 0; x < nx - 1; x++) {
            const tl = grid[y][x], tr = grid[y][x + 1], br = grid[y + 1][x + 1], bl = grid[y + 1][x];
            const mask = (tl > t ? 1 : 0) | (tr > t ? 2 : 0) | (br > t ? 4 : 0) | (bl > t ? 8 : 0);
            const segs = cases[mask];
            if (!segs) continue;
            const ptl = cellPt(x, y), ptr = cellPt(x + 1, y), pbr = cellPt(x + 1, y + 1), pbl = cellPt(x, y + 1);
            const edges = [
              edgeInterp(ptl, ptr, tl, tr, t),
              edgeInterp(ptr, pbr, tr, br, t),
              edgeInterp(pbl, pbr, bl, br, t),
              edgeInterp(ptl, pbl, tl, bl, t),
            ];
            for (const [a, b] of segs) {
              ctx.moveTo(edges[a].x, edges[a].y);
              ctx.lineTo(edges[b].x, edges[b].y);
            }
          }
        }
        ctx.stroke();
      });
      ctx.restore();
    }
    function densityGrid(src, ex, rect, nx = 58, ny = 58) {
      const grid = Array.from({length: ny}, () => Array(nx).fill(0));
      const sigma = 1.45, radius = 4;
      for (const p of src.filter(Boolean)) {
        const m = mapPt(p, ex, rect);
        const gx = (m.x - rect.x) / rect.w * (nx - 1);
        const gy = (m.y - rect.y) / rect.h * (ny - 1);
        const ix = Math.round(gx), iy = Math.round(gy);
        for (let yy = Math.max(0, iy - radius); yy <= Math.min(ny - 1, iy + radius); yy++) {
          for (let xx = Math.max(0, ix - radius); xx <= Math.min(nx - 1, ix + radius); xx++) {
            const d2 = (xx - gx) ** 2 + (yy - gy) ** 2;
            grid[yy][xx] += Math.exp(-d2 / (2 * sigma * sigma));
          }
        }
      }
      const norm = Math.max(1, src.length);
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) grid[y][x] /= norm;
      return grid;
    }
    function drawGridHeatmap(grid, rect, colorForValue) {
      const ny = grid.length, nx = grid[0]?.length || 0;
      if (!nx || !ny) return;
      let maxAbs = 0;
      for (const row of grid) for (const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));
      if (maxAbs <= 0) return;
      const cw = rect.w / Math.max(1, nx - 1), ch = rect.h / Math.max(1, ny - 1);
      ctx.save();
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const v = grid[y][x];
          if (Math.abs(v) <= 0) continue;
          ctx.fillStyle = colorForValue(v / maxAbs);
          ctx.fillRect(rect.x + x * cw, rect.y + y * ch, cw + 1, ch + 1);
        }
      }
      ctx.restore();
    }
    function drawObservedHeatmap(src, ex, rect) {
      if (src.length < 4) return;
      const grid = densityGrid(src, ex, rect, 72, 72);
      drawGridHeatmap(grid, rect, v => `rgba(0,0,0,${0.04 + clamp(v, 0, 1) * 0.32})`);
    }
    function drawSignedDifferenceHeatmap(aSrc, bSrc, f, ex, rect) {
      if (aSrc.length < 4 || bSrc.length < 4) return;
      const ga = densityGrid(aSrc, ex, rect, 72, 72), gb = densityGrid(bSrc, ex, rect, 72, 72);
      const diff = ga.map((row, y) => row.map((v, x) => v - gb[y][x]));
      const codeBg = f.bgMode === "code" && f.mapMode === "latent";
      drawGridHeatmap(diff, rect, v => {
        const a = 0.06 + Math.abs(v) * 0.38;
        if (codeBg) return v >= 0 ? `rgba(0,0,0,${a})` : `rgba(255,255,255,${Math.min(0.82, a + 0.12)})`;
        return v >= 0 ? `rgba(33,102,172,${a})` : `rgba(178,24,43,${a})`;
      });
    }
    function drawTopologyHeatmap(seriesRows, pts, f, ex, rect) {
      if (!f.showHeatmap || f.mapMode === "code_decomp") return;
      const src = contourPoints(seriesRows, pts, f).filter(Boolean);
      if (src.length < 4) return;
      if (f.topologyMode === "compare") {
        const aSrc = src.filter(p => comparePointPass(p, f.topologyA));
        const bSrc = src.filter(p => comparePointPass(p, f.topologyB));
        drawSignedDifferenceHeatmap(aSrc, bSrc, f, ex, rect);
      } else {
        drawObservedHeatmap(src, ex, rect);
      }
    }
    function drawThresholdGrid(grid, threshold, ex, rect, stroke, width) {
      const ny = grid.length, nx = grid[0]?.length || 0;
      if (!nx || !ny) return;
      const cellPt = (x, y) => ({ x:rect.x + x / (nx - 1) * rect.w, y:rect.y + y / (ny - 1) * rect.h });
      const cases = {
        1:[[3,0]], 2:[[0,1]], 3:[[3,1]], 4:[[1,2]], 5:[[3,2],[0,1]], 6:[[0,2]], 7:[[3,2]],
        8:[[2,3]], 9:[[0,2]], 10:[[0,3],[1,2]], 11:[[1,2]], 12:[[3,1]], 13:[[0,1]], 14:[[3,0]]
      };
      ctx.save();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      for (let y = 0; y < ny - 1; y++) {
        for (let x = 0; x < nx - 1; x++) {
          const tl = grid[y][x], tr = grid[y][x + 1], br = grid[y + 1][x + 1], bl = grid[y + 1][x];
          const mask = (tl > threshold ? 1 : 0) | (tr > threshold ? 2 : 0) | (br > threshold ? 4 : 0) | (bl > threshold ? 8 : 0);
          const segs = cases[mask];
          if (!segs) continue;
          const ptl = cellPt(x, y), ptr = cellPt(x + 1, y), pbr = cellPt(x + 1, y + 1), pbl = cellPt(x, y + 1);
          const edges = [
            edgeInterp(ptl, ptr, tl, tr, threshold),
            edgeInterp(ptr, pbr, tr, br, threshold),
            edgeInterp(pbl, pbr, bl, br, threshold),
            edgeInterp(ptl, pbl, tl, bl, threshold),
          ];
          for (const [a, b] of segs) {
            ctx.moveTo(edges[a].x, edges[a].y);
            ctx.lineTo(edges[b].x, edges[b].y);
          }
        }
      }
      ctx.stroke();
      ctx.restore();
    }
    function drawSignedDifferenceContours(aSrc, bSrc, f, ex, rect) {
      if (aSrc.length < 4 || bSrc.length < 4) return;
      const ga = densityGrid(aSrc, ex, rect), gb = densityGrid(bSrc, ex, rect);
      const diffA = ga.map((row, y) => row.map((v, x) => v - gb[y][x]));
      const diffB = diffA.map(row => row.map(v => -v));
      let maxAbs = 0;
      for (const row of diffA) for (const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));
      if (maxAbs <= 0) return;
      const levels = [0.22, 0.42, 0.62].map(v => v * maxAbs);
      const codeBg = f.bgMode === "code" && f.mapMode === "latent";
      const aColor = codeBg ? ["rgba(0,0,0,.42)","rgba(0,0,0,.66)","rgba(0,0,0,.9)"] : ["rgba(33,102,172,.42)","rgba(33,102,172,.66)","rgba(33,102,172,.9)"];
      const bColor = codeBg ? ["rgba(255,255,255,.55)","rgba(255,255,255,.78)","rgba(255,255,255,.98)"] : ["rgba(178,24,43,.42)","rgba(178,24,43,.66)","rgba(178,24,43,.9)"];
      levels.forEach((t, i) => drawThresholdGrid(diffA, t, ex, rect, aColor[i], i === 0 ? 1.1 : i === 1 ? 1.5 : 2.0));
      levels.forEach((t, i) => {
        if (codeBg) drawThresholdGrid(diffB, t, ex, rect, "rgba(0,0,0,.32)", i === 0 ? 3.3 : i === 1 ? 3.8 : 4.4);
        drawThresholdGrid(diffB, t, ex, rect, bColor[i], i === 0 ? 1.1 : i === 1 ? 1.5 : 2.0);
      });
    }
    function drawTopologyContours(seriesRows, pts, f, ex, rect) {
      if (!f.showTopology) return;
      const src = contourPoints(seriesRows, pts, f).filter(Boolean);
      if (src.length < 4) return;
      if (f.topologyMode === "compare") {
        const aSrc = src.filter(p => comparePointPass(p, f.topologyA));
        const bSrc = src.filter(p => comparePointPass(p, f.topologyB));
        drawSignedDifferenceContours(aSrc, bSrc, f, ex, rect);
      } else {
        drawContourSet(src, ex, rect, ["rgba(0,0,0,.26)","rgba(0,0,0,.38)","rgba(0,0,0,.52)"]);
      }
    }
    function drawStar(cx, cy, r, fill) {
      ctx.save();
      ctx.fillStyle = fill;
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let i = 0; i < 10; i++) {
        const rr = i % 2 === 0 ? r : r * 0.45;
        const a = -Math.PI / 2 + i * Math.PI / 5;
        const x = cx + Math.cos(a) * rr, y = cy + Math.sin(a) * rr;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
    function drawNamingStars(seriesRows, f, ex, rect) {
      if (!f.showNamingStars) return;
      const moving = true;
      const lifetime = moving ? Math.max(1 / 10000, f.windowsPerSecond / Math.max(1, typicalWindowCount(f) - 1)) : 1;
      const start = moving ? Math.max(0, f.progress - lifetime) : 0;
      const end = moving ? f.progress : 1;
      for (const row of seriesRows) {
        let lastAnchor = -Infinity;
        for (const p of row.series) {
          if (!p.window || data.phases[p.window[W.phase]] !== "during") continue;
          if (p.progress < start || p.progress > end) continue;
          const minGap = Math.max(0.5, scaleSeconds(p.scale) * 0.75);
          if (p.window[W.anchor] - lastAnchor < minGap) continue;
          lastAnchor = p.window[W.anchor];
          const m = mapPt(p, ex, rect);
          const age = moving ? clamp((p.progress - start) / Math.max(1e-9, end - start), 0, 1) : 1;
          ctx.save();
          ctx.globalAlpha = 0.35 + 0.65 * age;
          drawStar(m.x, m.y, 7.2, colorForPoint(p, f));
          ctx.restore();
        }
      }
    }
    function syncCanvasSize() {
      const box = canvas.getBoundingClientRect();
      const w = Math.max(600, Math.round(box.width));
      const h = Math.max(420, Math.round(box.height));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    }
    function plotRect(f = null) {
      const interactive = document.body.classList.contains("interactiveMode");
      let base;
      if (interactive) {
        const leftRail = document.querySelector(".explorerControls")?.getBoundingClientRect();
        const rightRail = legend?.getBoundingClientRect();
        const timeline = document.querySelector(".embedTimeline")?.getBoundingClientRect();
        const x0 = Math.max(70, Math.round((leftRail?.right || 330) + 36));
        const x1 = Math.min(canvas.width - 42, Math.round((rightRail?.left || canvas.width) - 36));
        const y0 = 34;
        const y1 = Math.min(canvas.height - 70, Math.round((timeline?.top || canvas.height - 86) - 26));
        base = { x:x0, y:y0, w:Math.max(180, x1 - x0), h:Math.max(220, y1 - y0) };
      } else {
        base = { x: 70, y: 34, w: canvas.width - 112, h: canvas.height - 88 };
      }
      if (!f || f.mapMode !== "code_layout") return base;
      const size = Math.max(180, Math.min(base.w, base.h));
      return { x: base.x + (base.w - size) / 2, y: base.y + (base.h - size) / 2, w: size, h: size };
    }
    function zoomView(factor) {
      if (!viewEx) draw();
      if (!viewEx) return;
      const cx = (viewEx.xmin + viewEx.xmax) / 2;
      const cy = (viewEx.ymin + viewEx.ymax) / 2;
      const hx = (viewEx.xmax - viewEx.xmin) * factor / 2;
      const hy = (viewEx.ymax - viewEx.ymin) * factor / 2;
      viewEx = { xmin:cx - hx, xmax:cx + hx, ymin:cy - hy, ymax:cy + hy };
      draw();
    }
    function draw() {
      syncCanvasSize();
      const f = filters();
      scheduleUrlStateSync();
      const rawSeriesRows = allSeries(f);
      const cb = codebookFor(f);
      const ring = codeRingLayout(rawSeriesRows, f, cb);
      const seriesRows = projectSeriesRows(rawSeriesRows, f, ring);
      if (f.mapMode === "code_usage_chart") {
        ctx.clearRect(0,0,canvas.width,canvas.height);
        const histogramWindows = drawCodeUsageHistogramView(f, rawSeriesRows, cb) || 0;
        currentCanvasMeta = filterSummary(f);
        renderLegend(f, cb, histogramWindows, 0);
        layoutRightRail();
        status.textContent = "Code usage view: histograms count filtered windows assigned to each selected codebook.";
        sessionList.innerHTML = `<b>Displayed participants at selected time</b><span class="muted">Code usage view hides session movement. Turn on dots, topology, transitions, or the background layer to return to the embedding view.</span>`;
        return;
      }
      if (f.mapMode === "heatmap_chart") {
        ctx.clearRect(0,0,canvas.width,canvas.height);
        const heatmapWindows = drawBehaviorHeatmapView(f, rawSeriesRows) || 0;
        currentCanvasMeta = filterSummary(f);
        renderLegend(f, cb, heatmapWindows, 0);
        layoutRightRail();
        status.textContent = "Heatmap view: visible windows summarized as behavior co-occurrence values.";
        sessionList.innerHTML = `<b>Displayed participants at selected time</b><span class="muted">Heatmap view hides session movement. Turn off Heatmap to return to the embedding view.</span>`;
        return;
      }
      if (f.mapMode === "code_decomp") {
        ctx.clearRect(0,0,canvas.width,canvas.height);
        currentDrawn = [];
        currentTailHits = [];
        currentTransitionHits = [];
        drawCodeDecompositionView(f, cb);
        currentCanvasMeta = filterSummary(f);
        renderLegend(f, cb, 0, 0);
        layoutRightRail();
        status.textContent = "Code decomposition view: hover over a code label for the same behavior profile shown in the hover inspector.";
        sessionList.innerHTML = `<b>Displayed participants at selected time</b><span class="muted">Code decomposition view hides session movement. Turn on session points or latent positions to inspect trajectories.</span>`;
        return;
      }
      const exBase = f.mapMode === "code_layout" ? ringExtent() : baseExtent(seriesRows, f, cb);
      const ex = ensureView(f, exBase);
      const rect = plotRect(f);
      const pts = currentPoints(seriesRows, f);
      ctx.clearRect(0,0,canvas.width,canvas.height);
      currentCodeMarks = [];
      currentTransitionHits = [];
      ctx.fillStyle = "#faf8f2"; ctx.fillRect(0,0,canvas.width,canvas.height);
      drawBackground(rect, ex, cb, f);
      drawAxes(rect, ex, f);
      drawTopologyHeatmap(seriesRows, pts, f, ex, rect);
      drawTopologyContours(seriesRows, pts, f, ex, rect);
      drawTransitions(seriesRows, f, ex, rect, cb, ring);
      drawPaths(seriesRows, f, ex, rect);
      drawNamingStars(seriesRows, f, ex, rect);
      currentDrawn = [];
      if (f.showKeypoints) {
        for (const pt of pts) {
          const m = mapPt(pt, ex, rect);
          const s = sessionByKey.get(key(pt.model, pt.sid));
          const highlighted = f.highlightSessions.has(sessionBaseKey(s));
          const hovered = hoveredSessionKey === sessionBaseKey(s);
          currentDrawn.push({pt, x:m.x, y:m.y, r:hovered ? 9 : highlighted ? 7 : 5, sessionKey:sessionBaseKey(s)});
          ctx.globalAlpha = hovered || highlighted ? 1 : 0.86;
          ctx.fillStyle = colorForPoint(pt, f);
          ctx.strokeStyle = hovered || highlighted ? "#000" : "rgba(0,0,0,.55)";
          ctx.lineWidth = hovered ? 3.0 : highlighted ? 2.2 : 0.8;
          ctx.beginPath(); ctx.arc(m.x, m.y, hovered ? 7.5 : highlighted ? 6.5 : 4.4, 0, Math.PI*2); ctx.fill(); ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
      if (f.mapMode === "code_layout") drawCodeRingNodes(rect, ex, ring, f);
      else if (f.showCodes || f.showTransitions) drawCodeMarks(rect, ex, cb, f);
      const source = data.models.find(m => m.idx === f.model)?.label || "model";
      currentCanvasMeta = filterSummary(f);
      const windows = seriesRows.reduce((a,b)=>a+b.series.length,0);
      renderLegend(f, cb, windows, pts.length);
      layoutRightRail();
      status.textContent = f.showKeypoints
        ? `${source}: ${pts.length} sessions shown, ${windows} windows available; speed ${visualizationSpeedLabel(f.windowsPerSecond)}. Drag to pan; use buttons to zoom.`
        : `${source}: session points hidden; ${windows} windows used for code regions and transition arrows. Drag to pan; use buttons to zoom.`;
      if (f.showKeypoints) renderSessionList(pts, f);
      else sessionList.innerHTML = `<b>Displayed participants at selected time</b><span class="muted">Moving session points are hidden. Turn them on to inspect per-session behavior at a selected time.</span>`;
    }
    function pointFromEvent(ev) {
      const r = canvas.getBoundingClientRect();
      return { x:(ev.clientX - r.left) / r.width * canvas.width, y:(ev.clientY - r.top) / r.height * canvas.height };
    }
    canvas.addEventListener("wheel", () => {}, { passive:true });
    canvas.addEventListener("mousedown", ev => { dragging = true; lastDrag = pointFromEvent(ev); });
    window.addEventListener("mouseup", () => { dragging = false; lastDrag = null; });
    function setHoverInspector(html, kind = "") {
      if (!hoverPanel) return;
      hoverPanel.classList.toggle("profileTip", kind === "profile");
      hoverPanel.classList.toggle("sessionTip", kind === "session");
      hoverPanel.classList.toggle("transitionTip", kind === "transition");
      hoverPanel.innerHTML = html || `<b>Code breakdown</b><span class="tipMuted">Hover a code diamond or code chip to inspect behavior composition. Hover a session point or path to inspect that window.</span>`;
      layoutRightRail();
    }
    function layoutRightRail() {
      if (!hoverPanel || !legend) return;
      const gap = 12;
      const top = Math.round(legend.getBoundingClientRect().bottom + gap);
      const bottom = 16;
      hoverPanel.style.top = `${top}px`;
      hoverPanel.style.bottom = `${bottom}px`;
    }
    function setHoveredSession(nextKey) {
      if (hoveredSessionKey === nextKey) return;
      hoveredSessionKey = nextKey;
      draw();
    }
    canvas.addEventListener("mousemove", ev => {
      const m = pointFromEvent(ev);
      if (dragging && viewEx && lastDrag) {
        const rect = plotRect(filters());
        const dx = (m.x - lastDrag.x) / rect.w * (viewEx.xmax - viewEx.xmin);
        const dy = (m.y - lastDrag.y) / rect.h * (viewEx.ymax - viewEx.ymin);
        viewEx = { xmin:viewEx.xmin - dx, xmax:viewEx.xmax - dx, ymin:viewEx.ymin + dy, ymax:viewEx.ymax + dy };
        lastDrag = m;
        draw();
        return;
      }
      let bestCode = null, bdCode = Infinity;
      for (const c of currentCodeMarks) {
        const d = (c.x - m.x) ** 2 + (c.y - m.y) ** 2;
        if (d < bdCode) { bdCode = d; bestCode = c; }
      }
      if (bestCode && bdCode < bestCode.r ** 2) {
        setHoveredSession(null);
        if (!bestCode.pair) renderCodeProfile(bestCode);
        setHoverInspector(codeMarkTooltip(bestCode), "profile");
        return;
      }
      let best = null, bd = Infinity;
      for (const d of currentDrawn) {
        const dist = (d.x - m.x) ** 2 + (d.y - m.y) ** 2;
        if (dist < bd) { bd = dist; best = d; }
      }
      let hit = null;
      if (best && bd < Math.max(144, best.r ** 2)) hit = { pt:best.pt, sessionKey:best.sessionKey };
      if (!hit) {
        let bestEdge = null, be = Infinity;
        for (const e of currentTransitionHits) {
          const d = distToQuadratic(m.x, m.y, e.start, e.ctrl, e.end);
          if (d < be) { be = d; bestEdge = e; }
        }
        if (bestEdge && be < 10) {
          setHoveredSession(null);
          let edgeHtml = "";
          if (Number.isFinite(bestEdge.diff)) {
            edgeHtml = `<b>${esc(bestEdge.from)} -> ${esc(bestEdge.to)}</b><br><span class="tipMuted">A-B normalized difference: ${fmt(bestEdge.diff, 3)}<br>A strength: ${fmt(bestEdge.probA, 3)} (${bestEdge.countA || 0} transitions)<br>B strength: ${fmt(bestEdge.probB, 3)} (${bestEdge.countB || 0} transitions)</span>`;
          } else {
            edgeHtml = `<b>${esc(bestEdge.from)} -> ${esc(bestEdge.to)}</b><br><span class="tipMuted">normalized transition strength: ${fmt(bestEdge.prob, 3)}<br>observed transitions: ${bestEdge.count}</span>`;
          }
          setHoverInspector(edgeHtml, "transition");
          return;
        }
      }
      if (!hit) {
        let bestTail = null, bt = Infinity;
        for (const seg of currentTailHits) {
          const d = distToSegment(m.x, m.y, seg.x1, seg.y1, seg.x2, seg.y2);
          if (d < bt) { bt = d; bestTail = seg; }
        }
        if (bestTail && bt < 7) hit = { pt:bestTail.pt, sessionKey:bestTail.sessionKey };
      }
      if (hit) {
        setHoveredSession(hit.sessionKey);
        renderWindowDetail(hit.pt);
        setHoverInspector(sessionTooltip(hit.pt), "session");
      } else {
        setHoveredSession(null);
      }
    });
    canvas.addEventListener("mouseleave", () => { setHoveredSession(null); });
    function wrapText(ctx2, text, x, y, maxWidth, lineHeight) {
      const words = text.split(/\s+/), lines = [];
      let line = "";
      for (const word of words) {
        const test = line ? `${line} ${word}` : word;
        if (ctx2.measureText(test).width > maxWidth && line) { lines.push(line); line = word; }
        else line = test;
      }
      if (line) lines.push(line);
      lines.forEach((ln, i) => ctx2.fillText(ln, x, y + i * lineHeight));
      return lines.length * lineHeight;
    }
    controls.export.addEventListener("click", () => {
      const out = document.createElement("canvas");
      out.width = canvas.width;
      out.height = canvas.height + 96;
      const ox = out.getContext("2d");
      ox.fillStyle = "#ffffff"; ox.fillRect(0,0,out.width,out.height);
      ox.drawImage(canvas,0,0);
      ox.fillStyle = "#111"; ox.font = "13px Segoe UI, Arial";
      wrapText(ox, currentCanvasMeta, 24, canvas.height + 28, out.width - 48, 18);
      const a = document.createElement("a");
      const f = filters();
      a.download = `interactive_${data.scales[f.scale]}_${data.components[f.xComp]}_x_${data.components[f.yComp]}.png`;
      a.href = out.toDataURL("image/png");
      a.click();
    });
    controls.play.addEventListener("click", () => {
      playing = !playing;
      controls.play.innerHTML = playing ? "&#10074;&#10074;" : "&#9658;";
      controls.play.setAttribute("aria-label", playing ? "Pause" : "Play");
      if (timer) clearInterval(timer);
      if (playing) {
        playProgress = Number(controls.progress.value || 0) / 10000;
        lastFrameMs = performance.now();
        timer = setInterval(() => {
          const now = performance.now();
          const dt = Math.max(0.001, (now - lastFrameMs) / 1000);
          lastFrameMs = now;
          const f = filters();
          const typical = Math.max(1, typicalWindowCount(f) - 1);
          playProgress = (playProgress ?? (Number(controls.progress.value || 0) / 10000)) + f.windowsPerSecond * dt / typical;
          if (playProgress >= 1) playProgress = playProgress % 1;
          controls.progress.value = String(Math.round(playProgress * 10000));
          draw();
        }, 33);
      }
    });
    document.querySelectorAll(".speedBtn").forEach(btn => {
      btn.addEventListener("click", () => setVisualizationSpeed(Number(btn.dataset.speed || 1)));
    });
    controls.reset.addEventListener("click", () => { viewEx = null; viewKey = ""; draw(); });
    controls.zoomIn.addEventListener("click", () => zoomView(0.78));
    controls.zoomOut.addEventListener("click", () => zoomView(1.28));
    if (relatedToggle && relatedPanel) {
      relatedToggle.addEventListener("click", () => { relatedPanel.classList.toggle("collapsed"); layoutRightRail(); });
    }
    function setRelatedMode(mode) {
      relatedMode = mode;
      if (relatedSimilar) {
        relatedSimilar.classList.toggle("active", mode === "similar");
        relatedSimilar.setAttribute("aria-pressed", mode === "similar" ? "true" : "false");
      }
      if (relatedAll) {
        relatedAll.classList.toggle("active", mode === "all");
        relatedAll.setAttribute("aria-pressed", mode === "all" ? "true" : "false");
      }
      renderRelatedFigures(filters());
    }
    if (relatedSearch) relatedSearch.addEventListener("input", () => renderRelatedFigures(filters()));
    if (relatedSimilar) relatedSimilar.addEventListener("click", () => setRelatedMode("similar"));
    if (relatedAll) relatedAll.addEventListener("click", () => setRelatedMode("all"));
    if (legend) {
      ["mousemove", "mouseover", "pointerover", "click"].forEach(evName => {
        legend.addEventListener(evName, ev => {
          const chip = ev.target?.closest ? ev.target.closest(".codeChip") : null;
          if (chip) showCodeChipProfile(chip);
        });
      });
    }
    $("copyGlobalToTopology")?.addEventListener("click", () => {
      ["topoA", "topoB"].forEach(copyGlobalToComparePrefix);
      syncCompareAccordions();
      draw();
    });
    $("copyGlobalToTransitions")?.addEventListener("click", () => {
      ["transA", "transB"].forEach(copyGlobalToComparePrefix);
      syncCompareAccordions();
      draw();
    });
    $("copyTopologyToTransitions")?.addEventListener("click", () => {
      copyComparePrefix("topoA", "transA");
      copyComparePrefix("topoB", "transB");
      controls.transitionMode.value = controls.topologyMode.value === "compare" ? "compare" : "observed";
      syncCompareAccordions();
      draw();
    });
    $("copyTransitionsToTopology")?.addEventListener("click", () => {
      copyComparePrefix("transA", "topoA");
      copyComparePrefix("transB", "topoB");
      controls.topologyMode.value = controls.transitionMode.value === "compare" ? "compare" : "density";
      syncCompareAccordions();
      draw();
    });
    $("swapTopologyAB")?.addEventListener("click", () => {
      swapComparePrefixes("topoA", "topoB");
      draw();
    });
    $("swapTransitionsAB")?.addEventListener("click", () => {
      swapComparePrefixes("transA", "transB");
      draw();
    });
    controls.resetSelectedSessions?.addEventListener("click", resetSelectedSessionsToGlobal);
    function setLayerOpen(group, open) {
      const body = group.querySelector(".layerBody");
      const header = group.querySelector(".layerHeader");
      const toggle = group.querySelector(".layerToggle");
      group.classList.toggle("open", open);
      if (header) header.setAttribute("aria-expanded", open ? "true" : "false");
      if (toggle) {
        toggle.textContent = open ? "-" : "+";
        const name = header?.querySelector("label")?.textContent?.trim() || "section";
        toggle.setAttribute("aria-label", `${open ? "Collapse" : "Expand"} ${name}`);
      }
      if (body) body.style.maxHeight = open ? `${body.scrollHeight}px` : "0px";
    }
    function openLayer(name) {
      const target = document.querySelector(`.layerGroup[data-layer="${name}"]`);
      document.querySelectorAll(".layerGroup").forEach(other => setLayerOpen(other, false));
      if (target) setLayerOpen(target, true);
    }
    function refreshOpenAccordions() {
      document.querySelectorAll(".layerGroup.open .layerBody").forEach(body => {
        body.style.maxHeight = `${body.scrollHeight}px`;
      });
    }
    function syncCompareAccordions() {
      document.querySelectorAll(".abAccordion").forEach(acc => {
        const target = acc.dataset.modeTarget ? $(acc.dataset.modeTarget) : null;
        const activeValue = acc.dataset.activeValue || "compare";
        const active = target ? target.value === activeValue : acc.classList.contains("compareActive");
        acc.classList.toggle("compareActive", active);
        acc.querySelectorAll(".abTab").forEach(tab => {
          const isB = tab.dataset.side === "B";
          const isActive = isB ? active : !active;
          tab.classList.toggle("active", isActive);
          tab.setAttribute("aria-pressed", isActive ? "true" : "false");
        });
      });
      refreshOpenAccordions();
    }
    document.querySelectorAll(".layerHeader").forEach(header => {
      header.addEventListener("click", ev => {
        if (ev.target && ev.target.closest && ev.target.closest("input")) return;
        const group = header.closest(".layerGroup");
        const willOpen = !group.classList.contains("open");
        document.querySelectorAll(".layerGroup").forEach(other => setLayerOpen(other, false));
        if (willOpen) setLayerOpen(group, true);
      });
    });
    document.querySelectorAll(".abAccordion").forEach(acc => {
      const tabs = [...acc.querySelectorAll(".abTab")];
      const target = acc.dataset.modeTarget ? $(acc.dataset.modeTarget) : null;
      const activeValue = acc.dataset.activeValue || "compare";
      const inactiveValue = acc.dataset.inactiveValue || "density";
      const setActive = active => {
        acc.classList.toggle("compareActive", active);
        if (target) target.value = active ? activeValue : inactiveValue;
        if (active && acc.dataset.ab === "topology") { clearChartModes(); if (!controls.showTopology.checked) controls.showTopology.checked = true; }
        if (active && acc.dataset.ab === "transitions") { clearChartModes(); if (!controls.showTransitions.checked) controls.showTransitions.checked = true; }
        tabs.forEach(tab => {
          const isB = tab.dataset.side === "B";
          const isActive = isB ? active : !active;
          tab.classList.toggle("active", isActive);
          tab.setAttribute("aria-pressed", isActive ? "true" : "false");
        });
        const group = acc.closest(".layerGroup");
        if (group && group.classList.contains("open")) setLayerOpen(group, true);
        draw();
      };
      tabs.forEach(tab => {
        tab.addEventListener("click", () => {
          const isB = tab.dataset.side === "B";
          setActive(isB ? !acc.classList.contains("compareActive") : false);
        });
      });
      setActive(target ? target.value === activeValue : false);
    });
    const HELP_TEXT = {
      "global selectors": "These filters decide which sessions and windows are visible across the live view.",
      "preset": "Jump to a saved axis/filter configuration that was useful for high-vs-low naming contrasts.",
      "window": "Temporal scale of each encoded window: shorter windows track local behavior; longer windows summarize broader interaction state.",
      "x axis": "Codebook/latent source plotted horizontally.",
      "y axis": "Codebook/latent source plotted vertically. Match X for the native 2D latent space; choose another source for a cross-modality view.",
      "language": "Session language modality: NGT signing or NL spoken Dutch.",
      "aud. status": "Child audiological status group.",
      "aud.": "Child audiological status group.",
      "session": "Recording session number.",
      "naming frequency": "Within-language low/high naming-frequency group.",
      "naming": "Within-language low/high naming-frequency group.",
      "proximity to naming event": "Whether a window is far from, before, during, or after a naming event.",
      "near": "Temporal proximity to naming events.",
      "session phase": "Early, middle, or late third of the session.",
      "phase": "Early, middle, or late third of the session.",
      "mutual attention": "none: no shared alignment; object-aligned: both oriented to object; person-aligned: dyad oriented to each other; coordinated JA: object/person coordination; naming-aligned JA: coordinated attention during naming.",
      "mutual": "Mutual attention category for the selected windows.",
      "code arrangement": "Controls whether codes use their learned latent geometry or a code-usage summary view.",
      "background + codes": "Colored code regions and code markers. Turn this off with other layers off to show code-usage histograms.",
      "latent positions": "Use learned 2D codebook/latent coordinates instead of arranging codes in a simple layout.",
      "context size": "When latent positions are off, this controls how much of each session contributes to the code-usage histogram.",
      "stacking": "Subdivide each code-usage bar by a selected categorical variable.",
      "dots": "Animated session points at the selected normalized session time.",
      "color": "Variable used to color moving session points.",
      "session trails": "Show recent path history behind each moving point.",
      "naming event stars": "Mark recent naming-event windows near the moving point.",
      "trail length": "How much recent path history is shown for each session point.",
      "selected sessions": "Manually filter or highlight specific sessions after applying the global filters.",
      "filter sessions": "Only selected sessions remain visible.",
      "highlight sessions": "Selected sessions stay visible but are emphasized.",
      "topology": "Density contours over the currently selected windows. In comparison mode, show the difference between A and B.",
      "context": "How much of each session contributes to topology or transition summaries: current frame through full session.",
      "transitions": "Directed transitions between codes over the selected windows. In comparison mode, arrows show A/B differences."
    };
    function helpLabelText(node) {
      const clone = node.cloneNode(true);
      clone.querySelectorAll("input, select, button, .helpIcon").forEach(x => x.remove());
      return clone.textContent.replace(/\s+/g, " ").trim().toLowerCase();
    }
    function helpForLabel(text) {
      if (HELP_TEXT[text]) return HELP_TEXT[text];
      if (text.includes("language")) return HELP_TEXT["language"];
      if (text.includes("aud")) return HELP_TEXT["aud. status"];
      if (text.includes("naming frequency")) return HELP_TEXT["naming frequency"];
      if (text === "naming") return HELP_TEXT["naming"];
      if (text.includes("proximity")) return HELP_TEXT["proximity to naming event"];
      if (text.includes("session phase")) return HELP_TEXT["session phase"];
      if (text.includes("mutual")) return HELP_TEXT["mutual attention"];
      if (text.includes("context")) return HELP_TEXT["context"];
      return "";
    }
    let controlHelpTooltip = null;
    function ensureControlHelpTooltip() {
      if (controlHelpTooltip) return controlHelpTooltip;
      controlHelpTooltip = document.createElement("div");
      controlHelpTooltip.className = "controlHelpTooltip";
      document.body.appendChild(controlHelpTooltip);
      return controlHelpTooltip;
    }
    function showControlHelp(icon) {
      const tip = ensureControlHelpTooltip();
      tip.textContent = icon.dataset.help || "";
      tip.style.display = "block";
      tip.style.opacity = "0";
      const r = icon.getBoundingClientRect();
      const tr = tip.getBoundingClientRect();
      let left = r.left + 14;
      let top = r.top - tr.height - 8;
      if (left + tr.width > window.innerWidth - 10) left = window.innerWidth - tr.width - 10;
      if (top < 10) top = r.bottom + 8;
      tip.style.left = `${Math.max(10, left)}px`;
      tip.style.top = `${Math.max(10, top)}px`;
      tip.style.opacity = "1";
    }
    function hideControlHelp() {
      if (controlHelpTooltip) controlHelpTooltip.style.opacity = "0";
    }
    function decorateHelpIcons() {
      const selector = [
        ".explorerControls h3",
        ".explorerControls .control > label",
        ".explorerControls .globalSelectors > label",
        ".explorerControls .globalSelectors .controlRow > div > label",
        ".explorerControls .layerHeader > label",
        ".explorerControls .layerInner > label",
        ".explorerControls .abCompareGrid > label",
        ".explorerControls .inlineChecks > label"
      ].join(",");
      document.querySelectorAll(selector).forEach(node => {
        if (node.querySelector(".helpIcon")) return;
        const text = helpLabelText(node);
        const help = helpForLabel(text);
        if (!help) return;
        const icon = document.createElement("span");
        icon.className = "helpIcon";
        icon.tabIndex = 0;
        icon.setAttribute("role", "button");
        icon.setAttribute("aria-label", help);
        icon.dataset.help = help;
        icon.addEventListener("click", ev => { ev.preventDefault(); ev.stopPropagation(); });
        icon.addEventListener("mouseenter", () => showControlHelp(icon));
        icon.addEventListener("mouseleave", hideControlHelp);
        icon.addEventListener("focus", () => showControlHelp(icon));
        icon.addEventListener("blur", hideControlHelp);
        node.appendChild(icon);
      });
    }
    controls.model.addEventListener("change", () => { fillScaleControls(); refreshSessionPickers(); viewEx = null; viewKey = ""; });
    controls.preset?.addEventListener("change", () => {
      const rank = controls.preset.value;
      if (!rank) return;
      const preset = (data.highLowPresets || []).find(p => Number(p.model ?? 0) === Number(controls.model.value || 0) && String(p.rank) === String(rank));
      if (preset) applyHighLowPreset(preset);
    });
    controls.scale.addEventListener("change", () => { fillAxisControls(); refreshSessionPickers(); viewEx = null; viewKey = ""; });
    controls.xComp.addEventListener("change", () => { viewEx = null; viewKey = ""; draw(); });
    controls.yComp.addEventListener("change", () => { viewEx = null; viewKey = ""; draw(); });
    controls.latentPositions?.addEventListener("change", updateArrangementControls);
    function clearChartModes() {
      if (controls.showHeatmap) controls.showHeatmap.checked = false;
      if (controls.showBarChart) controls.showBarChart.checked = false;
    }
    controls.topologyMode.addEventListener("change", () => { controls.showTopology.checked = true; clearChartModes(); draw(); });
    controls.transitionMode.addEventListener("change", () => { controls.showTransitions.checked = true; clearChartModes(); draw(); });
    controls.showKeypoints.addEventListener("change", () => { if (controls.showKeypoints.checked) clearChartModes(); updateMovementControls(); draw(); });
    controls.showTopology.addEventListener("change", () => { if (controls.showTopology.checked) clearChartModes(); });
    controls.showTransitions.addEventListener("change", () => { if (controls.showTransitions.checked) clearChartModes(); });
    controls.showHeatmap?.addEventListener("change", () => { if (controls.showHeatmap.checked && controls.showBarChart) controls.showBarChart.checked = false; });
    controls.showBarChart?.addEventListener("change", () => { if (controls.showBarChart.checked && controls.showHeatmap) controls.showHeatmap.checked = false; });
    controls.showCodes.addEventListener("change", draw);
    for (const node of [controls.color, controls.latentPositions, controls.showBg, controls.showHeatmap, controls.showBarChart, controls.showTransitions, controls.showTopology, controls.selectedSessionsActive, controls.showNamingStars, controls.showTrails, controls.tailLength, controls.topologyContext, controls.codeUsageContext, controls.codeUsageStack, controls.transitionContext, controls.filterSessions, controls.highlightSessions, controls.progress].filter(Boolean)) {
      node.addEventListener("input", draw);
      node.addEventListener("change", draw);
    }
    controls.progress.addEventListener("input", () => { playProgress = Number(controls.progress.value || 0) / 10000; });
    window.addEventListener("resize", () => { refreshOpenAccordions(); draw(); layoutRightRail(); });
    fillModelControls();
    fillSessionControls();
    applyUrlState();
    decorateHelpIcons();
    updateMovementControls();
    updateArrangementControls();
    setVisualizationSpeed(Number(controls.windowsPerSecond.value || 1));
    syncCompareAccordions();
    renderWindowDetail(null);
    renderCodeProfile(null);
    setHoverInspector(null);
    bootingUrlState = false;
    draw();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
