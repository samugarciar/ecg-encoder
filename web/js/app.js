// Interfaz: carga de archivos, opciones, análisis y presentación de resultados.

import { analyzeEcg, resultsToCsv } from "./analysis.js";
import { drawBeat, drawErrorChart, drawStrip, formatTime, nearest } from "./charts.js";
import {
  chooseSignalColumn,
  guessUnits,
  inferFsFromTime,
  parseAnnotations,
  parseSignalCsv,
} from "./dsp.js";
import { loadModel } from "./model.js";

const $ = (id) => document.getElementById(id);
const SAMPLE = { csv: "samples/ecg_sintetico.csv", ann: "samples/ecg_sintetico_annotations.txt",
  name: "ecg_sintetico.csv (sintético)" };
// Al cambiar de señal se descartan las anotaciones, para no usar los R de otro registro
const ANN_HINT = "Cárguelas después de la señal. Si las carga, se usan sus picos R en vez del "
  + "detector automático.";

const state = {
  model: null,
  modelPromise: null,
  csv: null, // { columns, values, skipped }
  fileName: "",
  annotations: null,
  annName: "",
  result: null,
  selected: null,
  view: { start: 0, end: 10 },
  errorPoints: [],
  stripMarkers: [],
  sort: { key: "time", dir: 1 },
  onlyAtypical: true,
};

const nf = (d) => new Intl.NumberFormat("es-CO", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmt = (v, d = 2) => (Number.isFinite(v) ? nf(d).format(v) : "—");
const pct = (v) => (v == null ? "—" : `${fmt(100 * v, 1)} %`);

function setStatus(message, isError = false) {
  const el = $("status");
  el.textContent = message;
  el.classList.toggle("is-error", isError);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- Modelo ---------- */

function renderModelInfo(card) {
  const badge = $("model-badge");
  badge.textContent = `${card.is_demo ? "Modelo de demostración" : "Modelo"} · ${card.run_id}`;
  badge.classList.toggle("is-demo", card.is_demo);
  $("demo-banner").hidden = !card.is_demo;
  $("footer-model").textContent = `Modelo ${card.run_id} · umbral ${fmt(card.threshold.value, 5)} `
    + `(percentil ${fmt(card.threshold.percentile, 0)} de los latidos normales de validación) · `
    + `exportado ${card.exported_at.slice(0, 10)} · commit ${card.git_commit.slice(0, 7)}`;

  const g = card.validation.global;
  const aami = card.validation.per_aami_sensitivity;
  const items = [
    ["Sensibilidad", pct(g.sensitivity)],
    ["Especificidad", pct(g.specificity)],
    ["Precisión (VPP)", pct(g.precision)],
    ["AUC-ROC", fmt(g.auc_roc, 3)],
    ["Sens. supraventricular (S)", pct(aami.S)],
    ["Sens. ventricular (V)", pct(aami.V)],
  ];
  $("model-metrics").innerHTML = `
    <dl class="metrics">${items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
    <p class="metrics-note">${card.is_demo
    ? "Métricas sobre datos <strong>sintéticos</strong>: no reflejan el desempeño en pacientes."
    : `Medido en el conjunto de validación (${fmt(card.validation.n.normal, 0)} latidos normales y `
      + `${fmt(card.validation.n.anomalous, 0)} anómalos de 4 registros que el modelo no vio al `
      + "entrenar). Todavía no se ha medido en el conjunto de prueba final."}</p>`;
}

async function ensureModel() {
  if (!state.modelPromise) {
    state.modelPromise = loadModel("model/").then((m) => {
      state.model = m;
      renderModelInfo(m.card);
      return m;
    }).catch((e) => {
      const badge = $("model-badge");
      badge.textContent = "No se pudo cargar el modelo";
      badge.classList.add("is-error");
      $("model-metrics").innerHTML = `<p class="muted">${escapeHtml(e.message)}</p>`;
      state.modelPromise = null;
      throw e;
    });
  }
  return state.modelPromise;
}

/* ---------- Carga de archivos ---------- */

async function readFile(file) {
  if (file.size > 60 * 1024 * 1024) throw new Error("El archivo supera 60 MB.");
  return file.text();
}

async function loadSignalText(text, name) {
  const model = await ensureModel();
  const csv = parseSignalCsv(text);
  state.csv = csv;
  state.fileName = name;
  $("signal-name").textContent = name;

  const column = $("opt-column");
  column.innerHTML = csv.columns.map((c, i) => `<option value="${i}">${escapeHtml(c)}</option>`).join("");
  const idx = chooseSignalColumn(csv.columns, model.prep.lead);
  column.value = String(idx);
  $("opt-units").value = guessUnits(csv.values[idx], model.prep);
  const fs = inferFsFromTime(csv.columns, csv.values);
  $("opt-fs").value = String(fs ?? model.prep.fs);
  $("options").hidden = false;
  updateSummary();
  setStatus(csv.skipped ? `Se omitieron ${csv.skipped} filas que no eran numéricas.` : "");
}

async function loadAnnotationText(text, name) {
  const model = await ensureModel();
  state.annotations = parseAnnotations(text, model.prep.aami_map);
  state.annName = name;
  $("ann-name").textContent = `${name}: ${fmt(state.annotations.samples.length, 0)} latidos anotados.`;
  $("opt-ann-wrap").hidden = false;
  $("opt-use-ann").checked = true;
}

function updateSummary() {
  if (!state.csv) return;
  const n = state.csv.values[0].length;
  const fs = Number($("opt-fs").value);
  $("file-summary").textContent = `${fmt(n, 0)} muestras · ${state.csv.columns.length} columnas`
    + (fs > 0 ? ` · ${formatTime(n / fs)} de registro` : "");
}

/* ---------- Análisis ---------- */

function robustRange(x) {
  const step = Math.max(1, Math.floor(x.length / 20000));
  const sample = [];
  for (let i = 0; i < x.length; i += step) sample.push(x[i]);
  sample.sort((a, b) => a - b);
  const lo = sample[Math.floor(sample.length * 0.002)];
  const hi = sample[Math.ceil(sample.length * 0.998) - 1];
  const pad = Math.max(0.2, (hi - lo) * 0.12);
  return [lo - pad, hi + pad];
}

async function analyze() {
  const button = $("analyze-btn");
  button.disabled = true;
  try {
    const model = await ensureModel();
    const fs = Number($("opt-fs").value);
    if (!(fs >= 50 && fs <= 5000)) throw new Error("Indique una frecuencia de muestreo entre 50 y 5000 Hz.");
    const column = Number($("opt-column").value);
    setStatus("Filtrando la señal y reconstruyendo los latidos…");
    await new Promise((r) => setTimeout(r, 20)); // deja pintar el estado
    const useAnn = state.annotations && $("opt-use-ann").checked;
    const result = await analyzeEcg({
      signal: state.csv.values[column],
      fs,
      units: $("opt-units").value,
      annotations: useAnn ? state.annotations : null,
    }, model.prep, model.card, model.infer);
    result.yRange = robustRange(result.filtered);
    result.lead = state.csv.columns[column];
    state.result = result;
    state.sort = { key: "time", dir: 1 };
    const firstAtypical = result.beats.find((b) => b.atypical);
    state.onlyAtypical = result.nAtypical > 0;
    $("only-atypical").checked = state.onlyAtypical;
    setStatus(`Listo: ${fmt(result.beats.length, 0)} latidos analizados.`);
    renderResults();
    selectBeat(firstAtypical ? firstAtypical.row : 0, true);
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    console.error(e);
    setStatus(e.message, true);
  } finally {
    button.disabled = false;
  }
}

/* ---------- Resultados ---------- */

function renderResults() {
  const r = state.result;
  $("results").hidden = false;
  $("result-file").textContent = state.fileName;
  $("strip-lead").textContent = r.lead;
  $("warnings").innerHTML = r.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")
    + (r.droppedEdge ? `<li>${r.droppedEdge} latido(s) en los bordes del registro no tienen ventana completa y no se analizaron.</li>` : "");

  const share = r.beats.length ? r.nAtypical / r.beats.length : 0;
  const tiles = [
    { label: "Latidos atípicos", value: `<span class="tri" aria-hidden="true"></span>${fmt(r.nAtypical, 0)}`,
      sub: `${fmt(100 * share, 1)} % de los latidos`, hero: true },
    { label: "Latidos analizados", value: fmt(r.beats.length, 0), sub: "con ventana completa" },
    { label: "Frecuencia cardíaca", value: Number.isFinite(r.heartRate) ? `${fmt(r.heartRate, 0)} lpm` : "—",
      sub: "mediana, por RR" },
    { label: "Duración", value: formatTime(r.duration), sub: `${fmt(r.duration, 0)} s a ${r.fs} Hz` },
    { label: "Picos R", value: r.rSource === "annotations" ? "Anotados" : "Detectados",
      sub: r.rSource === "annotations" ? state.annName : "detector automático (sin validar)" },
  ];
  $("tiles").innerHTML = tiles.map((t) => `
    <div class="tile${t.hero ? " tile-hero" : ""}">
      <div class="tile-label">${t.label}</div>
      <div class="tile-value">${t.value}</div>
      <div class="tile-sub">${escapeHtml(t.sub)}</div>
    </div>`).join("");

  renderAgreement();
  $("prev-atyp").disabled = r.nAtypical === 0;
  $("next-atyp").disabled = r.nAtypical === 0;
  renderTable();
}

function renderAgreement() {
  const r = state.result;
  const card = $("agreement-card");
  if (!r.agreement) {
    card.querySelector("#agreement").innerHTML = `<p class="hint">Cargue el archivo de anotaciones
      del registro para comparar los latidos marcados con la clasificación de un cardiólogo.</p>`;
    return;
  }
  const a = r.agreement;
  const rows = a.rows.map((row) => `<tr>
      <td><strong>${escapeHtml(row.symbol)}</strong></td><td>${escapeHtml(row.aami ?? "")}</td>
      <td class="num">${fmt(row.n, 0)}</td><td class="num">${fmt(row.flagged, 0)}</td>
      <td class="num">${pct(row.flagged / row.n)}</td></tr>`).join("");
  card.querySelector("#agreement").innerHTML = `
    <dl class="metrics agree-summary">
      <div><dt>Anómalos anotados marcados (sensibilidad)</dt><dd>${pct(a.sensitivity)}</dd></div>
      <div><dt>Normales (N) sin marcar (especificidad)</dt><dd>${pct(a.specificity)}</dd></div>
    </dl>
    <div class="table-wrap"><table class="agree-table">
      <thead><tr><th>Símbolo</th><th>Clase AAMI</th><th class="num">Latidos</th>
      <th class="num">Marcados</th><th class="num">% marcados</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <p class="hint">Solo los latidos con símbolo N cuentan como normales; L, R, e y j cuentan como anómalos (definición del proyecto).</p>`;
}

const COLUMNS = [
  { key: "row", label: "#", num: true, cell: (b) => b.row + 1 },
  { key: "time", label: "Tiempo", num: true, cell: (b) => formatTime(b.time) },
  { key: "symbol", label: "Anotación", cell: (b) => escapeHtml(b.symbol ?? "—"), annOnly: true },
  { key: "rrPrev", label: "RR previo (s)", num: true, cell: (b) => fmt(b.rrPrev, 3) },
  { key: "hr", label: "FC inst. (lpm)", num: true,
    cell: (b) => (Number.isFinite(b.rrPrev) ? fmt(60 / b.rrPrev, 0) : "—") },
  { key: "ratio", label: "Error / umbral", num: true, cell: (b) => `${fmt(b.ratio, 2)}×` },
  { key: "atypical", label: "Resultado",
    cell: (b) => (b.atypical ? '<span class="pill pill-atypical">Atípico</span>'
      : '<span class="pill pill-normal">Dentro del umbral</span>') },
];

function renderTable() {
  const r = state.result;
  const cols = COLUMNS.filter((c) => !c.annOnly || r.agreement);
  const { key, dir } = state.sort;
  const arrow = (k) => (k === key ? (dir > 0 ? " ▲" : " ▼") : "");
  $("beats-table").querySelector("thead").innerHTML = `<tr>${cols.map((c) =>
    `<th class="${c.num ? "num" : ""}" aria-sort="${c.key === key ? (dir > 0 ? "ascending" : "descending") : "none"}">
      <button type="button" data-sort="${c.key}">${c.label}${arrow(c.key)}</button></th>`).join("")}</tr>`;

  const value = (b) => (key === "hr" ? b.rrPrev : key === "time" ? b.time : b[key]);
  const rows = r.beats.filter((b) => !state.onlyAtypical || b.atypical).slice().sort((a, b) => {
    const va = value(a);
    const vb = value(b);
    if (va === vb) return a.row - b.row;
    if (va == null || Number.isNaN(va)) return 1;
    if (vb == null || Number.isNaN(vb)) return -1;
    return (va > vb ? 1 : -1) * (key === "hr" ? -dir : dir);
  });
  const body = rows.length ? rows.map((b) => `<tr data-row="${b.row}" class="${b.row === state.selected ? "is-selected" : ""}">
      ${cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.cell(b)}</td>`).join("")}</tr>`).join("")
    : `<tr class="empty-row"><td colspan="${cols.length}">Ningún latido supera el umbral.</td></tr>`;
  $("beats-table").querySelector("tbody").innerHTML = body;
}

function selectBeat(row, recenter = false) {
  const r = state.result;
  if (!r || row == null) return;
  state.selected = row;
  const b = r.beats[row];
  const span = state.view.end - state.view.start;
  if (recenter || b.time < state.view.start || b.time > state.view.end) {
    const start = Math.min(Math.max(0, b.time - span / 2), Math.max(0, r.duration - span));
    state.view = { start, end: start + span };
  }
  drawAll();
  const info = [
    ["Tiempo", formatTime(b.time)],
    ["Resultado", b.atypical ? '<span class="pill pill-atypical">Atípico</span>' : '<span class="pill pill-normal">Dentro del umbral</span>'],
    ["Error / umbral", `${fmt(b.ratio, 2)}×`],
    ["Error (MSE)", fmt(b.error, 5)],
    ["RR previo", Number.isFinite(b.rrPrev) ? `${fmt(b.rrPrev, 3)} s` : "—"],
    ["RR siguiente", Number.isFinite(b.rrNext) ? `${fmt(b.rrNext, 3)} s` : "—"],
  ];
  if (b.symbol) info.push(["Anotación", escapeHtml(b.symbol)]);
  $("beat-info").innerHTML = info.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  for (const tr of $("beats-table").querySelectorAll("tbody tr")) {
    tr.classList.toggle("is-selected", Number(tr.dataset.row) === row);
  }
}

function drawAll() {
  const r = state.result;
  if (!r || $("results").hidden) return;
  const view = { ...state.view, selected: state.selected };
  state.errorPoints = drawErrorChart($("error-chart"), r, view);
  state.stripMarkers = drawStrip($("strip-chart"), r, view);
  if (state.selected != null) drawBeat($("beat-chart"), r, state.selected);
  const pos = $("strip-pos");
  const span = state.view.end - state.view.start;
  pos.max = String(Math.max(0, Math.round((r.duration - span) * 10)));
  pos.value = String(Math.round(state.view.start * 10));
}

function setSpan(span) {
  const r = state.result;
  const s = Math.min(span, r.duration);
  const center = state.selected != null ? r.beats[state.selected].time
    : (state.view.start + state.view.end) / 2;
  const start = Math.min(Math.max(0, center - s / 2), Math.max(0, r.duration - s));
  state.view = { start, end: start + s };
  drawAll();
}

function pan(fraction) {
  const r = state.result;
  const span = state.view.end - state.view.start;
  const start = Math.min(Math.max(0, state.view.start + fraction * span), Math.max(0, r.duration - span));
  state.view = { start, end: start + span };
  drawAll();
}

function stepAtypical(direction) {
  const r = state.result;
  const atyp = r.beats.filter((b) => b.atypical);
  if (!atyp.length) return;
  const current = state.selected ?? (direction > 0 ? -1 : r.beats.length);
  const next = direction > 0 ? atyp.find((b) => b.row > current) ?? atyp[0]
    : [...atyp].reverse().find((b) => b.row < current) ?? atyp[atyp.length - 1];
  selectBeat(next.row);
}

/* ---------- Tooltips ---------- */

function showTip(tip, canvas, x, y, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const w = tip.offsetWidth;
  let left = x + 14;
  if (left + w > canvas.clientWidth) left = Math.max(0, x - w - 14);
  tip.style.left = `${left}px`;
  tip.style.top = `${Math.max(0, y - tip.offsetHeight - 10)}px`;
}

function beatTip(b) {
  return `<strong>Latido ${b.row + 1}</strong> · ${formatTime(b.time)}<br>`
    + `Error: ${fmt(b.ratio, 2)}× el umbral<br>`
    + (b.symbol ? `Anotación: ${escapeHtml(b.symbol)}<br>` : "")
    + (b.atypical ? "▲ Atípico" : "● Dentro del umbral");
}

function bindHover(canvasId, tipId, pointsKey) {
  const canvas = $(canvasId);
  const tip = $(tipId);
  const pick = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    return { x, y, p: nearest(state[pointsKey], x, y, pointsKey === "stripMarkers" ? 16 : 12) };
  };
  canvas.addEventListener("pointermove", (e) => {
    if (!state.result) return;
    const { x, y, p } = pick(e);
    if (p) showTip(tip, canvas, x, y, beatTip(state.result.beats[p.row]));
    else tip.hidden = true;
  });
  canvas.addEventListener("pointerleave", () => { tip.hidden = true; });
  canvas.addEventListener("click", (e) => {
    if (!state.result) return;
    const { p } = pick(e);
    if (p) selectBeat(p.row, pointsKey === "errorPoints");
  });
}

/* ---------- Eventos ---------- */

function bindEvents() {
  const dropzone = $("dropzone");
  for (const ev of ["dragenter", "dragover"]) {
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("is-over"); });
  }
  for (const ev of ["dragleave", "drop"]) {
    dropzone.addEventListener(ev, () => dropzone.classList.remove("is-over"));
  }
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    const [file] = e.dataTransfer.files;
    if (file) handleSignalFile(file);
  });
  $("signal-file").addEventListener("change", (e) => {
    const [file] = e.target.files;
    if (file) handleSignalFile(file);
  });
  $("ann-file").addEventListener("change", async (e) => {
    const [file] = e.target.files;
    if (!file) return;
    try {
      await loadAnnotationText(await readFile(file), file.name);
    } catch (err) {
      state.annotations = null;
      $("opt-ann-wrap").hidden = true;
      setStatus(err.message, true);
    }
  });
  $("sample-btn").addEventListener("click", runSample);
  $("opt-fs").addEventListener("input", updateSummary);
  $("options").addEventListener("submit", (e) => {
    e.preventDefault();
    analyze();
  });

  bindHover("error-chart", "error-tip", "errorPoints");
  bindHover("strip-chart", "strip-tip", "stripMarkers");
  const beatCanvas = $("beat-chart");
  const beatTipEl = $("beat-tip");
  beatCanvas.addEventListener("pointermove", (e) => {
    if (!state.result || state.selected == null) return;
    const rect = beatCanvas.getBoundingClientRect();
    const read = drawBeat(beatCanvas, state.result, state.selected, e.clientX - rect.left);
    if (read) {
      showTip(beatTipEl, beatCanvas, read.x, e.clientY - rect.top,
        `<strong>${fmt(read.ms, 0)} ms</strong> desde el R<br>Latido: ${fmt(read.input, 2)}<br>`
        + `Reconstrucción: ${fmt(read.output, 2)}`);
    } else {
      beatTipEl.hidden = true;
    }
  });
  beatCanvas.addEventListener("pointerleave", () => {
    beatTipEl.hidden = true;
    if (state.result && state.selected != null) drawBeat(beatCanvas, state.result, state.selected);
  });

  $("strip-prev").addEventListener("click", () => pan(-0.8));
  $("strip-next").addEventListener("click", () => pan(0.8));
  $("strip-span").addEventListener("change", (e) => setSpan(Number(e.target.value)));
  $("strip-pos").addEventListener("input", (e) => {
    const span = state.view.end - state.view.start;
    const start = Number(e.target.value) / 10;
    state.view = { start, end: start + span };
    drawAll();
  });
  $("prev-atyp").addEventListener("click", () => stepAtypical(-1));
  $("next-atyp").addEventListener("click", () => stepAtypical(1));
  $("only-atypical").addEventListener("change", (e) => {
    state.onlyAtypical = e.target.checked;
    renderTable();
  });
  $("beats-table").addEventListener("click", (e) => {
    const sortBtn = e.target.closest("button[data-sort]");
    if (sortBtn) {
      const key = sortBtn.dataset.sort;
      state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : key === "ratio" ? -1 : 1 };
      renderTable();
      return;
    }
    const tr = e.target.closest("tr[data-row]");
    if (tr) selectBeat(Number(tr.dataset.row));
  });
  $("download-btn").addEventListener("click", () => {
    const blob = new Blob([resultsToCsv(state.result)], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${state.fileName.replace(/\.[^.]+$/, "").replace(/\s+\(.*\)$/, "")}_latidos.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  });
  $("print-btn").addEventListener("click", () => window.print());

  // Impresión siempre en modo claro; redibuja los canvas con esos colores
  let previousTheme = null;
  window.addEventListener("beforeprint", () => {
    previousTheme = document.documentElement.getAttribute("data-theme");
    document.documentElement.setAttribute("data-theme", "light");
    drawAll();
  });
  window.addEventListener("afterprint", () => {
    if (previousTheme == null) document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", previousTheme);
    drawAll();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
  let resizeTimer;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(drawAll, 60);
  }).observe($("results"));
}

async function runSample() {
  try {
    setStatus("Cargando el ECG de ejemplo…");
    const [csv, ann] = await Promise.all([SAMPLE.csv, SAMPLE.ann].map(async (url) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`No se pudo cargar ${url}.`);
      return res.text();
    }));
    await loadSignalText(csv, SAMPLE.name);
    await loadAnnotationText(ann, "ecg_sintetico_annotations.txt");
    await analyze();
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function handleSignalFile(file) {
  try {
    setStatus(`Leyendo ${file.name}…`);
    state.annotations = null;
    state.annName = "";
    $("opt-ann-wrap").hidden = true;
    $("ann-file").value = "";
    $("ann-name").textContent = ANN_HINT;
    await loadSignalText(await readFile(file), file.name);
    setStatus("Revise las opciones y pulse «Analizar».");
  } catch (err) {
    setStatus(err.message, true);
  }
}

bindEvents();
ensureModel().catch(() => {});
// index.html?ejemplo abre la página con el ECG de ejemplo ya analizado
if (new URLSearchParams(window.location.search).has("ejemplo")) runSample();
