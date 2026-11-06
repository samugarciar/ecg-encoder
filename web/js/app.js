// Interfaz: carga de archivos, opciones, análisis y presentación de resultados.

import { analyzeEcg, resultsToCsv } from "./analysis.js";
import { drawBeat, drawErrorChart, drawMinimap, drawStrip, formatTime, nearest } from "./charts.js";
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
const NARROW = "(max-width: 1100px)";

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
  minimap: null,
  sort: { key: "time", dir: 1 },
  onlyAtypical: true,
  search: "",
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

const atypicalPill = '<span class="pill pill-atypical">Atípico</span>';
const normalPill = '<span class="pill pill-normal">Dentro del umbral</span>';
const symChip = (s) => `<span class="sym-chip" title="Anotación del cardiólogo">${escapeHtml(s)}</span>`;

/* ---------- Modelo ---------- */

function renderModelInfo(card) {
  const badge = $("model-badge");
  badge.textContent = `Run ${card.run_id}`;
  badge.title = card.is_demo ? "Modelo de demostración entrenado con datos sintéticos" : card.run_id;
  $("demo-banner").hidden = !card.is_demo;
  $("footer-model").textContent = `Modelo ${card.run_id}, exportado el ${card.exported_at.slice(0, 10)} `
    + `desde el commit ${card.git_commit.slice(0, 7)}. Umbral ${fmt(card.threshold.value, 5)}: percentil `
    + `${fmt(card.threshold.percentile, 0)} del error en los latidos normales de validación.`;

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

  const column = $("opt-column");
  column.innerHTML = csv.columns.map((c, i) => `<option value="${i}">${escapeHtml(c)}</option>`).join("");
  const idx = chooseSignalColumn(csv.columns, model.prep.lead);
  column.value = String(idx);
  $("opt-units").value = guessUnits(csv.values[idx], model.prep);
  const fs = inferFsFromTime(csv.columns, csv.values);
  $("opt-fs").value = String(fs ?? model.prep.fs);
  $("options").hidden = false;
  $("options-empty").hidden = true;
  $("signal-name").textContent = name;
  $("signal-detail").textContent = `${fmt(csv.values[0].length, 0)} muestras en ${csv.columns.length} `
    + `${csv.columns.length === 1 ? "columna" : "columnas"}`;
  setZoneLoaded("dropzone", "signal-dz-title", true);
  $("signal-chip").hidden = false;
  updateSummary();
  setStatus(csv.skipped ? `Se omitieron ${csv.skipped} filas que no eran numéricas.` : "");
}

async function loadAnnotationText(text, name) {
  const model = await ensureModel();
  state.annotations = parseAnnotations(text, model.prep.aami_map);
  state.annName = name;
  $("ann-chip-name").textContent = name;
  $("ann-chip-detail").textContent = `${fmt(state.annotations.samples.length, 0)} latidos anotados`;
  setZoneLoaded("ann-dropzone", "ann-dz-title", true);
  $("ann-chip").hidden = false;
  $("opt-ann-wrap").hidden = false;
  $("opt-use-ann").checked = true;
}

// Con un archivo cargado, la zona de carga sigue visible pero compacta, para poder cambiarlo
const ZONE_TITLES = {
  "signal-dz-title": ["Cargue la señal", "Cargar otra señal"],
  "ann-dz-title": ["Anotaciones de los latidos, si las tiene", "Cargar otras anotaciones"],
};
function setZoneLoaded(zoneId, titleId, loaded) {
  $(zoneId).classList.toggle("is-loaded", loaded);
  $(titleId).textContent = ZONE_TITLES[titleId][loaded ? 1 : 0];
}

// Al cambiar o quitar la señal se descartan las anotaciones, para no usar los R de otro registro
function clearAnnotations() {
  state.annotations = null;
  state.annName = "";
  $("ann-file").value = "";
  $("ann-chip").hidden = true;
  setZoneLoaded("ann-dropzone", "ann-dz-title", false);
  $("opt-ann-wrap").hidden = true;
}

function clearSignal() {
  state.csv = null;
  state.fileName = "";
  state.result = null;
  state.selected = null;
  $("signal-file").value = "";
  $("signal-chip").hidden = true;
  setZoneLoaded("dropzone", "signal-dz-title", false);
  $("options").hidden = true;
  $("options-empty").hidden = false;
  $("results").hidden = true;
  clearAnnotations();
  setStatus("");
}

function updateSummary() {
  if (!state.csv) return;
  const n = state.csv.values[0].length;
  const fs = Number($("opt-fs").value);
  $("file-summary").textContent = `${fmt(n, 0)} muestras en ${state.csv.columns.length} `
    + `${state.csv.columns.length === 1 ? "columna" : "columnas"}`
    + (fs > 0 ? `, ${formatTime(n / fs)} de registro.` : ".");
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
    state.search = "";
    $("beat-search").value = "";
    // En pantallas angostas, 10 s de trazado quedan muy apretados
    const span = window.innerWidth < 600 ? 5 : Number($("strip-span").value);
    $("strip-span").value = String(span);
    state.view = { start: 0, end: span };
    const firstAtypical = result.beats.find((b) => b.atypical);
    state.onlyAtypical = result.nAtypical > 0;
    $("only-atypical").checked = state.onlyAtypical;
    setStatus(`Registro analizado: ${fmt(result.beats.length, 0)} latidos.`);
    renderResults();
    selectBeat(firstAtypical ? firstAtypical.row : 0, true);
    if (window.matchMedia(NARROW).matches) setSidebar(false);
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
  $("result-meta").textContent = r.rSource === "annotations"
    ? `Picos R: anotados (${state.annName})`
    : "Picos R: detectados con el detector automático (sin validar)";
  $("strip-lead").textContent = r.lead;
  $("warnings").innerHTML = r.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")
    + (r.droppedEdge ? `<li>${r.droppedEdge} latido(s) en los bordes del registro no tienen ventana completa y no se analizaron.</li>` : "");

  const share = r.beats.length ? r.nAtypical / r.beats.length : 0;
  const none = r.nAtypical === 0;
  $("summary").innerHTML = `
    <div class="count-line">
      <span class="count-tri${none ? " is-none" : ""}" aria-hidden="true"></span>
      <span class="count-num">${fmt(r.nAtypical, 0)}</span>
    </div>
    <p class="count-label">${r.nAtypical === 1 ? "latido atípico" : "latidos atípicos"}</p>
    <p class="count-sub">${none ? "Ningún latido supera el umbral. Esto no descarta una arritmia."
    : `${fmt(100 * share, 1)} % de ${fmt(r.beats.length, 0)} latidos analizados`}</p>
    <div class="share-bar" role="img" aria-label="${fmt(100 * share, 1)} % de los latidos son atípicos">
      <span data-w="${(100 * share).toFixed(2)}"></span></div>`;
  applyBars($("summary"));

  const fields = [
    ["Duración", formatTime(r.duration), `${fmt(r.duration, 0)} s a ${r.fs} Hz`],
    ["Frecuencia cardíaca", Number.isFinite(r.heartRate) ? `${fmt(r.heartRate, 0)} lpm` : "—",
      "mediana de los RR"],
    ["Latidos analizados", fmt(r.beats.length, 0), "con ventana completa"],
  ];
  $("tiles").innerHTML = fields.map(([label, value, sub]) => `
    <div><dt>${label}</dt><dd>${value}<small>${escapeHtml(sub)}</small></dd></div>`).join("");

  renderAgreement();
  $("prev-atyp").disabled = r.nAtypical === 0;
  $("next-atyp").disabled = r.nAtypical === 0;
  renderTable();
}

function renderAgreement() {
  const r = state.result;
  const box = $("agreement");
  if (!r.agreement) {
    box.innerHTML = `<p class="hint">Cargue el archivo de anotaciones del registro para comparar los
      latidos marcados con la clasificación de un cardiólogo.</p>`;
    return;
  }
  const a = r.agreement;
  const normal = new Set(state.model.prep.normal_symbols);
  const rows = a.rows.map((row) => {
    const share = row.flagged / row.n;
    const kind = normal.has(row.symbol) ? "is-normal" : "is-atypical";
    return `<tr>
      <td>${symChip(row.symbol)}</td><td>${escapeHtml(row.aami ?? "")}</td>
      <td class="num">${fmt(row.n, 0)}</td><td class="num">${fmt(row.flagged, 0)}</td>
      <td class="num pct-cell"><i class="pct-bar ${kind}" data-w="${(100 * share).toFixed(1)}"></i><span>${pct(share)}</span></td></tr>`;
  }).join("");
  box.innerHTML = `
    <dl class="agree-metrics">
      <div><dt>Anómalos anotados marcados (sensibilidad)</dt><dd>${pct(a.sensitivity)}</dd></div>
      <div><dt>Normales (N) sin marcar (especificidad)</dt><dd>${pct(a.specificity)}</dd></div>
    </dl>
    <table class="agree-table">
      <thead><tr><th>Símbolo</th><th>Clase AAMI</th><th class="num">Latidos</th>
      <th class="num">Marcados</th><th class="num">% marcados</th></tr></thead>
      <tbody>${rows}</tbody></table>
    <p class="agree-note">Solo los latidos con símbolo N cuentan como normales; L, R, e y j cuentan como anómalos (definición del proyecto).</p>`;
  applyBars(box);
}

/**
 * La CSP del sitio (style-src 'self') bloquea los atributos style del HTML; los anchos y posiciones
 * de las barras se pasan como data-w y data-l y se aplican por el DOM, que sí está permitido.
 */
function applyBars(root) {
  for (const el of root.querySelectorAll("[data-w]")) el.style.width = `${el.dataset.w}%`;
  for (const el of root.querySelectorAll("[data-l]")) el.style.left = `${el.dataset.l}%`;
}

/** Mini trazo (80×24) del latido normalizado, para la columna "Forma". */
function sparkline(b) {
  const r = state.result;
  const row = r.inputs.subarray(b.row * r.len, (b.row + 1) * r.len);
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of row) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const n = 64;
  const pts = [];
  for (let k = 0; k < n; k += 1) {
    const v = row[Math.round((k / (n - 1)) * (r.len - 1))];
    pts.push(`${((k / (n - 1)) * 78 + 1).toFixed(1)},${(22 - ((v - lo) / (hi - lo || 1)) * 20).toFixed(1)}`);
  }
  return `<svg class="spark${b.atypical ? " is-atypical" : ""}" viewBox="0 0 80 24" aria-hidden="true">`
    + `<polyline points="${pts.join(" ")}"/></svg>`;
}

/** Número de error/umbral con una mini barra logarítmica y una marca en el umbral. */
function ratioCell(b, scale) {
  const pos = (v) => Math.min(100, Math.max(0, ((Math.log10(v) - scale.lo) / (scale.hi - scale.lo)) * 100));
  return `<span class="ratio-cell">${fmt(b.ratio, 2)}×<span class="ratio-bar${b.atypical ? " is-atypical" : ""}" aria-hidden="true">`
    + `<span data-w="${pos(b.ratio).toFixed(1)}"></span><i data-l="${pos(1).toFixed(1)}"></i></span></span>`;
}

function ratioScale(beats) {
  const ratios = beats.map((b) => b.ratio).filter((v) => v > 0);
  return { lo: Math.log10(Math.min(...ratios, 0.1)), hi: Math.log10(Math.max(...ratios, 2)) };
}

const COLUMNS = [
  { key: "row", label: "#", num: true, cell: (b) => b.row + 1 },
  { key: "time", label: "Tiempo", num: true, cell: (b) => formatTime(b.time) },
  { key: "symbol", label: "Anotación", cell: (b) => (b.symbol ? symChip(b.symbol) : "—"), annOnly: true },
  { key: "shape", label: "Forma", cls: "col-shape", sortable: false, cell: (b) => sparkline(b) },
  { key: "rrPrev", label: "RR previo (s)", num: true, cell: (b) => fmt(b.rrPrev, 3) },
  { key: "hr", label: "FC inst. (lpm)", num: true, cls: "col-hr",
    cell: (b) => (Number.isFinite(b.rrPrev) ? fmt(60 / b.rrPrev, 0) : "—") },
  { key: "ratio", label: "Error / umbral", num: true, cell: (b, scale) => ratioCell(b, scale) },
  { key: "atypical", label: "Resultado", cls: "col-fill", cell: (b) => (b.atypical ? atypicalPill : normalPill) },
];

function matchesSearch(b) {
  const q = state.search.trim();
  if (!q) return true;
  if (q.includes(":")) return formatTime(b.time).startsWith(q);
  if (/^\d+$/.test(q)) return String(b.row + 1).startsWith(q);
  return formatTime(b.time).includes(q);
}

function renderTable() {
  const r = state.result;
  const cols = COLUMNS.filter((c) => !c.annOnly || r.agreement);
  const { key, dir } = state.sort;
  const arrow = (k) => (k === key ? (dir > 0 ? " ▲" : " ▼") : "");
  $("beats-table").querySelector("thead").innerHTML = `<tr>${cols.map((c) => {
    const cls = [c.num ? "num" : "", c.cls ?? ""].join(" ").trim();
    if (c.sortable === false) return `<th class="${cls}">${c.label}</th>`;
    return `<th class="${cls}" aria-sort="${c.key === key ? (dir > 0 ? "ascending" : "descending") : "none"}">
      <button type="button" data-sort="${c.key}">${c.label}${arrow(c.key)}</button></th>`;
  }).join("")}</tr>`;

  const scale = ratioScale(r.beats);
  const value = (b) => (key === "hr" ? b.rrPrev : key === "time" ? b.time : b[key]);
  const rows = r.beats.filter((b) => (!state.onlyAtypical || b.atypical) && matchesSearch(b)).slice().sort((a, b) => {
    const va = value(a);
    const vb = value(b);
    if (va === vb) return a.row - b.row;
    if (va == null || Number.isNaN(va)) return 1;
    if (vb == null || Number.isNaN(vb)) return -1;
    return (va > vb ? 1 : -1) * (key === "hr" ? -dir : dir);
  });
  const empty = state.search.trim() ? "Ningún latido coincide con la búsqueda." : "Ningún latido supera el umbral.";
  const body = rows.length ? rows.map((b) => `<tr data-row="${b.row}" class="${b.row === state.selected ? "is-selected" : ""}">
      ${cols.map((c) => `<td class="${[c.num ? "num" : "", c.cls ?? ""].join(" ").trim()}">${c.cell(b, scale)}</td>`).join("")}</tr>`).join("")
    : `<tr class="empty-row"><td colspan="${cols.length}">${empty}</td></tr>`;
  $("beats-table").querySelector("tbody").innerHTML = body;
  applyBars($("beats-table"));
}

function updateCounter() {
  const r = state.result;
  const atyp = r.beats.filter((b) => b.atypical);
  const idx = atyp.findIndex((b) => b.row === state.selected);
  $("atyp-counter").textContent = idx >= 0 ? `${idx + 1} de ${atyp.length}`
    : `${atyp.length} ${atyp.length === 1 ? "atípico" : "atípicos"}`;
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
    ["Resultado", (b.atypical ? atypicalPill : normalPill) + (b.symbol ? symChip(b.symbol) : "")],
    ["Error / umbral", `${fmt(b.ratio, 2)}×`],
    ["Error (MSE)", fmt(b.error, 5)],
    ["RR previo", Number.isFinite(b.rrPrev) ? `${fmt(b.rrPrev, 3)} s` : "—"],
    ["RR siguiente", Number.isFinite(b.rrNext) ? `${fmt(b.rrNext, 3)} s` : "—"],
  ];
  $("beat-info").innerHTML = info.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  updateCounter();
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
  state.minimap = drawMinimap($("minimap"), r, view);
  if (state.selected != null) drawBeat($("beat-chart"), r, state.selected);
  const span = state.view.end - state.view.start;
  const maxStart = Math.max(0, r.duration - span);
  $("minimap").setAttribute("aria-valuenow", String(Math.round(maxStart ? (100 * state.view.start) / maxStart : 0)));
  $("minimap").setAttribute("aria-valuetext", `${formatTime(state.view.start)} a ${formatTime(state.view.end)}`);
}

function setViewStart(start) {
  const r = state.result;
  const span = state.view.end - state.view.start;
  const s = Math.min(Math.max(0, start), Math.max(0, r.duration - span));
  state.view = { start: s, end: s + span };
  drawAll();
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
  const span = state.view.end - state.view.start;
  setViewStart(state.view.start + fraction * span);
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
  return `<strong>Latido ${b.row + 1}</strong>, en ${formatTime(b.time)}<br>`
    + (b.symbol ? `Anotación: ${escapeHtml(b.symbol)}<br>` : "")
    + `Error: ${fmt(b.ratio, 2)}× el umbral<br>`
    + (b.atypical ? '<span class="tip-atyp">▲ Atípico</span>' : '<span class="tip-normal">● Dentro del umbral</span>');
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
    canvas.style.cursor = p ? "pointer" : "default";
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

/** Minimapa: clic o arrastre para mover la ventana visible; flechas del teclado para desplazarla. */
function bindMinimap() {
  const canvas = $("minimap");
  let offset = null;
  const px = (e) => e.clientX - canvas.getBoundingClientRect().left;
  canvas.addEventListener("pointerdown", (e) => {
    if (!state.result || !state.minimap) return;
    const p = px(e);
    const span = state.view.end - state.view.start;
    const { x0, x1, toTime } = state.minimap;
    offset = p >= x0 && p <= x1 ? toTime(p) - state.view.start : span / 2;
    setViewStart(toTime(p) - offset);
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // puntero sintético o ya liberado: el arrastre sigue funcionando sin captura
    }
    canvas.classList.add("is-dragging");
  });
  canvas.addEventListener("pointermove", (e) => {
    if (offset == null || !state.minimap) return;
    setViewStart(state.minimap.toTime(px(e)) - offset);
  });
  const end = () => {
    offset = null;
    canvas.classList.remove("is-dragging");
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("keydown", (e) => {
    if (!state.result) return;
    if (e.key === "ArrowLeft") pan(-0.25);
    else if (e.key === "ArrowRight") pan(0.25);
    else if (e.key === "Home") setViewStart(0);
    else if (e.key === "End") setViewStart(state.result.duration);
    else return;
    e.preventDefault();
  });
}

/* ---------- Eventos ---------- */

function bindDrop(zoneId, onFile) {
  const zone = $(zoneId);
  for (const ev of ["dragenter", "dragover"]) {
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("is-over"); });
  }
  for (const ev of ["dragleave", "drop"]) {
    zone.addEventListener(ev, () => zone.classList.remove("is-over"));
  }
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    const [file] = e.dataTransfer.files;
    if (file) onFile(file);
  });
}

function setSidebar(open) {
  $("sidebar").classList.toggle("is-collapsed", !open);
  $("sidebar-toggle").setAttribute("aria-expanded", String(open));
}

function bindEvents() {
  bindDrop("dropzone", handleSignalFile);
  bindDrop("ann-dropzone", handleAnnotationFile);
  $("signal-file").addEventListener("change", (e) => {
    const [file] = e.target.files;
    if (file) handleSignalFile(file);
  });
  $("ann-file").addEventListener("change", (e) => {
    const [file] = e.target.files;
    if (file) handleAnnotationFile(file);
  });
  $("signal-clear").addEventListener("click", clearSignal);
  $("ann-clear").addEventListener("click", clearAnnotations);
  $("sample-btn").addEventListener("click", runSample);
  $("opt-fs").addEventListener("input", updateSummary);
  $("options").addEventListener("submit", (e) => {
    e.preventDefault();
    analyze();
  });

  // Pastilla del modelo de demostración: el aviso completo se abre al pasar el cursor o al hacer clic
  const demo = $("demo-banner");
  $("demo-pill").addEventListener("click", (e) => {
    e.stopPropagation();
    const open = !demo.classList.contains("is-open");
    demo.classList.toggle("is-open", open);
    $("demo-pill").setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", () => {
    demo.classList.remove("is-open");
    $("demo-pill").setAttribute("aria-expanded", "false");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") demo.classList.remove("is-open");
  });

  $("sidebar-toggle").addEventListener("click", () => {
    setSidebar($("sidebar").classList.contains("is-collapsed"));
  });

  bindHover("error-chart", "error-tip", "errorPoints");
  bindHover("strip-chart", "strip-tip", "stripMarkers");
  bindMinimap();

  // Detalle del latido: el tooltip queda fijo arriba a la izquierda, nunca sobre las curvas
  const beatCanvas = $("beat-chart");
  const beatTipEl = $("beat-tip");
  beatCanvas.addEventListener("pointermove", (e) => {
    if (!state.result || state.selected == null) return;
    const rect = beatCanvas.getBoundingClientRect();
    const read = drawBeat(beatCanvas, state.result, state.selected, e.clientX - rect.left);
    if (read) {
      beatTipEl.innerHTML = `<strong>${fmt(read.ms, 0)} ms</strong> desde el R<br>`
        + `Latido: ${fmt(read.input, 2)}<br>Reconstrucción: ${fmt(read.output, 2)}`;
      beatTipEl.style.left = `${read.left + 8}px`;
      beatTipEl.style.top = `${read.top + 4}px`;
      beatTipEl.hidden = false;
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
  $("prev-atyp").addEventListener("click", () => stepAtypical(-1));
  $("next-atyp").addEventListener("click", () => stepAtypical(1));
  $("only-atypical").addEventListener("change", (e) => {
    state.onlyAtypical = e.target.checked;
    renderTable();
  });
  $("beat-search").addEventListener("input", (e) => {
    state.search = e.target.value;
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

  // Impresión en la variante clara; redibuja los canvas con esos colores
  window.addEventListener("beforeprint", () => {
    document.documentElement.setAttribute("data-theme", "print");
    drawAll();
  });
  window.addEventListener("afterprint", () => {
    document.documentElement.removeAttribute("data-theme");
    drawAll();
  });
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
    clearAnnotations();
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
    clearAnnotations();
    await loadSignalText(await readFile(file), file.name);
    setStatus("Revise cómo se va a leer el archivo y pulse «Analizar el registro».");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function handleAnnotationFile(file) {
  try {
    await loadAnnotationText(await readFile(file), file.name);
  } catch (err) {
    clearAnnotations();
    setStatus(err.message, true);
  }
}

bindEvents();
ensureModel().catch(() => {});
document.fonts?.ready.then(drawAll);
// index.html?ejemplo abre la página con el ECG de ejemplo ya analizado
if (new URLSearchParams(window.location.search).has("ejemplo")) runSample();
