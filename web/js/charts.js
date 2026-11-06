// Gráficos en canvas: trazado ECG, minimapa, error por latido y detalle de un latido con su
// reconstrucción. Los colores salen de las variables CSS (styles.css): el rojo (--atypical) se usa
// solo para los latidos atípicos.

function tokens() {
  const css = getComputedStyle(document.documentElement);
  const get = (name) => css.getPropertyValue(name).trim();
  return {
    surface: get("--surface"),
    chartBg: get("--chart-bg"),
    border: get("--border"),
    text: get("--text"),
    muted: get("--text-muted"),
    subtle: get("--text-subtle"),
    accent: get("--accent"),
    accentSoft: get("--accent-soft"),
    atypical: get("--atypical"),
    atypicalBand: get("--atypical-band"),
    atypicalFill: get("--atypical-fill"),
    normal: get("--normal"),
    recon: get("--reconstruction"),
    trace: get("--ecg-trace"),
    gridMinor: get("--ecg-grid-minor"),
    gridMajor: get("--ecg-grid-major"),
    font: "12px Inter, 'Segoe UI', system-ui, sans-serif",
    fontSmall: "11px Inter, 'Segoe UI', system-ui, sans-serif",
  };
}

/** Ajusta el canvas a su tamaño en pantalla y a la densidad de píxeles; devuelve el contexto. */
function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

export function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function niceStep(range, target) {
  const raw = range / target;
  const pow = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].find((m) => m * pow >= raw) * pow;
}

const fmtNum = (v, d) => v.toLocaleString("es-CO", { minimumFractionDigits: d, maximumFractionDigits: d });

/** Menor valor de la serie 1-2-5 que es >= v, y mayor que es <= v (escala logarítmica). */
function niceAbove(v) {
  const d = Math.floor(Math.log10(v));
  for (const k of [d, d + 1]) for (const m of [1, 2, 5]) if (m * 10 ** k >= v) return m * 10 ** k;
  return 10 ** (d + 1);
}
function niceBelow(v) {
  const d = Math.floor(Math.log10(v));
  for (const k of [d, d - 1]) for (const m of [5, 2, 1]) if (m * 10 ** k <= v) return m * 10 ** k;
  return 10 ** (d - 1);
}

/** Marcador de latido: triángulo rojo si es atípico, punto apagado si está dentro del umbral. */
function drawMarker(ctx, x, y, atypical, t, size = 1) {
  ctx.beginPath();
  if (atypical) {
    const r = 6 * size;
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r * 0.9, y + r * 0.65);
    ctx.lineTo(x - r * 0.9, y + r * 0.65);
    ctx.closePath();
  } else {
    ctx.arc(x, y, 3 * size, 0, Math.PI * 2);
  }
  ctx.lineWidth = 2;
  ctx.strokeStyle = t.surface;
  ctx.stroke();
  ctx.fillStyle = atypical ? t.atypical : t.normal;
  ctx.fill();
}

function ring(ctx, x, y, r, t) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.lineWidth = 2;
  ctx.strokeStyle = t.accent;
  ctx.stroke();
}

function timeAxis(ctx, t, x, start, end, plotW, yText, width) {
  ctx.font = t.font;
  ctx.fillStyle = t.muted;
  ctx.textBaseline = "top";
  const step = niceStep(end - start, Math.max(2, Math.floor(plotW / 90)));
  for (let k = Math.ceil(start / step); k * step <= end + 1e-9; k += 1) {
    const xx = x(k * step);
    // Las etiquetas de los bordes se alinean hacia adentro para no salirse del canvas
    ctx.textAlign = xx + 24 > width ? "right" : xx - 24 < 0 ? "left" : "center";
    ctx.fillText(formatTime(k * step), xx, yText);
  }
}

/* ---------- Error por latido ---------- */

/**
 * Error relativo al umbral por latido, en el tiempo (escala logarítmica, umbral = 1×). El eje llega
 * al siguiente valor redondo por encima del máximo, así que ningún punto queda recortado.
 * Devuelve las posiciones de los puntos para el hover y el clic.
 */
export function drawErrorChart(canvas, result, view) {
  const t = tokens();
  const { ctx, width, height } = setup(canvas);
  const PAD = { top: 26, right: 16, bottom: 28, left: 48 };
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const ratios = result.beats.map((b) => b.ratio).filter((v) => v > 0);
  const top = niceAbove(Math.max(...ratios, 1) * 1.02);
  const bottom = niceBelow(Math.min(...ratios, 1) / 1.02);
  const lo = Math.log10(bottom);
  const hi = Math.log10(top);
  const x = (s) => PAD.left + (s / result.duration) * plotW;
  const y = (ratio) => PAD.top + (1 - (Math.log10(Math.max(ratio, bottom)) - lo) / (hi - lo)) * plotH;

  ctx.font = t.font;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let d = Math.floor(lo); d <= Math.ceil(hi); d += 1) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** d;
      if (v < bottom * 0.999 || v > top * 1.001) continue;
      const yy = Math.round(y(v)) + 0.5;
      ctx.strokeStyle = t.border;
      ctx.beginPath();
      ctx.moveTo(PAD.left, yy);
      ctx.lineTo(width - PAD.right, yy);
      ctx.stroke();
      ctx.fillStyle = t.muted;
      ctx.fillText(`${fmtNum(Number(v.toPrecision(2)), v < 1 ? (v < 0.1 ? 2 : 1) : 0)}×`, PAD.left - 8, yy);
    }
  }
  timeAxis(ctx, t, x, 0, result.duration, plotW, height - PAD.bottom + 8, width);

  // Vista actual del visor
  if (view) {
    const a = x(view.start);
    const w = Math.max(2, x(view.end) - a);
    ctx.fillStyle = t.accentSoft;
    ctx.fillRect(a, PAD.top, w, plotH);
    ctx.font = t.fontSmall;
    ctx.fillStyle = t.muted;
    ctx.textBaseline = "bottom";
    ctx.textAlign = "center";
    const label = "Vista actual";
    const half = ctx.measureText(label).width / 2 + 2;
    ctx.fillText(label, Math.min(Math.max(a + w / 2, half), width - half), PAD.top - 6);
  }

  // Umbral
  const yThr = Math.round(y(1)) + 0.5;
  ctx.save();
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = t.muted;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.moveTo(PAD.left, yThr);
  ctx.lineTo(width - PAD.right, yThr);
  ctx.stroke();
  ctx.restore();
  ctx.font = t.font;
  ctx.fillStyle = t.muted;
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText("umbral", width - PAD.right - 2, yThr - 3);

  const points = result.beats.map((b) => ({ x: x(b.time), y: y(b.ratio), row: b.row }));
  for (const pass of [false, true]) {
    for (const b of result.beats) {
      if (b.atypical !== pass) continue;
      if (b.ratio > top) {
        // Nunca debería pasar con el eje redondeado; si pasa, una flecha hacia arriba
        const xx = x(b.time);
        ctx.fillStyle = t.atypical;
        ctx.beginPath();
        ctx.moveTo(xx, PAD.top - 2);
        ctx.lineTo(xx + 5, PAD.top + 6);
        ctx.lineTo(xx - 5, PAD.top + 6);
        ctx.closePath();
        ctx.fill();
      } else {
        drawMarker(ctx, x(b.time), y(b.ratio), b.atypical, t);
      }
    }
  }
  if (view && view.selected != null) {
    const p = points[view.selected];
    ring(ctx, p.x, Math.max(p.y, PAD.top), 9, t);
  }
  return points;
}

/* ---------- Trazado ECG ---------- */

/** Camino de la señal entre i0 e i1; con más muestras que píxeles, pasa por el mín. y máx. de cada columna. */
function signalPath(sig, i0, i1, fs, x, y, left, plotW) {
  const path = new Path2D();
  const perPx = (i1 - i0) / plotW;
  if (perPx <= 1.5) {
    for (let i = i0; i < i1; i += 1) {
      if (i === i0) path.moveTo(x(i / fs), y(sig[i]));
      else path.lineTo(x(i / fs), y(sig[i]));
    }
    return path;
  }
  let started = false;
  for (let px = 0; px < plotW; px += 1) {
    const a = i0 + Math.floor(px * perPx);
    const e = Math.min(i1, i0 + Math.floor((px + 1) * perPx));
    if (e <= a) continue;
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = a; i < e; i += 1) {
      if (sig[i] < mn) mn = sig[i];
      if (sig[i] > mx) mx = sig[i];
    }
    // Recorre mín. y máx. en el orden en que aparecen, para que la línea no se cruce
    const [p, q] = sig[a] >= (mn + mx) / 2 ? [mx, mn] : [mn, mx];
    if (!started) {
      path.moveTo(left + px, y(p));
      started = true;
    } else {
      path.lineTo(left + px, y(p));
    }
    path.lineTo(left + px + 0.5, y(q));
  }
  return path;
}

/** Trazado de la señal filtrada entre view.start y view.end (s), con los atípicos resaltados. */
export function drawStrip(canvas, result, view) {
  const t = tokens();
  const { ctx, width, height } = setup(canvas);
  const PAD = { top: 28, right: 16, bottom: 28, left: 48 };
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const { fs, filtered, pre, len } = result;
  const span = view.end - view.start;
  const i0 = Math.max(0, Math.floor(view.start * fs));
  const i1 = Math.min(filtered.length, Math.ceil(view.end * fs));
  const [yMin, yMax] = result.yRange;
  const x = (s) => PAD.left + ((s - view.start) / span) * plotW;
  const y = (mv) => PAD.top + (1 - (mv - yMin) / (yMax - yMin)) * plotH;

  // Cuadrícula de papel de ECG en tonos neutros; la menor solo hasta 10 s de ventana
  const vLines = (stepS, color) => {
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (let k = Math.ceil(view.start / stepS); k * stepS <= view.end; k += 1) {
      const xx = Math.round(x(k * stepS)) + 0.5;
      ctx.moveTo(xx, PAD.top);
      ctx.lineTo(xx, PAD.top + plotH);
    }
    ctx.stroke();
  };
  const hLines = (stepMv, color) => {
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (let k = Math.ceil(yMin / stepMv); k * stepMv <= yMax; k += 1) {
      const yy = Math.round(y(k * stepMv)) + 0.5;
      ctx.moveTo(PAD.left, yy);
      ctx.lineTo(width - PAD.right, yy);
    }
    ctx.stroke();
  };
  ctx.lineWidth = 1;
  if (span <= 10 && x(view.start + 0.04) - x(view.start) >= 3) {
    vLines(0.04, t.gridMinor);
    hLines(0.1, t.gridMinor);
  }
  vLines(0.2, t.gridMajor);
  hLines(0.5, t.gridMajor);

  ctx.font = t.font;
  ctx.fillStyle = t.muted;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let k = Math.ceil(yMin / 0.5); k * 0.5 <= yMax; k += 1) {
    ctx.fillText(fmtNum(k * 0.5, 1), PAD.left - 8, y(k * 0.5));
  }
  ctx.save();
  ctx.translate(12, PAD.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("mV", 0, 0);
  ctx.restore();
  timeAxis(ctx, t, x, view.start, view.end, plotW, height - PAD.bottom + 8, width);

  // Ventanas de los latidos visibles: [r - pre, r - pre + len)
  const visible = result.beats.filter((b) => b.time >= view.start - 1 && b.time <= view.end + 1);
  const windowRect = (b) => {
    const a = Math.max(PAD.left, x((b.r - pre) / fs));
    const e = Math.min(width - PAD.right, x((b.r - pre + len) / fs));
    return [a, e - a];
  };
  const selected = view.selected != null ? result.beats[view.selected] : null;
  if (selected && !selected.atypical) {
    const [a, w] = windowRect(selected);
    if (w > 0) {
      ctx.fillStyle = t.accentSoft;
      ctx.fillRect(a, PAD.top, w, plotH);
    }
  }
  for (const b of visible) {
    if (!b.atypical) continue;
    const [a, w] = windowRect(b);
    if (w <= 0) continue;
    ctx.fillStyle = t.atypicalBand;
    ctx.fillRect(a, PAD.top, w, plotH);
  }

  // Señal; el tramo dentro de la ventana de un atípico se redibuja en rojo
  const path = signalPath(filtered, i0, i1, fs, x, y, PAD.left, plotW);
  ctx.save();
  ctx.beginPath();
  ctx.rect(PAD.left, PAD.top, plotW, plotH);
  ctx.clip();
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = t.trace;
  ctx.stroke(path);
  for (const b of visible) {
    if (!b.atypical) continue;
    const [a, w] = windowRect(b);
    if (w <= 0) continue;
    ctx.save();
    ctx.beginPath();
    ctx.rect(a, PAD.top, w, plotH);
    ctx.clip();
    ctx.strokeStyle = t.atypical;
    ctx.stroke(path);
    ctx.restore();
  }
  ctx.restore();

  // Marcadores sobre los picos R
  const markers = [];
  const my = PAD.top - 13;
  for (const b of visible) {
    if (b.time < view.start || b.time > view.end) continue;
    const mx = x(b.time);
    drawMarker(ctx, mx, my, b.atypical, t);
    markers.push({ x: mx, y: my, row: b.row });
  }
  if (selected && selected.time >= view.start && selected.time <= view.end) {
    ring(ctx, x(selected.time), my, 9, t);
  }
  return markers;
}

/* ---------- Minimapa ---------- */

/** Registro completo en miniatura, con los atípicos y la ventana visible. Devuelve su geometría. */
export function drawMinimap(canvas, result, view) {
  const t = tokens();
  const { ctx, width, height } = setup(canvas);
  const left = 3;
  const plotW = width - 6;
  const { filtered, fs } = result;
  const x = (s) => left + (s / result.duration) * plotW;
  const [yMin, yMax] = result.yRange;
  const y = (mv) => 4 + (1 - (mv - yMin) / (yMax - yMin)) * (height - 8);

  const path = signalPath(filtered, 0, filtered.length, fs, (s) => x(s), y, left, plotW);
  ctx.strokeStyle = t.subtle;
  ctx.lineWidth = 1;
  ctx.stroke(path);

  ctx.strokeStyle = t.atypical;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (const b of result.beats) {
    if (!b.atypical) continue;
    const xx = Math.round(x(b.time)) + 0.5;
    ctx.moveTo(xx, 3);
    ctx.lineTo(xx, height - 3);
  }
  ctx.stroke();

  const a = x(view.start);
  const w = Math.max(6, x(view.end) - a);
  ctx.fillStyle = t.accentSoft;
  ctx.fillRect(a, 1, w, height - 2);
  ctx.strokeStyle = t.accent;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(a + 0.75, 1.75, w - 1.5, height - 3.5, 3);
  else ctx.rect(a + 0.75, 1.75, w - 1.5, height - 3.5);
  ctx.stroke();

  return {
    x0: a,
    x1: a + w,
    toTime: (px) => Math.min(Math.max(0, ((px - left) / plotW) * result.duration), result.duration),
  };
}

/* ---------- Detalle del latido ---------- */

/** Latido normalizado contra su reconstrucción, con la diferencia sombreada. */
export function drawBeat(canvas, result, row, hoverX = null) {
  const t = tokens();
  const { ctx, width, height } = setup(canvas);
  const PAD = { top: 12, right: 16, bottom: 28, left: 40 };
  const { len, pre, fs } = result;
  const inp = result.inputs.subarray(row * len, (row + 1) * len);
  const out = result.outputs.subarray(row * len, (row + 1) * len);
  // El eje sale del mínimo y el máximo de ambas curvas, con un 10 % de margen
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < len; i += 1) {
    lo = Math.min(lo, inp[i], out[i]);
    hi = Math.max(hi, inp[i], out[i]);
  }
  const margin = (hi - lo || 1) * 0.1;
  lo -= margin;
  hi += margin;
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const ms = (i) => ((i - pre) * 1000) / fs;
  const x = (i) => PAD.left + (i / (len - 1)) * plotW;
  const y = (v) => PAD.top + (1 - (v - lo) / (hi - lo)) * plotH;

  ctx.font = t.font;
  ctx.lineWidth = 1;
  ctx.fillStyle = t.muted;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const stepY = niceStep(hi - lo, 4);
  for (let k = Math.ceil(lo / stepY); k * stepY <= hi; k += 1) {
    const v = Math.abs(k * stepY) < 1e-9 ? 0 : k * stepY; // sin "-0"
    const yy = Math.round(y(v)) + 0.5;
    ctx.strokeStyle = t.border;
    ctx.beginPath();
    ctx.moveTo(PAD.left, yy);
    ctx.lineTo(width - PAD.right, yy);
    ctx.stroke();
    ctx.fillText(fmtNum(v, stepY < 1 ? 1 : 0), PAD.left - 8, yy);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const labelEvery = plotW < 420 ? 200 : 100;
  for (let m = -200; m <= 400; m += 100) {
    const i = pre + (m * fs) / 1000;
    const xx = Math.round(x(i)) + 0.5;
    ctx.strokeStyle = m === 0 ? t.subtle : t.border;
    ctx.beginPath();
    ctx.moveTo(xx, PAD.top);
    ctx.lineTo(xx, PAD.top + plotH);
    ctx.stroke();
    if (m % labelEvery !== 0) continue;
    ctx.fillText(m === 0 ? "R" : `${m} ms`, xx, height - PAD.bottom + 8);
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(PAD.left, PAD.top, plotW, plotH);
  ctx.clip();
  ctx.fillStyle = t.atypicalFill;
  ctx.beginPath();
  for (let i = 0; i < len; i += 1) ctx[i ? "lineTo" : "moveTo"](x(i), y(inp[i]));
  for (let i = len - 1; i >= 0; i -= 1) ctx.lineTo(x(i), y(out[i]));
  ctx.closePath();
  ctx.fill();
  const line = (arr, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let i = 0; i < len; i += 1) ctx[i ? "lineTo" : "moveTo"](x(i), y(arr[i]));
    ctx.stroke();
  };
  line(inp, t.trace);
  line(out, t.recon);
  ctx.restore();

  // Cursor de lectura: línea vertical fina y un punto en cada curva
  if (hoverX != null && hoverX >= PAD.left && hoverX <= width - PAD.right) {
    const i = Math.round(((hoverX - PAD.left) / plotW) * (len - 1));
    const xx = Math.round(x(i)) + 0.5;
    ctx.strokeStyle = t.subtle;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xx, PAD.top);
    ctx.lineTo(xx, PAD.top + plotH);
    ctx.stroke();
    for (const [v, c] of [[inp[i], t.trace], [out[i], t.recon]]) {
      ctx.beginPath();
      ctx.arc(xx, y(v), 3.5, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = t.surface;
      ctx.stroke();
      ctx.fillStyle = c;
      ctx.fill();
    }
    return { ms: ms(i), input: inp[i], output: out[i], left: PAD.left, top: PAD.top };
  }
  return null;
}

/** Punto más cercano a (px, py) dentro de `radius` píxeles, o null. */
export function nearest(points, px, py, radius = 12) {
  let best = null;
  let bestD = radius * radius;
  for (const p of points) {
    const d = (p.x - px) ** 2 + (p.y - py) ** 2;
    if (d <= bestD) {
      best = p;
      bestD = d;
    }
  }
  return best;
}
