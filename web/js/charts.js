// Gráficos en canvas: error por latido, trazado ECG y detalle de un latido con su reconstrucción.
// Los colores salen de las variables CSS (styles.css), así que respetan el modo claro y oscuro.

const PAD = { top: 16, right: 16, bottom: 30, left: 52 };

function tokens() {
  const css = getComputedStyle(document.documentElement);
  const get = (name) => css.getPropertyValue(name).trim();
  return {
    surface: get("--surface"),
    paper: get("--paper"),
    text: get("--ink"),
    text2: get("--ink-2"),
    muted: get("--muted"),
    grid: get("--line"),
    gridStrong: get("--line-strong"),
    paperMinor: get("--grid-minor"),
    paperMajor: get("--grid-major"),
    trace: get("--ink"),
    accent: get("--primary"),
    critical: get("--atypical"),
    criticalWash: get("--atypical-wash"),
    accentWash: get("--primary-wash"),
    font: "13px 'Atkinson Hyperlegible Next', 'Segoe UI', system-ui, sans-serif",
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
  const unit = [1, 2, 5, 10].find((m) => m * pow >= raw) * pow;
  return unit;
}

function drawMarker(ctx, x, y, atypical, t, radius = 4) {
  ctx.beginPath();
  if (atypical) {
    const r = radius + 1.5;
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r * 0.95, y + r * 0.7);
    ctx.lineTo(x - r * 0.95, y + r * 0.7);
    ctx.closePath();
  } else {
    ctx.arc(x, y, radius, 0, Math.PI * 2);
  }
  ctx.lineWidth = 2;
  ctx.strokeStyle = t.surface; // anillo del color de la superficie
  ctx.stroke();
  ctx.fillStyle = atypical ? t.critical : t.accent;
  ctx.fill();
}

/**
 * Error relativo al umbral por latido, en el tiempo (escala logarítmica, umbral = 1).
 * Devuelve las posiciones de los puntos para el hover y el clic.
 */
export function drawErrorChart(canvas, result, view) {
  const t = tokens();
  const { ctx, width, height } = setup(canvas);
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const ratios = result.beats.map((b) => b.ratio);
  const lo = Math.log10(Math.max(1e-3, Math.min(...ratios, 1) * 0.8));
  const hi = Math.log10(Math.max(...ratios, 1) * 1.25);
  const x = (s) => PAD.left + (s / result.duration) * plotW;
  const y = (ratio) => PAD.top + (1 - (Math.log10(ratio) - lo) / (hi - lo)) * plotH;

  ctx.font = t.font;
  ctx.textBaseline = "middle";
  // Rejilla y eje y en múltiplos 1-2-5 de cada década
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  for (let d = Math.floor(lo); d <= Math.ceil(hi); d += 1) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** d;
      if (Math.log10(v) < lo || Math.log10(v) > hi) continue;
      const yy = Math.round(y(v)) + 0.5;
      ctx.strokeStyle = t.grid;
      ctx.beginPath();
      ctx.moveTo(PAD.left, yy);
      ctx.lineTo(width - PAD.right, yy);
      ctx.stroke();
      ctx.fillStyle = t.muted;
      ctx.fillText(`${Number(v.toPrecision(2)).toLocaleString("es-CO")}×`, PAD.left - 8, yy);
    }
  }
  // Eje x en mm:ss
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const step = niceStep(result.duration, Math.max(2, Math.floor(plotW / 90)));
  ctx.fillStyle = t.muted;
  for (let s = 0; s <= result.duration; s += step) {
    // La última etiqueta se alinea a la derecha para no salirse del canvas
    ctx.textAlign = x(s) + 30 > width ? "right" : "center";
    ctx.fillText(formatTime(s), Math.min(x(s), width - 2), height - PAD.bottom + 8);
  }

  // Ventana visible del trazado
  if (view) {
    ctx.fillStyle = t.accentWash;
    ctx.fillRect(x(view.start), PAD.top, Math.max(2, x(view.end) - x(view.start)), plotH);
  }

  // Umbral
  const yThr = Math.round(y(1)) + 0.5;
  ctx.strokeStyle = t.text2;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(PAD.left, yThr);
  ctx.lineTo(width - PAD.right, yThr);
  ctx.stroke();
  ctx.fillStyle = t.text2;
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText("umbral", PAD.left + 4, yThr - 3);

  const points = result.beats.map((b) => ({ x: x(b.time), y: y(b.ratio), row: b.row }));
  // Primero los normales, luego los atípicos encima
  for (const pass of [false, true]) {
    for (const b of result.beats) {
      if (b.atypical !== pass) continue;
      drawMarker(ctx, x(b.time), y(b.ratio), b.atypical, t, 3.5);
    }
  }
  if (view && view.selected != null) {
    const p = points[view.selected];
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
    ctx.lineWidth = 2;
    ctx.strokeStyle = t.text;
    ctx.stroke();
  }
  return points;
}

/** Trazado de la señal filtrada entre view.start y view.end (s), con los latidos marcados. */
export function drawStrip(canvas, result, view) {
  const t = tokens();
  const { ctx, width, height } = setup(canvas);
  const top = PAD.top + 14; // banda superior para los marcadores
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - top - PAD.bottom;
  const { fs, filtered } = result;
  const i0 = Math.max(0, Math.floor(view.start * fs));
  const i1 = Math.min(filtered.length, Math.ceil(view.end * fs));
  const [yMin, yMax] = result.yRange;
  const x = (s) => PAD.left + ((s - view.start) / (view.end - view.start)) * plotW;
  const y = (mv) => top + (1 - (mv - yMin) / (yMax - yMin)) * plotH;

  ctx.font = t.font;
  // Papel de ECG: cuadros pequeños de 0,04 s × 0,1 mV y grandes de 0,2 s × 0,5 mV
  ctx.fillStyle = t.paper;
  ctx.fillRect(PAD.left, top, plotW, plotH);
  const vLines = (stepS, color) => {
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (let k = Math.ceil(view.start / stepS); k * stepS <= view.end; k += 1) {
      const xx = Math.round(x(k * stepS)) + 0.5;
      ctx.moveTo(xx, top);
      ctx.lineTo(xx, top + plotH);
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
  if (x(view.start + 0.04) - x(view.start) >= 3) vLines(0.04, t.paperMinor);
  if (y(yMin) - y(yMin + 0.1) >= 3) hLines(0.1, t.paperMinor);
  vLines(0.2, t.paperMajor);
  hLines(0.5, t.paperMajor);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillStyle = t.muted;
  for (let k = Math.ceil(yMin / 0.5); k * 0.5 <= yMax; k += 1) {
    ctx.fillText((k * 0.5).toLocaleString("es-CO", { minimumFractionDigits: 1 }),
      PAD.left - 8, y(k * 0.5));
  }
  ctx.save();
  ctx.translate(14, top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("mV", 0, 0);
  ctx.restore();
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const step = niceStep(view.end - view.start, Math.max(2, Math.floor(plotW / 80)));
  for (let s = Math.ceil(view.start / step) * step; s <= view.end; s += step) {
    ctx.fillStyle = t.muted;
    ctx.fillText(formatTime(s), x(s), height - PAD.bottom + 8);
  }

  // Latido seleccionado: su ventana [r - pre, r + post)
  if (view.selected != null) {
    const b = result.beats[view.selected];
    const a = (b.r - result.pre) / fs;
    const e = (b.r - result.pre + result.len) / fs;
    ctx.fillStyle = t.accentWash;
    ctx.fillRect(x(a), top, x(e) - x(a), plotH);
  }

  // Señal: si hay más muestras que píxeles, la línea pasa por el mínimo y el máximo de cada columna
  ctx.save();
  ctx.beginPath();
  ctx.rect(PAD.left, top, plotW, plotH);
  ctx.clip();
  ctx.strokeStyle = t.trace;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.beginPath();
  const perPx = (i1 - i0) / plotW;
  if (perPx <= 1.5) {
    for (let i = i0; i < i1; i += 1) {
      const xx = x(i / fs);
      if (i === i0) ctx.moveTo(xx, y(filtered[i]));
      else ctx.lineTo(xx, y(filtered[i]));
    }
  } else {
    for (let px = 0; px < plotW; px += 1) {
      const a = i0 + Math.floor(px * perPx);
      const e = Math.min(i1, i0 + Math.floor((px + 1) * perPx));
      let mn = Infinity;
      let mx = -Infinity;
      let first = 0;
      for (let i = a; i < e; i += 1) {
        if (filtered[i] < mn) mn = filtered[i];
        if (filtered[i] > mx) mx = filtered[i];
        if (i === a) first = filtered[i];
      }
      if (e <= a) continue;
      // Recorre min y max en el orden en que aparecen, para que la línea no se cruce
      const [p, q] = first >= (mn + mx) / 2 ? [mx, mn] : [mn, mx];
      if (px === 0) ctx.moveTo(PAD.left + px, y(p));
      else ctx.lineTo(PAD.left + px, y(p));
      ctx.lineTo(PAD.left + px + 0.5, y(q));
    }
  }
  ctx.stroke();
  ctx.restore();

  const markers = [];
  for (const b of result.beats) {
    if (b.time < view.start || b.time > view.end) continue;
    const mx = x(b.time);
    drawMarker(ctx, mx, PAD.top + 4, b.atypical, t, 4);
    markers.push({ x: mx, y: PAD.top + 4, row: b.row });
  }
  return markers;
}

/** Latido normalizado contra su reconstrucción, con la diferencia sombreada. */
export function drawBeat(canvas, result, row, hoverX = null) {
  const t = tokens();
  const { ctx, width, height } = setup(canvas);
  const { len, pre, fs } = result;
  const inp = result.inputs.subarray(row * len, (row + 1) * len);
  const out = result.outputs.subarray(row * len, (row + 1) * len);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < len; i += 1) {
    lo = Math.min(lo, inp[i], out[i]);
    hi = Math.max(hi, inp[i], out[i]);
  }
  const padY = (hi - lo) * 0.08;
  lo -= padY;
  hi += padY;
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const ms = (i) => ((i - pre) * 1000) / fs;
  const x = (i) => PAD.left + (i / (len - 1)) * plotW;
  const y = (v) => PAD.top + (1 - (v - lo) / (hi - lo)) * plotH;

  ctx.font = t.font;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const stepY = niceStep(hi - lo, 4);
  for (let v = Math.ceil(lo / stepY) * stepY; v <= hi; v += stepY) {
    const yy = Math.round(y(v)) + 0.5;
    ctx.strokeStyle = Math.abs(v) < 1e-9 ? t.gridStrong : t.grid;
    ctx.beginPath();
    ctx.moveTo(PAD.left, yy);
    ctx.lineTo(width - PAD.right, yy);
    ctx.stroke();
    ctx.fillStyle = t.muted;
    ctx.fillText(v.toFixed(Math.abs(stepY) < 1 ? 1 : 0), PAD.left - 8, yy);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const labelEvery = plotW < 420 ? 200 : 100; // en pantallas angostas, una etiqueta cada 200 ms
  for (let m = -200; m <= 400; m += 100) {
    const i = pre + (m * fs) / 1000;
    const xx = Math.round(x(i)) + 0.5;
    ctx.strokeStyle = m === 0 ? t.gridStrong : t.grid;
    ctx.beginPath();
    ctx.moveTo(xx, PAD.top);
    ctx.lineTo(xx, PAD.top + plotH);
    ctx.stroke();
    if (m % labelEvery !== 0) continue;
    ctx.fillStyle = t.muted;
    ctx.fillText(m === 0 ? "R" : `${m} ms`, xx, height - PAD.bottom + 8);
  }

  // Diferencia entre ambas curvas
  ctx.fillStyle = t.criticalWash;
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
  line(out, t.accent);

  // Cruz de lectura
  if (hoverX != null && hoverX >= PAD.left && hoverX <= width - PAD.right) {
    const i = Math.round(((hoverX - PAD.left) / plotW) * (len - 1));
    const xx = Math.round(x(i)) + 0.5;
    ctx.strokeStyle = t.muted;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xx, PAD.top);
    ctx.lineTo(xx, PAD.top + plotH);
    ctx.stroke();
    for (const [v, c] of [[inp[i], t.trace], [out[i], t.accent]]) {
      ctx.beginPath();
      ctx.arc(xx, y(v), 4, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = t.surface;
      ctx.stroke();
      ctx.fillStyle = c;
      ctx.fill();
    }
    return { ms: ms(i), input: inp[i], output: out[i], x: xx };
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
