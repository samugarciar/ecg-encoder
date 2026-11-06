// Procesamiento de señal de la interfaz. Funciones puras (sin DOM) para poder probarlas con Node
// contra el pipeline de Python (tests/test_web_dsp.py). Los parámetros (fs, ventana, filtro)
// llegan en `prep`, que es web/model/preprocessing.json, generado desde src/ecg/config.py.

const INDEX_COLUMNS = ["sample #", "sample", "muestra", "index", "indice", "índice", "n"];
const TIME_COLUMNS = ["time", "tiempo", "t", "time (s)", "tiempo (s)", "seconds", "segundos"];

function cleanCell(cell) {
  return cell.trim().replace(/^['"]+|['"]+$/g, "").trim();
}

function detectDelimiter(line) {
  const counts = { ",": 0, ";": 0, "\t": 0 };
  for (const ch of line) if (ch in counts) counts[ch] += 1;
  const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return best[1] > 0 ? best[0] : null; // null: separado por espacios o una sola columna
}

function splitLine(line, delimiter) {
  return delimiter ? line.split(delimiter) : line.trim().split(/\s+/);
}

function toNumber(cell, decimalComma) {
  const text = cleanCell(cell);
  if (text === "") return NaN;
  return Number(decimalComma ? text.replace(",", ".") : text);
}

/**
 * Lee un CSV de señal: una o varias columnas numéricas, con o sin encabezado.
 * Devuelve { columns, values (Float64Array por columna), skipped }.
 */
export function parseSignalCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "" && !l.trim().startsWith("#"));
  if (lines.length < 2) throw new Error("El archivo no tiene suficientes filas de datos.");
  const delimiter = detectDelimiter(lines[0]);
  const decimalComma = delimiter === ";" || delimiter === "\t";

  const first = splitLine(lines[0], delimiter);
  const hasHeader = first.some((cell) => Number.isNaN(toNumber(cell, decimalComma)));
  const width = first.length;
  const columns = hasHeader ? first.map(cleanCell) : first.map((_, i) => `columna ${i + 1}`);
  const rows = hasHeader ? lines.slice(1) : lines;

  const values = columns.map(() => new Float64Array(rows.length));
  let kept = 0;
  let skipped = 0;
  for (const line of rows) {
    const cells = splitLine(line, delimiter);
    if (cells.length !== width) {
      skipped += 1;
      continue;
    }
    const nums = cells.map((c) => toNumber(c, decimalComma));
    if (nums.some((v) => !Number.isFinite(v))) {
      skipped += 1;
      continue;
    }
    for (let j = 0; j < width; j += 1) values[j][kept] = nums[j];
    kept += 1;
  }
  if (kept < 2) throw new Error("No se encontraron filas numéricas válidas en el archivo.");
  if (skipped > 0.01 * rows.length) {
    throw new Error(`${skipped} de ${rows.length} filas no se pudieron leer como números.`);
  }
  return { columns, values: values.map((v) => v.slice(0, kept)), skipped };
}

/** Elige la columna de la señal: la derivación `lead` si existe, si no la primera que no sea índice ni tiempo. */
export function chooseSignalColumn(columns, lead) {
  const lower = columns.map((c) => c.toLowerCase());
  const exact = lower.indexOf(lead.toLowerCase());
  if (exact >= 0) return exact;
  const partial = lower.findIndex((c) => c.includes(lead.toLowerCase()));
  if (partial >= 0) return partial;
  const other = lower.findIndex((c) => !INDEX_COLUMNS.includes(c) && !TIME_COLUMNS.includes(c));
  return other >= 0 ? other : 0;
}

/** Frecuencia de muestreo a partir de una columna de tiempo en segundos, o null si no hay. */
export function inferFsFromTime(columns, values) {
  const idx = columns.findIndex((c) => TIME_COLUMNS.includes(c.toLowerCase()));
  if (idx < 0) return null;
  const t = values[idx];
  const n = Math.min(t.length - 1, 1000);
  const diffs = [];
  for (let i = 0; i < n; i += 1) diffs.push(t[i + 1] - t[i]);
  diffs.sort((a, b) => a - b);
  const dt = diffs[Math.floor(diffs.length / 2)];
  return dt > 0 && dt < 1 ? Math.round(1 / dt) : null;
}

/** Adivina las unidades: "adc" si son enteros alrededor del cero del conversor de MIT-BIH, si no "mv". */
export function guessUnits(x, prep) {
  const step = Math.max(1, Math.floor(x.length / 5000));
  const sample = [];
  let allIntegers = true;
  for (let i = 0; i < x.length; i += step) {
    sample.push(x[i]);
    if (!Number.isInteger(x[i])) allIntegers = false;
  }
  sample.sort((a, b) => a - b);
  const median = sample[Math.floor(sample.length / 2)];
  if (allIntegers && Math.abs(median - prep.adc_zero) < 400) return "adc";
  if (allIntegers && Math.abs(median) > 50) return "uv";
  return "mv";
}

/** Convierte la señal a mV: "adc" (MIT-BIH, (x - 1024) / 200), "uv" (µV) o "mv". */
export function toMillivolts(x, units, prep) {
  const out = new Float64Array(x.length);
  if (units === "adc") {
    for (let i = 0; i < x.length; i += 1) out[i] = (x[i] - prep.adc_zero) / prep.adc_gain;
  } else if (units === "uv") {
    for (let i = 0; i < x.length; i += 1) out[i] = x[i] / 1000;
  } else if (units === "mv") {
    out.set(x);
  } else {
    throw new Error(`Unidades desconocidas: ${units}`);
  }
  return out;
}

/**
 * Lee anotaciones con el formato de MIT-BIH (Time, Sample #, Type, ...) o un CSV "muestra,símbolo".
 * Devuelve solo los latidos (símbolos de AAMI_MAP, D-014), ordenados por muestra.
 */
export function parseAnnotations(text, aamiMap) {
  const beats = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.split("\t")[0].trim();
    if (line === "") continue;
    const fields = line.includes(",") ? line.split(",").map(cleanCell) : line.split(/\s+/);
    let sample;
    let symbol;
    if (fields.length >= 3 && /^\d+:\d/.test(fields[0])) {
      sample = Number(fields[1]);
      symbol = fields[2];
    } else if (fields.length >= 2) {
      sample = Number(fields[0]);
      symbol = fields[1];
    } else {
      continue;
    }
    if (!Number.isInteger(sample) || !(symbol in aamiMap)) continue; // encabezado y no-latidos
    beats.push([sample, symbol]);
  }
  if (beats.length === 0) throw new Error("El archivo de anotaciones no contiene latidos reconocibles.");
  beats.sort((a, b) => a[0] - b[0]);
  return {
    samples: Int32Array.from(beats.map((b) => b[0])),
    symbols: beats.map((b) => b[1]),
  };
}

/** Remuestreo por interpolación lineal de fsIn a fsOut. */
export function resampleLinear(x, fsIn, fsOut) {
  if (fsIn === fsOut) return Float64Array.from(x);
  const nOut = Math.floor(((x.length - 1) * fsOut) / fsIn) + 1;
  const out = new Float64Array(nOut);
  const ratio = fsIn / fsOut;
  for (let i = 0; i < nOut; i += 1) {
    const t = i * ratio;
    const k = Math.min(Math.floor(t), x.length - 2);
    const frac = t - k;
    out[i] = x[k] * (1 - frac) + x[k + 1] * frac;
  }
  return out;
}

/** scipy.signal.sosfilt con estado inicial `zi` (forma directa II transpuesta, sección por sección). */
export function sosfilt(sos, x, zi) {
  const y = Float64Array.from(x);
  for (let s = 0; s < sos.length; s += 1) {
    const [b0, b1, b2, , a1, a2] = sos[s];
    let z0 = zi[s][0];
    let z1 = zi[s][1];
    for (let i = 0; i < y.length; i += 1) {
      const xi = y[i];
      const yi = b0 * xi + z0;
      z0 = b1 * xi - a1 * yi + z1;
      z1 = b2 * xi - a2 * yi;
      y[i] = yi;
    }
  }
  return y;
}

/**
 * scipy.signal.sosfiltfilt con sus valores por defecto: extensión impar de `padlen` muestras en
 * cada borde, estado inicial zi · (primera muestra) en la ida y en la vuelta, y fase cero (D-007).
 */
export function sosfiltfilt(filter, x) {
  const { sos, zi, padlen } = filter;
  const n = x.length;
  if (n <= padlen) {
    throw new Error(`La señal es demasiado corta para filtrar: ${n} muestras (mínimo ${padlen + 1}).`);
  }
  const ext = new Float64Array(n + 2 * padlen);
  for (let i = 0; i < padlen; i += 1) ext[i] = 2 * x[0] - x[padlen - i];
  ext.set(x, padlen);
  for (let i = 0; i < padlen; i += 1) ext[padlen + n + i] = 2 * x[n - 1] - x[n - 2 - i];

  const scaled = (v) => zi.map((row) => [row[0] * v, row[1] * v]);
  const forward = sosfilt(sos, ext, scaled(ext[0]));
  forward.reverse();
  const backward = sosfilt(sos, forward, scaled(forward[0]));
  backward.reverse();
  return backward.slice(padlen, padlen + n);
}

/**
 * Detector de picos R tipo Pan-Tompkins sobre la señal ya filtrada a 0,5-40 Hz.
 * Derivada de 5 puntos, cuadrado, integración en ventana de 150 ms, umbrales adaptativos con
 * período refractario, descarte de ondas T y búsqueda hacia atrás si falta un latido. El R se ubica
 * en la deflexión principal del QRS. NO está validado contra las anotaciones de MIT-BIH: solo se
 * usa cuando el archivo no trae anotaciones.
 */
export function detectRPeaks(y, fs) {
  const n = y.length;
  const slope = new Float64Array(n);
  for (let i = 4; i < n; i += 1) {
    const d = (2 * y[i] + y[i - 1] - y[i - 3] - 2 * y[i - 4]) / 8;
    slope[i] = d * d;
  }
  const w = Math.max(1, Math.round(0.15 * fs));
  const integ = new Float64Array(n);
  let acc = 0;
  for (let i = 0; i < n; i += 1) {
    acc += slope[i];
    if (i >= w) acc -= slope[i - w];
    integ[i] = acc / w;
  }

  // Candidatos: máximos locales separados por al menos 200 ms
  const refractory = Math.round(0.2 * fs);
  const candidates = [];
  for (let i = 1; i < n - 1; i += 1) {
    if (integ[i] > 0 && integ[i] >= integ[i - 1] && integ[i] > integ[i + 1]) {
      const last = candidates[candidates.length - 1];
      if (last !== undefined && i - last < refractory) {
        if (integ[i] > integ[last]) candidates[candidates.length - 1] = i;
      } else {
        candidates.push(i);
      }
    }
  }

  const learn = Math.min(n, 2 * fs);
  let maxLearn = 0;
  let meanLearn = 0;
  for (let i = 0; i < learn; i += 1) {
    maxLearn = Math.max(maxLearn, integ[i]);
    meanLearn += integ[i] / learn;
  }
  let spk = 0.25 * maxLearn;
  let npk = 0.5 * meanLearn;
  const maxSlope = (peak) => {
    let m = 0;
    for (let i = Math.max(4, peak - w); i <= peak; i += 1) m = Math.max(m, slope[i]);
    return m;
  };

  const qrs = [];
  const rr = [];
  let lastSlope = 0;
  let searchFrom = 0;
  const accept = (peak, factor) => {
    if (qrs.length > 0) {
      rr.push(peak - qrs[qrs.length - 1]);
      if (rr.length > 8) rr.shift();
    }
    qrs.push(peak);
    lastSlope = maxSlope(peak);
    spk = factor * integ[peak] + (1 - factor) * spk;
  };

  for (let c = 0; c < candidates.length; c += 1) {
    const peak = candidates[c];
    const threshold = npk + 0.25 * (spk - npk);
    const prev = qrs[qrs.length - 1];

    // Búsqueda hacia atrás: si pasó 1,66 RR sin latido, recupera el mejor candidato perdido
    if (rr.length >= 2 && prev !== undefined) {
      const rrMean = rr.reduce((a, b) => a + b, 0) / rr.length;
      if (peak - prev > 1.66 * rrMean) {
        let best = -1;
        for (let k = searchFrom; k < c; k += 1) {
          const cand = candidates[k];
          if (cand - prev >= refractory && integ[cand] > threshold / 2
              && (best < 0 || integ[cand] > integ[best])) best = cand;
        }
        if (best >= 0) accept(best, 0.25);
      }
    }

    const last = qrs[qrs.length - 1];
    if (integ[peak] > threshold) {
      const isTWave = last !== undefined && peak - last < Math.round(0.36 * fs)
        && maxSlope(peak) < 0.5 * lastSlope;
      if (last !== undefined && peak - last < refractory) {
        npk = 0.125 * integ[peak] + 0.875 * npk;
      } else if (isTWave) {
        npk = 0.125 * integ[peak] + 0.875 * npk;
      } else {
        accept(peak, 0.125);
        searchFrom = c + 1;
      }
    } else {
      npk = 0.125 * integ[peak] + 0.875 * npk;
    }
  }

  // Ubica el R en la deflexión principal dentro del QRS que terminó en el pico integrado
  const margin = Math.round(0.03 * fs);
  const rPeaks = [];
  for (const peak of qrs) {
    const lo = Math.max(0, peak - w - margin);
    const hi = Math.min(n - 1, peak + margin);
    let pos = lo;
    let neg = lo;
    for (let i = lo; i <= hi; i += 1) {
      if (y[i] > y[pos]) pos = i;
      if (y[i] < y[neg]) neg = i;
    }
    const r = -y[neg] > 1.5 * Math.max(y[pos], 0) ? neg : pos;
    const prevR = rPeaks[rPeaks.length - 1];
    if (prevR !== undefined && r - prevR < refractory) {
      if (Math.abs(y[r]) > Math.abs(y[prevR])) rPeaks[rPeaks.length - 1] = r;
    } else {
      rPeaks.push(r);
    }
  }
  return Int32Array.from(rPeaks);
}

/** RR previo y siguiente de cada latido, en segundos; NaN donde no hay vecino (D-017). */
export function rrIntervals(r, fs) {
  const prev = new Float64Array(r.length).fill(NaN);
  const next = new Float64Array(r.length).fill(NaN);
  for (let i = 1; i < r.length; i += 1) {
    prev[i] = (r[i] - r[i - 1]) / fs;
    next[i - 1] = prev[i];
  }
  return { prev, next };
}

/**
 * Recorta y[r - pre : r + post] para cada R y descarta los que no caben (como extract_windows del
 * notebook). Devuelve { windows (Float32Array plano, fila por latido), keep (índices de r) }.
 */
export function extractWindows(y, r, pre, post) {
  const len = pre + post;
  const keep = [];
  for (let i = 0; i < r.length; i += 1) {
    if (r[i] - pre >= 0 && r[i] + post <= y.length) keep.push(i);
  }
  const windows = new Float32Array(keep.length * len);
  keep.forEach((idx, row) => {
    windows.set(y.subarray(r[idx] - pre, r[idx] + post), row * len);
  });
  return { windows, keep: Int32Array.from(keep) };
}

/** z-score por latido (D-008), igual que ecg.models.preprocessing.zscore_beats. */
export function zscoreRows(windows, len, eps = 1e-6) {
  const out = new Float32Array(windows.length);
  for (let start = 0; start < windows.length; start += len) {
    let mean = 0;
    for (let j = 0; j < len; j += 1) mean += windows[start + j];
    mean = Math.fround(mean / len);
    let variance = 0;
    for (let j = 0; j < len; j += 1) {
      const d = windows[start + j] - mean;
      variance += d * d;
    }
    const std = Math.max(Math.fround(Math.sqrt(variance / len)), eps);
    for (let j = 0; j < len; j += 1) out[start + j] = (windows[start + j] - mean) / std;
  }
  return out;
}

/** Normalización configurada en el run (model.json): "zscore" o "none". */
export function prepareInputs(windows, len, normalization) {
  if (normalization === "zscore") return zscoreRows(windows, len);
  if (normalization === "none") return Float32Array.from(windows);
  throw new Error(`Normalización desconocida: ${normalization}`);
}

/** MSE por latido entre la entrada normalizada y su reconstrucción (columna `error` de C2). */
export function mseRows(inputs, outputs, len) {
  const out = new Float32Array(inputs.length / len);
  for (let row = 0; row < out.length; row += 1) {
    let acc = 0;
    for (let j = 0; j < len; j += 1) {
      const d = inputs[row * len + j] - outputs[row * len + j];
      acc += d * d;
    }
    out[row] = acc / len;
  }
  return out;
}
