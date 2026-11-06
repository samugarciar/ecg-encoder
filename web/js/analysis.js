// Análisis completo de un registro: de la señal cruda al error de reconstrucción por latido.
// Repite el pipeline de C1 (mV, filtro, ventana) y de ecg.score (normalización, MSE). La inferencia
// llega como función `infer(inputs, n, len)` para poder probar este módulo sin navegador.

import {
  detectRPeaks,
  extractWindows,
  mseRows,
  prepareInputs,
  resampleLinear,
  rrIntervals,
  sosfiltfilt,
  toMillivolts,
} from "./dsp.js";

/**
 * @param {object} input
 * @param {Float64Array} input.signal   señal de la derivación, en las unidades de `units`
 * @param {number} input.fs             frecuencia de muestreo del archivo (Hz)
 * @param {string} input.units          "adc" | "mv" | "uv"
 * @param {{samples: Int32Array, symbols: string[]}|null} input.annotations  latidos anotados
 * @param {object} prep   web/model/preprocessing.json
 * @param {object} card   web/model/model.json
 * @param {Function} infer  (Float32Array, n, len) => Promise<Float32Array>
 */
export async function analyzeEcg(input, prep, card, infer) {
  const warnings = [];
  const { pre, post, total: len } = prep.window;
  if (card.input_length !== len) {
    throw new Error(`El modelo espera ${card.input_length} muestras y la ventana tiene ${len}.`);
  }

  let mv = toMillivolts(input.signal, input.units, prep);
  let annotations = input.annotations;
  if (input.fs !== prep.fs) {
    mv = resampleLinear(mv, input.fs, prep.fs);
    warnings.push(`La señal venía a ${input.fs} Hz y se remuestreó a ${prep.fs} Hz. El modelo `
      + `se entrenó con registros a ${prep.fs} Hz; el remuestreo puede alterar levemente el QRS.`);
    if (annotations) {
      const factor = prep.fs / input.fs;
      annotations = {
        samples: Int32Array.from(annotations.samples, (s) => Math.round(s * factor)),
        symbols: annotations.symbols,
      };
    }
  }
  const fs = prep.fs;
  const filtered = sosfiltfilt(prep.filter, mv);

  let r;
  let symbols = null;
  let rSource;
  if (annotations) {
    r = annotations.samples;
    symbols = annotations.symbols;
    rSource = "annotations";
    const outside = r.filter((s) => s < 0 || s >= mv.length).length;
    if (outside > 0) warnings.push(`${outside} anotaciones caen fuera de la señal y se ignoran.`);
  } else {
    r = detectRPeaks(filtered, fs);
    rSource = "detector";
    warnings.push("Los latidos se ubicaron con el detector automático de picos R, que no está "
      + "validado contra las anotaciones de cardiólogos de MIT-BIH.");
  }
  if (r.length === 0) throw new Error("No se encontró ningún latido en la señal.");

  // El RR se calcula antes de descartar los latidos de borde (D-017)
  const rr = rrIntervals(r, fs);
  const { windows, keep } = extractWindows(filtered, r, pre, post);
  const n = keep.length;
  if (n === 0) throw new Error("Ningún latido tiene una ventana completa dentro de la señal.");
  const droppedEdge = r.length - n;

  const inputs = prepareInputs(windows, len, card.normalization);
  const outputs = await infer(inputs, n, len);
  const error = mseRows(inputs, outputs, len);
  const threshold = card.threshold.value;

  const beats = Array.from(keep, (idx, row) => ({
    row,
    r: r[idx],
    time: r[idx] / fs,
    symbol: symbols ? symbols[idx] : null,
    rrPrev: rr.prev[idx],
    rrNext: rr.next[idx],
    error: error[row],
    ratio: error[row] / threshold,
    atypical: error[row] > threshold,
  }));

  const rrValues = Array.from(rr.prev).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  const medianRr = rrValues.length ? rrValues[Math.floor(rrValues.length / 2)] : NaN;

  return {
    fs,
    duration: mv.length / fs,
    filtered,
    rSource,
    beats,
    windows,
    inputs,
    outputs,
    len,
    pre,
    threshold,
    droppedEdge,
    heartRate: Number.isFinite(medianRr) ? 60 / medianRr : NaN,
    nAtypical: beats.filter((b) => b.atypical).length,
    agreement: symbols ? agreementBySymbol(beats, prep) : null,
    warnings,
  };
}

/** Si hay anotaciones: cuántos latidos de cada símbolo se marcaron como atípicos. */
export function agreementBySymbol(beats, prep) {
  const normal = new Set(prep.normal_symbols);
  const bySymbol = new Map();
  for (const b of beats) {
    const entry = bySymbol.get(b.symbol) || { symbol: b.symbol, aami: prep.aami_map[b.symbol],
      n: 0, flagged: 0 };
    entry.n += 1;
    if (b.atypical) entry.flagged += 1;
    bySymbol.set(b.symbol, entry);
  }
  const rows = [...bySymbol.values()].sort((a, b) => b.n - a.n);
  const tp = beats.filter((b) => !normal.has(b.symbol) && b.atypical).length;
  const fn = beats.filter((b) => !normal.has(b.symbol) && !b.atypical).length;
  const fp = beats.filter((b) => normal.has(b.symbol) && b.atypical).length;
  const tn = beats.filter((b) => normal.has(b.symbol) && !b.atypical).length;
  return {
    rows,
    confusion: { tp, fp, fn, tn },
    sensitivity: tp + fn ? tp / (tp + fn) : null,
    specificity: tn + fp ? tn / (tn + fp) : null,
  };
}

/** CSV con una fila por latido, para descargar. */
export function resultsToCsv(result) {
  const head = ["latido", "muestra_r", "tiempo_s", "simbolo_anotado", "rr_prev_s", "rr_next_s",
    "error", "error_relativo_al_umbral", "atipico"];
  const fmt = (v, d) => (Number.isFinite(v) ? v.toFixed(d) : "");
  const lines = result.beats.map((b) => [
    b.row + 1, b.r, fmt(b.time, 3), b.symbol ?? "", fmt(b.rrPrev, 3), fmt(b.rrNext, 3),
    fmt(b.error, 6), fmt(b.ratio, 3), b.atypical ? 1 : 0,
  ].join(","));
  return `${head.join(",")}\n${lines.join("\n")}\n`;
}
