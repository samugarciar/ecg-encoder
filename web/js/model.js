// Carga del modelo ONNX y de sus metadatos, e inferencia por lotes con onnxruntime-web (WASM).
// Todo corre en el navegador: la señal del paciente no sale del equipo.

export const ORT_VERSION = "1.20.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const BATCH = 256;

async function fetchJson(url) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`No se pudo cargar ${url} (HTTP ${res.status}).`);
  return res.json();
}

/** Descarga model.json, preprocessing.json y model.onnx de `base` y abre la sesión de inferencia. */
export async function loadModel(base = "model/") {
  const [card, prep, ort] = await Promise.all([
    fetchJson(`${base}model.json`),
    fetchJson(`${base}preprocessing.json`),
    import(`${ORT_BASE}ort.wasm.min.mjs`),
  ]);
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = 1; // sin hilos: no exige aislamiento entre orígenes (COOP/COEP)
  const session = await ort.InferenceSession.create(`${base}${card.onnx.file}`, {
    executionProviders: ["wasm"],
  });

  /** Reconstruye `n` latidos de `len` muestras (Float32Array plano) en lotes. */
  async function infer(inputs, n, len) {
    const out = new Float32Array(n * len);
    for (let start = 0; start < n; start += BATCH) {
      const b = Math.min(BATCH, n - start);
      const tensor = new ort.Tensor("float32", inputs.slice(start * len, (start + b) * len),
        [b, 1, len]);
      const result = await session.run({ [card.onnx.input]: tensor });
      out.set(result[card.onnx.output].data, start * len);
    }
    return out;
  }

  return { card, prep, infer };
}
