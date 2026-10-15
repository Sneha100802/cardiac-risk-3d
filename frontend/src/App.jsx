import { useEffect, useMemo, useState } from "react";
import HeartViewer from "./components/HeartViewer";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const VESSELS = ["LAD", "LCX", "RCA"];

async function post(path, features) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) throw new Error(`${path} failed (${res.status})`);
  return res.json();
}

export default function App() {
  const [schema, setSchema] = useState([]);
  const [values, setValues] = useState({});          // what the user typed
  const [probs, setProbs] = useState({});            // { CAD, LAD, LCX, RCA }
  const [thr, setThr] = useState({});                // per-target decision thresholds
  const [flagged, setFlagged] = useState({});        // { LAD: true, ... } = above threshold
  const [selected, setSelected] = useState(null);    // "LAD" | "LCX" | "RCA" | null
  const [explanation, setExplanation] = useState(null);
  const [error, setError] = useState("");
  const [examples, setExamples] = useState([]);
  const [truth, setTruth] = useState(null);          // recorded angiography of the loaded example

  // Only send fields the user actually filled in; the backend imputes the rest.
  const record = useMemo(
    () => Object.fromEntries(Object.entries(values).filter(([, v]) => v !== "" && v != null)),
    [values]
  );
  const recordKey = JSON.stringify(record);

  useEffect(() => {
    fetch(`${API}/examples`).then((r) => r.json()).then((d) => setExamples(d.examples)).catch(() => {});
  }, []);

  const loadExample = (ex) => {
    // Inputs hold strings; the API accepts numbers sent as text.
    setValues(Object.fromEntries(Object.entries(ex.record).map(([k, v]) => [k, String(v)])));
    setTruth(ex.truth);
  };

  useEffect(() => {
    fetch(`${API}/schema`).then((r) => r.json()).then((d) => setSchema(d.features))
      .catch(() => setError("Cannot reach the API. Is `uvicorn backend.app:app` running?"));
  }, []);

  // Re-predict 300 ms after the user stops typing, so the vessels recolour live.
  useEffect(() => {
    const t = setTimeout(() => {
      post("/predict", record).then((d) => { setProbs(d.probabilities); setThr(d.thresholds); setFlagged(d.flagged); setError(""); })
        .catch((e) => setError(e.message));
    }, 300);
    return () => clearTimeout(t);
  }, [recordKey]);

  // Fetch SHAP for the clicked vessel (or overall CAD when nothing is selected).
  useEffect(() => {
    const target = selected ?? "CAD";
    const t = setTimeout(() => {
      post(`/explain/${target}`, record).then(setExplanation).catch(() => setExplanation(null));
    }, 300);
    return () => clearTimeout(t);
  }, [selected, recordKey]);

  const maxShap = Math.max(0.0001, ...(explanation?.top_features ?? []).map((f) => Math.abs(f.shap)));

  return (
    <div className="app">
      <div className="disclaimer" role="note">
        Decision-support and educational use only. Not a substitute for formal diagnostic imaging
        or clinical judgement.
      </div>

      <main className="layout">
        <section className="viewer">
          <HeartViewer risk={probs} thresholds={thr} selected={selected} onSelect={setSelected} />
          <p className="hint">Drag to rotate, scroll to zoom, click a vessel for its explanation.</p>
        </section>

        <section className="panel">
          {error && <p className="error">{error}</p>}

          <h2>Predicted risk</h2>
          <div className="examples">
            <span>Load a sample patient:</span>
            {examples.map((ex) => (
              <button key={ex.label} onClick={() => loadExample(ex)}>{ex.label.replace("Recorded: ", "")}</button>
            ))}
            <button className="ghost" onClick={() => { setValues({}); setTruth(null); }}>Clear</button>
          </div>
          {truth && (
            <p className="units">
              Recorded angiography: {Object.entries(truth).map(([k, v]) => `${k} ${v.toLowerCase()}`).join(", ")}.
              Sample patients were in the training data, so these predictions look better than they would on new patients.
            </p>
          )}
          <div className="cad">
            <span>Overall CAD{flagged.CAD ? " · above threshold" : ""}</span>
            <strong>{probs.CAD != null ? `${Math.round(probs.CAD * 100)}%` : "–"}</strong>
          </div>
          <ul className="vessel-list">
            {VESSELS.map((v) => (
              <li key={v}>
                <button className={selected === v ? "on" : ""} onClick={() => setSelected(selected === v ? null : v)}>
                  <span>{v} stenosis{flagged[v] ? " · above threshold" : ""}</span>
                  <strong>{probs[v] != null ? `${Math.round(probs[v] * 100)}%` : "–"}</strong>
                </button>
              </li>
            ))}
          </ul>

          <h2>Why: {selected ?? "overall CAD"}</h2>
          {Object.keys(record).length < 5 && (
            <p className="units">Few values entered. Missing fields are filled with typical values, so this
              explanation says little. Load a sample patient or enter more data.</p>
          )}
          {explanation && (
            <>
              <p className="units">Contribution to the prediction ({explanation.units}). Red pushes risk up, blue pushes it down.</p>
              <ul className="shap">
                {explanation.top_features.map((f) => (
                  <li key={f.feature}>
                    <span className="name">
                      {f.feature} <em>{f.was_missing ? "(not entered, filled with typical value)" : f.value}</em>
                    </span>
                    <span className="bar">
                      <i style={{ width: `${(Math.abs(f.shap) / maxShap) * 100}%`, background: f.shap >= 0 ? "#d4352f" : "#2f6fd4" }} />
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>

        <section className="form">
          <h2>Patient data</h2>
          <div className="fields">
            {schema.map((f) => (
              <label key={f.name}>
                {f.name}
                {f.type === "categorical" ? (
                  <select value={values[f.name] ?? ""} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}>
                    <option value="">not entered</option>
                    {f.options.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : (
                  <input type="number" step="any" value={values[f.name] ?? ""}
                         onChange={(e) => setValues({ ...values, [f.name]: e.target.value })} />
                )}
              </label>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
