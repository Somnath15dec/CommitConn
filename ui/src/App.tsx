import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import type { DemoUser, Health, RecommendResponse, Track } from "./api";

const SUGGESTIONS = [
  "something calm for a 3 a.m. study session",
  "rainy afternoon staring out the window",
  "pump me up for leg day",
  "lazy sunday with a coffee",
  "songs for a long drive at night",
  "something upbeat, not sad",
];
const COMPARE_DIALS = [-1, 0, 1] as const;
const DIAL_LABEL: Record<number, string> = { [-1]: "Familiar", 0: "Neutral", 1: "Discover" };

interface Run {
  dial: number;
  resp?: RecommendResponse;
  error?: string;
}

const GUARD_TITLE: Record<string, string> = {
  rephrase: "Could you rephrase that?",
  redirect: "Shelf only does music",
  refuse_or_ignore: "That's not something Shelf can do",
  self_harm: "You're not alone",
};

function healthColor(h: Health | null | "down"): string {
  if (h === "down" || h === null) return "var(--bad)";
  if (h.sid_model.loaded && h.faiss.loaded && h.explainer.reachable) return "var(--good)";
  return "var(--warn)";
}

export default function App() {
  const [health, setHealth] = useState<Health | null | "down">(null);
  const [users, setUsers] = useState<DemoUser[]>([]);
  const [userId, setUserId] = useState<string>("");
  const [dial, setDial] = useState(0);
  const [query, setQuery] = useState("");
  const [compare, setCompare] = useState(false);
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [last, setLast] = useState<{ system: string; ms: number } | null>(null);
  const reqId = useRef(0);

  useEffect(() => {
    const ping = () => api.getHealth().then(setHealth).catch(() => setHealth("down"));
    ping();
    const t = setInterval(ping, 15000);
    api.getUsers().then(setUsers).catch(() => setUsers([]));
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2200);
    return () => clearTimeout(t);
  }, [toast]);

  const user = users.find((u) => u.user_id === userId) ?? null;

  async function run(q: string) {
    if (!q.trim()) return;
    const id = ++reqId.current;
    const dials = compare && user ? [...COMPARE_DIALS] : [user ? dial : 0];
    setLoading(true);
    setRuns(dials.map((d) => ({ dial: d })));
    const done: Run[] = [];
    for (const d of dials) {                       // sequential: the API serializes GPU work anyway
      try {
        const resp = await api.recommend(q, user?.user_id ?? null, d);
        done.push({ dial: d, resp });
        setLast({ system: resp.meta.system ?? `guard: ${resp.meta.guard}`, ms: resp.meta.latency_ms });
      } catch (e) {
        done.push({ dial: d, error: String(e) });
      }
      if (id !== reqId.current) return;
      setRuns([...done, ...dials.slice(done.length).map((x) => ({ dial: x }))]);
    }
    setLoading(false);
    // phase 2: replace templates with grounded LLM explanations
    for (const r of done) {
      if (!r.resp || r.resp.results.length === 0) continue;
      try {
        const ex = await api.explain(r.resp.query, r.resp.results.map((t) => t.track_id));
        if (id !== reqId.current) return;
        const by = new Map(ex.items.map((i) => [i.track_id, i]));
        setRuns((prev) =>
          prev.map((p) =>
            p.dial === r.dial && p.resp
              ? { ...p, resp: { ...p.resp, results: p.resp.results.map((t) => ({ ...t, ...by.get(t.track_id) })) } }
              : p,
          ),
        );
      } catch {
        /* keep templates */
      }
    }
  }

  async function sendFeedback(t: Track, action: "save" | "skip", runDial: number, q: string) {
    try {
      await api.feedback(user?.user_id ?? "anonymous", q, t.track_id, action, runDial);
      setToast(action === "save" ? `Saved “${t.track_name}”` : `Skipped “${t.track_name}”`);
    } catch {
      setToast("Couldn't send feedback");
    }
  }

  const single = runs.length === 1 ? runs[0] : null;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo" aria-hidden>▤</span> Shelf
        </div>
        <div className="status">
          <span className="dot" style={{ background: healthColor(health) }} title="API health" />
          {last ? (
            <span>
              {last.system === "sid" ? "SID model" : last.system === "faiss_fallback" ? "FAISS fallback" : last.system}
              {" · "}
              {Math.round(last.ms)} ms
            </span>
          ) : (
            <span className="muted">{health === "down" ? "API offline" : "ready"}</span>
          )}
        </div>
      </header>

      <section className="controls">
        <label className="field">
          <span className="label">Listener</span>
          <select value={userId} onChange={(e) => setUserId(e.target.value)}>
            <option value="">No user (neutral)</option>
            {users.map((u, i) => (
              <option key={u.user_id} value={u.user_id}>
                Listener {i + 1} · {u.top_clusters.join(" · ")}
              </option>
            ))}
          </select>
        </label>

        <label className={`field dial ${user ? "" : "disabled"}`}>
          <span className="label">
            Familiar ⟷ Discover <b>{(user ? dial : 0).toFixed(2)}</b>
          </span>
          <input type="range" min={-1} max={1} step={0.25} value={user ? dial : 0} disabled={!user || compare}
                 onChange={(e) => setDial(Number(e.target.value))} />
        </label>

        <label className={`toggle ${user ? "" : "disabled"}`} title={user ? "" : "Pick a listener to compare"}>
          <input type="checkbox" checked={compare && !!user} disabled={!user} onChange={(e) => setCompare(e.target.checked)} />
          Compare −1 / 0 / +1
        </label>
      </section>

      <form className="search" onSubmit={(e) => { e.preventDefault(); run(query); }}>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Describe what you want to hear…" />
        <button type="submit" disabled={loading || !query.trim()}>{loading ? "Finding…" : "Find music"}</button>
      </form>
      <div className="chips">
        {SUGGESTIONS.map((s) => (
          <button key={s} className="chip" onClick={() => { setQuery(s); run(s); }}>{s}</button>
        ))}
      </div>

      {single && <Column run={single} user={user} onFeedback={sendFeedback} />}
      {runs.length > 1 && (
        <div className="compare">
          {runs.map((r) => (
            <div key={r.dial} className="compare-col">
              <h3>{DIAL_LABEL[r.dial]} <span className="muted">dial {r.dial > 0 ? "+" : ""}{r.dial}</span></h3>
              <Column run={r} user={user} onFeedback={sendFeedback} compact />
            </div>
          ))}
        </div>
      )}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

function Column({ run, user, onFeedback, compact }: {
  run: Run; user: DemoUser | null; compact?: boolean;
  onFeedback: (t: Track, a: "save" | "skip", dial: number, q: string) => void;
}) {
  if (run.error) return <div className="message bad">Request failed: {run.error}</div>;
  if (!run.resp) return <div className="loading">Loading…</div>;
  const { resp } = run;
  if (resp.meta.guard) {
    return (
      <div className={`message ${resp.meta.guard === "self_harm" ? "care" : ""}`}>
        <h3>{GUARD_TITLE[resp.meta.guard] ?? "Can't help with that"}</h3>
        <p>{resp.message}</p>
      </div>
    );
  }
  const newCount = resp.results.filter((t) => t.new_to_user).length;
  return (
    <>
      {resp.meta.system === "faiss_fallback" && (
        <div className="banner">The main model was slow or unavailable, so these come from the backup search.</div>
      )}
      {resp.meta.notes.includes("instruction_stripped") && (
        <div className="banner subtle">Shelf ignored the instructions in your request and only used the music part.</div>
      )}
      {user && compact && <div className="muted small">{newCount} of {resp.results.length} from clusters new to this listener</div>}
      <div className={compact ? "cards compact" : "cards"}>
        {resp.results.map((t) => (
          <Card key={t.track_id} t={t} showNew={!!user} onFeedback={(a) => onFeedback(t, a, run.dial, resp.query)} />
        ))}
      </div>
    </>
  );
}

function Card({ t, showNew, onFeedback }: { t: Track; showNew: boolean; onFeedback: (a: "save" | "skip") => void }) {
  const ai = t.explanation_source === "llm";
  return (
    <article className="card">
      <div className="card-top">
        <div>
          <div className="title">{t.track_name}</div>
          <div className="artists">{t.artists}</div>
        </div>
        {showNew && t.new_to_user && <span className="new">new</span>}
      </div>
      <div className="meta-row">
        <span className="cluster">{t.cluster_name}</span>
        <code className="sid">{t.sid}</code>
      </div>
      <p className="why" key={t.explanation}>
        <span className={`badge ${ai ? "ai" : ""}`} title={t.explanation_fallback_reason ?? ""}>{ai ? "AI" : "template"}</span>
        {t.explanation}
      </p>
      <div className="actions">
        <button onClick={() => onFeedback("save")}>Save</button>
        <button className="ghost" onClick={() => onFeedback("skip")}>Skip</button>
      </div>
    </article>
  );
}
