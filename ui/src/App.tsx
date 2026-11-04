import { useEffect, useRef, useState, type CSSProperties } from "react";
import * as api from "./api";
import type { DemoUser, Health, RecommendResponse, Track } from "./api";
import { Arrow, Heart, Logo, Search, Skip, Sparkle, User } from "./icons";

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
const DIAL_BLURB: Record<number, string> = {
  [-1]: "Stays inside this listener's usual clusters",
  0: "No history bias",
  1: "Pushes toward clusters they haven't heard",
};

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

/* ---------- colour helpers: covers are tinted by level-1 SID cluster ---------- */
function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
function clusterHue(level1: number): number {
  return Math.round((level1 * 137.508) % 360); // golden-angle spread so neighbours differ
}
function coverStyle(t: Pick<Track, "level1" | "track_id">): CSSProperties {
  const h = clusterHue(t.level1);
  const v = hash(t.track_id);
  const angle = 100 + (v % 120);
  const shift = 20 + ((v >> 8) % 40);
  return {
    background: `linear-gradient(${angle}deg, hsl(${h} 72% 58%), hsl(${(h + shift) % 360} 68% 36%))`,
  };
}
function userStyle(id: string): CSSProperties {
  const h = hash(id) % 360;
  return { background: `linear-gradient(135deg, hsl(${h} 70% 62%), hsl(${(h + 50) % 360} 65% 40%))` };
}

function healthState(h: Health | null | "down"): { cls: string; text: string } {
  if (h === null) return { cls: "", text: "Connecting…" };
  if (h === "down") return { cls: "bad", text: "API offline" };
  if (h.sid_model.loaded && h.faiss.loaded && h.explainer.reachable) return { cls: "good", text: "All systems ready" };
  return { cls: "warn", text: "Running degraded" };
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
  const userIndex = user ? users.indexOf(user) + 1 : 0;

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

  async function sendFeedback(t: Track, action: "save" | "skip", runDial: number, q: string): Promise<boolean> {
    try {
      await api.feedback(user?.user_id ?? "anonymous", q, t.track_id, action, runDial);
      setToast(action === "save" ? `Added “${t.track_name}” to Saved` : `Skipped “${t.track_name}”`);
      return true;
    } catch {
      setToast("Couldn't send feedback");
      return false;
    }
  }

  const single = runs.length === 1 ? runs[0] : null;
  const hs = healthState(health);
  const effDial = user ? dial : 0;

  return (
    <div className="shell">
      {/* ------------------------------ sidebar ------------------------------ */}
      <aside className="sidebar">
        <div className="brand">
          <Logo />
          <span>Shelf</span>
        </div>

        <section className="side-block">
          <h2 className="side-title">Listener</h2>
          <ul className="library" role="listbox" aria-label="Listener">
            <li>
              <button role="option" aria-selected={!user} className={`lib-item ${!user ? "active" : ""}`}
                      onClick={() => setUserId("")}>
                <span className="lib-art anon"><User /></span>
                <span className="lib-text">
                  <span className="lib-name">No listener</span>
                  <span className="lib-sub">Neutral recommendations</span>
                </span>
              </button>
            </li>
            {users.map((u, i) => (
              <li key={u.user_id}>
                <button role="option" aria-selected={u.user_id === userId}
                        className={`lib-item ${u.user_id === userId ? "active" : ""}`}
                        onClick={() => setUserId(u.user_id)}>
                  <span className="lib-art round" style={userStyle(u.user_id)}>{i + 1}</span>
                  <span className="lib-text">
                    <span className="lib-name">Listener {i + 1}</span>
                    <span className="lib-sub">{u.top_clusters.join(" · ")}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className={`side-block ${user ? "" : "is-off"}`}>
          <div className="dial-head">
            <h2 className="side-title">Familiar ⟷ Discover</h2>
            <span className="dial-val">{effDial > 0 ? "+" : ""}{effDial.toFixed(2)}</span>
          </div>
          <input
            className="slider" type="range" min={-1} max={1} step={0.25} value={effDial}
            disabled={!user || compare} aria-label="Familiar to discover dial"
            style={{ "--p": `${((effDial + 1) / 2) * 100}%` } as CSSProperties}
            onChange={(e) => setDial(Number(e.target.value))}
          />
          <div className="dial-ends"><span>Familiar</span><span>Discover</span></div>

          <label className="switch-row" title={user ? "" : "Pick a listener to compare"}>
            <span>
              <span className="switch-label">Compare side by side</span>
              <span className="switch-sub">Run −1, 0 and +1 together</span>
            </span>
            <input type="checkbox" className="switch" checked={compare && !!user} disabled={!user}
                   onChange={(e) => setCompare(e.target.checked)} />
          </label>
          {!user && <p className="hint">Pick a listener to use the dial.</p>}
        </section>

        <footer className="side-foot">
          <span className={`dot ${hs.cls}`} />
          <span>
            {hs.text}
            {last && (
              <span className="foot-sub">
                {last.system === "sid" ? "SID model" : last.system === "faiss_fallback" ? "FAISS fallback" : last.system}
                {" · "}{Math.round(last.ms)} ms
              </span>
            )}
          </span>
        </footer>
      </aside>

      {/* ------------------------------- main ------------------------------- */}
      <main className="main">
        <div className="hero">
          <h1>What do you want to hear?</h1>
          <form className="search" onSubmit={(e) => { e.preventDefault(); run(query); }}>
            <Search className="search-icon" />
            <input value={query} onChange={(e) => setQuery(e.target.value)}
                   placeholder="Describe a moment, a mood, a place…" aria-label="Describe what you want to hear" />
            <button type="submit" className="go" disabled={loading || !query.trim()} aria-label="Find music">
              {loading ? <span className="spinner" /> : <Arrow />}
            </button>
          </form>
          <div className="chips">
            {SUGGESTIONS.map((s) => (
              <button key={s} className={`chip ${s === query ? "on" : ""}`} onClick={() => { setQuery(s); run(s); }}>
                {s}
              </button>
            ))}
          </div>
        </div>

        {runs.length === 0 && (
          <div className="empty">
            <p>Type anything — <em>“music for cooking dinner with friends”</em> — and Shelf builds a list for it,
              with a short note on why each track fits.</p>
          </div>
        )}

        {single && (
          <PlaylistView run={single} user={user} userIndex={userIndex} onFeedback={sendFeedback} />
        )}

        {runs.length > 1 && (
          <section className="compare">
            {runs.map((r) => (
              <div key={r.dial} className="compare-col">
                <header className="compare-head">
                  <span className={`dial-pill d${r.dial + 1}`}>{r.dial > 0 ? "+" : ""}{r.dial}</span>
                  <div>
                    <h3>{DIAL_LABEL[r.dial]}</h3>
                    <p>{DIAL_BLURB[r.dial]}</p>
                  </div>
                </header>
                <CompareList run={r} user={user} onFeedback={sendFeedback} />
              </div>
            ))}
          </section>
        )}
      </main>

      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}

type FeedbackFn = (t: Track, a: "save" | "skip", dial: number, q: string) => Promise<boolean>;

/* --------------------------- shared state views --------------------------- */
function StateView({ run }: { run: Run }) {
  if (run.error) {
    return (
      <div className="notice bad">
        <h3>Something went wrong</h3>
        <p>{run.error}</p>
      </div>
    );
  }
  const { resp } = run;
  if (!resp?.meta.guard) return null;
  return (
    <div className={`notice ${resp.meta.guard === "self_harm" ? "care" : ""}`}>
      <h3>{GUARD_TITLE[resp.meta.guard] ?? "Can't help with that"}</h3>
      <p>{resp.message}</p>
    </div>
  );
}

function Banners({ resp }: { resp: RecommendResponse }) {
  return (
    <>
      {resp.meta.system === "faiss_fallback" && (
        <div className="banner warn">The main model was slow or unavailable, so these come from the backup search.</div>
      )}
      {resp.meta.notes.includes("instruction_stripped") && (
        <div className="banner">Shelf ignored the instructions in your request and only used the music part.</div>
      )}
    </>
  );
}

function Skeleton({ rows, compact }: { rows: number; compact?: boolean }) {
  return (
    <div className={`skeleton ${compact ? "compact" : ""}`} aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="sk-row" style={{ animationDelay: `${i * 60}ms` }}>
          <span className="sk-art" />
          <span className="sk-lines"><span /><span /></span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------ playlist view ------------------------------ */
function PlaylistView({ run, user, userIndex, onFeedback }: {
  run: Run; user: DemoUser | null; userIndex: number; onFeedback: FeedbackFn;
}) {
  if (run.error || run.resp?.meta.guard) return <StateView run={run} />;
  const resp = run.resp;
  const tracks = resp?.results ?? [];
  const newCount = tracks.filter((t) => t.new_to_user).length;

  return (
    <section className="playlist">
      <header className="pl-head">
        <div className="mosaic">
          {resp
            ? Array.from({ length: 4 }, (_, i) => {
                const t = tracks[i % Math.max(tracks.length, 1)];
                return <span key={i} style={t ? coverStyle(t) : undefined} />;
              })
            : Array.from({ length: 4 }, (_, i) => <span key={i} className="sk" />)}
        </div>
        <div className="pl-info">
          <span className="eyebrow">Made for {user ? `Listener ${userIndex}` : "you"}</span>
          <h2 className="pl-title">{resp ? resp.query : "Building your list…"}</h2>
          {resp && (
            <p className="pl-meta">
              <b>Shelf</b>
              <span>{tracks.length} songs</span>
              {user && <span>{newCount} new to this listener</span>}
              <span>{resp.meta.system === "sid" ? "Semantic-ID model" : "Backup search"}</span>
              <span>{Math.round(resp.meta.latency_ms)} ms</span>
            </p>
          )}
        </div>
      </header>

      {!resp ? (
        <Skeleton rows={6} />
      ) : (
        <>
          <Banners resp={resp} />
          {tracks.length === 0 ? (
            <div className="notice"><h3>No tracks found</h3><p>Try describing it a different way.</p></div>
          ) : (
            <div className="table" role="table">
              <div className="t-row t-head" role="row">
                <span className="c-num">#</span>
                <span>Title</span>
                <span className="c-why">Why it fits</span>
                <span className="c-cluster">Cluster</span>
                <span className="c-act" />
              </div>
              {tracks.map((t, i) => (
                <TrackRow key={t.track_id} t={t} n={i + 1} showNew={!!user}
                          onFeedback={(a) => onFeedback(t, a, run.dial, resp.query)} />
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}

function useFeedbackState(onFeedback: (a: "save" | "skip") => Promise<boolean>) {
  const [state, setState] = useState<"none" | "save" | "skip">("none");
  const act = async (a: "save" | "skip") => {
    if (state === a) return;
    if (await onFeedback(a)) setState(a);
  };
  return [state, act] as const;
}

function Why({ t }: { t: Track }) {
  const ai = t.explanation_source === "llm";
  return (
    <span className="why" key={t.explanation}>
      {ai && (
        <span className="ai-tag" title="Written by the explainer model">
          <Sparkle /> AI
        </span>
      )}
      <span title={!ai ? t.explanation_fallback_reason ?? "Template explanation" : undefined}>{t.explanation}</span>
    </span>
  );
}

function Actions({ state, act, name }: {
  state: "none" | "save" | "skip"; act: (a: "save" | "skip") => void; name: string;
}) {
  return (
    <span className="act">
      <button className={`icon-btn heart ${state === "save" ? "on" : ""}`} onClick={() => act("save")}
              aria-label={`Save ${name}`} aria-pressed={state === "save"} title="Save">
        <Heart filled={state === "save"} />
      </button>
      <button className={`icon-btn ${state === "skip" ? "on-skip" : ""}`} onClick={() => act("skip")}
              aria-label={`Skip ${name}`} aria-pressed={state === "skip"} title="Not for me">
        <Skip />
      </button>
    </span>
  );
}

function TrackRow({ t, n, showNew, onFeedback }: {
  t: Track; n: number; showNew: boolean; onFeedback: (a: "save" | "skip") => Promise<boolean>;
}) {
  const [state, act] = useFeedbackState(onFeedback);
  return (
    <div className={`t-row ${state === "skip" ? "skipped" : ""}`} role="row">
      <span className="c-num">{n}</span>
      <span className="c-title">
        <span className="art" style={coverStyle(t)} />
        <span className="tt">
          <span className="name">{t.track_name}</span>
          <span className="artist">
            {showNew && t.new_to_user && <span className="new-tag">New</span>}
            {t.artists}
          </span>
        </span>
      </span>
      <span className="c-why"><Why t={t} /></span>
      <span className="c-cluster">
        <span className="cluster">
          <i style={{ background: `hsl(${clusterHue(t.level1)} 65% 48%)` }} />
          {t.cluster_name}
        </span>
        <code className="sid">{t.sid}</code>
      </span>
      <span className="c-act"><Actions state={state} act={act} name={t.track_name} /></span>
    </div>
  );
}

/* ------------------------------ compare view ------------------------------ */
function CompareList({ run, user, onFeedback }: { run: Run; user: DemoUser | null; onFeedback: FeedbackFn }) {
  if (run.error || run.resp?.meta.guard) return <StateView run={run} />;
  if (!run.resp) return <Skeleton rows={5} compact />;
  const resp = run.resp;
  const newCount = resp.results.filter((t) => t.new_to_user).length;
  return (
    <>
      <Banners resp={resp} />
      {user && (
        <div className="new-meter">
          <span className="bar"><span style={{ width: `${(newCount / Math.max(resp.results.length, 1)) * 100}%` }} /></span>
          <span>{newCount} of {resp.results.length} from new clusters</span>
        </div>
      )}
      <ul className="clist">
        {resp.results.map((t) => (
          <CompareItem key={t.track_id} t={t} showNew={!!user}
                       onFeedback={(a) => onFeedback(t, a, run.dial, resp.query)} />
        ))}
      </ul>
    </>
  );
}

function CompareItem({ t, showNew, onFeedback }: {
  t: Track; showNew: boolean; onFeedback: (a: "save" | "skip") => Promise<boolean>;
}) {
  const [state, act] = useFeedbackState(onFeedback);
  return (
    <li className={`citem ${state === "skip" ? "skipped" : ""}`}>
      <span className="art" style={coverStyle(t)} />
      <div className="citem-body">
        <div className="citem-top">
          <span className="tt">
            <span className="name">{t.track_name}</span>
            <span className="artist">
              {showNew && t.new_to_user && <span className="new-tag">New</span>}
              {t.artists}
            </span>
          </span>
          <Actions state={state} act={act} name={t.track_name} />
        </div>
        <Why t={t} />
      </div>
    </li>
  );
}
