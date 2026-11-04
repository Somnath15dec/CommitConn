export const API = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

export type Guard = "rephrase" | "redirect" | "self_harm" | "refuse_or_ignore";

export interface Track {
  track_id: string;
  track_name: string;
  artists: string;
  sid: string;
  cluster_name: string;
  level1: number;
  new_to_user: boolean | null;
  explanation: string;
  explanation_source: "template" | "llm" | "fallback";
  explanation_fallback_reason: string | null;
}

export interface RecommendResponse {
  query: string;
  message: string | null;
  results: Track[];
  meta: {
    system: "sid" | "faiss_fallback" | null;
    guard: Guard | null;
    notes: string[];
    dial: number;
    latency_ms: number;
    fallback_reason?: string;
  };
}

export interface ExplainResponse {
  items: Pick<Track, "track_id" | "explanation" | "explanation_source" | "explanation_fallback_reason">[];
}

export interface DemoUser {
  user_id: string;
  n_history: number;
  top_clusters: string[];
}

export interface Health {
  sid_model: { loaded: boolean };
  faiss: { loaded: boolean; size: number };
  explainer: { reachable: boolean; model_present?: boolean };
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${API}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export const getHealth = () => call<Health>("/health");
export const getUsers = () => call<{ users: DemoUser[] }>("/users").then((r) => r.users);
export const recommend = (query: string, user_id: string | null, dial: number, k = 10) =>
  call<RecommendResponse>("/recommend", { query, user_id, dial, k, explain: "template" });
export const explain = (query: string, track_ids: string[]) => call<ExplainResponse>("/explain", { query, track_ids });
export const feedback = (user_id: string, query: string, track_id: string, action: "save" | "skip", dial: number) =>
  call<{ ok: boolean }>("/feedback", { user_id, query, track_id, action, dial, timestamp: new Date().toISOString() });
