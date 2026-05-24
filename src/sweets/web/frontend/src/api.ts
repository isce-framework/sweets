import type {
  BowserHandoff,
  Job,
  Manifest,
  SearchRequest,
  SearchResponse,
} from "./types";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${body}`);
  }
  if (r.status === 204) return undefined as T;
  return r.json();
}

export const api = {
  schema: () => http<unknown>("/api/schema"),
  search: (req: SearchRequest) =>
    http<SearchResponse>("/api/search", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  listJobs: () => http<Job[]>("/api/jobs/"),
  createJob: (name: string, config: Record<string, unknown>) =>
    http<Job>("/api/jobs/", {
      method: "POST",
      body: JSON.stringify({ name, config }),
    }),
  startJob: (id: number) =>
    http<Job>(`/api/jobs/${id}/start`, { method: "POST" }),
  cancelJob: (id: number) =>
    http<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  deleteJob: (id: number) =>
    http<void>(`/api/jobs/${id}`, { method: "DELETE" }),
  manifest: (id: number) => http<Manifest>(`/api/jobs/${id}/manifest`),
  bowser: (id: number, autostart: boolean) =>
    http<BowserHandoff>(`/api/jobs/${id}/view?autostart=${autostart}`, {
      method: "POST",
    }),
};

export function jobLogsWs(id: number): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return new WebSocket(`${proto}//${host}/api/ws/jobs/${id}/logs`);
}
