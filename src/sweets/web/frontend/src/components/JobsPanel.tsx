import { useEffect, useState } from "react";
import { useAppState } from "../state";
import type { Job, Manifest, BowserHandoff } from "../types";
import { api, jobLogsWs } from "../api";

const STEPS = [
  "download",
  "geocode",
  "ifg",
  "stitch",
  "unwrap",
];

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function JobsPanel() {
  const { selectedJobId, setSelectedJobId } = useAppState();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let alive = true;
    const tick = () =>
      api.listJobs().then((j) => {
        if (alive) setJobs(j);
      });
    tick();
    const id = setInterval(tick, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [refreshKey]);

  const selected = jobs.find((j) => j.id === selectedJobId) ?? null;

  return (
    <div>
      <h2>Jobs ({jobs.length})</h2>
      <ul className="job-list">
        {jobs.map((j) => (
          <li
            key={j.id}
            className={
              "job-item" + (j.id === selectedJobId ? " selected" : "")
            }
            onClick={() => setSelectedJobId(j.id)}
          >
            <div className="job-name">{j.name}</div>
            <div className="job-meta">
              <span className={"status " + j.status}>{j.status}</span>{" "}
              #{j.id} · step {j.current_step}/5
            </div>
          </li>
        ))}
        {jobs.length === 0 && <li className="muted">No jobs yet.</li>}
      </ul>

      {selected && (
        <JobDetail
          job={selected}
          onChange={() => setRefreshKey((k) => k + 1)}
        />
      )}
    </div>
  );
}

function JobDetail({ job, onChange }: { job: Job; onChange: () => void }) {
  return (
    <JobDetailBody job={job} onChange={onChange} />
  );
}

function JobDetailBody({ job, onChange }: { job: Job; onChange: () => void }) {
  // `job.current_step` from the DB only updates at completion (the executor
  // commits the final step in its `finally` block). The WebSocket already
  // streams the live step alongside each log line, so we lift that into
  // local state and take the max with the DB value — the bar animates in
  // real time while a job is running and stays correct after refresh.
  const [liveStep, setLiveStep] = useState(0);
  useEffect(() => {
    setLiveStep(0);
  }, [job.id]);
  const step = Math.max(job.current_step, liveStep);
  return (
    <>
      <h2>Detail · #{job.id}</h2>
      <div className="muted">
        work_dir: <code>{job.work_dir ?? "—"}</code>
      </div>

      <div className="step-bar">
        {STEPS.map((label, i) => {
          const idx = i + 1;
          const cls =
            step > idx
              ? " done"
              : step === idx && job.status === "running"
                ? " active"
                : step >= idx
                  ? " done"
                  : "";
          return <div key={label} className={"step" + cls} title={label} />;
        })}
      </div>

      <JobActions job={job} onChange={onChange} />
      <JobLogs
        jobId={job.id}
        status={job.status}
        onStep={(s) => setLiveStep((prev) => (s > prev ? s : prev))}
      />
      <JobManifest jobId={job.id} />
      <BowserButton jobId={job.id} />
    </>
  );
}

function JobActions({ job, onChange }: { job: Job; onChange: () => void }) {
  const [busy, setBusy] = useState(false);

  async function start() {
    setBusy(true);
    try {
      await api.startJob(job.id);
    } finally {
      setBusy(false);
      onChange();
    }
  }
  async function cancel() {
    setBusy(true);
    try {
      await api.cancelJob(job.id);
    } finally {
      setBusy(false);
      onChange();
    }
  }
  async function del() {
    if (!confirm(`Delete job ${job.name}?`)) return;
    setBusy(true);
    try {
      await api.deleteJob(job.id);
    } finally {
      setBusy(false);
      onChange();
    }
  }

  return (
    <div style={{ display: "flex", gap: 6, margin: "8px 0" }}>
      {job.status === "pending" && (
        <button onClick={start} disabled={busy}>
          Start
        </button>
      )}
      {job.status === "running" && (
        <button className="danger" onClick={cancel} disabled={busy}>
          Cancel
        </button>
      )}
      {job.status !== "running" && (
        <button className="secondary" onClick={del} disabled={busy}>
          Delete
        </button>
      )}
    </div>
  );
}

function JobLogs({
  jobId,
  status,
  onStep,
}: {
  jobId: number;
  status: Job["status"];
  onStep?: (step: number) => void;
}) {
  const [lines, setLines] = useState<string[]>([]);

  useEffect(() => {
    setLines([]);
    const ws = jobLogsWs(jobId);
    ws.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data);
        if (m.type === "history") {
          setLines(m.lines);
          if (typeof m.step === "number") onStep?.(m.step);
        } else if (m.type === "log") {
          setLines((prev) => [...prev, m.line]);
          if (typeof m.step === "number") onStep?.(m.step);
        }
      } catch {
        // ignore malformed frames
      }
    };
    return () => ws.close();
  }, [jobId, status, onStep]);

  return (
    <>
      <h2>Logs</h2>
      <div className="log-window">
        {lines.length === 0 ? (
          <span className="muted">(no output yet)</span>
        ) : (
          lines.join("\n")
        )}
      </div>
    </>
  );
}

function JobManifest({ jobId }: { jobId: number }) {
  const [m, setM] = useState<Manifest | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = () =>
      api.manifest(jobId).then((mm) => {
        if (alive) setM(mm);
      });
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [jobId]);

  return (
    <>
      <h2>Results</h2>
      {m == null ? (
        <p className="muted">Loading...</p>
      ) : !m.exists ? (
        <p className="muted">work_dir does not exist yet.</p>
      ) : m.entries.length === 0 ? (
        <p className="muted">No outputs yet.</p>
      ) : (
        <div className="manifest">
          {m.entries.map((e) => (
            <div className="file" key={e.path}>
              <span className="path" title={e.kind}>
                {e.path}
              </span>
              <span className="size">{fmtBytes(e.size)}</span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function BowserButton({ jobId }: { jobId: number }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<BowserHandoff | null>(null);

  async function go(autostart: boolean) {
    setBusy(true);
    try {
      const r = await api.bowser(jobId, autostart);
      setResult(r);
      if (r.url) window.open(r.url, "_blank");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <h2>View in bowser</h2>
      <div style={{ display: "flex", gap: 6 }}>
        <button onClick={() => go(false)} disabled={busy}>
          Setup
        </button>
        <button
          className="secondary"
          onClick={() => go(true)}
          disabled={busy}
        >
          Setup &amp; open
        </button>
      </div>
      {result && (
        <div className="muted" style={{ marginTop: 6 }}>
          <code>{result.command}</code>
          {!result.ran && (
            <div className="error">{result.stderr || "not run"}</div>
          )}
        </div>
      )}
    </>
  );
}
