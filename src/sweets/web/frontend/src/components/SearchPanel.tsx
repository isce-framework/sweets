import { useState } from "react";
import { useAppState } from "../state";
import type {
  SearchFeature,
  SearchRequest,
  SearchResponse,
  SourceKind,
  TrackSummary,
} from "../types";
import { api } from "../api";

const SOURCES: { id: SourceKind; label: string }[] = [
  { id: "opera-cslc", label: "OPERA CSLC (S1)" },
  { id: "safe", label: "S1 burst SAFE" },
  { id: "nisar-gslc", label: "NISAR GSLC" },
];

function dirLabel(d: TrackSummary["flight_direction"]): string {
  if (d === "ASCENDING") return "ASC";
  if (d === "DESCENDING") return "DESC";
  return "—";
}

export function SearchPanel() {
  const {
    source,
    setSource,
    bbox,
    setBbox,
    searchParams,
    setSearchParams,
    setSearchResults,
    searchResults,
  } = useAppState();
  const { start, end, track, frame } = searchParams;
  const setField = (k: keyof typeof searchParams, v: string) =>
    setSearchParams((p) => ({ ...p, [k]: v }));

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showGranules, setShowGranules] = useState(false);

  function setBboxField(idx: 0 | 1 | 2 | 3, raw: string) {
    const next = (bbox ?? [-118.5, 34.0, -118.0, 34.3]).slice() as [
      number,
      number,
      number,
      number,
    ];
    const n = parseFloat(raw);
    if (Number.isFinite(n)) next[idx] = n;
    setBbox(next);
  }

  async function runSearch(overrideTrack?: number | null) {
    if (!bbox) {
      setError("Draw an AOI on the map (or type a bbox).");
      return;
    }
    setError(null);
    setBusy(true);
    const trackVal =
      overrideTrack !== undefined
        ? overrideTrack
        : track
          ? Number(track)
          : null;
    const req: SearchRequest = {
      source,
      bbox,
      start,
      end,
      track: trackVal,
      frame: frame ? Number(frame) : null,
    };
    try {
      const r = await api.search(req);
      setSearchResults(r);
      // Persist the override so the Config tab picks it up too.
      if (overrideTrack !== undefined) {
        setField("track", overrideTrack == null ? "" : String(overrideTrack));
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h2>Source</h2>
      <select
        value={source}
        onChange={(e) => setSource(e.target.value as SourceKind)}
      >
        {SOURCES.map((s) => (
          <option key={s.id} value={s.id}>
            {s.label}
          </option>
        ))}
      </select>

      <h2>AOI</h2>
      <p className="muted">Draw on map, or type a bbox below.</p>
      <div className="row">
        <label>
          West
          <input
            type="number"
            step="0.01"
            value={bbox?.[0] ?? ""}
            onChange={(e) => setBboxField(0, e.target.value)}
          />
        </label>
        <label>
          South
          <input
            type="number"
            step="0.01"
            value={bbox?.[1] ?? ""}
            onChange={(e) => setBboxField(1, e.target.value)}
          />
        </label>
      </div>
      <div className="row">
        <label>
          East
          <input
            type="number"
            step="0.01"
            value={bbox?.[2] ?? ""}
            onChange={(e) => setBboxField(2, e.target.value)}
          />
        </label>
        <label>
          North
          <input
            type="number"
            step="0.01"
            value={bbox?.[3] ?? ""}
            onChange={(e) => setBboxField(3, e.target.value)}
          />
        </label>
      </div>

      <h2>Dates</h2>
      <div className="row">
        <label>
          Start
          <input
            type="date"
            value={start}
            onChange={(e) => setField("start", e.target.value)}
          />
        </label>
        <label>
          End
          <input
            type="date"
            value={end}
            onChange={(e) => setField("end", e.target.value)}
          />
        </label>
      </div>

      <h2>Filters</h2>
      <div className="row">
        <label>
          Track
          <input
            type="number"
            value={track}
            placeholder={source === "safe" ? "required" : "any"}
            onChange={(e) => setField("track", e.target.value)}
          />
        </label>
        {source === "nisar-gslc" && (
          <label>
            Frame
            <input
              type="number"
              value={frame}
              placeholder="any"
              onChange={(e) => setField("frame", e.target.value)}
            />
          </label>
        )}
      </div>

      <div style={{ marginTop: 16 }}>
        <button onClick={() => runSearch()} disabled={busy}>
          {busy ? "Searching..." : "Search"}
        </button>
        <button
          className="secondary"
          style={{ marginLeft: 6 }}
          onClick={() => {
            setSearchResults(null);
            setField("track", "");
          }}
        >
          Clear
        </button>
      </div>

      {error && <div className="error">{error}</div>}
      {searchResults && (
        <ResultsPanel
          results={searchResults}
          activeTrack={track ? Number(track) : null}
          onPickTrack={(t) => runSearch(t)}
          showGranules={showGranules}
          setShowGranules={setShowGranules}
        />
      )}
    </div>
  );
}

function ResultsPanel({
  results,
  activeTrack,
  onPickTrack,
  showGranules,
  setShowGranules,
}: {
  results: SearchResponse;
  activeTrack: number | null;
  onPickTrack: (t: number | null) => void;
  showGranules: boolean;
  setShowGranules: (b: boolean) => void;
}) {
  const cov = results.coverage;
  return (
    <div style={{ marginTop: 12 }}>
      <div className="muted">
        {results.count} granule{results.count === 1 ? "" : "s"} returned.
      </div>
      {cov?.num_bursts != null && (
        <div className="muted" style={{ marginTop: 4 }}>
          Missing-data filter: {cov.num_bursts} burst
          {cov.num_bursts === 1 ? "" : "s"} &times; {cov.num_dates} date
          {cov.num_dates === 1 ? "" : "s"} (
          <span style={{ color: "var(--ok)" }}>
            {cov.num_features_in_coverage} kept
          </span>
          {cov.num_features_excluded ? (
            <>
              ,{" "}
              <span style={{ color: "var(--err)" }}>
                {cov.num_features_excluded} excluded
              </span>
            </>
          ) : null}
          ).
        </div>
      )}

      {results.tracks.length > 0 && (
        <>
          <h2>Tracks</h2>
          <p className="muted">
            Sweets processes one track per job. Pick one to narrow the search.
          </p>
          <ul className="track-list">
            {results.tracks.map((t) => {
              const isActive = t.track === activeTrack;
              return (
                <li
                  key={`${t.track}-${t.flight_direction}`}
                  className={"track-item" + (isActive ? " selected" : "")}
                  onClick={() => onPickTrack(isActive ? null : t.track)}
                  title={
                    isActive ? "Click to clear filter" : "Filter to this track"
                  }
                >
                  <span className="track-num">T{t.track}</span>
                  <span className={"track-dir " + (t.flight_direction || "")}>
                    {dirLabel(t.flight_direction)}
                  </span>
                  <span className="track-count">{t.count}</span>
                </li>
              );
            })}
          </ul>
        </>
      )}

      <h2>
        Granules{" "}
        <button
          className="secondary"
          style={{ fontSize: 11, padding: "1px 6px", marginLeft: 6 }}
          onClick={() => setShowGranules(!showGranules)}
        >
          {showGranules ? "hide" : "show"}
        </button>
      </h2>

      <div style={{ display: "flex", gap: 6 }}>
        <button
          className="secondary"
          onClick={() => downloadFile("granules.csv", featuresToCsv(results.features))}
        >
          CSV
        </button>
        <button
          className="secondary"
          onClick={() => downloadFile("urls.txt", featuresToUrlList(results.features))}
        >
          URLs (wget -i)
        </button>
      </div>

      {showGranules && (
        <div className="granule-list">
          {results.features.map((f) => (
            <div
              key={f.properties.name}
              className={
                "granule-item" +
                (f.properties.in_coverage === false ? " excluded" : "")
              }
              title={f.properties.url || f.properties.name}
            >
              <span className="granule-name">{f.properties.name}</span>
              <span className="granule-meta">
                {(f.properties.date || "").slice(0, 10)} · T
                {f.properties.track ?? "?"}{" "}
                {f.properties.flight_direction === "ASCENDING"
                  ? "ASC"
                  : f.properties.flight_direction === "DESCENDING"
                    ? "DESC"
                    : ""}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function featuresToCsv(features: SearchFeature[]): string {
  const header = [
    "name",
    "date",
    "track",
    "flight_direction",
    "burst_id",
    "frame",
    "in_coverage",
    "url",
  ];
  const rows = features.map((f) => {
    const p = f.properties;
    return [
      p.name,
      (p.date || "").slice(0, 10),
      p.track ?? "",
      p.flight_direction ?? "",
      p.burst_id ?? "",
      p.frame ?? "",
      p.in_coverage,
      p.url ?? "",
    ]
      .map(csvCell)
      .join(",");
  });
  return [header.join(","), ...rows].join("\n") + "\n";
}

function csvCell(v: unknown): string {
  const s = String(v ?? "");
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function featuresToUrlList(features: SearchFeature[]): string {
  return (
    features
      .map((f) => f.properties.url)
      .filter((u): u is string => !!u)
      .join("\n") + "\n"
  );
}

function downloadFile(name: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
