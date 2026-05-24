import { useState } from "react";
import { useAppState } from "../state";
import type { SourceKind } from "../types";
import { api } from "../api";

const SOURCES: { id: SourceKind; label: string }[] = [
  { id: "opera-cslc", label: "OPERA CSLC (S1)" },
  { id: "safe", label: "S1 burst SAFE" },
  { id: "nisar-gslc", label: "NISAR GSLC" },
];

export function SearchPanel() {
  const {
    source,
    setSource,
    bbox,
    setBbox,
    setSearchResults,
    searchResults,
  } = useAppState();

  const [start, setStart] = useState("2024-01-01");
  const [end, setEnd] = useState("2024-12-31");
  const [track, setTrack] = useState<string>("");
  const [frame, setFrame] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  async function runSearch() {
    if (!bbox) {
      setError("Draw an AOI on the map (or type a bbox).");
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const r = await api.search({
        source,
        bbox,
        start,
        end,
        track: track ? Number(track) : null,
        frame: frame ? Number(frame) : null,
      });
      setSearchResults(r);
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
        <div>
          <label>West</label>
          <input
            type="number"
            step="0.01"
            value={bbox?.[0] ?? ""}
            onChange={(e) => setBboxField(0, e.target.value)}
          />
        </div>
        <div>
          <label>South</label>
          <input
            type="number"
            step="0.01"
            value={bbox?.[1] ?? ""}
            onChange={(e) => setBboxField(1, e.target.value)}
          />
        </div>
      </div>
      <div className="row">
        <div>
          <label>East</label>
          <input
            type="number"
            step="0.01"
            value={bbox?.[2] ?? ""}
            onChange={(e) => setBboxField(2, e.target.value)}
          />
        </div>
        <div>
          <label>North</label>
          <input
            type="number"
            step="0.01"
            value={bbox?.[3] ?? ""}
            onChange={(e) => setBboxField(3, e.target.value)}
          />
        </div>
      </div>

      <h2>Dates</h2>
      <div className="row">
        <div>
          <label>Start</label>
          <input
            type="date"
            value={start}
            onChange={(e) => setStart(e.target.value)}
          />
        </div>
        <div>
          <label>End</label>
          <input
            type="date"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
          />
        </div>
      </div>

      <h2>Filters</h2>
      <div className="row">
        <div>
          <label>Track</label>
          <input
            type="number"
            value={track}
            placeholder="any"
            onChange={(e) => setTrack(e.target.value)}
          />
        </div>
        {source === "nisar-gslc" && (
          <div>
            <label>Frame</label>
            <input
              type="number"
              value={frame}
              placeholder="any"
              onChange={(e) => setFrame(e.target.value)}
            />
          </div>
        )}
      </div>

      <div style={{ marginTop: 16 }}>
        <button onClick={runSearch} disabled={busy}>
          {busy ? "Searching..." : "Search"}
        </button>
        <button
          className="secondary"
          style={{ marginLeft: 6 }}
          onClick={() => setSearchResults(null)}
        >
          Clear
        </button>
      </div>

      {error && <div className="error">{error}</div>}
      {searchResults && (
        <div className="muted" style={{ marginTop: 8 }}>
          <div>{searchResults.count} granule(s) returned.</div>
          {searchResults.coverage?.num_bursts != null && (
            <div style={{ marginTop: 4 }}>
              Missing-data filter: {searchResults.coverage.num_bursts} burst
              {searchResults.coverage.num_bursts === 1 ? "" : "s"} &times;{" "}
              {searchResults.coverage.num_dates} date
              {searchResults.coverage.num_dates === 1 ? "" : "s"} (
              <span style={{ color: "var(--ok)" }}>
                {searchResults.coverage.num_features_in_coverage} kept
              </span>
              {searchResults.coverage.num_features_excluded ? (
                <>
                  ,{" "}
                  <span style={{ color: "var(--err)" }}>
                    {searchResults.coverage.num_features_excluded} excluded
                  </span>
                </>
              ) : null}
              ).
            </div>
          )}
        </div>
      )}
    </div>
  );
}
