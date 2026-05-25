import { createContext, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { SearchResponse, SourceKind } from "./types";

export type Tab = "search" | "config" | "jobs";

type Bbox = [number, number, number, number];

// Shared search params. These live above SearchPanel so the Config tab can
// fold them into the discriminated-union `Workflow.search` payload —
// otherwise a job created from the UI would be missing start/end/track and
// fail at Workflow validation in the executor subprocess.
export interface SearchParams {
  start: string;
  end: string;
  track: string;
  frame: string;
}

const DEFAULT_SEARCH: SearchParams = {
  start: "2024-01-01",
  end: "2024-12-31",
  track: "",
  frame: "",
};

interface AppState {
  tab: Tab;
  setTab: (t: Tab) => void;
  bbox: Bbox | null;
  setBbox: (b: Bbox | null) => void;
  source: SourceKind;
  setSource: (s: SourceKind) => void;
  searchParams: SearchParams;
  setSearchParams: (
    updater: SearchParams | ((prev: SearchParams) => SearchParams),
  ) => void;
  searchResults: SearchResponse | null;
  setSearchResults: (r: SearchResponse | null) => void;
  selectedJobId: number | null;
  setSelectedJobId: (n: number | null) => void;
}

const Ctx = createContext<AppState | null>(null);

export function StateProvider({ children }: { children: ReactNode }) {
  const [tab, setTab] = useState<Tab>("search");
  const [bbox, setBbox] = useState<Bbox | null>(null);
  const [source, setSource] = useState<SourceKind>("opera-cslc");
  const [searchParams, setSearchParams] = useState<SearchParams>(DEFAULT_SEARCH);
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(
    null,
  );
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);

  const value = useMemo(
    () => ({
      tab,
      setTab,
      bbox,
      setBbox,
      source,
      setSource,
      searchParams,
      setSearchParams,
      searchResults,
      setSearchResults,
      selectedJobId,
      setSelectedJobId,
    }),
    [tab, bbox, source, searchParams, searchResults, selectedJobId],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAppState(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAppState outside StateProvider");
  return v;
}
