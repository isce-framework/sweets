import { createContext, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { SearchResponse, SourceKind } from "./types";

export type Tab = "search" | "config" | "jobs";

type Bbox = [number, number, number, number];

interface AppState {
  tab: Tab;
  setTab: (t: Tab) => void;
  bbox: Bbox | null;
  setBbox: (b: Bbox | null) => void;
  source: SourceKind;
  setSource: (s: SourceKind) => void;
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
      searchResults,
      setSearchResults,
      selectedJobId,
      setSelectedJobId,
    }),
    [tab, bbox, source, searchResults, selectedJobId],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAppState(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAppState outside StateProvider");
  return v;
}
