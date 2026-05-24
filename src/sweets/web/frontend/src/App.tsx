import { StateProvider, useAppState } from "./state";
import type { Tab } from "./state";
import { MapView } from "./components/MapView";
import { SearchPanel } from "./components/SearchPanel";
import { ConfigPanel } from "./components/ConfigPanel";
import { JobsPanel } from "./components/JobsPanel";

const TABS: { id: Tab; label: string }[] = [
  { id: "search", label: "Search" },
  { id: "config", label: "Config" },
  { id: "jobs", label: "Jobs" },
];

function Sidebar() {
  const { tab, setTab } = useAppState();
  return (
    <aside className="sidebar">
      <h1>sweets</h1>
      <div className="tabs">
        {TABS.map((t) => (
          <div
            key={t.id}
            className={"tab" + (tab === t.id ? " active" : "")}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </div>
        ))}
      </div>
      <div className="panel">
        {tab === "search" && <SearchPanel />}
        {tab === "config" && <ConfigPanel />}
        {tab === "jobs" && <JobsPanel />}
      </div>
    </aside>
  );
}

function Shell() {
  return (
    <div className="app">
      <Sidebar />
      <div className="map">
        <MapView />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <StateProvider>
      <Shell />
    </StateProvider>
  );
}
