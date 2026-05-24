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

function TopNav() {
  const { tab, setTab } = useAppState();
  return (
    <header className="topbar">
      <div className="title">sweets</div>
      <nav className="topbar-tabs">
        {TABS.map((t) => (
          <div
            key={t.id}
            className={"tab" + (tab === t.id ? " active" : "")}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </div>
        ))}
      </nav>
    </header>
  );
}

function Body() {
  const { tab } = useAppState();
  if (tab === "config") {
    return (
      <main className="content full">
        <ConfigPanel />
      </main>
    );
  }
  return (
    <main className="content split">
      <aside className="sidebar">
        <div className="panel">
          {tab === "search" && <SearchPanel />}
          {tab === "jobs" && <JobsPanel />}
        </div>
      </aside>
      <div className="map">
        <MapView />
      </div>
    </main>
  );
}

function Shell() {
  return (
    <div className="app">
      <TopNav />
      <Body />
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
