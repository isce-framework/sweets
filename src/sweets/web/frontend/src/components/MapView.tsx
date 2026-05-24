import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet-draw";
import { useAppState } from "../state";

// Fix Leaflet's default marker icons under bundlers — irrelevant here since
// we don't use point markers, but RJSF/Leaflet plugins sometimes choke
// without this. Keeping a single CSS-only basemap from CARTO's free tier.
const BASEMAP =
  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
const ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>, &copy; <a href="https://carto.com/attributions">CARTO</a>';

const COVERED_STYLE: L.PathOptions = {
  color: "#5cd97a",
  weight: 1,
  fillOpacity: 0.12,
  fillColor: "#5cd97a",
};

const EXCLUDED_STYLE: L.PathOptions = {
  color: "#ff6b6b",
  weight: 1,
  fillOpacity: 0.08,
  fillColor: "#ff6b6b",
  dashArray: "3 3",
};

const AOI_STYLE: L.PathOptions = {
  color: "#ffb454",
  weight: 2,
  fill: false,
  dashArray: "6 4",
};

export function MapView() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const aoiLayerRef = useRef<L.Rectangle | null>(null);
  const searchLayerRef = useRef<L.GeoJSON | null>(null);
  const drawerRef = useRef<unknown>(null);

  const { bbox, setBbox, searchResults } = useAppState();

  // Initialize the map once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, {
      center: [34.05, -118.25],
      zoom: 5,
      zoomControl: true,
    });
    L.tileLayer(BASEMAP, { attribution: ATTRIBUTION, maxZoom: 18 }).addTo(map);

    const searchLayer = L.geoJSON(undefined, {
      style: (feat) => {
        const p = (feat?.properties ?? {}) as { in_coverage?: boolean };
        return p.in_coverage === false ? EXCLUDED_STYLE : COVERED_STYLE;
      },
      onEachFeature: (feat, layer) => {
        const p = (feat.properties ?? {}) as Record<string, unknown>;
        const lines = [
          `<b>${p.name ?? "?"}</b>`,
          p.date ? `date: ${String(p.date).slice(0, 10)}` : null,
          p.track != null ? `track: ${p.track}` : null,
          p.frame != null ? `frame: ${p.frame}` : null,
          p.burst_id ? `burst: ${p.burst_id}` : null,
          p.in_coverage === false
            ? "<i>excluded by missing-data filter</i>"
            : null,
        ].filter(Boolean);
        layer.bindPopup(lines.join("<br/>"));
      },
    }).addTo(map);

    // Leaflet.draw via dynamic any-cast — the typing is patchy under v1+v4
    // mixes and we only use rectangle.enable() / disable().
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const Ldraw = (L as any).Draw;
    if (Ldraw?.Rectangle) {
      drawerRef.current = new Ldraw.Rectangle(map, {
        shapeOptions: AOI_STYLE,
        showArea: false,
      });
      map.on((L as unknown as { Draw: { Event: { CREATED: string } } }).Draw.Event.CREATED, (e: unknown) => {
        const ev = e as { layer: L.Rectangle };
        if (aoiLayerRef.current) {
          map.removeLayer(aoiLayerRef.current);
        }
        aoiLayerRef.current = ev.layer;
        ev.layer.setStyle(AOI_STYLE);
        ev.layer.addTo(map);
        const b = ev.layer.getBounds();
        setBbox([
          b.getWest(),
          b.getSouth(),
          b.getEast(),
          b.getNorth(),
        ] as [number, number, number, number]);
      });
    }

    mapRef.current = map;
    searchLayerRef.current = searchLayer;
  }, [setBbox]);

  // Sync AOI rectangle from state (e.g. typed-in bbox).
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!bbox) {
      if (aoiLayerRef.current) {
        map.removeLayer(aoiLayerRef.current);
        aoiLayerRef.current = null;
      }
      return;
    }
    const [w, s, e, n] = bbox;
    if (aoiLayerRef.current) {
      aoiLayerRef.current.setBounds([
        [s, w],
        [n, e],
      ]);
    } else {
      aoiLayerRef.current = L.rectangle(
        [
          [s, w],
          [n, e],
        ],
        AOI_STYLE,
      ).addTo(map);
    }
  }, [bbox]);

  // Replace the search-results overlay whenever results change.
  useEffect(() => {
    const layer = searchLayerRef.current;
    const map = mapRef.current;
    if (!layer || !map) return;
    layer.clearLayers();
    if (searchResults && searchResults.features.length > 0) {
      layer.addData(searchResults as unknown as GeoJSON.FeatureCollection);
      try {
        map.fitBounds(layer.getBounds(), { padding: [40, 40] });
      } catch {
        // empty bounds — ignore
      }
    }
  }, [searchResults]);

  function startDraw() {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (drawerRef.current as any)?.enable?.();
  }

  return (
    <>
      <div ref={containerRef} style={{ height: "100%", width: "100%" }} />
      <div
        style={{
          position: "absolute",
          top: 10,
          right: 10,
          zIndex: 1000,
          display: "flex",
          gap: 6,
        }}
      >
        <button onClick={startDraw}>Draw AOI</button>
        <button
          className="secondary"
          onClick={() => setBbox(null)}
          disabled={!bbox}
        >
          Clear
        </button>
      </div>
    </>
  );
}
