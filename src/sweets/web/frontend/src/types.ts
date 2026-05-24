export type SourceKind = "safe" | "opera-cslc" | "nisar-gslc";

export interface SearchRequest {
  source: SourceKind;
  bbox: [number, number, number, number];
  start: string;
  end: string;
  track?: number | null;
  frame?: number | null;
}

export interface SearchFeatureProps {
  name: string;
  date: string;
  track?: number | null;
  burst_id?: string | null;
  frame?: number | null;
  url?: string | null;
  in_coverage: boolean;
}

export interface SearchFeature {
  type: "Feature";
  properties: SearchFeatureProps;
  geometry: GeoJSON.Polygon | GeoJSON.MultiPolygon;
}

export interface CoverageSummary {
  num_bursts?: number;
  num_dates?: number;
  num_features_in_coverage?: number;
  num_features_excluded?: number;
  num_options?: number;
}

export interface SearchResponse {
  type: "FeatureCollection";
  features: SearchFeature[];
  count: number;
  source: SourceKind;
  coverage: CoverageSummary;
}

export type JobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface Job {
  id: number;
  name: string;
  status: JobStatus;
  current_step: number;
  config: Record<string, unknown>;
  work_dir: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface ManifestEntry {
  path: string;
  size: number;
  kind: string;
}

export interface Manifest {
  job_id: number;
  work_dir: string | null;
  exists: boolean;
  entries: ManifestEntry[];
}

export interface BowserHandoff {
  job_id: number;
  dolphin_dir: string;
  command: string;
  url: string | null;
  ran: boolean;
  stdout: string;
  stderr: string;
}
