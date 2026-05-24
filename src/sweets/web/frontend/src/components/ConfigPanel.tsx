import { useEffect, useMemo, useState } from "react";
import Form from "@rjsf/core";
import type { RJSFSchema, UiSchema } from "@rjsf/utils";
import validator from "@rjsf/validator-ajv8";
import { api } from "../api";
import { useAppState } from "../state";

// Fields shown in the default "basic" view. Trimmed to the knobs that
// actually matter when configuring a new run from the UI; everything else
// (AOI, source, computed paths, deep dolphin/tropo internals) is either
// surfaced elsewhere or derived from work_dir.
const BASIC_FIELDS = [
  "work_dir",
  "slc_posting",
  "pol_type",
  "gpu_enabled",
  "n_workers",
  "threads_per_worker",
  "overwrite",
];

// Same as BASIC_FIELDS plus the nested dolphin / tropo configs, shown when
// the user flips the "Show advanced" toggle.
const ADVANCED_FIELDS = [...BASIC_FIELDS, "dolphin", "tropo"];

// Field-specific UI hints. RJSF picks reasonable widgets from the schema
// types already; this only customizes the few cases worth tightening.
const UI_SCHEMA: UiSchema = {
  work_dir: { "ui:help": "Root output directory. Created if missing." },
  slc_posting: {
    "ui:help":
      "Geocoded posting (y, x) in meters. Ignored for OPERA/NISAR sources.",
  },
  pol_type: { "ui:help": "Only honored for the SAFE / BurstSearch path." },
  n_workers: { "ui:help": "COMPASS geocoding process pool size." },
  threads_per_worker: { "ui:help": "OMP threads per geocoding worker." },
  overwrite: { "ui:help": "Re-run steps even if outputs already exist." },
  // Nested objects rendered with a less-busy default field template.
  dolphin: { "ui:options": { label: false } },
  tropo: { "ui:options": { label: false } },
};

function pickFields(schema: RJSFSchema, fields: string[]): RJSFSchema {
  const src = schema as RJSFSchema & {
    properties?: Record<string, unknown>;
    required?: string[];
  };
  const props: Record<string, unknown> = {};
  for (const k of fields) {
    if (src.properties && k in src.properties) {
      props[k] = src.properties[k];
    }
  }
  const required = (src.required ?? []).filter((r: string) =>
    fields.includes(r),
  );
  return { ...schema, properties: props, required };
}

// Pydantic emits `tuple[float, float]` as JSON Schema 2020-12, which uses
// `prefixItems` for per-position element schemas. RJSF v5 + ajv8 only know
// how to render `items` (draft-7 / array-of-schemas style), so we rewrite
// the schema in place. Without this, fixed-length tuples like
// `Workflow.slc_posting` render as a raw schema dump with the message
// "Unsupported field schema for field …: Missing items definition".
function rewriteTuplesInPlace(node: unknown): void {
  if (node == null || typeof node !== "object") return;
  if (Array.isArray(node)) {
    for (const child of node) rewriteTuplesInPlace(child);
    return;
  }
  const obj = node as Record<string, unknown>;
  if (Array.isArray(obj.prefixItems) && obj.items == null) {
    obj.items = obj.prefixItems;
    delete obj.prefixItems;
  }
  for (const key of Object.keys(obj)) {
    rewriteTuplesInPlace(obj[key]);
  }
}

export function ConfigPanel() {
  const [fullSchema, setFullSchema] = useState<RJSFSchema | null>(null);
  const [formData, setFormData] = useState<Record<string, unknown>>({});
  const [name, setName] = useState("sweets-job");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { bbox, source, setTab, setSelectedJobId } = useAppState();

  useEffect(() => {
    api
      .schema()
      .then((s) => {
        rewriteTuplesInPlace(s);
        setFullSchema(s as RJSFSchema);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const visibleSchema = useMemo<RJSFSchema | null>(() => {
    if (!fullSchema) return null;
    return pickFields(fullSchema, showAdvanced ? ADVANCED_FIELDS : BASIC_FIELDS);
  }, [fullSchema, showAdvanced]);

  async function submit() {
    if (!bbox) {
      setError("No AOI set. Draw one on the Search tab first.");
      return;
    }
    setError(null);
    setBusy(true);
    try {
      // Merge the map AOI + currently-selected source into the form payload
      // before sending it to the backend. The Workflow validator on the
      // server side will reject anything malformed.
      const config: Record<string, unknown> = {
        ...formData,
        bbox,
        search: {
          ...((formData.search as Record<string, unknown>) ?? {}),
          kind: source,
          bbox,
        },
      };
      const job = await api.createJob(name, config);
      setSelectedJobId(job.id);
      setTab("jobs");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  if (error && !visibleSchema) return <div className="error">{error}</div>;
  if (!visibleSchema) return <p className="muted">Loading schema...</p>;

  return (
    <div className="config-page">
      <div className="config-header">
        <h2 style={{ margin: 0 }}>Configure a new job</h2>
        <p className="muted" style={{ margin: "4px 0 0" }}>
          AOI comes from the Search tab map ({bbox ? bboxText(bbox) : "not set"}
          ). Source: <code>{source}</code>.
        </p>
      </div>

      <div className="config-grid">
        <label className="config-name">
          Job name
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>

        <label className="config-toggle">
          <input
            type="checkbox"
            checked={showAdvanced}
            onChange={(e) => setShowAdvanced(e.target.checked)}
          />
          Show advanced (dolphin / tropo)
        </label>
      </div>

      <Form
        schema={visibleSchema}
        uiSchema={UI_SCHEMA}
        validator={validator}
        formData={formData}
        onChange={(e) => setFormData(e.formData as Record<string, unknown>)}
        onSubmit={submit}
        liveValidate={false}
        showErrorList={false}
      >
        <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
          <button type="submit" disabled={busy}>
            {busy ? "Creating..." : "Create job"}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => setTab("search")}
          >
            Back to search
          </button>
        </div>
      </Form>

      {error && <div className="error">{error}</div>}
    </div>
  );
}

function bboxText(b: [number, number, number, number]): string {
  return b.map((n) => n.toFixed(3)).join(", ");
}
