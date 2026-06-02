import { useEffect, useMemo, useState } from "react";
import Form from "@rjsf/core";
import type { RJSFSchema, UiSchema } from "@rjsf/utils";
import { getDefaultFormState } from "@rjsf/utils";
import validator from "@rjsf/validator-ajv8";
import { api } from "../api";
import { useAppState } from "../state";
import type { SearchParams } from "../state";
import type { SourceKind } from "../types";

type WorkflowType = "displacement" | "interferogram";

// Fields shown in the default "basic" view for the displacement workflow.
const DISPLACEMENT_BASIC_FIELDS = [
  "work_dir",
  "slc_posting",
  "pol_type",
  "gpu_enabled",
  "n_workers",
  "threads_per_worker",
  "overwrite",
];
const DISPLACEMENT_ADVANCED_FIELDS = [
  ...DISPLACEMENT_BASIC_FIELDS,
  "dolphin",
  "tropo",
];

// Fields for the interferogram workflow.
const IFG_BASIC_FIELDS = [
  "work_dir",
  "slc_posting",
  "pol_type",
  "overwrite",
  "network",
  "crossmul",
  "unwrap",
];
const IFG_ADVANCED_FIELDS = [
  ...IFG_BASIC_FIELDS,
  "gpu_enabled",
  "n_workers",
  "threads_per_worker",
];

// Within the IFG unwrap sub-schema, hide the advanced SNAPHU tile knobs
// (snaphu_ntiles, snaphu_tile_overlap) from the basic view — they render
// poorly because Union[tuple,Literal] can't be represented as a simple widget.
const IFG_UI_SCHEMA_UNWRAP_ADVANCED: UiSchema = {
  snaphu_ntiles: { "ui:widget": "hidden" },
  snaphu_tile_overlap: { "ui:widget": "hidden" },
};

const DISPLACEMENT_UI_SCHEMA: UiSchema = {
  work_dir: { "ui:help": "Root output directory. Created if missing." },
  slc_posting: {
    "ui:help":
      "Geocoded posting (y, x) in meters. Ignored for OPERA/NISAR sources.",
  },
  pol_type: { "ui:help": "Only honored for the SAFE / BurstSearch path." },
  n_workers: { "ui:help": "COMPASS geocoding process pool size." },
  threads_per_worker: { "ui:help": "OMP threads per geocoding worker." },
  overwrite: { "ui:help": "Re-run steps even if outputs already exist." },
  dolphin: { "ui:options": { label: false } },
  tropo: { "ui:options": { label: false } },
};

const IFG_UI_SCHEMA: UiSchema = {
  work_dir: { "ui:help": "Root output directory. Created if missing." },
  slc_posting: {
    "ui:help":
      "Geocoded posting (y, x) in meters. Ignored for OPERA/NISAR sources.",
  },
  pol_type: { "ui:help": "Only honored for the SAFE / BurstSearch path." },
  overwrite: { "ui:help": "Re-run steps even if outputs already exist." },
  // No label:false here — RJSF renders the legend title above the fieldset
  // by default, which is correct. ui:help would appear below the block.
  unwrap: IFG_UI_SCHEMA_UNWRAP_ADVANCED,
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

// Rewrite Pydantic JSON Schema 2020-12 constructs that RJSF v5 can't render:
//
// 1. prefixItems (fixed-length tuples) → items  (RJSF only knows draft-7 style)
// 2. anyOf: [T, {type:"null"}]  → T             (Optional[X] renders as a
//    schema-picker dropdown instead of a plain X widget without this)
function rewriteSchemaInPlace(node: unknown): void {
  if (node == null || typeof node !== "object") return;
  if (Array.isArray(node)) {
    for (const child of node) rewriteSchemaInPlace(child);
    return;
  }
  const obj = node as Record<string, unknown>;

  // 1. tuple rewrite
  if (Array.isArray(obj.prefixItems) && obj.items == null) {
    obj.items = obj.prefixItems;
    delete obj.prefixItems;
  }

  // 2. $ref + siblings → allOf:[{$ref}] + siblings
  // In draft-07, siblings of $ref are ignored; RJSF sees an object with no
  // properties and renders the "add new key" button. Wrapping in allOf fixes it.
  if (typeof obj.$ref === "string" && Object.keys(obj).length > 1) {
    const ref = obj.$ref;
    delete obj.$ref;
    obj.allOf = [{ $ref: ref }];
  }

  // 3. nullable union collapse: anyOf: [T, null] → T
  if (Array.isArray(obj.anyOf)) {
    const variants = obj.anyOf as unknown[];
    const nonNull = variants.filter(
      (s) =>
        !(
          typeof s === "object" &&
          s !== null &&
          (s as Record<string, unknown>).type === "null"
        ),
    );
    if (nonNull.length < variants.length && nonNull.length === 1) {
      const inner = nonNull[0] as Record<string, unknown>;
      for (const [k, v] of Object.entries(inner)) {
        if (!(k in obj)) obj[k] = v;
      }
      delete obj.anyOf;
    }
  }

  for (const key of Object.keys(obj)) {
    rewriteSchemaInPlace(obj[key]);
  }
}

// Build the discriminated-union `Workflow.search` dict from the shared
// Search-tab state.
function buildSearch(
  source: SourceKind,
  bbox: [number, number, number, number],
  sp: SearchParams,
  burstIds: string[] | null,
): { search: Record<string, unknown> } | { error: string } {
  const search: Record<string, unknown> = { kind: source, bbox };
  if (!sp.start || !sp.end) {
    return { error: "Set Start and End dates on the Search tab." };
  }
  search.start = sp.start;
  search.end = sp.end;
  if (source === "safe") {
    if (!sp.track) {
      return { error: "Track is required for the S1 burst SAFE source." };
    }
    search.track = Number(sp.track);
  } else if (source === "opera-cslc") {
    if (sp.track) search.track = Number(sp.track);
    if (burstIds?.length) search.burst_ids = burstIds;
  } else if (source === "nisar-gslc") {
    if (sp.track) search.track = Number(sp.track);
    if (sp.frame) search.frame = Number(sp.frame);
  }
  return { search };
}

export function ConfigPanel() {
  const [workflowType, setWorkflowType] = useState<WorkflowType>("displacement");
  const [schemas, setSchemas] = useState<
    Partial<Record<WorkflowType, RJSFSchema>>
  >({});
  const [formData, setFormData] = useState<Record<string, unknown>>({});
  const [name, setName] = useState("sweets-job");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { bbox, source, searchParams, selectedBurstIds, setTab, setSelectedJobId } =
    useAppState();

  // Compute defaults only for the fields in `fields`, using the full schema
  // as rootSchema so $refs resolve correctly. This avoids extra keys
  // (bbox, search, orbit_dir…) bleeding into formData and triggering RJSF's
  // additionalProperties widget for fields not in the visible schema.
  function applyDefaults(
    fullSchema: RJSFSchema,
    fields: string[],
  ): Record<string, unknown> {
    const visible = pickFields(fullSchema, fields);
    return (
      (getDefaultFormState(
        validator,
        visible,
        {},
        fullSchema,
      ) as Record<string, unknown>) ?? {}
    );
  }

  // Fetch both schemas once on mount.
  useEffect(() => {
    api
      .schema()
      .then((s) => {
        rewriteSchemaInPlace(s);
        const schema = s as RJSFSchema;
        setSchemas((prev) => ({ ...prev, displacement: schema }));
        if (workflowType === "displacement")
          setFormData(applyDefaults(schema, DISPLACEMENT_BASIC_FIELDS));
      })
      .catch((e) => setError(String(e)));

    api
      .ifgSchema()
      .then((s) => {
        rewriteSchemaInPlace(s);
        const schema = s as RJSFSchema;
        setSchemas((prev) => ({ ...prev, interferogram: schema }));
        if (workflowType === "interferogram")
          setFormData(applyDefaults(schema, IFG_BASIC_FIELDS));
      })
      .catch((e) => setError(String(e)));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Reset form data (with defaults) when workflow type changes.
  function switchWorkflowType(t: WorkflowType) {
    setWorkflowType(t);
    const s = schemas[t];
    const fields =
      t === "interferogram" ? IFG_BASIC_FIELDS : DISPLACEMENT_BASIC_FIELDS;
    setFormData(s ? applyDefaults(s, fields) : {});
    setShowAdvanced(false);
  }

  const fullSchema = schemas[workflowType] ?? null;

  const basicFields =
    workflowType === "interferogram" ? IFG_BASIC_FIELDS : DISPLACEMENT_BASIC_FIELDS;
  const advancedFields =
    workflowType === "interferogram"
      ? IFG_ADVANCED_FIELDS
      : DISPLACEMENT_ADVANCED_FIELDS;
  const uiSchema =
    workflowType === "interferogram" ? IFG_UI_SCHEMA : DISPLACEMENT_UI_SCHEMA;
  const advancedLabel =
    workflowType === "interferogram"
      ? "Show advanced (workers)"
      : "Show advanced (dolphin / tropo)";

  const visibleSchema = useMemo<RJSFSchema | null>(() => {
    if (!fullSchema) return null;
    return pickFields(fullSchema, showAdvanced ? advancedFields : basicFields);
  }, [fullSchema, showAdvanced, basicFields, advancedFields]);

  async function submit(opts: { start: boolean }) {
    if (!bbox) {
      setError("No AOI set. Draw one on the Search tab first.");
      return;
    }
    const built = buildSearch(source, bbox, searchParams, selectedBurstIds);
    if ("error" in built) {
      setError(built.error);
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const config: Record<string, unknown> = {
        ...formData,
        bbox,
        ...built,
      };
      const job = await api.createJob(name, config);
      if (opts.start) {
        await api.startJob(job.id);
      }
      setSelectedJobId(job.id);
      setTab("jobs");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  if (error && !fullSchema) return <div className="error">{error}</div>;
  if (!visibleSchema) return <p className="muted">Loading schema...</p>;

  const sp = searchParams;
  const trackLabel = sp.track
    ? `T${sp.track}`
    : source === "safe"
      ? "missing!"
      : "any";

  return (
    <div className="config-page">
      <div className="config-header">
        <h2 style={{ margin: 0 }}>Configure a new job</h2>
        <p className="muted" style={{ margin: "4px 0 0" }}>
          From the Search tab: source <code>{source}</code>, AOI{" "}
          <code>{bbox ? bboxText(bbox) : "not set"}</code>, dates{" "}
          <code>
            {sp.start || "?"} → {sp.end || "?"}
          </code>
          , track <code>{trackLabel}</code>
          {source === "nisar-gslc" && (
            <>
              , frame <code>{sp.frame || "any"}</code>
            </>
          )}
          .
        </p>
      </div>

      <div className="config-grid">
        <label className="config-name">
          Job name
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>

        <div className="config-workflow-type">
          <span className="config-workflow-label">Workflow</span>
          <div className="workflow-toggle">
            <button
              type="button"
              className={workflowType === "displacement" ? "active" : "secondary"}
              onClick={() => switchWorkflowType("displacement")}
            >
              Displacement
            </button>
            <button
              type="button"
              className={workflowType === "interferogram" ? "active" : "secondary"}
              onClick={() => switchWorkflowType("interferogram")}
            >
              Interferogram
            </button>
          </div>
        </div>

        <label className="config-toggle">
          <input
            type="checkbox"
            checked={showAdvanced}
            onChange={(e) => {
              const next = e.target.checked;
              setShowAdvanced(next);
              if (fullSchema) {
                const fields = next ? advancedFields : basicFields;
                setFormData((prev) => ({
                  ...applyDefaults(fullSchema, fields),
                  ...prev,
                }));
              }
            }}
          />
          {advancedLabel}
        </label>
      </div>

      <Form
        schema={visibleSchema}
        uiSchema={uiSchema}
        validator={validator}
        formData={formData}
        onChange={(e) => setFormData(e.formData as Record<string, unknown>)}
        onSubmit={() => submit({ start: false })}
        liveValidate={false}
        showErrorList={false}
      >
        <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
          <button type="submit" disabled={busy}>
            {busy ? "Working..." : "Create job"}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => submit({ start: true })}
          >
            Create &amp; start
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
