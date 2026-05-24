import { useEffect, useState } from "react";
import Form from "@rjsf/core";
import type { RJSFSchema } from "@rjsf/utils";
import validator from "@rjsf/validator-ajv8";
import { api } from "../api";
import { useAppState } from "../state";

// Fields the user picks elsewhere (map AOI, search panel) or that aren't
// useful in a form (computed paths). Hide them from the auto-generated form
// so the visible surface area stays manageable.
const HIDDEN_FIELDS = new Set([
  "bbox",
  "wkt",
  "log_dir",
  "gslc_dir",
  "geom_dir",
  "dolphin_dir",
]);

function stripHidden(schema: RJSFSchema): RJSFSchema {
  // Shallow clone — RJSF only reads `properties`, so this is safe.
  const out = { ...schema } as RJSFSchema & {
    properties?: Record<string, unknown>;
  };
  if (out.properties) {
    const props: Record<string, unknown> = { ...out.properties };
    for (const k of HIDDEN_FIELDS) delete props[k];
    out.properties = props;
  }
  return out;
}

export function ConfigPanel() {
  const [schema, setSchema] = useState<RJSFSchema | null>(null);
  const [formData, setFormData] = useState<Record<string, unknown>>({});
  const [name, setName] = useState("sweets-job");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { bbox, source, setTab, setSelectedJobId } = useAppState();

  useEffect(() => {
    api
      .schema()
      .then((s) => setSchema(stripHidden(s as RJSFSchema)))
      .catch((e) => setError(String(e)));
  }, []);

  async function submit() {
    if (!bbox) {
      setError("No AOI set. Draw one on the map first.");
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

  if (error && !schema) return <div className="error">{error}</div>;
  if (!schema) return <p className="muted">Loading schema...</p>;

  return (
    <div>
      <h2>Job name</h2>
      <input value={name} onChange={(e) => setName(e.target.value)} />

      <h2>Configuration</h2>
      <p className="muted">
        Auto-generated from <code>Workflow.model_json_schema()</code>. AOI
        comes from the map; source from the Search tab.
      </p>
      <Form
        schema={schema}
        validator={validator}
        formData={formData}
        onChange={(e) =>
          setFormData(e.formData as Record<string, unknown>)
        }
        onSubmit={submit}
        liveValidate={false}
        showErrorList={false}
      >
        <button type="submit" disabled={busy}>
          {busy ? "Creating..." : "Create job"}
        </button>
      </Form>

      {error && <div className="error">{error}</div>}
    </div>
  );
}
