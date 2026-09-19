import type { EndpointId } from "@/lib/endpoints";

const BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export type RunStep = {
  endpoint: EndpointId;
  config: Record<string, unknown>;
  reference?: Blob;
  lut?: Blob;
  outputFormat: "png" | "jpeg";
  includeExif: boolean;
};

export type StepMetadata = {
  contentType: string;
  xOutputFormat: string | null;
  xProcess: string | null;
  xForensicMetadata: string | null;
};

export type StepSuccess = { blob: Blob; meta: StepMetadata };

export type StepEvent =
  | { kind: "running" }
  | { kind: "success"; blob: Blob; meta: StepMetadata }
  | { kind: "error"; message: string }
  | { kind: "cancelled" };

export type RunOutcome =
  | { ok: true; results: StepSuccess[] }
  | { ok: false; status: "error"; index: number; message: string; results: StepSuccess[] }
  | { ok: false; status: "aborted"; index: number; results: StepSuccess[] };

type StructuredDetail = { loc?: unknown; msg?: unknown };

export function normalizeApiError(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((d: StructuredDetail) => {
      const loc = Array.isArray(d?.loc) ? d.loc.join(".") : d?.loc;
      const msg = typeof d?.msg === "string" ? d.msg : String(d?.msg ?? "validation error");
      return loc ? `${String(loc)}: ${msg}` : msg;
    });
    if (parts.length) return parts.join("; ");
  }
  return `Request failed with HTTP ${status}`;
}

function isAbortError(err: unknown): boolean {
  return (err as { name?: string } | null)?.name === "AbortError";
}

function errorMessage(err: unknown): string {
  return err instanceof Error && err.message ? err.message : `Request failed: ${String(err)}`;
}

export async function runSteps(
  source: Blob,
  steps: RunStep[],
  signal: AbortSignal,
  onEvent: (index: number, event: StepEvent) => void,
): Promise<RunOutcome> {
  const results: StepSuccess[] = [];
  let current = source;

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    onEvent(i, { kind: "running" });

    const fd = new FormData();
    fd.append("image", current);
    fd.append("config", JSON.stringify(step.config ?? {}));
    if (step.endpoint !== "forensic-camera") {
      if (step.reference) fd.append("reference_image", step.reference);
      if (step.lut) fd.append("lut_file", step.lut);
      fd.append("output_format", step.outputFormat);
      fd.append("include_exif", step.includeExif ? "true" : "false");
    }

    let res: Response;
    try {
      res = await fetch(`${BASE}/api/v1/${step.endpoint}`, {
        method: "POST",
        body: fd,
        signal,
      });
    } catch (err) {
      if (isAbortError(err)) {
        onEvent(i, { kind: "cancelled" });
        return { ok: false, status: "aborted", index: i, results };
      }
      const message = errorMessage(err);
      onEvent(i, { kind: "error", message });
      return { ok: false, status: "error", index: i, message, results };
    }

    if (!res.ok) {
      let message = `Request failed with HTTP ${res.status}`;
      try {
        message = normalizeApiError(await res.json(), res.status);
      } catch {
        // non-JSON error body: keep the fallback
      }
      onEvent(i, { kind: "error", message });
      return { ok: false, status: "error", index: i, message, results };
    }

    const meta: StepMetadata = {
      contentType: res.headers.get("Content-Type") ?? "",
      xOutputFormat: res.headers.get("X-Output-Format"),
      xProcess: res.headers.get("X-Process"),
      xForensicMetadata: res.headers.get("X-Forensic-Metadata"),
    };

    let blob: Blob;
    try {
      blob = await res.blob();
    } catch (err) {
      if (isAbortError(err)) {
        onEvent(i, { kind: "cancelled" });
        return { ok: false, status: "aborted", index: i, results };
      }
      const message = errorMessage(err);
      onEvent(i, { kind: "error", message });
      return { ok: false, status: "error", index: i, message, results };
    }
    onEvent(i, { kind: "success", blob, meta });
    results.push({ blob, meta });
    current = blob;
  }

  return { ok: true, results };
}
