"use client";

import { useEffect, useRef, useState, type ChangeEvent } from "react";
import StepList, { type StepStatus } from "../components/StepList";
import StepConfig, {
  type ConfigurableStep,
  type ConfigValue,
} from "../components/StepConfig";
import Preview from "../components/Preview";
import { getEndpoint, type EndpointId } from "../lib/endpoints";
import { runSteps, type RunOutcome, type RunStep, type StepMetadata } from "../lib/run";

const BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
const MAX_FILE_BYTES = 67108864;
const DATETIME_RE = /^\d{4}(-|:)\d{2}\1\d{2} \d{2}:\d{2}:\d{2}$/;

type Step = ConfigurableStep & {
  status: StepStatus;
  error?: string;
  blobUrl?: string;
  metadata?: StepMetadata;
};

type StatusMessage = { kind: "error" | "status"; text: string };
type Health = "checking" | "ok" | "down";

const PANEL = "min-w-0 rounded-md border border-border bg-surface p-5 shadow-sm";
const HEADING = "font-heading text-[15px] font-semibold text-text";

export default function Workspace() {
  const [file, setFile] = useState<File | undefined>(undefined);
  const [sourceUrl, setSourceUrl] = useState<string | undefined>(undefined);
  const [uploadError, setUploadError] = useState<string | undefined>(undefined);
  const [steps, setSteps] = useState<Step[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>(undefined);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<StatusMessage | undefined>(undefined);
  const [health, setHealth] = useState<Health>("checking");
  const [dark, setDark] = useState(false);

  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  };

  const abortRef = useRef<AbortController | undefined>(undefined);
  const settingsDialogRef = useRef<HTMLDialogElement>(null);
  const urlsRef = useRef<{ source?: string; steps: Map<string, string> }>({
    steps: new Map(),
  });

  const setStepUrl = (id: string, url: string | undefined) => {
    const prev = urlsRef.current.steps.get(id);
    if (prev && prev !== url) URL.revokeObjectURL(prev);
    if (url) urlsRef.current.steps.set(id, url);
    else urlsRef.current.steps.delete(id);
  };

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (urlsRef.current.source) URL.revokeObjectURL(urlsRef.current.source);
      for (const url of urlsRef.current.steps.values()) URL.revokeObjectURL(url);
      urlsRef.current.steps.clear();
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const check = async () => {
      try {
        const res = await fetch(`${BASE}/health`, { signal: controller.signal });
        setHealth(res.ok ? "ok" : "down");
      } catch {
        if (!controller.signal.aborted) setHealth("down");
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => {
      controller.abort();
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!running) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [running]);

  // Shared invalidation: edited step and everything downstream goes stale,
  // errors clear, and their output object URLs are revoked.
  const invalidateFrom = (list: Step[], from: number): Step[] => {
    for (let i = from; i < list.length; i++) setStepUrl(list[i].id, undefined);
    return list.map((s, i) => {
      if (i < from) return s;
      const nextStatus: StepStatus =
        s.status === "done" || s.status === "stale"
          ? "stale"
          : s.status === "error"
            ? "idle"
            : s.status;
      return {
        ...s,
        status: nextStatus,
        error: undefined,
        blobUrl: undefined,
        metadata: undefined,
      };
    });
  };

  const selectedStep = steps.find((s) => s.id === selectedId);

  const selectedIndexOf = () => steps.findIndex((s) => s.id === selectedId);

  const handleUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    if (f.size > MAX_FILE_BYTES) {
      setUploadError("image exceeds 64 MiB upload cap");
      return;
    }
    setUploadError(undefined);
    if (urlsRef.current.source) URL.revokeObjectURL(urlsRef.current.source);
    const url = URL.createObjectURL(f);
    urlsRef.current.source = url;
    setFile(f);
    setSourceUrl(url);
    setSteps(invalidateFrom(steps, 0));
  };

  const handleAdd = (endpoint: EndpointId) => {
    const step: Step = {
      id: crypto.randomUUID(),
      endpoint,
      config: {},
      outputFormat: "png",
      includeExif: false,
      status: "idle",
    };
    setSteps([...steps, step]);
    setSelectedId(step.id);
  };

  const handleMove = (index: number, direction: -1 | 1) => {
    const j = index + direction;
    if (j < 0 || j >= steps.length) return;
    const next = invalidateFrom(steps, 0);
    [next[index], next[j]] = [next[j], next[index]];
    setSteps(next);
  };

  const handleRemove = (index: number) => {
    const removed = steps[index];
    const remaining = invalidateFrom(steps, 0).filter((s) => s.id !== removed.id);
    setSteps(remaining);
    if (selectedId === removed.id) {
      const neighbour = steps[index + 1] ?? steps[index - 1];
      setSelectedId(neighbour?.id);
    }
  };

  const handleConfigChange = (key: string, value: ConfigValue) => {
    const index = selectedIndexOf();
    if (index === -1) return;
    const next = invalidateFrom(steps, index);
    const config = { ...next[index].config };
    if (value === undefined) delete config[key];
    else config[key] = value;
    next[index] = { ...next[index], config };
    setSteps(next);
  };

  const handleStepFileChange = (which: "reference" | "lut", f: File | undefined) => {
    const index = selectedIndexOf();
    if (index === -1) return;
    const next = invalidateFrom(steps, index);
    next[index] = which === "reference" ? { ...next[index], reference: f } : { ...next[index], lut: f };
    setSteps(next);
  };

  const handleOutputFormatChange = (format: "png" | "jpeg") => {
    const index = selectedIndexOf();
    if (index === -1) return;
    const next = invalidateFrom(steps, index);
    next[index] = { ...next[index], outputFormat: format };
    setSteps(next);
  };

  const handleIncludeExifChange = (include: boolean) => {
    const index = selectedIndexOf();
    if (index === -1) return;
    const next = invalidateFrom(steps, index);
    next[index] = { ...next[index], includeExif: include };
    setSteps(next);
  };

  const validatePipeline = (): { message: string; stepIndex?: number } | null => {
    if (!file) return { message: "Upload a source image before running." };
    if (steps.length === 0) return { message: "Add at least one step before running." };
    for (let i = 0; i < steps.length; i++) {
      const s = steps[i];
      const def = getEndpoint(s.endpoint);
      const name = def ? def.label : s.endpoint;
      if (s.lut && s.lut.size > MAX_FILE_BYTES)
        return { message: `Step ${i + 1} (${name}): LUT file exceeds 64 MiB upload cap.`, stepIndex: i };
      if (def?.lutUpload && !s.lut)
        return { message: `Step ${i + 1} (${name}): LUT file is required.`, stepIndex: i };
      if (s.reference && s.reference.size > MAX_FILE_BYTES)
        return {
          message: `Step ${i + 1} (${name}): reference image exceeds 64 MiB upload cap.`,
          stepIndex: i,
        };
      const dt = s.config.datetime_original;
      if (s.endpoint === "forensic-camera" && typeof dt === "string" && !DATETIME_RE.test(dt))
        return {
          message: `Step ${i + 1} (${name}): datetime_original must match YYYY-MM-DD HH:MM:SS or YYYY:MM:DD HH:MM:SS.`,
          stepIndex: i,
        };
    }
    return null;
  };

  const handleRun = async () => {
    if (running) return;
    const problem = validatePipeline();
    if (problem) {
      const badIndex = problem.stepIndex;
      if (badIndex !== undefined)
        setSteps((prev) =>
          prev.map((s, i) => (i === badIndex ? { ...s, status: "error", error: problem.message } : s)),
        );
      setStatus({ kind: "error", text: problem.message });
      return;
    }
    if (!file) return;

    const snapshot = steps;
    snapshot.forEach((s) => setStepUrl(s.id, undefined));
    setSteps(
      snapshot.map((s) => ({
        ...s,
        status: "idle" as const,
        error: undefined,
        blobUrl: undefined,
        metadata: undefined,
      })),
    );
    setStatus({ kind: "status", text: "Running pipeline…" });

    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);

    const requests: RunStep[] = snapshot.map((s) => ({
      endpoint: s.endpoint,
      config: s.config,
      reference: s.reference,
      lut: s.lut,
      outputFormat: s.outputFormat,
      includeExif: s.includeExif,
    }));

    let outcome: RunOutcome | undefined;
    let lastRunningIndex = -1;
    try {
      outcome = await runSteps(file, requests, controller.signal, (index, event) => {
        if (event.kind === "running") lastRunningIndex = index;
        if (event.kind === "error") setStatus({ kind: "error", text: event.message });
        const id = snapshot[index]?.id;
        if (!id) return;
        if (event.kind === "success") {
          const url = URL.createObjectURL(event.blob);
          setStepUrl(id, url);
          setSteps((prev) =>
            prev.map((s, i) =>
              i === index
                ? { ...s, status: "done", error: undefined, blobUrl: url, metadata: event.meta }
                : s,
            ),
          );
          return;
        }
        setSteps((prev) =>
          prev.map((s, i) => {
            if (i !== index) return s;
            if (event.kind === "running") return { ...s, status: "running", error: undefined };
            if (event.kind === "error") return { ...s, status: "error", error: event.message };
            return { ...s, status: "cancelled" };
          }),
        );
      });
    } catch (err) {
      const message =
        err instanceof Error && err.message ? err.message : `Unexpected failure: ${String(err)}`;
      setStatus({ kind: "error", text: message });
      if (lastRunningIndex >= 0)
        setSteps((prev) =>
          prev.map((s, i) =>
            i === lastRunningIndex ? { ...s, status: "error", error: message } : s,
          ),
        );
    } finally {
      abortRef.current = undefined;
      setRunning(false);
    }

    if (!outcome) return;
    if (outcome.ok) setStatus({ kind: "status", text: "Pipeline completed." });
    else if (outcome.status === "aborted") setStatus({ kind: "status", text: "Run cancelled." });
  };

  const handleCancel = () => abortRef.current?.abort();

  let lastDoneIndex = -1;
  for (let i = steps.length - 1; i >= 0; i--) {
    if (steps[i].status === "done" && steps[i].blobUrl) {
      lastDoneIndex = i;
      break;
    }
  }
  const previewStep = lastDoneIndex >= 0 ? steps[lastDoneIndex] : undefined;
  const previewIsFinalizer = previewStep?.endpoint === "forensic-camera";
  const metadataStatus = previewIsFinalizer
    ? previewStep?.metadata?.xForensicMetadata ?? null
    : null;
  const previewLabel = previewStep
    ? `Output of step ${lastDoneIndex + 1} · ${
        previewStep.metadata?.xProcess ||
        getEndpoint(previewStep.endpoint)?.label ||
        previewStep.endpoint
      }`
    : null;

  const handleDownload = () => {
    if (!previewStep?.blobUrl || !file) return;
    const meta = previewStep.metadata;
    let ext = "png";
    const fmt = meta?.xOutputFormat?.toLowerCase();
    if (fmt === "jpg" || fmt === "jpeg") ext = "jpg";
    else if (fmt === "png") ext = "png";
    else {
      const ct = meta?.contentType?.toLowerCase() ?? "";
      if (ct.includes("jpeg")) ext = "jpg";
      else if (ct.includes("png")) ext = "png";
    }
    const base = file.name.replace(/\.[^.]+$/, "") || "image";
    const anchor = document.createElement("a");
    anchor.href = previewStep.blobUrl;
    anchor.download = `${base}-processed.${ext}`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  };

  const healthLabel =
    health === "ok" ? "API online" : health === "down" ? "API unreachable" : "Checking API…";
  const healthDot =
    health === "ok" ? "bg-success" : health === "down" ? "bg-error" : "bg-border";

  return (
    <div className="min-h-screen">
      <header className="border-b border-border bg-surface">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-4 py-3 md:px-6">
          <h1 className="min-w-0 truncate font-heading text-xl font-semibold tracking-tight">
            Image Detection Bypass Utility
          </h1>
          <div className="flex shrink-0 items-center gap-3">
            <div className="flex items-center gap-2" role="status" aria-live="polite">
              <span aria-hidden className={`h-2 w-2 rounded-full ${healthDot}`} />
              <span className="text-sm text-text-muted">{healthLabel}</span>
            </div>
            <button
              type="button"
              onClick={toggleDark}
              aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
              aria-pressed={dark}
              className="rounded-md border border-border bg-surface p-1.5 text-text-muted hover:bg-border/30 aria-pressed:text-text"
            >
              {dark ? (
                <svg aria-hidden width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4m11.4-11.4 1.4-1.4" />
                </svg>
              ) : (
                <svg aria-hidden width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </header>

      {health === "down" && (
        <div
          role="alert"
          className="border-b border-[var(--color-warning)] bg-surface px-4 py-2 text-sm text-[var(--color-warning)] md:px-6"
        >
          The API is unreachable. Runs are disabled until it is back online.
        </div>
      )}

      <main className="mx-auto w-full max-w-7xl px-4 py-6 md:px-6">
        <div className="grid grid-cols-1 items-start gap-5 lg:grid-cols-12 lg:gap-6">
          <div className="flex min-w-0 flex-col gap-5 lg:col-span-5">
            <section aria-labelledby="upload-heading" className={PANEL}>
              <h3 id="upload-heading" className={HEADING}>
                Source image
              </h3>
              <label
                htmlFor="source-upload"
                className="mt-3 flex cursor-pointer flex-col items-center justify-center gap-1 rounded-sm border border-dashed border-border px-4 py-6 text-center hover:border-violet-400 hover:bg-violet-50 dark:hover:bg-violet-900/20"
              >
                <span className="text-sm font-medium text-text">
                  {file ? "Replace image" : "Click to upload"}
                </span>
                <span className="min-w-0 truncate text-xs text-text-muted">
                  {file ? file.name : "PNG, JPEG, WebP · up to 64 MiB"}
                </span>
              </label>
              <input
                id="source-upload"
                type="file"
                accept="image/*"
                disabled={running}
                aria-invalid={uploadError ? true : undefined}
                aria-describedby={uploadError ? "source-upload-error" : undefined}
                onChange={handleUpload}
                className="sr-only"
              />
              {uploadError && (
                <p id="source-upload-error" role="alert" className="mt-2 text-xs text-error break-words">
                  {uploadError}
                </p>
              )}
            </section>

            <section aria-labelledby="list-heading" className={PANEL}>
              <h3 id="list-heading" className={HEADING}>
                Process list
              </h3>
              <div className="mt-3">
                <StepList
                  steps={steps}
                  selectedId={selectedId}
                  disabled={running}
                  onAdd={handleAdd}
                  onSelect={(id) => {
                    setSelectedId(id);
                    settingsDialogRef.current?.showModal();
                  }}
                  onMove={handleMove}
                  onRemove={handleRemove}
                />
              </div>
            </section>
          </div>

          <div className="flex min-w-0 flex-col gap-5 lg:col-span-7">
            <Preview
              beforeUrl={sourceUrl ?? null}
              afterUrl={previewStep?.blobUrl ?? null}
              metadataStatus={metadataStatus}
              finalizer={previewIsFinalizer}
              label={previewLabel}
              disabled={running}
              onDownload={handleDownload}
            />

            <section
              aria-labelledby="run-heading"
              className="flex flex-wrap items-center gap-3 rounded-md border border-border bg-surface p-4 shadow-sm"
            >
              <h3 id="run-heading" className="sr-only">
                Run
              </h3>
              <button
                type="button"
                onClick={handleRun}
                disabled={running || health !== "ok"}
                className="rounded-md bg-violet-500 px-5 py-2 text-sm font-medium text-white hover:bg-violet-600 active:bg-violet-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {running ? "Running…" : "Run pipeline"}
              </button>
              {running && (
                <button
                  type="button"
                  onClick={handleCancel}
                  className="rounded-md border border-violet-500 px-5 py-2 text-sm font-medium text-violet-700 hover:bg-violet-50 dark:text-violet-300 dark:hover:bg-violet-900/30"
                >
                  Cancel
                </button>
              )}
              {status?.kind === "error" ? (
                <p role="alert" className="min-w-0 flex-1 text-sm text-error break-words">
                  {status.text}
                </p>
              ) : status ? (
                <p
                  role="status"
                  aria-live="polite"
                  className="min-w-0 flex-1 text-sm text-text-muted break-words"
                >
                  {status.text}
                </p>
              ) : null}
            </section>
          </div>
        </div>

        <dialog
          ref={settingsDialogRef}
          aria-label="Step configuration"
          onClick={(e) => {
            if (e.target === e.currentTarget) settingsDialogRef.current?.close();
          }}
          className={`${PANEL} fixed inset-0 m-auto max-h-[85vh] w-[min(90vw,32rem)] overflow-y-auto shadow-md backdrop:bg-black/50`}
        >
          <StepConfig
            step={selectedStep}
            disabled={running}
            onConfigChange={handleConfigChange}
            onReferenceChange={(f) => handleStepFileChange("reference", f)}
            onLutChange={(f) => handleStepFileChange("lut", f)}
            onOutputFormatChange={handleOutputFormatChange}
            onIncludeExifChange={handleIncludeExifChange}
          />
          <div className="mt-4 flex justify-end">
            <button
              type="button"
              onClick={() => settingsDialogRef.current?.close()}
              className="rounded-md border border-border bg-surface px-4 py-2 text-sm hover:bg-border/30"
            >
              Close
            </button>
          </div>
        </dialog>
      </main>
    </div>
  );
}
