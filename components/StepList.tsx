"use client";

import { ENDPOINTS, getEndpoint, type EndpointId } from "../lib/endpoints";

export type StepStatus = "idle" | "running" | "done" | "error" | "stale" | "cancelled";

export type StepListItem = {
  id: string;
  endpoint: EndpointId;
  status: StepStatus;
  error?: string;
};

type Props = {
  steps: StepListItem[];
  selectedId?: string;
  disabled: boolean;
  onAdd: (endpoint: EndpointId) => void;
  onSelect: (id: string) => void;
  onMove: (index: number, direction: -1 | 1) => void;
  onRemove: (index: number) => void;
};

const STATUS_DOT: Record<StepStatus, string> = {
  idle: "bg-[var(--border)]",
  running: "bg-violet-500",
  done: "bg-[var(--color-success)]",
  error: "bg-[var(--color-error)]",
  stale: "bg-[var(--color-warning)]",
  cancelled: "bg-[var(--border)]",
};

const STATUS_TEXT: Record<StepStatus, string> = {
  idle: "text-[var(--text-muted)]",
  running: "text-violet-500",
  done: "text-[var(--color-success)]",
  error: "text-[var(--color-error)]",
  stale: "text-[var(--color-warning)]",
  cancelled: "text-[var(--text-muted)]",
};

export default function StepList({
  steps,
  selectedId,
  disabled,
  onAdd,
  onSelect,
  onMove,
  onRemove,
}: Props) {
  const forensicMisplaced = steps.some(
    (s, i) => s.endpoint === "forensic-camera" && i < steps.length - 1,
  );

  return (
    <div className="border border-[var(--border)] rounded-[var(--radius-sm)] bg-surface p-3">
      <label className="flex items-center gap-2 text-sm">
        <span>Add step</span>
        <select
          className="min-w-0 flex-1 border border-[var(--border)] rounded-[var(--radius-sm)] bg-bg px-2 py-1"
          value=""
          disabled={disabled}
          onChange={(e) => {
            if (e.target.value) onAdd(e.target.value as EndpointId);
            e.target.value = "";
          }}
        >
          <option value="" disabled>
            Choose an endpoint…
          </option>
          {ENDPOINTS.map((e) => (
            <option key={e.id} value={e.id}>
              {e.label}
            </option>
          ))}
        </select>
      </label>

      {forensicMisplaced && (
        <p
          role="alert"
          className="mt-2 border-l-2 border-[var(--color-warning)] pl-2 text-sm text-[var(--color-warning)] break-words"
        >
          Warning: forensic output fed into another endpoint loses metadata. Move Forensic Camera
          last to keep it.
        </p>
      )}

      <ol className="mt-3 flex flex-col gap-1">
        {steps.length === 0 && (
          <li className="text-sm text-[var(--text-muted)] py-2">No steps yet. Add one above.</li>
        )}
        {steps.map((step, index) => {
          const def = getEndpoint(step.endpoint);
          const isForensic = step.endpoint === "forensic-camera";
          const selected = step.id === selectedId;
          return (
            <li
              key={step.id}
              className={
                "min-w-0 border rounded-[var(--radius-sm)] p-2 " +
                (isForensic
                  ? "border-l-4 border-l-pink-500 border-[var(--border)]"
                  : "border-[var(--border)]")
              }
            >
              <button
                type="button"
                aria-current={selected ? "true" : undefined}
                onClick={() => onSelect(step.id)}
                disabled={disabled}
                aria-label={
                  "Step " +
                  (index + 1) +
                  ": " +
                  (def ? def.label : step.endpoint) +
                  ", status " +
                  step.status +
                  (selected ? ", selected" : "")
                }
                className="flex items-center gap-2 min-w-0 w-full text-left disabled:opacity-50"
              >
                <span className="text-sm text-[var(--text-muted)] tabular-nums shrink-0">
                  {index + 1}.
                </span>
                <span
                  className={"h-2 w-2 rounded-full shrink-0 " + STATUS_DOT[step.status]}
                  aria-hidden
                />
                <span className="truncate min-w-0 flex-1 text-sm">
                  {def ? def.label : step.endpoint}
                </span>
                <span role="status" className={"text-xs shrink-0 " + STATUS_TEXT[step.status]}>
                  {step.status}
                </span>
                {isForensic && (
                  <span className="text-xs text-[var(--text-muted)] shrink-0 hidden sm:inline">
                    Finalizer · recommended last
                  </span>
                )}
              </button>

              {step.status === "stale" && (
                <p className="mt-1 text-xs text-[var(--color-warning)]">
                  Stale: settings changed, re-run to refresh this step.
                </p>
              )}

              {step.error && (
                <p role="alert" className="mt-1 text-xs text-[var(--color-error)] break-words">
                  {step.error}
                </p>
              )}

              <div className="mt-1 flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => onMove(index, -1)}
                  disabled={disabled || index === 0}
                  aria-label={"Move " + (def ? def.label : step.endpoint) + " up"}
                  className="ml-auto px-2 py-0.5 border border-[var(--border)] rounded-[var(--radius-sm)] text-sm disabled:opacity-50"
                >
                  ↑
                </button>
                <button
                  type="button"
                  onClick={() => onMove(index, 1)}
                  disabled={disabled || index === steps.length - 1}
                  aria-label={"Move " + (def ? def.label : step.endpoint) + " down"}
                  className="px-2 py-0.5 border border-[var(--border)] rounded-[var(--radius-sm)] text-sm disabled:opacity-50"
                >
                  ↓
                </button>
                <button
                  type="button"
                  onClick={() => onRemove(index)}
                  disabled={disabled}
                  aria-label={"Remove " + (def ? def.label : step.endpoint)}
                  className="px-2 py-0.5 border border-[var(--border)] rounded-[var(--radius-sm)] text-sm disabled:opacity-50"
                >
                  Remove
                </button>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
