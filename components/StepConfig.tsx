"use client";

import { useRef, useState, type ChangeEvent, type RefObject } from "react";
import { Field } from "./Field";
import { getEndpoint, type EndpointId } from "../lib/endpoints";

export type ConfigValue =
  | number
  | string
  | boolean
  | number[]
  | undefined;

export interface ConfigurableStep {
  id: string;
  endpoint: EndpointId;
  config: Record<string, unknown>;
  reference?: File;
  lut?: File;
  outputFormat: "png" | "jpeg";
  includeExif: boolean;
}

export interface StepConfigProps {
  step?: ConfigurableStep;
  disabled?: boolean;
  onConfigChange?: (key: string, value: ConfigValue) => void;
  onReferenceChange?: (file: File | undefined) => void;
  onLutChange?: (file: File | undefined) => void;
  onOutputFormatChange?: (format: "png" | "jpeg") => void;
  onIncludeExifChange?: (include: boolean) => void;
}

const MAX_FILE_BYTES = 67108864;
const DATETIME_RE = /^\d{4}(-|:)\d{2}\1\d{2} \d{2}:\d{2}:\d{2}$/;

const UPLOAD_LABEL =
  "text-sm font-medium border border-[var(--border)] rounded-[var(--radius-sm)] bg-bg px-2 py-1.5 hover:bg-surface disabled:opacity-60";

export default function StepConfig({
  step,
  disabled = false,
  onConfigChange,
  onReferenceChange,
  onLutChange,
  onOutputFormatChange,
  onIncludeExifChange,
}: StepConfigProps) {
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [fileErrors, setFileErrors] = useState<Record<string, string>>({});
  const referenceInputRef = useRef<HTMLInputElement>(null);
  const lutInputRef = useRef<HTMLInputElement>(null);

  if (!step) {
    return (
      <section className="flex flex-col gap-3">
        <h3 className="text-base font-semibold text-[var(--text)]">Step configuration</h3>
        <p className="text-sm text-[var(--text-muted)]">
          No step selected. Select a step in the list to edit its settings.
        </p>
      </section>
    );
  }

  const def = getEndpoint(step.endpoint);
  if (!def) return null;

  const isForensic = def.finalizer && def.id === "forensic-camera";
  // Draft/error keys are per step so switching selected steps cannot leak local edits.
  const pkey = (key: string) => `${step.id}:${key}`;
  const showError = (key: string, message: string) =>
    setFieldErrors((prev) => {
      const k = pkey(key);
      return prev[k] === message ? prev : { ...prev, [k]: message };
    });
  const clearError = (key: string) =>
    setFieldErrors((prev) => {
      const k = pkey(key);
      if (!(k in prev)) return prev;
      const next = { ...prev };
      delete next[k];
      return next;
    });

  const effectiveString = (key: string, defaultDisplay = ""): string => {
    const k = pkey(key);
    if (k in drafts) return drafts[k];
    const val = step.config[key];
    if (val !== undefined && val !== null) return String(val);
    return defaultDisplay;
  };

  const handleNumberText = (
    key: string,
    raw: string,
    fieldDefault: number | null,
    isInt: boolean,
  ) => {
    setDrafts((prev) => ({ ...prev, [pkey(key)]: raw }));
    if (raw.trim() === "") {
      clearError(key);
      onConfigChange?.(key, undefined);
      return;
    }
    const num = Number(raw.trim());
    if (!Number.isFinite(num)) {
      showError(key, "Enter a valid number.");
      return;
    }
    if (isInt && !Number.isInteger(num)) {
      showError(key, "Enter a whole number.");
      return;
    }
    clearError(key);
    onConfigChange?.(key, fieldDefault !== null && num === fieldDefault ? undefined : num);
  };

  const handleArray = (
    key: string,
    raw: string,
    fieldDefault: number[],
    isInt: boolean,
  ) => {
    setDrafts((prev) => ({ ...prev, [pkey(key)]: raw }));
    if (raw.trim() === "") {
      clearError(key);
      onConfigChange?.(key, undefined);
      return;
    }
    const parsed: number[] = [];
    for (const token of raw.split(",")) {
      const num = Number(token.trim());
      if (!Number.isFinite(num) || (isInt && !Number.isInteger(num))) {
        showError(
          key,
          isInt
            ? "Enter comma-separated integers, e.g. 1, 2."
            : "Enter comma-separated numbers, e.g. 0, 0.5.",
        );
        return;
      }
      parsed.push(num);
    }
    clearError(key);
    const sameAsDefault =
      parsed.length === fieldDefault.length && parsed.every((v, i) => v === fieldDefault[i]);
    onConfigChange?.(key, sameAsDefault ? undefined : parsed);
  };

  const fieldDisabled = (key: string): boolean => {
    if (key !== "phase_perturb") return disabled;
    const variant = String(step.config["fft_variant"] ?? "v2");
    return disabled || variant === "v3" || variant === "v4";
  };

  const fieldNode = (field: (typeof def.fields)[number]) => {
    const id = `${step.id}-${field.key}`;
    const fieldDefault = field.default;
    const error = fieldErrors[pkey(field.key)];
    const common = {
      id,
      label: field.label ?? field.key,
      disabled: fieldDisabled(field.key),
      note: field.note,
      error,
    };

    switch (field.kind) {
      case "boolean": {
        const checked = (step.config[field.key] ?? fieldDefault) === true;
        return (
          <Field
            key={field.key}
            {...common}
            type="checkbox"
            checked={checked}
            onChange={(e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
              const next = (e.target as HTMLInputElement).checked;
              clearError(field.key);
              onConfigChange?.(
                field.key,
                next === (fieldDefault === true) ? undefined : next,
              );
            }}
          />
        );
      }
      case "select": {
        const value = String(
          step.config[field.key] !== undefined && step.config[field.key] !== null
            ? step.config[field.key]
            : fieldDefault,
        );
        return (
          <Field
            key={field.key}
            {...common}
            type="select"
            value={value}
            options={field.options?.map((o) => o.value) ?? []}
            onChange={(e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
              clearError(field.key);
              onConfigChange?.(
                field.key,
                e.target.value === fieldDefault ? undefined : e.target.value,
              );
            }}
          />
        );
      }
      case "text": {
        const fieldDefault = field.default as string | null;
        return (
          <Field
            key={field.key}
            {...common}
            type="text"
            value={effectiveString(field.key, field.nullable && fieldDefault === null ? "" : String(fieldDefault ?? ""))}
            onChange={(e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
              const raw = e.target.value;
              setDrafts((prev) => ({ ...prev, [pkey(field.key)]: raw }));
              const trimmed = raw.trim();
              if (trimmed === "") {
                clearError(field.key);
                if (field.nullable) onConfigChange?.(field.key, undefined);
                else showError(field.key, "Value is required.");
                return;
              }
              if (field.key === "datetime_original") {
                if (!DATETIME_RE.test(trimmed)) {
                  showError(
                    field.key,
                    "Format: YYYY-MM-DD HH:MM:SS or YYYY:MM:DD HH:MM:SS.",
                  );
                  return;
                }
              }
              clearError(field.key);
              onConfigChange?.(
                field.key,
                fieldDefault !== null && trimmed === fieldDefault ? undefined : trimmed,
              );
            }}
          />
        );
      }
      case "int-array":
      case "number-array": {
        const arrDefault = (fieldDefault as number[] | null) ?? [];
        const raw = effectiveString(field.key, arrDefault.join(", "));
        return (
          <Field
            key={field.key}
            {...common}
            type="text"
            value={raw}
            onChange={(e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
              handleArray(
                field.key,
                e.target.value,
                (fieldDefault as number[]) ?? [],
                field.kind === "int-array",
              )
            }
          />
        );
      }
      default: {
        // number / integer / nullable integer (seed, blend_n_jobs, iso, gps_*)
        const numDefault = fieldDefault === null ? "" : String(fieldDefault);
        return (
          <Field
            key={field.key}
            {...common}
            type="text"
            value={effectiveString(field.key, numDefault)}
            onChange={(e) =>
              handleNumberText(
                field.key,
                e.target.value,
                (fieldDefault as number) ?? null,
                field.kind === "integer",
              )
            }
          />
        );
      }
    }
  };

  const fileInput = (
    key: "reference" | "lut",
    label: string,
    required: boolean,
    file: File | undefined,
    ref: RefObject<HTMLInputElement | null>,
    onChange: ((file: File | undefined) => void) | undefined,
  ) => (
    <div className="flex flex-col gap-1">
      <label className="text-sm">
        <span>
          {label}{" "}
          {required ? (
            <span className="text-[var(--color-error)]">(required)</span>
          ) : (
            <span className="text-[var(--text-muted)]">(optional)</span>
          )}
        </span>
      </label>
      <div className="flex items-center gap-2">
        <input
          type="file"
          accept={key === "lut" ? ".cube" : "image/*"}
          disabled={disabled}
          ref={ref}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f && f.size > MAX_FILE_BYTES) {
              e.currentTarget.value = "";
              setFileErrors((prev) => ({
                ...prev,
                [step.id + ":" + key]: `${
                  key === "lut" ? "LUT file" : "reference image"
                } exceeds 64 MiB upload cap`,
              }));
              return;
            }
            setFileErrors((prev) => {
              const k = step.id + ":" + key;
              if (!(k in prev)) return prev;
              const next = { ...prev };
              delete next[k];
              return next;
            });
            onChange?.(f);
          }}
          className={UPLOAD_LABEL + " min-w-0"}
        />
        {file ? (
          <button
            type="button"
            disabled={disabled}
            aria-label={`Clear ${key === "lut" ? "LUT file" : "reference image"}`}
            onClick={() => {
              if (ref.current) ref.current.value = "";
              onChange?.(undefined);
            }}
            className="px-2 py-1.5 border border-[var(--border)] rounded-[var(--radius-sm)] text-sm disabled:opacity-60"
          >
            Clear
          </button>
        ) : null}
      </div>
      {file ? (
        <p className="text-xs text-[var(--text-muted)] truncate min-w-0">{file.name}</p>
      ) : null}
      {fileErrors[step.id + ":" + key] ? (
        <p role="alert" className="text-xs text-[var(--color-error)] break-words">
          {fileErrors[step.id + ":" + key]}
        </p>
      ) : null}
      {required && !file ? (
        <p role="alert" className="text-xs text-[var(--color-error)] break-words">
          LUT file is required.
        </p>
      ) : null}
    </div>
  );

  return (
    <section className="flex flex-col gap-4 min-w-0" aria-label="Step configuration">
      <h3 className="text-base font-semibold text-[var(--text)] min-w-0 truncate">
        {def.label}
      </h3>

      {def.referenceUpload
        ? fileInput("reference", "Reference image", false, step.reference, referenceInputRef, onReferenceChange)
        : null}
      {def.lutUpload
        ? fileInput("lut", "LUT file", true, step.lut, lutInputRef, onLutChange)
        : null}
      {step.endpoint === "fft" ? (
        <p className="text-xs text-[var(--text-muted)] break-words">
          The reference image drives reference mode matching.
        </p>
      ) : null}
      {step.endpoint === "awb" ? (
        <p className="text-xs text-[var(--text-muted)] break-words">
          Without a reference image, grey-world balancing is used as a fallback.
        </p>
      ) : null}

      <div className="flex flex-col gap-3">
        {def.fields.map(fieldNode)}
      </div>

      {!isForensic ? (
        <div className="flex flex-col gap-3 border-t border-[var(--border)] pt-3">
          <Field
            id={`${step.id}-output_format`}
            label="output_format"
            type="select"
            value={step.outputFormat}
            options={["png", "jpeg"]}
            disabled={disabled}
            onChange={(e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
              onOutputFormatChange?.(e.target.value as "png" | "jpeg")
            }
          />
          <Field
            id={`${step.id}-include_exif`}
            label="include_exif"
            type="checkbox"
            checked={step.includeExif}
            disabled={disabled}
            onChange={(e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
              onIncludeExifChange?.((e.target as HTMLInputElement).checked)
            }
          />
        </div>
      ) : (
        <p className="text-xs text-[var(--text-muted)] break-words">
          Forensic Camera always returns JPEG; this step does not accept
          output_format, include_exif, reference images, or LUT files.
        </p>
      )}

      {step.endpoint === "non-semantic" ? (
        <p className="text-xs text-[var(--text-muted)] break-words">
          Runtime depends on the optional backend optimization stack being installed.
        </p>
      ) : null}
    </section>
  );
}
