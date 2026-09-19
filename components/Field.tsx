import type { ChangeEvent, FocusEvent } from "react";

export type FieldType = "text" | "number" | "select" | "checkbox";

export interface FieldProps {
  id: string;
  label: string;
  type: FieldType;
  value?: string;
  checked?: boolean;
  options?: readonly string[];
  disabled?: boolean;
  note?: string;
  error?: string;
  step?: string;
  onChange?: (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => void;
  onBlur?: (e: FocusEvent<HTMLInputElement | HTMLSelectElement>) => void;
}

const controlClass =
  "w-full rounded-sm border border-border bg-surface px-2 py-1.5 text-sm text-text disabled:text-text-muted disabled:opacity-60";
const errorClass = "border-error";

export function Field({
  id,
  label,
  type,
  value,
  checked,
  options,
  disabled,
  note,
  error,
  step,
  onChange,
  onBlur,
}: FieldProps) {
  const describedBy =
    [note ? `${id}-note` : null, error ? `${id}-error` : null]
      .filter(Boolean)
      .join(" ") || undefined;

  if (type === "checkbox") {
    return (
      <div className="flex items-center gap-2">
        <input
          id={id}
          type="checkbox"
          checked={checked ?? false}
          disabled={disabled}
          aria-describedby={describedBy}
          aria-invalid={error ? true : undefined}
          onChange={onChange}
          onBlur={onBlur}
          className="h-4 w-4 shrink-0 accent-violet-500"
        />
        <label htmlFor={id} className="text-sm">
          {label}
        </label>
        <FieldMessages id={id} note={note} error={error} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm">
        {label}
      </label>
      {type === "select" ? (
        <select
          id={id}
          value={value ?? ""}
          disabled={disabled}
          aria-describedby={describedBy}
          aria-invalid={error ? true : undefined}
          onChange={onChange}
          onBlur={onBlur}
          className={`${controlClass} ${error ? errorClass : ""}`}
        >
          {options?.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          type={type}
          value={value ?? ""}
          step={type === "number" ? step : undefined}
          disabled={disabled}
          aria-describedby={describedBy}
          aria-invalid={error ? true : undefined}
          onChange={onChange}
          onBlur={onBlur}
          className={`${controlClass} ${error ? errorClass : ""}`}
        />
      )}
      <FieldMessages id={id} note={note} error={error} />
    </div>
  );
}

function FieldMessages({
  id,
  note,
  error,
}: {
  id: string;
  note?: string;
  error?: string;
}) {
  return (
    <>
      {note ? (
        <p id={`${id}-note`} className="text-xs text-text-muted break-words">
          {note}
        </p>
      ) : null}
      {error ? (
        <p
          id={`${id}-error`}
          role="alert"
          className="text-xs text-error break-words"
        >
          {error}
        </p>
      ) : null}
    </>
  );
}
