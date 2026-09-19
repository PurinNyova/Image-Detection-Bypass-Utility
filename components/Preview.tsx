"use client";

type PreviewProps = {
  beforeUrl?: string | null;
  afterUrl?: string | null;
  metadataStatus?: string | null;
  finalizer: boolean;
  label?: string | null;
  disabled: boolean;
  onDownload: () => void;
};

function Pane({
  title,
  url,
  alt,
}: {
  title: string;
  url?: string | null;
  alt: string;
}) {
  return (
    <figure className="min-w-0">
      <figcaption className="mb-2 font-heading text-sm font-semibold text-text">
        {title}
      </figcaption>
      <div className="flex h-64 items-center justify-center overflow-hidden rounded-sm border border-border bg-surface md:h-80">
        {url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={url}
            alt={alt}
            className="max-h-full max-w-full object-contain"
          />
        ) : (
          <span className="text-xs text-text-muted">No image</span>
        )}
      </div>
    </figure>
  );
}

export default function Preview({
  beforeUrl,
  afterUrl,
  metadataStatus,
  finalizer,
  label,
  disabled,
  onDownload,
}: PreviewProps) {
  const metaFull = metadataStatus === "full";
  const metaDegraded = metadataStatus === "degraded";

  return (
    <section
      aria-label="Preview"
      className="rounded-md border border-border bg-surface p-6 shadow-sm"
    >
      <h2 className="font-heading text-[20px] font-semibold text-violet-700 dark:text-violet-300">
        Preview
      </h2>

      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <Pane
          title="Before"
          url={beforeUrl}
          alt="Source image preview"
        />
        <Pane
          title="After"
          url={afterUrl}
          alt="Processed output preview"
        />
      </div>

      {(metaFull || metaDegraded) && (
        <p
          role="status"
          className={
            metaFull
              ? "mt-4 text-xs text-success"
              : "mt-4 text-xs text-cyan-500"
          }
        >
          {metaFull
            ? "X-Forensic-Metadata: full - metadata written"
            : "X-Forensic-Metadata: degraded - metadata written without ExifTool"}
        </p>
      )}

      {label && (
        <p className="mt-4 text-xs text-text-muted break-words">{label}</p>
      )}

      {afterUrl && (
        <div className="mt-4">
          <button
            type="button"
            onClick={onDownload}
            disabled={disabled}
            className={
              finalizer
                ? "rounded-md bg-pink-600 px-5 py-2.5 text-base font-medium text-white hover:bg-pink-700 active:bg-pink-800 disabled:cursor-not-allowed disabled:opacity-50"
                : "rounded-md bg-violet-500 px-5 py-2.5 text-base font-medium text-white hover:bg-violet-600 active:bg-violet-800 disabled:cursor-not-allowed disabled:opacity-50"
            }
          >
            Download
          </button>
        </div>
      )}
    </section>
  );
}
