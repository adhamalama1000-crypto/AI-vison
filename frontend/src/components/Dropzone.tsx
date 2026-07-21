import { useRef, useState, type ReactNode } from "react";
import { UploadCloud, X } from "lucide-react";
import { cx } from "../lib/format";

/**
 * Reusable drag-&-drop image dropzone with click-to-browse and a live preview.
 * Validates that the dropped file is an image before handing it up.
 */
export function Dropzone({
  onFile, preview, label = "Drag & drop an image, or click to browse",
  hint = "JPG, PNG, WEBP, BMP · up to 8K", onClear, className,
}: {
  onFile: (file: File) => void;
  preview?: string | null;
  label?: string;
  hint?: string;
  onClear?: () => void;
  className?: string;
}) {
  const ref = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  const take = (files?: FileList | null) => {
    const f = files?.[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files); }}
      onClick={() => ref.current?.click()}
      className={cx(
        "relative grid cursor-pointer place-items-center overflow-hidden rounded-2xl border-2 border-dashed p-6 text-center transition-all",
        drag ? "border-brand-500 bg-brand-600/[0.06]" : "border-[rgb(var(--border))] hover:border-brand-500/60 hover:bg-brand-600/[0.03]",
        className || "min-h-[220px]",
      )}
    >
      <input ref={ref} type="file" accept="image/*" className="hidden"
             onChange={(e) => take(e.target.files)} />
      {preview ? (
        <>
          <img src={preview} alt="preview" className="max-h-[320px] w-auto rounded-xl object-contain" />
          {onClear && (
            <button
              onClick={(e) => { e.stopPropagation(); onClear(); }}
              className="absolute right-3 top-3 grid h-8 w-8 place-items-center rounded-lg bg-black/60 text-white hover:bg-black/80"
              aria-label="Clear image"
            ><X className="h-4 w-4" /></button>
          )}
        </>
      ) : (
        <div className="flex flex-col items-center gap-2 text-muted">
          <span className="grid h-14 w-14 place-items-center rounded-2xl bg-brand-600/[0.08] text-brand-600">
            <UploadCloud className="h-7 w-7" />
          </span>
          <p className="text-sm font-semibold text-[rgb(var(--text))]">{label}</p>
          <p className="text-xs text-faint">{hint}</p>
        </div>
      )}
    </div>
  );
}

export function preview(file: File): Promise<string> {
  return new Promise((res) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result));
    r.readAsDataURL(file);
  });
}

export function ProgressBar({ active, label }: { active: boolean; label?: ReactNode }) {
  if (!active) return null;
  return (
    <div className="space-y-1.5">
      {label && <p className="text-xs font-medium text-muted">{label}</p>}
      <div className="h-1.5 w-full overflow-hidden rounded-full surface-2">
        <div className="h-full w-1/3 animate-[indeterminate_1.2s_ease-in-out_infinite] rounded-full bg-brand-600" />
      </div>
    </div>
  );
}
