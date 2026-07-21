import { cx } from "../lib/format";

/**
 * AI Human Vision brand mark.
 *
 * NOTE: this is an ORIGINAL "vision" mark in the ACUD-inspired red/black style.
 * No official ACUD logo file was supplied, so this is a clean, production-ready
 * placeholder. To use your real logo, either drop your file at
 * `frontend/public/logo.svg` and render it with <img src="/app/logo.svg">, or
 * replace the <svg> in `LogoMark` below — spacing/proportions are preserved by
 * the surrounding layout, so nothing else needs to change.
 */
export function LogoMark({ size = 40, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 64 64"
      className={cx("shrink-0", className)} role="img" aria-label="AI Human Vision logo"
    >
      <rect width="64" height="64" rx="15" fill="rgb(var(--brand))" />
      {/* stylised eye = "vision" */}
      <path d="M11 32c9-12 33-12 42 0-9 12-33 12-42 0z" fill="none"
            stroke="white" strokeWidth="3.5" strokeLinejoin="round" />
      <circle cx="32" cy="32" r="7.5" fill="white" />
      <circle cx="32" cy="32" r="3" fill="rgb(var(--brand))" />
    </svg>
  );
}

export function Logo({
  size = 40, showText = true, subtitle, className, textClassName,
}: {
  size?: number; showText?: boolean; subtitle?: string;
  className?: string; textClassName?: string;
}) {
  return (
    <div className={cx("flex items-center gap-3", className)}>
      <LogoMark size={size} className="shadow-sm shadow-brand-600/25" />
      {showText && (
        <div className="leading-tight">
          <p className={cx("font-extrabold tracking-tight", textClassName || "text-[15px]")}>
            AI Human <span className="text-brand-600">Vision</span>
          </p>
          {subtitle && <p className="text-[11px] text-muted">{subtitle}</p>}
        </div>
      )}
    </div>
  );
}
