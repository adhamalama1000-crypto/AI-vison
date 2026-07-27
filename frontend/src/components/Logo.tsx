import { cx } from "../lib/format";

/**
 * Madkour AI Panel Inspector brand mark.
 *
 * ORIGINAL artwork. No Madkour brand asset was supplied or reproduced: this is a
 * clean, industrial mark drawn from the subject matter itself — a DIN rail
 * carrying modular devices, with a scanning line across it — in the engineering
 * blue / signal amber palette defined in `index.css`.
 *
 * To use the real corporate logo instead, drop the file at
 * `frontend/public/logo.svg` and render it with `<img src="/app/logo.svg">`, or
 * replace the <svg> in `LogoMark`. Surrounding layout preserves the proportions,
 * so nothing else needs to change.
 */
export function LogoMark({ size = 40, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 64 64"
      className={cx("shrink-0", className)} role="img"
      aria-label="Madkour AI Panel Inspector logo"
    >
      <rect width="64" height="64" rx="14" fill="rgb(var(--surface-2))" />
      <rect x="0.75" y="0.75" width="62.5" height="62.5" rx="13.4"
            fill="none" stroke="rgb(var(--brand))" strokeWidth="1.5" opacity="0.55" />
      {/* DIN rail */}
      <rect x="9" y="41" width="46" height="5" rx="1.5" fill="rgb(var(--faint))" />
      {/* modular devices seated on the rail */}
      <rect x="12" y="20" width="10" height="22" rx="2" fill="rgb(var(--brand))" />
      <rect x="24" y="24" width="14" height="18" rx="2" fill="rgb(var(--muted))" />
      <rect x="40" y="17" width="13" height="25" rx="2" fill="rgb(var(--brand))" opacity="0.62" />
      {/* toggle lever + status LED: the two things an inspector reads first */}
      <rect x="15" y="27" width="4" height="8" rx="1.4" fill="rgb(var(--bg))" />
      <circle cx="46.5" cy="23.5" r="2.4" fill="rgb(var(--accent))" />
      {/* scan line */}
      <path d="M6 33.5h52" stroke="rgb(var(--accent))" strokeWidth="1.6"
            strokeLinecap="round" opacity="0.9" />
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
            MADKOUR <span className="text-brand-400">AI</span>
          </p>
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">
            Panel Inspector
          </p>
          {subtitle && <p className="mt-0.5 text-[11px] text-faint">{subtitle}</p>}
        </div>
      )}
    </div>
  );
}
