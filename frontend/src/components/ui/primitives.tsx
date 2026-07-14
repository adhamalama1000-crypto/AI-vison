import type { ReactNode } from "react";
import { cx } from "../../lib/format";

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cx("card", className)}>{children}</div>;
}

export function Badge({ tone = "gray", children, className }: {
  tone?: "green" | "red" | "amber" | "blue" | "gray"; children: ReactNode; className?: string;
}) {
  const map = { green: "badge-green", red: "badge-red", amber: "badge-amber", blue: "badge-blue", gray: "badge-gray" };
  return <span className={cx(map[tone], className)}>{children}</span>;
}

export function Dot({ tone = "gray", pulse }: { tone?: "green" | "red" | "amber" | "blue" | "gray"; pulse?: boolean }) {
  const map = { green: "bg-emerald-500", red: "bg-rose-500", amber: "bg-amber-500", blue: "bg-brand-500", gray: "bg-slate-400" };
  return <span className={cx("inline-block h-2 w-2 rounded-full", map[tone], pulse && "animate-pulse-dot")} />;
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg className={cx("animate-spin", className || "h-5 w-5")} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("skeleton rounded-lg", className)} />;
}

export function EmptyState({ icon, title, hint }: { icon?: ReactNode; title: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-14 text-center animate-fade-in">
      {icon && <div className="mb-3 text-faint">{icon}</div>}
      <p className="text-base font-semibold">{title}</p>
      {hint && <p className="mt-1 max-w-sm text-sm text-muted">{hint}</p>}
    </div>
  );
}

export function Toggle({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button
      type="button" role="switch" aria-checked={checked} disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cx(
        "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500/50 disabled:opacity-50",
        checked ? "bg-brand-600" : "bg-slate-300 dark:bg-slate-600",
      )}
    >
      <span className={cx("inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform duration-200", checked ? "translate-x-[22px]" : "translate-x-0.5")} />
    </button>
  );
}

export function Segmented<T extends string>({ value, options, onChange }: {
  value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-xl p-1 surface-2 border" role="tablist">
      {options.map((o) => (
        <button
          key={o.value} onClick={() => onChange(o.value)}
          className={cx(
            "rounded-lg px-3 py-1.5 text-xs font-semibold transition-all",
            value === o.value ? "bg-brand-600 text-white shadow-sm" : "text-muted hover:text-[rgb(var(--text))]",
          )}
        >{o.label}</button>
      ))}
    </div>
  );
}

export function StatCard({ icon, label, value, sub, tone = "blue" }: {
  icon: ReactNode; label: string; value: ReactNode; sub?: ReactNode;
  tone?: "blue" | "green" | "amber" | "red" | "violet";
}) {
  const tint: Record<string, string> = {
    blue: "bg-brand-500/12 text-brand-400", green: "bg-emerald-500/12 text-emerald-500",
    amber: "bg-amber-500/12 text-amber-500", red: "bg-rose-500/12 text-rose-500",
    violet: "bg-violet-500/12 text-violet-400",
  };
  return (
    <Card className="card-hover p-5 animate-slide-up">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</p>
          <p className="mt-2 text-3xl font-bold tabular-nums">{value}</p>
          {sub && <p className="mt-1 text-xs text-muted">{sub}</p>}
        </div>
        <div className={cx("grid h-11 w-11 place-items-center rounded-xl", tint[tone])}>{icon}</div>
      </div>
    </Card>
  );
}

export function SectionTitle({ title, action }: { title: string; action?: ReactNode }) {
  return (
    <div className="mb-3 flex items-center justify-between">
      <h2 className="text-sm font-bold uppercase tracking-wide text-muted">{title}</h2>
      {action}
    </div>
  );
}
