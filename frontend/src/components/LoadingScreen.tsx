import { cx } from "../lib/format";
import { LogoMark } from "./Logo";
import { Spinner } from "./ui/primitives";

/**
 * Branded splash / loading screen. Shown briefly on first app mount and while
 * async chunks load, then fades out. Pure presentation — no data dependencies.
 */
export function LoadingScreen({ fading = false }: { fading?: boolean }) {
  return (
    <div
      className={cx(
        "fixed inset-0 z-[100] flex flex-col items-center justify-center gap-6 transition-opacity duration-500",
        fading ? "pointer-events-none opacity-0" : "opacity-100",
      )}
      style={{ backgroundColor: "rgb(var(--bg))" }}
    >
      <div className="grid-dots absolute inset-0 opacity-60" />
      <div className="relative flex flex-col items-center gap-5 animate-scale-in">
        <div className="animate-logo-pulse">
          <LogoMark size={72} className="shadow-soft-lg" />
        </div>
        <div className="text-center">
          <h1 className="text-2xl font-extrabold tracking-tight">
            AI Human <span className="text-brand-600">Vision</span>
          </h1>
          <p className="mt-1 text-sm text-muted">AI-Powered Human Vision Platform</p>
        </div>
        <Spinner className="h-5 w-5 text-brand-600" />
      </div>
    </div>
  );
}
