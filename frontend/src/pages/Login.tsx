import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { LogIn, Lock, User, ArrowRight } from "lucide-react";
import { Logo } from "../components/Logo";
import { Spinner } from "../components/ui/primitives";

/**
 * Branded sign-in screen for the Madkour AI Panel Inspector.
 *
 * The backend currently ships without an interactive login (access is an
 * optional API key at the reverse proxy), so this page is presentational: it
 * provides the corporate entry experience and lands the operator on the
 * dashboard. No route is gated and no authentication behaviour is changed — if
 * a real auth API is added later, wire it into `onSubmit` only.
 */
export default function Login() {
  const nav = useNavigate();
  const [busy, setBusy] = useState(false);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setTimeout(() => nav("/"), 500);
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden p-4"
         style={{ backgroundColor: "rgb(var(--bg))" }}>
      {/* professional background: soft dot grid + brand glow */}
      <div className="grid-dots absolute inset-0 opacity-70" />
      <div className="pointer-events-none absolute -top-40 -right-40 h-96 w-96 rounded-full blur-3xl"
           style={{ background: "rgb(var(--brand) / 0.14)" }} />
      <div className="pointer-events-none absolute -bottom-40 -left-40 h-96 w-96 rounded-full blur-3xl"
           style={{ background: "rgb(17 17 17 / 0.06)" }} />

      <div className="relative w-full max-w-md animate-slide-up">
        <div className="card p-8 sm:p-10">
          <div className="flex flex-col items-center text-center">
            <Logo size={64} showText={false} />
            <h1 className="mt-5 text-2xl font-extrabold tracking-tight">
              MADKOUR <span className="text-brand-400">AI</span>
            </h1>
            <p className="mt-1 text-sm text-muted">Panel Inspector · Industrial Electrical Intelligence</p>
          </div>

          <form onSubmit={onSubmit} className="mt-8 space-y-4">
            <div>
              <label className="label">Username</label>
              <div className="relative">
                <User className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-faint" />
                <input className="input pl-10" placeholder="operator" autoComplete="username" defaultValue="operator" />
              </div>
            </div>
            <div>
              <label className="label">Password</label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-faint" />
                <input className="input pl-10" type="password" placeholder="••••••••" autoComplete="current-password" defaultValue="password" />
              </div>
            </div>
            <button type="submit" className="btn-primary w-full" disabled={busy}>
              {busy ? <Spinner className="h-4 w-4" /> : <LogIn className="h-4 w-4" />}
              Sign in to dashboard
              {!busy && <ArrowRight className="h-4 w-4" />}
            </button>
          </form>

          <p className="mt-6 text-center text-xs text-faint">
            Administrative Capital for Urban Development · Enterprise Vision Suite
          </p>
        </div>
        <p className="mt-4 text-center text-xs text-muted">© {new Date().getFullYear()} Madkour AI Panel Inspector</p>
      </div>
    </div>
  );
}
