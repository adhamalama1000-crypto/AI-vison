import { useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard, Video, Users, Bell, Cpu, Settings as SettingsIcon,
  Sun, Moon, Menu, X, ScanFace, Activity, Zap, Waypoints,
  Database, GitCompare, FileText, ScanLine, Layers, ClipboardCheck,
  GaugeCircle, CalendarCheck,
} from "lucide-react";
import { useTheme } from "../hooks/useTheme";
import { useEventSocket } from "../hooks/useEventSocket";
import { useHealth } from "../hooks/useData";
import { cx } from "../lib/format";
import { Dot } from "./ui/primitives";

const NAV = [
  { section: "Overview", items: [
    { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
    { to: "/live", label: "Live Cameras", icon: Video },
    { to: "/metrics", label: "Metrics", icon: GaugeCircle },
  ]},
  { section: "Recognition", items: [
    { to: "/employees", label: "Employees", icon: Users },
    { to: "/attendance", label: "Attendance", icon: CalendarCheck },
    { to: "/events", label: "Events", icon: Bell },
  ]},
  { section: "Electrical AI", items: [
    { to: "/datasets", label: "Electrical Dataset", icon: Database },
    { to: "/training", label: "Training", icon: GitCompare },
    { to: "/reference", label: "Reference Design", icon: Layers },
    { to: "/panel", label: "Panel Analysis", icon: ScanLine },
    { to: "/inspection", label: "Inspection", icon: ClipboardCheck },
  ]},
  { section: "System", items: [
    { to: "/reports", label: "Reports", icon: FileText },
    { to: "/models", label: "AI Models", icon: Cpu },
    { to: "/settings", label: "Settings", icon: SettingsIcon },
  ]},
];

export function Layout({ children }: { children: ReactNode }) {
  const { theme, toggle } = useTheme();
  const { state } = useEventSocket();
  const { data: health } = useHealth();
  const [open, setOpen] = useState(false);
  const loc = useLocation();

  const conn = state === "live"
    ? { tone: "green" as const, text: "Live" }
    : state === "connecting" ? { tone: "amber" as const, text: "Connecting" } : { tone: "red" as const, text: "Offline" };

  const Sidebar = (
    <aside className={cx(
      "flex h-full w-64 shrink-0 flex-col border-r surface",
      "fixed inset-y-0 left-0 z-40 transition-transform lg:static lg:translate-x-0",
      open ? "translate-x-0" : "-translate-x-full",
    )}>
      <div className="flex items-center gap-3 border-b px-5 py-[18px]">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 shadow-lg shadow-brand-600/30">
          <ScanFace className="h-5 w-5 text-white" />
        </div>
        <div className="leading-tight">
          <p className="text-sm font-bold">AI Vision</p>
          <p className="text-[11px] text-muted">Platform v3.1.5</p>
        </div>
        <button className="ml-auto lg:hidden text-muted" onClick={() => setOpen(false)}><X className="h-5 w-5" /></button>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {NAV.map((group) => (
          <div key={group.section} className="mb-5">
            <p className="px-3 pb-2 text-[10px] font-bold uppercase tracking-widest text-faint">{group.section}</p>
            <div className="space-y-1">
              {group.items.map((it) => (
                <NavLink key={it.to} to={it.to} end={(it as any).end} onClick={() => setOpen(false)}
                  className={({ isActive }) => cx(
                    "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all",
                    isActive
                      ? "bg-brand-600 text-white shadow-sm shadow-brand-600/30"
                      : "text-muted hover:bg-[rgb(var(--surface-2))] hover:text-[rgb(var(--text))]",
                  )}>
                  <it.icon className="h-[18px] w-[18px]" /> {it.label}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t p-4">
        <div className="flex items-center justify-between rounded-xl surface-2 px-3 py-2.5">
          <span className="flex items-center gap-2 text-xs font-semibold"><Dot tone={conn.tone} pulse={state === "live"} /> {conn.text}</span>
          <span className="text-[11px] text-muted">{health ? `${health.cameras_connected}/${health.cameras_total} cams` : "—"}</span>
        </div>
      </div>
    </aside>
  );

  const title = titleForPath(loc.pathname);

  return (
    <div className="flex h-screen overflow-hidden">
      {open && <div className="fixed inset-0 z-30 bg-slate-950/50 lg:hidden" onClick={() => setOpen(false)} />}
      {Sidebar}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b surface px-4 py-3 sm:px-6">
          <button className="btn-icon btn-ghost lg:hidden" onClick={() => setOpen(true)}><Menu className="h-5 w-5" /></button>
          <div>
            <h1 className="text-lg font-bold leading-tight sm:text-xl">{title.label}</h1>
            <p className="hidden text-xs text-muted sm:block">{title.sub}</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <TopGauge icon={<Activity className="h-3.5 w-3.5" />} label="FPS" val={health ? "live" : ""} />
            <button onClick={toggle} className="btn-icon btn-outline" title="Toggle theme" aria-label="Toggle theme">
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-4 sm:p-6">{children}</main>
      </div>
    </div>
  );
}

function TopGauge({ icon, label }: { icon: ReactNode; label: string; val?: string }) {
  return (
    <div className="hidden items-center gap-2 rounded-xl surface-2 px-3 py-2 text-xs font-semibold sm:flex">
      <span className="text-brand-400">{icon}</span>
      <span className="text-muted">{label}</span>
    </div>
  );
}

function titleForPath(p: string): { label: string; sub: string } {
  if (p.startsWith("/live")) return { label: "Live Cameras", sub: "Real-time RTSP streams with AI overlays" };
  if (p.startsWith("/metrics")) return { label: "Metrics", sub: "System & AI performance dashboard" };
  if (p.startsWith("/employees")) return { label: "Employees", sub: "Enrolment & face recognition management" };
  if (p.startsWith("/attendance")) return { label: "Attendance", sub: "Face-recognition attendance log & summary" };
  if (p.startsWith("/events")) return { label: "Events", sub: "Recognition & system event log" };
  if (p.startsWith("/datasets")) return { label: "Electrical Dataset", sub: "Upload & validate training datasets" };
  if (p.startsWith("/training")) {
    if (p.includes("/comparison")) return { label: "Model Comparison", sub: "Trained model evaluation & selection" };
    if (/\/training\/[^/]+$/.test(p)) return { label: "Training Progress", sub: "Live training metrics & controls" };
    return { label: "Training", sub: "Configure, launch & monitor training jobs" };
  }
  if (p.startsWith("/reference")) return { label: "Reference Design", sub: "Reference designs & expected specifications" };
  if (p.startsWith("/panel")) return { label: "Panel Analysis", sub: "Component & wiring analysis of control panels" };
  if (p.startsWith("/inspection")) return { label: "Inspection", sub: "Compare panels against reference designs" };
  if (p.startsWith("/reports")) return { label: "Reports", sub: "Generated analysis & inspection reports" };
  if (p.startsWith("/models")) return { label: "AI Models", sub: "Model configuration & live metrics" };
  if (p.startsWith("/settings")) return { label: "Settings", sub: "Cameras & platform configuration" };
  return { label: "Dashboard", sub: "System overview & live analytics" };
}

// re-export icons used by pages to keep imports centralised
export { Zap, Waypoints };
