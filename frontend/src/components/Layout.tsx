import { useState, type ReactNode } from "react";
import { NavLink, useLocation, useNavigate, Link } from "react-router-dom";
import {
  LayoutDashboard, Video, Users, Bell, Cpu, Settings as SettingsIcon,
  Sun, Moon, Menu, X, Zap, Waypoints, LogOut, ChevronDown,
  Database, GitCompare, FileText, ScanLine, Layers, ClipboardCheck,
  GaugeCircle, CalendarCheck, CircuitBoard, ScanText, ScanFace,
  ScanEye, GitCompareArrows,
} from "lucide-react";
import { useTheme } from "../hooks/useTheme";
import { useEventSocket } from "../hooks/useEventSocket";
import { useHealth, useEvents } from "../hooks/useData";
import { cx, timeAgo, EVENT_LABEL } from "../lib/format";
import { Dot } from "./ui/primitives";
import { Logo, LogoMark } from "./Logo";

const NAV = [
  { section: "Overview", items: [
    { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
    { to: "/live", label: "Live Cameras", icon: Video },
    { to: "/metrics", label: "Metrics", icon: GaugeCircle },
  ]},
  { section: "Recognition", items: [
    { to: "/face-recognition", label: "Face Recognition", icon: ScanFace },
    { to: "/employees", label: "Employees", icon: Users },
    { to: "/attendance", label: "Attendance", icon: CalendarCheck },
    { to: "/events", label: "Events", icon: Bell },
  ]},
  { section: "Image AI", items: [
    { to: "/image-analysis", label: "Image Analysis", icon: ScanEye },
    { to: "/image-comparison", label: "Image Comparison", icon: GitCompareArrows },
  ]},
  { section: "Industrial Inspection", items: [
    { to: "/reference-panels", label: "Reference Panels", icon: CircuitBoard },
    { to: "/topology", label: "Topology Viewer", icon: Waypoints },
    { to: "/datasheets", label: "Datasheets", icon: ScanText },
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
      <div className="flex items-center gap-3 border-b px-5 py-4">
        <Logo size={38} subtitle="Vision Platform" textClassName="text-[15px]" />
        <button className="ml-auto lg:hidden text-muted" onClick={() => setOpen(false)} aria-label="Close menu">
          <X className="h-5 w-5" />
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {NAV.map((group) => (
          <div key={group.section} className="mb-5">
            <p className="px-3 pb-2 text-[10px] font-bold uppercase tracking-widest text-faint">{group.section}</p>
            <div className="space-y-1">
              {group.items.map((it) => (
                <NavLink key={it.to} to={it.to} end={(it as any).end} onClick={() => setOpen(false)}
                  className={({ isActive }) => cx(
                    "group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-150",
                    isActive
                      ? "bg-brand-600 text-white shadow-sm shadow-brand-600/30"
                      : "text-muted hover:bg-brand-600/[0.08] hover:text-brand-600",
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
        <p className="mt-3 text-center text-[10px] text-faint">AI Human Vision · v5.0</p>
      </div>
    </aside>
  );

  const title = titleForPath(loc.pathname);

  return (
    <div className="flex h-screen overflow-hidden">
      {open && <div className="fixed inset-0 z-30 bg-black/40 backdrop-blur-sm lg:hidden" onClick={() => setOpen(false)} />}
      {Sidebar}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b glass px-4 py-3 sm:px-6">
          <button className="btn-icon btn-ghost lg:hidden" onClick={() => setOpen(true)} aria-label="Open menu"><Menu className="h-5 w-5" /></button>
          {/* compact logo on mobile where the sidebar is hidden */}
          <LogoMark size={30} className="lg:hidden" />
          <div className="min-w-0">
            <h1 className="truncate text-lg font-extrabold leading-tight sm:text-xl">{title.label}</h1>
            <p className="hidden truncate text-xs text-muted sm:block">{title.sub}</p>
          </div>
          <div className="ml-auto flex items-center gap-1.5 sm:gap-2">
            <NotificationsMenu />
            <button onClick={toggle} className="btn-icon btn-ghost" title="Toggle theme" aria-label="Toggle theme">
              {theme === "dark" ? <Sun className="h-[18px] w-[18px]" /> : <Moon className="h-[18px] w-[18px]" />}
            </button>
            <ProfileMenu />
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-4 sm:p-6">
          <div key={loc.pathname} className="animate-fade-in">{children}</div>
        </main>
      </div>
    </div>
  );
}

/* ---- notifications popover (recent events) ---- */
function NotificationsMenu() {
  const [open, setOpen] = useState(false);
  const { data } = useEvents("?limit=6");
  const events = data?.events ?? [];
  return (
    <div className="relative">
      <button className="btn-icon btn-ghost relative" onClick={() => setOpen((o) => !o)} aria-label="Notifications">
        <Bell className="h-[18px] w-[18px]" />
        {events.length > 0 && (
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-brand-600 ring-2 ring-[rgb(var(--surface))]" />
        )}
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-40 mt-2 w-80 origin-top-right animate-scale-in card p-2 shadow-soft-lg">
            <div className="flex items-center justify-between px-2 py-1.5">
              <p className="text-sm font-bold">Notifications</p>
              <Link to="/events" onClick={() => setOpen(false)} className="text-xs font-semibold text-brand-600 hover:underline">View all</Link>
            </div>
            <div className="max-h-80 overflow-y-auto">
              {events.length === 0 ? (
                <p className="px-2 py-6 text-center text-sm text-muted">No recent events</p>
              ) : events.map((e) => (
                <div key={e.id} className="flex items-start gap-3 rounded-lg px-2 py-2 hover:bg-[rgb(var(--surface-2))]">
                  <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-brand-600" />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{e.label || EVENT_LABEL[e.type] || e.type}</p>
                    <p className="text-[11px] text-muted">{e.camera_name || e.camera_id || "—"} · {timeAgo(e.created_at)}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/* ---- profile popover ---- */
function ProfileMenu() {
  const [open, setOpen] = useState(false);
  const nav = useNavigate();
  return (
    <div className="relative">
      <button onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 rounded-xl border px-1.5 py-1.5 pr-2 transition-colors hover:bg-[rgb(var(--surface-2))]"
        aria-label="Profile">
        <span className="grid h-7 w-7 place-items-center rounded-lg bg-brand-600 text-xs font-bold text-white">OP</span>
        <ChevronDown className="hidden h-4 w-4 text-muted sm:block" />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-40 mt-2 w-56 origin-top-right animate-scale-in card p-2 shadow-soft-lg">
            <div className="flex items-center gap-3 px-2 py-2">
              <span className="grid h-9 w-9 place-items-center rounded-lg bg-brand-600 text-sm font-bold text-white">OP</span>
              <div className="min-w-0">
                <p className="truncate text-sm font-bold">Operator</p>
                <p className="truncate text-[11px] text-muted">AI Human Vision</p>
              </div>
            </div>
            <div className="my-1 border-t" />
            <Link to="/settings" onClick={() => setOpen(false)}
              className="flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium hover:bg-[rgb(var(--surface-2))]">
              <SettingsIcon className="h-4 w-4 text-muted" /> Settings
            </Link>
            <button onClick={() => { setOpen(false); nav("/login"); }}
              className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium text-brand-600 hover:bg-brand-600/[0.08]">
              <LogOut className="h-4 w-4" /> Sign out
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function titleForPath(p: string): { label: string; sub: string } {
  if (p.startsWith("/live")) return { label: "Live Cameras", sub: "Real-time RTSP streams with AI overlays" };
  if (p.startsWith("/metrics")) return { label: "Metrics", sub: "System & AI performance dashboard" };
  if (p.startsWith("/face-recognition")) return { label: "Face Recognition", sub: "Live SCRFD + ArcFace recognition, threshold & history" };
  if (p.startsWith("/employees")) return { label: "Employees", sub: "Enrolment & face recognition management" };
  if (p.startsWith("/attendance")) return { label: "Attendance", sub: "Face-recognition attendance log & summary" };
  if (p.startsWith("/events")) return { label: "Events", sub: "Recognition & system event log" };
  if (p.startsWith("/datasets")) return { label: "Electrical Dataset", sub: "Upload & validate training datasets" };
  if (p.startsWith("/training")) {
    if (p.includes("/comparison")) return { label: "Model Comparison", sub: "Trained model evaluation & selection" };
    if (/\/training\/[^/]+$/.test(p)) return { label: "Training Progress", sub: "Live training metrics & controls" };
    return { label: "Training", sub: "Configure, launch & monitor training jobs" };
  }
  if (p.startsWith("/image-analysis")) return { label: "AI Image Analysis", sub: "Upload any image — objects, colours, text, tags & summary" };
  if (p.startsWith("/image-comparison")) return { label: "Image Comparison", sub: "Detect every difference between two images" };
  if (p.startsWith("/reference-panels")) return { label: "Reference Panels", sub: "Learn a correct panel, then inspect against it" };
  if (p.startsWith("/topology")) return { label: "Topology Viewer", sub: "Component / terminal / wire electrical graph" };
  if (p.startsWith("/datasheets")) return { label: "Datasheets", sub: "OCR schematics into an expected wiring graph" };
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
