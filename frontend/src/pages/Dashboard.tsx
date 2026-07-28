import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Users, ScanFace, Video, Bell, Cpu, MemoryStick, Activity, ArrowRight, CheckCircle2, HelpCircle,
} from "lucide-react";
import { useDashboard, useEvents } from "../hooks/useData";
import { useEventSocket } from "../hooks/useEventSocket";
import { Card, StatCard, SectionTitle, Badge, Dot, Skeleton, EmptyState } from "../components/ui/primitives";
import { LiveLineChart, RadialGauge, DonutChart } from "../components/ui/charts";
import { fmt, timeAgo, EVENT_LABEL, STATE_TONE, cx } from "../lib/format";

interface Point { t: string; fps: number; cpu: number; ram: number; }

export default function Dashboard() {
  const { data, isLoading } = useDashboard();
  const { data: eventsData } = useEvents("?limit=8");
  const [series, setSeries] = useState<Point[]>([]);
  const cpuRef = useRef(0); const ramRef = useRef(0);

  useEffect(() => {
    if (data) { cpuRef.current = data.resources.cpu_percent ?? cpuRef.current; ramRef.current = data.resources.ram_percent ?? ramRef.current; }
  }, [data]);

  // Live FPS series driven by camera "stats" websocket messages (falls back to poll).
  useEventSocket({
    onMessage: (m) => {
      if (m.type === "stats") {
        const now = new Date();
        setSeries((s) => [...s.slice(-39), {
          t: now.toLocaleTimeString([], { minute: "2-digit", second: "2-digit" }),
          fps: Math.round((m as any).fps ?? 0),
          cpu: Math.round(cpuRef.current), ram: Math.round(ramRef.current),
        }]);
      }
    },
  });

  useEffect(() => {
    const id = setInterval(() => {
      if (!data) return;
      const now = new Date();
      setSeries((s) => {
        if (s.length && Date.now() - (s as any)._last < 2500) return s;
        const next = [...s.slice(-39), {
          t: now.toLocaleTimeString([], { minute: "2-digit", second: "2-digit" }),
          fps: Math.round(data.cameras.total_fps), cpu: Math.round(data.resources.cpu_percent ?? 0), ram: Math.round(data.resources.ram_percent ?? 0),
        }];
        (next as any)._last = Date.now();
        return next;
      });
    }, 2500);
    return () => clearInterval(id);
  }, [data]);

  if (isLoading || !data) return <DashboardSkeleton />;

  const recog = data.recognition.recognized_events;
  const unknown = data.recognition.unknown_events;
  const donut = [
    { name: "Recognized", value: recog, color: "#10b981" },
    { name: "Unknown", value: unknown, color: "#f43f5e" },
  ];
  const events = eventsData?.events ?? [];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={<Users className="h-5 w-5" />} tone="blue" label="Employees" value={fmt(data.employees.total)} sub={`${fmt(data.employees.enrolled_faces)} face vectors enrolled`} />
        <StatCard icon={<Video className="h-5 w-5" />} tone="green" label="Cameras Online" value={`${data.cameras.connected}/${data.cameras.total}`} sub={`${fmt(data.cameras.total_fps, 1)} fps combined`} />
        <StatCard icon={<CheckCircle2 className="h-5 w-5" />} tone="violet" label="Recognitions" value={fmt(recog)} sub={`${fmt(unknown)} unknown detections`} />
        <StatCard icon={<Bell className="h-5 w-5" />} tone="amber" label="Total Events" value={fmt(data.events_total)} sub="across all cameras" />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Card className="p-5 xl:col-span-2">
          <SectionTitle title="Live throughput (FPS)" action={<Badge tone="blue"><Activity className="h-3 w-3" /> real-time</Badge>} />
          {series.length ? <LiveLineChart data={series} dataKey="fps" unit=" fps" label="FPS" height={240} />
            : <div className="flex h-[240px] items-center justify-center text-sm text-muted">Waiting for telemetry…</div>}
        </Card>

        <Card className="p-5">
          <SectionTitle title="System resources" />
          <div className="grid grid-cols-2 gap-2">
            <RadialGauge value={data.resources.cpu_percent} label="CPU" color="#2D8CDC" />
            <RadialGauge value={data.resources.ram_percent} label="RAM" color="#111111" />
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 text-center text-xs">
            <div className="rounded-xl surface-2 p-2.5"><Cpu className="mx-auto mb-1 h-4 w-4 text-brand-400" /><span className="text-muted">{fmt(data.resources.cpu_percent, 0)}% used</span></div>
            <div className="rounded-xl surface-2 p-2.5"><MemoryStick className="mx-auto mb-1 h-4 w-4 text-violet-400" /><span className="text-muted">{fmt(data.resources.ram_used_mb, 0)} MB</span></div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Card className="p-5">
          <SectionTitle title="Recognition breakdown" />
          {recog + unknown > 0 ? (
            <>
              <DonutChart data={donut} />
              <div className="mt-3 space-y-2">
                {donut.map((d) => (
                  <div key={d.name} className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full" style={{ background: d.color }} />{d.name}</span>
                    <span className="font-semibold tabular-nums">{fmt(d.value)}</span>
                  </div>
                ))}
              </div>
            </>
          ) : <EmptyState title="No recognition events yet" hint="Enrol an employee and stand in front of a camera." />}
        </Card>

        <Card className="p-5">
          <SectionTitle title="AI tasks" />
          <div className="space-y-2.5">
            {Object.entries(data.ai_tasks).map(([task, t]) => (
              <div key={task} className="flex items-center justify-between rounded-xl surface-2 px-3.5 py-3">
                <div>
                  <p className="text-sm font-semibold capitalize">{task}</p>
                  <p className="text-[11px] text-muted">{t.backend}</p>
                </div>
                <Badge tone={t.enabled ? (t.ready ? "green" : "amber") : "gray"}>
                  <Dot tone={t.enabled ? (t.ready ? "green" : "amber") : "gray"} />
                  {/* A disabled task is "off" whether or not its backend loaded —
                      reporting a loaded-but-disabled backend as "ready" reads as
                      though it were running. */}
                  {t.enabled ? (t.ready ? "running" : "loading") : "off"}
                </Badge>
              </div>
            ))}
          </div>
        </Card>

        <Card className="p-5">
          <SectionTitle title="Recent events" action={<Link to="/events" className="text-xs font-semibold text-brand-400 hover:underline">View all <ArrowRight className="inline h-3 w-3" /></Link>} />
          {events.length ? (
            <div className="space-y-2">
              {events.map((e) => (
                <div key={e.id} className="flex items-center gap-3 rounded-xl surface-2 px-3 py-2.5">
                  <div className={cx("grid h-8 w-8 shrink-0 place-items-center rounded-lg",
                    e.type === "face_recognized" ? "bg-emerald-500/15 text-emerald-500" : e.type === "unknown_person" ? "bg-rose-500/15 text-rose-500" : "bg-brand-500/15 text-brand-400")}>
                    {e.type === "face_recognized" ? <ScanFace className="h-4 w-4" /> : e.type === "unknown_person" ? <HelpCircle className="h-4 w-4" /> : <Bell className="h-4 w-4" />}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">{e.label || EVENT_LABEL[e.type] || e.type}</p>
                    <p className="text-[11px] text-muted">{e.camera_name || e.camera_id || "—"} · {timeAgo(e.created_at)}</p>
                  </div>
                  {e.confidence != null && <span className="text-xs font-semibold tabular-nums text-muted">{(e.confidence * 100).toFixed(0)}%</span>}
                </div>
              ))}
            </div>
          ) : <EmptyState title="No events yet" />}
        </Card>
      </div>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">{[0,1,2,3].map((i) => <Skeleton key={i} className="h-32" />)}</div>
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3"><Skeleton className="h-80 xl:col-span-2" /><Skeleton className="h-80" /></div>
    </div>
  );
}
