import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CalendarCheck, Users, Clock, Trash2, Save, ImageOff } from "lucide-react";
import { Card, Badge, EmptyState, Skeleton, SectionTitle, StatCard } from "../components/ui/primitives";
import { ConfirmDialog, Dialog } from "../components/ui/Dialog";
import { SimpleBarChart } from "../components/ui/charts";
import { api, mediaUrl } from "../lib/api";
import type { AttendanceRecord } from "../lib/types";
import { fmt, clockTime, dateTime } from "../lib/format";
import { useInvalidate } from "../hooks/useData";
import { useToast } from "../hooks/useToast";

export default function Attendance() {
  const invalidate = useInvalidate();
  const { push } = useToast();
  const today = useQuery({ queryKey: ["attendance-today"], queryFn: api.attendanceToday, refetchInterval: 10000 });
  const log = useQuery({ queryKey: ["attendance-log"], queryFn: () => api.attendance("?limit=100"), refetchInterval: 10000 });
  const summary = useQuery({ queryKey: ["attendance-summary"], queryFn: () => api.attendanceSummary(7), refetchInterval: 30000 });
  const config = useQuery({ queryKey: ["attendance-config"], queryFn: api.attendanceConfig });

  const [preview, setPreview] = useState<AttendanceRecord | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [timeoutOpen, setTimeoutOpen] = useState(false);

  const t = today.data;
  const records = log.data?.attendance ?? [];
  const chartData = (summary.data?.summary ?? []).map((s) => ({ name: s.day.slice(5), value: s.present }));

  const doClear = async () => {
    setClearing(true);
    try { await api.clearAttendance(); push("Attendance cleared", "success"); invalidate("attendance-today", "attendance-log", "attendance-summary"); }
    catch (e: any) { push(e.message, "error"); }
    finally { setClearing(false); setConfirmClear(false); }
  };

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard icon={<CalendarCheck className="h-5 w-5" />} label="Present today" tone="green"
          value={today.isLoading ? "—" : fmt(t?.present)} sub={t ? `of ${fmt(t.employees_total)} employees` : undefined} />
        <StatCard icon={<Users className="h-5 w-5" />} label="Total employees" tone="blue" value={fmt(t?.employees_total)} />
        <StatCard icon={<Clock className="h-5 w-5" />} label="Recognition timeout" tone="violet"
          value={config.data ? `${fmt(config.data.timeout_seconds)}s` : "—"}
          sub={<button className="text-brand-400 hover:underline" onClick={() => setTimeoutOpen(true)}>Configure</button>} />
      </div>

      <Card className="p-5">
        <SectionTitle title="Last 7 days" />
        {summary.isLoading ? <Skeleton className="h-52" /> :
          !chartData.length ? <EmptyState title="No attendance history yet" /> :
          <SimpleBarChart data={chartData} color="#3366ff" height={220} />}
      </Card>

      <Card className="overflow-hidden p-0">
        <div className="flex items-center justify-between border-b p-4">
          <h2 className="text-sm font-bold uppercase tracking-wide text-muted">Attendance log</h2>
          <button className="btn-outline btn-sm text-rose-500" onClick={() => setConfirmClear(true)} disabled={!records.length}>
            <Trash2 className="h-3.5 w-3.5" /> Clear
          </button>
        </div>
        {log.isLoading ? (
          <div className="space-y-2 p-4">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-14" />)}</div>
        ) : !records.length ? (
          <EmptyState icon={<CalendarCheck className="h-10 w-10" />} title="No attendance yet"
            hint="Recognised employees will be logged here automatically." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs font-semibold uppercase tracking-wide text-muted">
                  <th className="px-4 py-3">Employee</th>
                  <th className="hidden px-4 py-3 sm:table-cell">Camera</th>
                  <th className="px-4 py-3">Time</th>
                  <th className="hidden px-4 py-3 md:table-cell">Day</th>
                  <th className="px-4 py-3 text-right">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {records.map((r) => (
                  <tr key={r.id} className="border-b border-[rgb(var(--border))] transition-colors last:border-0 hover:bg-[rgb(var(--surface-2))]">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        {r.snapshot ? (
                          <button onClick={() => setPreview(r)} className="h-10 w-14 shrink-0 overflow-hidden rounded-lg bg-slate-950">
                            <img src={mediaUrl(r.snapshot)} className="h-full w-full object-cover" />
                          </button>
                        ) : (
                          <div className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-brand-500/15 text-brand-400 text-xs font-bold">
                            {(r.employee_name || "?").split(" ").map((w) => w[0]).slice(0, 2).join("").toUpperCase()}
                          </div>
                        )}
                        <span className="font-semibold">{r.employee_name || "Unknown"}</span>
                      </div>
                    </td>
                    <td className="hidden px-4 py-3 text-muted sm:table-cell">{r.camera_name || "—"}</td>
                    <td className="px-4 py-3 tabular-nums">{clockTime(r.created_at)}</td>
                    <td className="hidden px-4 py-3 text-muted md:table-cell">{r.day}</td>
                    <td className="px-4 py-3 text-right">
                      {r.confidence != null ? <Badge tone="green">{(r.confidence * 100).toFixed(1)}%</Badge> : <span className="text-faint">—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Dialog open={Boolean(preview)} onClose={() => setPreview(null)} size="md"
        title={preview?.employee_name || "Attendance snapshot"}
        subtitle={preview ? `${preview.camera_name || "—"} · ${dateTime(preview.created_at)}` : ""}>
        {preview?.snapshot ? <img src={mediaUrl(preview.snapshot)} className="w-full rounded-xl" /> :
          <div className="grid h-48 place-items-center text-muted"><ImageOff className="h-8 w-8" /></div>}
      </Dialog>

      {timeoutOpen && <TimeoutDialog current={config.data?.timeout_seconds ?? 0}
        onClose={() => setTimeoutOpen(false)} onSaved={() => invalidate("attendance-config")} />}

      <ConfirmDialog open={confirmClear} onClose={() => setConfirmClear(false)} onConfirm={doClear} loading={clearing}
        title="Clear attendance" confirmLabel="Clear all" message="This permanently removes every attendance record. This cannot be undone." />
    </div>
  );
}

function TimeoutDialog({ current, onClose, onSaved }: { current: number; onClose: () => void; onSaved: () => void }) {
  const { push } = useToast();
  const [seconds, setSeconds] = useState(String(current));
  const [saving, setSaving] = useState(false);
  const save = async () => {
    const n = Number(seconds);
    if (!Number.isFinite(n) || n < 0) return push("Enter a valid number of seconds", "error");
    setSaving(true);
    try { await api.setAttendanceConfig(n); push("Timeout updated", "success"); onSaved(); onClose(); }
    catch (e: any) { push(e.message, "error"); } finally { setSaving(false); }
  };
  return (
    <Dialog open onClose={onClose} size="sm" title="Recognition timeout"
      subtitle="Minimum seconds between logging the same employee again"
      footer={<>
        <button className="btn-ghost" onClick={onClose}>Cancel</button>
        <button className="btn-primary" onClick={save} disabled={saving}><Save className="h-4 w-4" /> Save</button>
      </>}>
      <label className="label">Timeout (seconds)</label>
      <input className="input" type="number" min={0} value={seconds} onChange={(e) => setSeconds(e.target.value)} />
    </Dialog>
  );
}
