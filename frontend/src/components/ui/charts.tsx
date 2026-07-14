import {
  Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
  RadialBar, RadialBarChart, PieChart, Pie, Cell, CartesianGrid,
} from "recharts";
import { useTheme } from "../../hooks/useTheme";

function axisColor(theme: string) { return theme === "dark" ? "#64748b" : "#94a3b8"; }
function gridColor(theme: string) { return theme === "dark" ? "#242d42" : "#e2e8f0"; }

export function LiveLineChart({ data, dataKey, color = "#3366ff", unit = "", height = 220, label }: {
  data: any[]; dataKey: string; color?: string; unit?: string; height?: number; label?: string;
}) {
  const { theme } = useTheme();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <defs>
          <linearGradient id={`g-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.35} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={gridColor(theme)} vertical={false} />
        <XAxis dataKey="t" tick={{ fill: axisColor(theme), fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={40} />
        <YAxis tick={{ fill: axisColor(theme), fontSize: 11 }} tickLine={false} axisLine={false} width={44} />
        <Tooltip
          contentStyle={{ background: theme === "dark" ? "#111624" : "#fff", border: `1px solid ${gridColor(theme)}`, borderRadius: 12, fontSize: 12 }}
          labelStyle={{ color: axisColor(theme) }} formatter={(v: any) => [`${v}${unit}`, label || dataKey]}
        />
        <Area type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} fill={`url(#g-${dataKey})`} isAnimationActive={false} dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function RadialGauge({ value, max = 100, label, unit = "%", color = "#3366ff", size = 150 }: {
  value: number | null; max?: number; label: string; unit?: string; color?: string; size?: number;
}) {
  const { theme } = useTheme();
  const v = value == null ? 0 : Math.max(0, Math.min(max, value));
  const pct = (v / max) * 100;
  const data = [{ name: label, value: pct, fill: color }];
  return (
    <div className="relative grid place-items-center" style={{ height: size }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadialBarChart innerRadius="72%" outerRadius="100%" data={data} startAngle={220} endAngle={-40}>
          <defs><linearGradient id={`rg-${label}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor={color} /><stop offset="100%" stopColor={color} stopOpacity={0.6} />
          </linearGradient></defs>
          <RadialBar background={{ fill: gridColor(theme) }} dataKey="value" cornerRadius={20} fill={`url(#rg-${label})`} isAnimationActive={false} />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold tabular-nums">{value == null ? "—" : Math.round(v)}<span className="text-sm text-muted">{unit}</span></span>
        <span className="mt-0.5 text-[11px] font-semibold uppercase tracking-wide text-muted">{label}</span>
      </div>
    </div>
  );
}

export function DonutChart({ data, height = 200 }: { data: { name: string; value: number; color: string }[]; height?: number }) {
  const { theme } = useTheme();
  const total = data.reduce((s, d) => s + d.value, 0);
  return (
    <div className="relative" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius="62%" outerRadius="92%" paddingAngle={3} stroke="none" isAnimationActive={false}>
            {data.map((d, i) => <Cell key={i} fill={d.color} />)}
          </Pie>
          <Tooltip contentStyle={{ background: theme === "dark" ? "#111624" : "#fff", border: `1px solid ${gridColor(theme)}`, borderRadius: 12, fontSize: 12 }} />
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <span className="text-2xl font-bold tabular-nums">{total.toLocaleString()}</span>
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">Total</span>
      </div>
    </div>
  );
}
