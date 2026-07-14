import {
  Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
  RadialBar, RadialBarChart, PieChart, Pie, Cell, CartesianGrid,
  Bar, BarChart, Line, LineChart, Legend,
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

export function SimpleBarChart({ data, dataKey = "value", xKey = "name", color = "#3366ff", height = 220, unit = "" }: {
  data: any[]; dataKey?: string; xKey?: string; color?: string; height?: number; unit?: string;
}) {
  const { theme } = useTheme();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={gridColor(theme)} vertical={false} />
        <XAxis dataKey={xKey} tick={{ fill: axisColor(theme), fontSize: 11 }} tickLine={false} axisLine={false} interval={0} angle={data.length > 8 ? -30 : 0} textAnchor={data.length > 8 ? "end" : "middle"} height={data.length > 8 ? 52 : 30} />
        <YAxis tick={{ fill: axisColor(theme), fontSize: 11 }} tickLine={false} axisLine={false} width={44} allowDecimals={false} />
        <Tooltip cursor={{ fill: gridColor(theme), opacity: 0.35 }}
          contentStyle={{ background: theme === "dark" ? "#111624" : "#fff", border: `1px solid ${gridColor(theme)}`, borderRadius: 12, fontSize: 12 }}
          formatter={(v: any) => [`${v}${unit}`, dataKey]} />
        <Bar dataKey={dataKey} fill={color} radius={[6, 6, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function MultiLineChart({ data, series, height = 240, xKey = "t" }: {
  data: any[]; series: { key: string; color: string; label?: string }[]; height?: number; xKey?: string;
}) {
  const { theme } = useTheme();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={gridColor(theme)} vertical={false} />
        <XAxis dataKey={xKey} tick={{ fill: axisColor(theme), fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={30} />
        <YAxis tick={{ fill: axisColor(theme), fontSize: 11 }} tickLine={false} axisLine={false} width={44} />
        <Tooltip contentStyle={{ background: theme === "dark" ? "#111624" : "#fff", border: `1px solid ${gridColor(theme)}`, borderRadius: 12, fontSize: 12 }}
          labelStyle={{ color: axisColor(theme) }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((s) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.label || s.key} stroke={s.color} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
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
