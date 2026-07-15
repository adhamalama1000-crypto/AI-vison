import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Waypoints, Cpu, CircuitBoard } from "lucide-react";
import { Card, EmptyState, Skeleton, SectionTitle, Badge } from "../components/ui/primitives";
import { api } from "../lib/api";

const WIRE_COLORS: Record<string, string> = {
  red: "#ef4444", green: "#22c55e", blue: "#3b82f6", yellow: "#eab308",
  orange: "#f97316", black: "#334155", "white/grey": "#94a3b8", grey: "#94a3b8",
  brown: "#92400e", violet: "#8b5cf6", cyan: "#06b6d4", "yellow-green": "#84cc16",
};

export default function TopologyViewer() {
  const panels = useQuery({ queryKey: ["refPanels"], queryFn: api.refPanels });
  const [selId, setSelId] = useState<number | null>(null);
  const detail = useQuery({
    queryKey: ["refPanel", selId], queryFn: () => api.refPanel(selId as number), enabled: selId != null,
  });

  const ready = (panels.data?.panels ?? []).filter((p) => p.status === "ready");
  const graph = detail.data?.graph;

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <SectionTitle title="Electrical topology" />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <select className="input max-w-[320px]" value={selId ?? ""} onChange={(e) => setSelId(e.target.value ? Number(e.target.value) : null)}>
            <option value="">Select a learned reference panel…</option>
            {ready.map((p) => <option key={p.id} value={p.id}>{p.name} {p.version}</option>)}
          </select>
          {graph && (
            <div className="flex items-center gap-2 text-sm text-muted">
              <Badge tone="blue">{graph.nodes.length} nodes</Badge>
              <Badge tone="green">{graph.edges.length} edges</Badge>
            </div>
          )}
        </div>
      </Card>

      <Card className="p-5">
        {panels.isLoading ? <Skeleton className="h-96" />
          : !ready.length ? (
            <EmptyState icon={<Waypoints className="h-10 w-10" />} title="No learned panels"
              hint="Learn a reference panel first — its component/terminal/wire graph appears here." />
          ) : selId == null ? (
            <EmptyState icon={<Waypoints className="h-10 w-10" />} title="Pick a panel" hint="Choose a learned reference panel above." />
          ) : detail.isLoading || !graph ? <Skeleton className="h-96" />
          : <GraphSVG nodes={graph.nodes} edges={graph.edges} />}
      </Card>
    </div>
  );
}

function GraphSVG({ nodes, edges }: { nodes: any[]; edges: any[] }) {
  const layout = useMemo(() => {
    const pts = nodes.filter((n) => n.x != null && n.y != null);
    if (!pts.length) return null;
    const xs = pts.map((n) => n.x), ys = pts.map((n) => n.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 40;
    const w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);
    const byId: Record<string, any> = {};
    nodes.forEach((n) => { byId[n.id] = n; });
    return { minX, minY, w, h, pad, byId };
  }, [nodes, edges]);

  if (!layout) {
    return <EmptyState icon={<Waypoints className="h-10 w-10" />} title="No positioned nodes"
      hint="This panel's graph has no spatial coordinates (component model may be absent)." />;
  }
  const { minX, minY, w, h, pad, byId } = layout;
  const vw = w + pad * 2, vh = h + pad * 2;
  const X = (x: number) => x - minX + pad;
  const Y = (y: number) => y - minY + pad;

  return (
    <div className="overflow-auto">
      <svg viewBox={`0 0 ${vw} ${vh}`} className="mx-auto w-full" style={{ maxHeight: 620, background: "rgb(var(--surface-2))", borderRadius: 12 }}>
        {edges.map((e, i) => {
          const a = byId[e.from], b = byId[e.to];
          if (!a || !b || a.x == null || b.x == null) return null;
          return <line key={i} x1={X(a.x)} y1={Y(a.y)} x2={X(b.x)} y2={Y(b.y)}
            stroke={WIRE_COLORS[e.color] ?? "#64748b"} strokeWidth={2.5} strokeOpacity={0.85} />;
        })}
        {nodes.map((n, i) => {
          if (n.x == null || n.y == null) return null;
          const isComp = n.kind === "component";
          return (
            <g key={i}>
              {isComp
                ? <rect x={X(n.x) - 9} y={Y(n.y) - 9} width={18} height={18} rx={4} fill="#8b5cf6" stroke="#fff" strokeWidth={1.5} />
                : <circle cx={X(n.x)} cy={Y(n.y)} r={5} fill="#f59e0b" stroke="#fff" strokeWidth={1.2} />}
              {isComp && n.label && (
                <text x={X(n.x) + 12} y={Y(n.y) + 4} fontSize={11} fill="rgb(var(--text))">{n.label}</text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="mt-3 flex items-center gap-4 text-xs text-muted">
        <span className="flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5 text-violet-500" /> Component</span>
        <span className="flex items-center gap-1.5"><CircuitBoard className="h-3.5 w-3.5 text-amber-500" /> Terminal</span>
        <span>Lines = wires (coloured by insulation)</span>
      </div>
    </div>
  );
}
