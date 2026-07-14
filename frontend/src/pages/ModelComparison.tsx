import { useParams, useSearchParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, GitCompare, CheckCircle2, ShieldCheck, ShieldOff, Trophy } from "lucide-react";
import { Card, Badge, Skeleton, EmptyState } from "../components/ui/primitives";
import { api } from "../lib/api";
import type { ComparisonEntry } from "../lib/types";
import { fmt } from "../lib/format";

function pctOrDash(v: number | null | undefined) {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}

export default function ModelComparison() {
  const params = useParams();
  const [search] = useSearchParams();
  const id = params.id ?? search.get("job") ?? "";

  const cmp = useQuery({ queryKey: ["training-comparison", id], queryFn: () => api.trainingComparison(id), enabled: Boolean(id) });

  if (!id) return <EmptyState icon={<GitCompare className="h-10 w-10" />} title="No job selected" />;
  if (cmp.isLoading || !cmp.data) return <Card className="p-5"><Skeleton className="h-72" /></Card>;

  const rows = cmp.data.comparison ?? [];
  const best = cmp.data.best_model;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link to={`/training/${id}`} className="btn-icon btn-outline"><ArrowLeft className="h-4 w-4" /></Link>
        <div className="flex-1">
          <h1 className="text-lg font-bold">Model comparison</h1>
          <p className="text-xs text-muted">Job #{id}{best ? ` · best model: ${best}` : ""}</p>
        </div>
        {best && <Badge tone="green"><Trophy className="h-3.5 w-3.5" /> {best}</Badge>}
      </div>

      <Card className="overflow-hidden p-0">
        {!rows.length ? (
          <EmptyState icon={<GitCompare className="h-10 w-10" />} title="No comparison data" hint="Comparison appears once training completes." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs font-semibold uppercase tracking-wide text-muted">
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3 text-right">Accuracy</th>
                  <th className="px-4 py-3 text-right">Precision</th>
                  <th className="px-4 py-3 text-right">Recall</th>
                  <th className="px-4 py-3 text-right">F1</th>
                  <th className="px-4 py-3 text-center">ONNX</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r: ComparisonEntry) => {
                  const skipped = r.status !== "ok" && r.status !== "completed" && r.status !== "trained";
                  const test = r.metrics?.test;
                  const onnxOk = r.onnx?.verification?.ok;
                  return (
                    <tr key={r.model} className={`border-b border-[rgb(var(--border))] last:border-0 ${r.selected ? "bg-emerald-500/10" : "hover:bg-[rgb(var(--surface-2))]"}`}>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {r.selected && <Trophy className="h-4 w-4 text-emerald-500" />}
                          <span className="font-semibold">{r.model}</span>
                          {r.selected && <Badge tone="green"><CheckCircle2 className="h-3 w-3" /> best</Badge>}
                        </div>
                        {skipped && r.reason && <p className="mt-0.5 text-xs text-muted">{r.reason}</p>}
                      </td>
                      <td className="px-4 py-3"><Badge tone={skipped ? "amber" : "green"}>{r.status}</Badge></td>
                      {skipped ? (
                        <td className="px-4 py-3 text-center text-faint" colSpan={4}>skipped — {r.reason || "not evaluated"}</td>
                      ) : (
                        <>
                          <td className="px-4 py-3 text-right font-bold tabular-nums">{pctOrDash(test?.accuracy)}</td>
                          <td className="px-4 py-3 text-right tabular-nums">{pctOrDash(test?.precision)}</td>
                          <td className="px-4 py-3 text-right tabular-nums">{pctOrDash(test?.recall)}</td>
                          <td className="px-4 py-3 text-right tabular-nums">{pctOrDash(test?.f1)}</td>
                        </>
                      )}
                      <td className="px-4 py-3 text-center">
                        {r.onnx == null ? <span className="text-faint">—</span> :
                          onnxOk ? <Badge tone="green"><ShieldCheck className="h-3 w-3" /> verified</Badge>
                                 : <Badge tone="red"><ShieldOff className="h-3 w-3" /> failed</Badge>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
      {rows.length > 0 && <p className="text-xs text-faint">{fmt(rows.length)} model(s) evaluated · metrics shown are on the held-out test split.</p>}
    </div>
  );
}
