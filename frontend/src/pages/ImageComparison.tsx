import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  GitCompareArrows, Play, FileDown, FileJson, ImageDown, CheckCircle2,
  AlertTriangle, XCircle,
} from "lucide-react";
import { Card, Badge, EmptyState, Skeleton, SectionTitle } from "../components/ui/primitives";
import { RadialGauge } from "../components/ui/charts";
import { Dropzone, preview as toDataUrl, ProgressBar } from "../components/Dropzone";
import { api, mediaUrl } from "../lib/api";
import type { ComparisonResult } from "../lib/types";
import { useInvalidate } from "../hooks/useData";
import { timeAgo } from "../lib/format";
import { useToast } from "../hooks/useToast";

const SEV_TONE: Record<string, "red" | "amber" | "gray"> = { major: "red", minor: "amber", info: "gray" };

function StatusIcon({ s }: { s: string }) {
  if (s === "identical") return <CheckCircle2 className="h-6 w-6 text-emerald-500" />;
  if (s === "minor") return <AlertTriangle className="h-6 w-6 text-amber-500" />;
  return <XCircle className="h-6 w-6 text-red-500" />;
}

export default function ImageComparison() {
  const { push } = useToast();
  const invalidate = useInvalidate();
  const history = useQuery({ queryKey: ["imageComparisons"], queryFn: api.imageComparisons });

  const [refFile, setRefFile] = useState<File | null>(null);
  const [curFile, setCurFile] = useState<File | null>(null);
  const [refPrev, setRefPrev] = useState<string | null>(null);
  const [curPrev, setCurPrev] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ComparisonResult | null>(null);

  const pickRef = async (f: File) => { setRefFile(f); setRefPrev(await toDataUrl(f)); };
  const pickCur = async (f: File) => { setCurFile(f); setCurPrev(await toDataUrl(f)); };

  const run = async () => {
    if (!refFile || !curFile) return push("Upload both reference & current images", "error");
    const form = new FormData();
    form.append("reference", refFile);
    form.append("current", curFile);
    form.append("make_pdf", "true");
    setBusy(true);
    try {
      const r = await api.compareImages(form);
      setResult(r);
      push(`Comparison: ${r.status} (${r.similarity.toFixed(1)}%)`,
        r.status === "identical" ? "success" : r.status === "major" ? "error" : "warning");
      invalidate("imageComparisons");
    } catch (e: any) { push(e.message, "error"); } finally { setBusy(false); }
  };

  const downloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result.report ?? result, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = `comparison_${result.id}.json`; a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="p-5">
          <SectionTitle title="Reference image" />
          <Dropzone onFile={pickRef} preview={refPrev} onClear={() => { setRefFile(null); setRefPrev(null); }} />
        </Card>
        <Card className="p-5">
          <SectionTitle title="Current image" />
          <Dropzone onFile={pickCur} preview={curPrev} onClear={() => { setCurFile(null); setCurPrev(null); }} />
        </Card>
      </div>

      <div className="flex items-center gap-3">
        <button className="btn-primary" onClick={run} disabled={busy || !refFile || !curFile}>
          <Play className="h-4 w-4" /> {busy ? "Comparing…" : "Compare images"}
        </button>
        <div className="flex-1"><ProgressBar active={busy} label="Aligning, differencing & scoring…" /></div>
      </div>

      {!result ? (
        <Card className="grid min-h-[200px] place-items-center p-6">
          <EmptyState icon={<GitCompareArrows className="h-10 w-10" />} title="Image comparison"
            hint="Upload a reference and a current image — the AI aligns them and reports every difference." />
        </Card>
      ) : (
        <>
          <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
            <Card className="flex flex-col items-center justify-center p-5 text-center">
              <RadialGauge value={result.similarity} max={100} label="Similarity" unit="%" color="#2D8CDC" />
              <div className="mt-2 flex items-center gap-2">
                <StatusIcon s={result.status} />
                <span className="text-lg font-bold capitalize">{result.status}</span>
              </div>
              <p className="mt-1 text-sm text-muted">{result.n_diffs} difference(s)</p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {result.report_pdf && (
                  <a className="btn-outline btn-sm" href={mediaUrl(result.report_pdf)} target="_blank" rel="noreferrer">
                    <FileDown className="h-3.5 w-3.5" /> PDF
                  </a>
                )}
                <button className="btn-outline btn-sm" onClick={downloadJson}><FileJson className="h-3.5 w-3.5" /> JSON</button>
                {result.overlay_path && (
                  <a className="btn-outline btn-sm" href={mediaUrl(result.overlay_path)} download>
                    <ImageDown className="h-3.5 w-3.5" /> Image
                  </a>
                )}
              </div>
            </Card>

            <Card className="p-5">
              <SectionTitle title="AI report" />
              <p className="text-sm">{result.report?.summary}</p>
              <div className="mt-3 max-h-[260px] space-y-1.5 overflow-auto">
                {result.diffs.length === 0 ? (
                  <p className="rounded-lg surface-2 p-3 text-sm text-muted">No differences detected.</p>
                ) : result.diffs.map((d, i) => (
                  <div key={i} className="flex items-center gap-3 rounded-lg surface-2 px-3 py-2 text-sm">
                    <Badge tone={SEV_TONE[d.severity] ?? "gray"}>{d.diff_type.replace(/_/g, " ")}</Badge>
                    <span className="min-w-0 flex-1 truncate">{d.detail}</span>
                    {d.confidence != null && <span className="text-xs text-muted">{(d.confidence * 100).toFixed(0)}%</span>}
                  </div>
                ))}
              </div>
              {result.report?.notes?.length > 0 && (
                <p className="mt-3 rounded-lg surface-2 p-2 text-xs text-muted">{result.report.notes.join(" · ")}</p>
              )}
            </Card>
          </div>

          {result.overlay_path && (
            <Card className="p-5">
              <SectionTitle title="Side-by-side & difference overlay" />
              <img src={mediaUrl(result.overlay_path)} className="w-full rounded-xl ring-1 ring-[rgb(var(--border))]" />
            </Card>
          )}
        </>
      )}

      <Card className="overflow-hidden p-0">
        <div className="border-b p-4"><h2 className="text-sm font-bold uppercase tracking-wide text-muted">Comparison history</h2></div>
        {history.isLoading ? (
          <div className="space-y-2 p-4">{Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-12" />)}</div>
        ) : !history.data?.comparisons.length ? (
          <EmptyState icon={<GitCompareArrows className="h-9 w-9" />} title="No comparisons yet" hint="Run a comparison to see it here." />
        ) : (
          <div className="divide-y divide-[rgb(var(--border))]">
            {history.data.comparisons.map((cmp) => (
              <div key={cmp.id} className="flex items-center gap-3 px-4 py-2.5 hover:bg-[rgb(var(--surface-2))]">
                {cmp.overlay_path && <img src={mediaUrl(cmp.overlay_path)} className="h-10 w-20 rounded-md object-cover ring-1 ring-[rgb(var(--border))]" />}
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold">Comparison #{cmp.id} · {cmp.similarity.toFixed(1)}% similar</p>
                  <p className="text-xs text-muted">{cmp.n_diffs} diffs · {timeAgo(cmp.created_at)}</p>
                </div>
                <Badge tone={cmp.status === "identical" ? "green" : cmp.status === "major" ? "red" : "amber"}>{cmp.status}</Badge>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
