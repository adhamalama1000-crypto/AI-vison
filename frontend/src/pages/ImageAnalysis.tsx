import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ScanEye, Sparkles, Tag, Palette, Type, Boxes, AlertTriangle } from "lucide-react";
import { Card, Badge, EmptyState, Skeleton, SectionTitle, StatCard } from "../components/ui/primitives";
import { Dropzone, preview as toDataUrl, ProgressBar } from "../components/Dropzone";
import { api, mediaUrl } from "../lib/api";
import type { ImageDetail } from "../lib/types";
import { useInvalidate } from "../hooks/useData";
import { timeAgo } from "../lib/format";
import { useToast } from "../hooks/useToast";

export default function ImageAnalysis() {
  const { push } = useToast();
  const invalidate = useInvalidate();
  const recent = useQuery({ queryKey: ["imagesList"], queryFn: api.imagesList });

  const [file, setFile] = useState<File | null>(null);
  const [prev, setPrev] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ImageDetail | null>(null);

  const pick = async (f: File) => { setFile(f); setPrev(await toDataUrl(f)); setResult(null); };
  const clear = () => { setFile(null); setPrev(null); setResult(null); };

  const run = async () => {
    if (!file) return push("Choose an image first", "error");
    const form = new FormData();
    form.append("file", file);
    setBusy(true);
    try {
      const r = await api.analyzeImage(form);
      setResult(r);
      push("Image analysed", "success");
      invalidate("imagesList");
    } catch (e: any) { push(e.message, "error"); } finally { setBusy(false); }
  };

  const imgUrl = result ? mediaUrl(result.path) : prev;

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        {/* upload + preview with bbox overlay */}
        <Card className="p-5">
          <SectionTitle title="Upload image" />
          {result ? (
            <BoxOverlay src={imgUrl!} detail={result} />
          ) : (
            <Dropzone onFile={pick} preview={prev} onClear={clear} />
          )}
          <div className="mt-4 flex items-center gap-2">
            <button className="btn-primary" onClick={run} disabled={busy || !file}>
              <Sparkles className="h-4 w-4" /> {busy ? "Analysing…" : "Analyze with AI"}
            </button>
            {result && <button className="btn-outline" onClick={clear}>New image</button>}
          </div>
          <div className="mt-3"><ProgressBar active={busy} label="Running AI analysis…" /></div>
        </Card>

        {/* results */}
        <div className="space-y-4">
          {!result ? (
            <Card className="grid min-h-[260px] place-items-center p-6">
              <EmptyState icon={<ScanEye className="h-10 w-10" />} title="AI analysis"
                hint="Upload any image — objects, colours, text, tags and a summary appear here." />
            </Card>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatCard icon={<Boxes className="h-4 w-4" />} label="Objects" value={result.n_objects} tone="blue" />
                <StatCard icon={<Palette className="h-4 w-4" />} label="Colours" value={result.dominant_colors?.length ?? 0} tone="violet" />
                <StatCard icon={<Type className="h-4 w-4" />} label="OCR items" value={result.ocr_items?.length ?? 0} tone="amber" />
                <StatCard icon={<AlertTriangle className="h-4 w-4" />} label="Defects" value={result.analysis?.defects?.length ?? 0} tone="red" />
              </div>

              <Card className="p-5">
                <SectionTitle title="AI summary" />
                <p className="text-sm leading-relaxed">{result.summary}</p>
                {result.tags?.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {result.tags.map((t) => <Badge key={t} tone="blue"><Tag className="h-3 w-3" /> {t}</Badge>)}
                  </div>
                )}
                {result.analysis?.notes?.length > 0 && (
                  <p className="mt-3 rounded-lg surface-2 p-2 text-xs text-muted">{result.analysis.notes.join(" · ")}</p>
                )}
              </Card>

              <Card className="p-5">
                <SectionTitle title="Dominant colours" />
                <div className="flex flex-wrap gap-2">
                  {result.dominant_colors?.map((c, i) => (
                    <div key={i} className="flex items-center gap-2 rounded-lg surface-2 px-2 py-1.5">
                      <span className="h-6 w-6 rounded-md ring-1 ring-[rgb(var(--border))]" style={{ background: c.hex }} />
                      <div className="text-xs">
                        <p className="font-semibold">{c.name}</p>
                        <p className="text-faint">{c.hex} · {(c.ratio * 100).toFixed(0)}%</p>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>

              {result.objects?.length > 0 && (
                <Card className="p-5">
                  <SectionTitle title="Detected objects" />
                  <div className="space-y-1.5">
                    {result.objects.map((o, i) => (
                      <div key={i} className="flex items-center justify-between rounded-lg surface-2 px-3 py-2 text-sm">
                        <span className="font-medium">{o.label}</span>
                        <Badge tone="green">{(o.confidence * 100).toFixed(0)}%</Badge>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {result.ocr_text && (
                <Card className="p-5">
                  <SectionTitle title="Extracted text (OCR)" />
                  <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded-lg surface-2 p-3 text-xs">{result.ocr_text}</pre>
                </Card>
              )}
            </>
          )}
        </div>
      </div>

      <Card className="overflow-hidden p-0">
        <div className="border-b p-4"><h2 className="text-sm font-bold uppercase tracking-wide text-muted">Recent images</h2></div>
        {recent.isLoading ? (
          <div className="space-y-2 p-4">{Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-12" />)}</div>
        ) : !recent.data?.images.length ? (
          <EmptyState icon={<ScanEye className="h-9 w-9" />} title="No images yet" hint="Analysed images appear here." />
        ) : (
          <div className="divide-y divide-[rgb(var(--border))]">
            {recent.data.images.map((im) => (
              <div key={im.id} className="flex items-center gap-3 px-4 py-2.5 hover:bg-[rgb(var(--surface-2))]">
                <img src={mediaUrl(im.path)} className="h-10 w-10 rounded-lg object-cover ring-1 ring-[rgb(var(--border))]" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold">{im.name || `image #${im.id}`}</p>
                  <p className="text-xs text-muted">{im.width}×{im.height} · {im.n_objects} objects · {timeAgo(im.created_at)}</p>
                </div>
                <Badge tone={im.status === "analyzed" ? "green" : "gray"}>{im.status}</Badge>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

/* image with responsive SVG bounding-box overlay (scales with the image) */
function BoxOverlay({ src, detail }: { src: string; detail: ImageDetail }) {
  const w = detail.width || detail.analysis?.image_size?.[0] || 100;
  const h = detail.height || detail.analysis?.image_size?.[1] || 100;
  return (
    <div className="relative overflow-hidden rounded-2xl ring-1 ring-[rgb(var(--border))]">
      <img src={src} className="block w-full" />
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
           className="pointer-events-none absolute inset-0 h-full w-full">
        {detail.objects?.map((o, i) => (
          <g key={i}>
            <rect x={o.x1} y={o.y1} width={o.x2 - o.x1} height={o.y2 - o.y1}
                  fill="none" stroke="#2D8CDC" strokeWidth={Math.max(2, w / 300)} />
            <text x={o.x1 + 3} y={o.y1 - 4} fontSize={Math.max(11, w / 45)} fill="#2D8CDC" fontWeight="700">{o.label}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}
