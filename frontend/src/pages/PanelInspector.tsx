import { useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle, Boxes, CircuitBoard, ClipboardList, Cpu, FileText, Gauge,
  Info, Layers, Play, ScanLine, ShieldAlert, Tag, UploadCloud, Wrench,
} from "lucide-react";
import {
  Badge, Card, EmptyState, SectionTitle, Skeleton, Spinner,
} from "../components/ui/primitives";
import { api, mediaUrl } from "../lib/api";
import type {
  BomEntry, MaintenanceNote, MissingComponent, PanelAnalyzeResult,
  PanelComponent, PanelResult,
} from "../lib/types";
import { useCameras, useInvalidate } from "../hooks/useData";
import { cx, fmt, summaryText, timeAgo } from "../lib/format";
import { useToast } from "../hooks/useToast";

/**
 * Panel Inspector — the primary workflow.
 *
 * Upload or capture a control-panel photograph and read the engineering result:
 * what every component is, what the panel is for, what it should also contain,
 * and how much of that to trust. The layout follows the order an inspector
 * actually works in: the annotated image and the verdict first, then the
 * component list, then the caveats.
 */
export default function PanelInspector() {
  const invalidate = useInvalidate();
  const { push } = useToast();
  const { data: cams } = useCameras();
  const reports = useQuery({ queryKey: ["panels"], queryFn: api.panels });

  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [cameraId, setCameraId] = useState("");
  const [running, setRunning] = useState(false);
  const [res, setRes] = useState<PanelAnalyzeResult | null>(null);

  const run = async () => {
    if (!file && !cameraId) return push("Choose an image or a camera", "error");
    const form = new FormData();
    if (file) form.append("file", file);
    else form.append("camera_id", cameraId);
    form.append("make_pdf", "true");
    setRunning(true);
    try {
      const r = await api.analyzePanel(form);
      setRes(r);
      const n = r.result?.component_total ?? 0;
      push(n ? `Inspected — ${n} component(s) recognised`
             : "Inspection complete — no components recognised", n ? "success" : "info");
      invalidate("panels");
    } catch (e: any) {
      push(e.message, "error");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="relative overflow-hidden p-5">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-din-rail opacity-60" />
        <div className="relative">
          <SectionTitle title="Inspect a control panel" />
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="label">Upload panel photograph</label>
              <button
                className="input flex items-center gap-2 text-left transition-colors hover:border-brand-500/60"
                onClick={() => fileRef.current?.click()}
              >
                <UploadCloud className="h-4 w-4 text-faint" />
                <span className="truncate">{file ? file.name : "Choose a panel photo…"}</span>
              </button>
              <input
                ref={fileRef} type="file" accept="image/*" className="hidden"
                onChange={(e) => { setFile(e.target.files?.[0] ?? null); setCameraId(""); }}
              />
              <p className="mt-1.5 text-[11px] text-faint">
                Best results: door open, one device row filling the frame.
              </p>
            </div>
            <div>
              <label className="label">…or capture from a live camera</label>
              <select
                className="input" value={cameraId}
                onChange={(e) => { setCameraId(e.target.value); setFile(null); }}
              >
                <option value="">Select a camera</option>
                {(cams?.cameras ?? []).map((c) => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
            </div>
          </div>
          <button className="mt-4 btn-primary" onClick={run} disabled={running}>
            {running
              ? <><Spinner className="h-4 w-4" /> Inspecting…</>
              : <><Play className="h-4 w-4" /> Run inspection</>}
          </button>
        </div>
      </Card>

      {running && <ScanningPlaceholder />}
      {res && !running && <InspectionResult res={res} />}

      <Card className="overflow-hidden p-0">
        <div className="border-b p-4">
          <h2 className="text-sm font-bold uppercase tracking-wide text-muted">
            Past inspections
          </h2>
        </div>
        {reports.isLoading ? (
          <div className="space-y-2 p-4">
            {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-14" />)}
          </div>
        ) : !reports.data?.panels.length ? (
          <EmptyState
            icon={<ScanLine className="h-10 w-10" />}
            title="No inspections yet"
            hint="Run an inspection to generate a report."
          />
        ) : (
          <div className="divide-y divide-[rgb(var(--border))]">
            {reports.data.panels.map((p) => (
              <div key={p.id}
                   className="flex items-center gap-3 px-4 py-3.5 transition-colors hover:bg-[rgb(var(--surface-2))]">
                <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-brand-500/15 text-brand-400">
                  <CircuitBoard className="h-5 w-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-semibold">{p.title}</p>
                  <p className="truncate text-xs text-muted">
                    {summaryText(p.summary)} · {timeAgo(p.created_at)}
                  </p>
                </div>
                <a className="btn-outline btn-sm" href={mediaUrl(p.path)}
                   target="_blank" rel="noreferrer">
                  <FileText className="h-3.5 w-3.5" /> Report
                </a>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function ScanningPlaceholder() {
  return (
    <Card className="relative overflow-hidden p-0">
      <div className="relative h-56 bg-[rgb(var(--surface-2))]">
        <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-transparent via-brand-400 to-transparent animate-sweep" />
        <div className="grid h-full place-items-center text-sm text-muted">
          <div className="flex items-center gap-2">
            <Spinner className="h-4 w-4" /> Recognising components…
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ====================================================================== */

function InspectionResult({ res }: { res: PanelAnalyzeResult }) {
  const r = res.result;
  const hasComponents = (r.component_total ?? 0) > 0;

  return (
    <div className="space-y-6 animate-slide-up">
      <div className="grid gap-6 lg:grid-cols-[1.15fr_1fr]">
        <Card className="overflow-hidden p-0">
          <div className="flex items-center justify-between gap-2 border-b px-4 py-3">
            <h2 className="text-sm font-bold uppercase tracking-wide text-muted">
              Annotated panel
            </h2>
            <div className="flex items-center gap-2">
              {res.pdf && (
                <a className="btn-outline btn-sm" href={mediaUrl(res.pdf)}
                   target="_blank" rel="noreferrer">
                  <FileText className="h-3.5 w-3.5" /> PDF
                </a>
              )}
              {res.json && (
                <a className="btn-outline btn-sm" href={mediaUrl(res.json)}
                   target="_blank" rel="noreferrer">JSON</a>
              )}
            </div>
          </div>
          <div className="bg-[rgb(var(--bg))]">
            {res.annotated
              ? <img src={mediaUrl(res.annotated)} alt="Annotated panel"
                     className="w-full object-contain" />
              : <div className="grid h-64 place-items-center text-muted">
                  No annotated image
                </div>}
          </div>
          {r.layout?.description?.length > 0 && (
            <div className="border-t px-4 py-3">
              <p className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted">
                <Layers className="h-3.5 w-3.5" /> Layout · {r.layout.rows} device row(s)
              </p>
              <ul className="space-y-0.5 text-xs text-muted">
                {r.layout.description.map((l, i) => <li key={i}>{l}</li>)}
              </ul>
            </div>
          )}
        </Card>

        <div className="space-y-6">
          <PanelVerdict r={r} />
          <MetricStrip r={r} />
        </div>
      </div>

      {r.notes?.length > 0 && <NotesCard notes={r.notes} loaded={r.component_model_loaded} />}

      {hasComponents && <BillOfMaterials bom={r.bill_of_materials} total={r.component_total} />}
      {hasComponents && <ComponentTable components={r.components} />}

      <div className="grid gap-6 lg:grid-cols-2">
        <MissingCard items={r.missing_components} />
        <MaintenanceCard items={r.maintenance_notes} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <ConfidenceCard r={r} />
        <GateCard r={r} />
      </div>

      <WireNotice r={r} />
    </div>
  );
}

/* -------------------------------- verdict ------------------------------- */

function PanelVerdict({ r }: { r: PanelResult }) {
  const p = r.panel;
  const unclassified = !p?.panel_type || p.panel_type === "unclassified";
  const pct = Math.round((p?.confidence ?? 0) * 100);

  return (
    <Card className="p-5">
      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted">
        Panel type
      </p>
      <div className="mt-1.5 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className={cx("text-2xl font-extrabold tracking-tight animate-count-up",
                          unclassified ? "text-muted" : "text-[rgb(var(--text))]")}>
          {p?.panel_type_name ?? "Unclassified Panel"}
        </h2>
        {!unclassified && (
          <Badge tone={pct >= 60 ? "green" : pct >= 35 ? "amber" : "gray"}>
            {pct}% confidence
          </Badge>
        )}
      </div>

      <ConfidenceBar value={p?.confidence ?? 0} muted={unclassified} />

      {p?.function && <p className="mt-3 text-sm leading-relaxed text-muted">{p.function}</p>}

      {r.application?.application && (
        <div className="mt-3 flex items-start gap-2 rounded-lg surface-2 p-3">
          <Gauge className="mt-0.5 h-4 w-4 shrink-0 text-signal-400" />
          <div className="text-sm">
            <span className="font-semibold capitalize">{r.application.application}</span>
            <span className="text-muted"> — likely controlled process</span>
            {r.application.evidence?.length > 0 && (
              <p className="mt-0.5 text-xs text-faint">{r.application.evidence[0]}</p>
            )}
          </div>
        </div>
      )}

      {p?.evidence?.length > 0 && (
        <div className="mt-3">
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted">
            Evidence
          </p>
          <div className="flex flex-wrap gap-1.5">
            {p.evidence.map((e, i) => (
              <span key={i}
                    className="rounded-md surface-2 px-2 py-1 text-[11px] text-muted">{e}</span>
            ))}
          </div>
        </div>
      )}

      {p?.candidates?.length > 1 && (
        <div className="mt-3 border-t pt-3">
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted">
            Alternatives considered
          </p>
          <ul className="space-y-1 text-xs text-muted">
            {p.candidates.slice(1).map((c) => (
              <li key={c.id} className="flex items-center justify-between gap-2">
                <span className="truncate">{c.name}</span>
                <span className="tabular-nums text-faint">
                  {Math.round(c.confidence * 100)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}

function ConfidenceBar({ value, muted }: { value: number; muted?: boolean }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-[rgb(var(--surface-2))]">
      <div
        className={cx("h-full rounded-full transition-[width] duration-700 ease-out",
                      muted ? "bg-[rgb(var(--faint))]"
                            : pct >= 60 ? "bg-[rgb(var(--success))]"
                            : pct >= 35 ? "bg-[rgb(var(--warning))]"
                            : "bg-[rgb(var(--faint))]")}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function MetricStrip({ r }: { r: PanelResult }) {
  const identified = (r.component_total ?? 0) - (r.confidence?.unknown ?? 0);
  const items = [
    { icon: <Boxes className="h-4 w-4" />, label: "Devices", value: fmt(r.component_total) },
    { icon: <Tag className="h-4 w-4" />, label: "Identified", value: fmt(identified) },
    { icon: <Cpu className="h-4 w-4" />, label: "Types", value: fmt(r.bill_of_materials?.length ?? 0) },
    {
      icon: <Gauge className="h-4 w-4" />, label: "Mean conf.",
      value: r.confidence?.mean != null ? `${Math.round(r.confidence.mean * 100)}%` : "—",
    },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map((it) => (
        <Card key={it.label} className="p-3.5">
          <div className="flex items-center gap-1.5 text-muted">{it.icon}
            <span className="text-[10px] font-semibold uppercase tracking-wide">{it.label}</span>
          </div>
          <p className="mt-1 text-xl font-extrabold tabular-nums animate-count-up">{it.value}</p>
        </Card>
      ))}
    </div>
  );
}

/* ------------------------------ components ----------------------------- */

const CATEGORY_TONE: Record<string, string> = {
  protection: "text-danger", switching: "text-signal-400", control: "text-brand-300",
  automation: "text-success", hmi: "text-brand-400", drives: "text-signal-300",
  power: "text-success", instrumentation: "text-signal-400", network: "text-brand-300",
  infrastructure: "text-muted", cooling: "text-muted",
};

function BillOfMaterials({ bom, total }: { bom: BomEntry[]; total: number }) {
  if (!bom?.length) return null;
  const max = Math.max(...bom.map((b) => b.quantity), 1);
  return (
    <Card className="p-5">
      <SectionTitle
        title="Component count"
        action={<span className="text-xs text-muted">{total} device(s) · {bom.length} type(s)</span>}
      />
      <div className="space-y-2.5">
        {bom.map((b) => (
          <div key={b.class_id} className="group">
            <div className="flex items-baseline justify-between gap-3">
              <span className="truncate text-sm font-medium">
                <span className={cx("mr-1.5 font-mono text-xs", CATEGORY_TONE[b.category] ?? "text-muted")}>
                  {b.category}
                </span>
                {b.name}
              </span>
              <span className="shrink-0 text-sm font-bold tabular-nums">{b.quantity}</span>
            </div>
            <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-[rgb(var(--surface-2))]">
              <div className="h-full rounded-full bg-brand-500/70 transition-[width] duration-700"
                   style={{ width: `${(b.quantity / max) * 100}%` }} />
            </div>
            {b.manufacturers?.length > 0 && (
              <p className="mt-0.5 text-[11px] text-faint">{b.manufacturers.join(", ")}</p>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
}

function ComponentTable({ components }: { components: PanelComponent[] }) {
  const [open, setOpen] = useState<number | null>(null);
  const sorted = useMemo(() => components, [components]);

  return (
    <Card className="overflow-hidden p-0">
      <div className="flex items-center justify-between gap-2 border-b px-4 py-3">
        <h2 className="text-sm font-bold uppercase tracking-wide text-muted">
          Detected components
        </h2>
        <span className="text-xs text-muted">{components.length} row(s) · click for detail</span>
      </div>
      <div className="overflow-x-auto">
        <table className="table w-full min-w-[860px]">
          <thead>
            <tr>
              <th className="w-12">#</th>
              <th>Component</th>
              <th className="w-28">Confidence</th>
              <th className="w-32">Position</th>
              <th className="w-40">Centre</th>
              <th className="w-56">Bounding box</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((c) => {
              const unknown = c.class_id === "unknown_industrial_component";
              return (
                <>
                  <tr key={c.index}
                      className="cursor-pointer"
                      onClick={() => setOpen(open === c.index ? null : c.index)}>
                    <td className="font-mono text-xs text-faint">{c.index}</td>
                    <td>
                      <p className={cx("font-semibold", unknown && "text-muted")}>{c.title}</p>
                      <p className="text-[11px] text-faint">
                        <span className={CATEGORY_TONE[c.category] ?? "text-muted"}>{c.category}</span>
                        {c.part_number ? ` · ${c.part_number}` : ""}
                        {c.identification_basis !== "detector" ? ` · ${c.identification_basis}` : ""}
                      </p>
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <span className="tabular-nums text-sm">{c.confidence_pct.toFixed(1)}%</span>
                        <div className="h-1 w-10 overflow-hidden rounded-full bg-[rgb(var(--surface-2))]">
                          <div className={cx("h-full rounded-full",
                                             c.confidence >= 0.7 ? "bg-[rgb(var(--success))]"
                                             : c.confidence >= 0.45 ? "bg-[rgb(var(--warning))]"
                                             : "bg-[rgb(var(--danger))]")}
                               style={{ width: `${c.confidence_pct}%` }} />
                        </div>
                      </div>
                    </td>
                    <td className="text-xs text-muted">
                      {c.row ? `row ${c.row}, pos ${c.row_position}` : c.position}
                    </td>
                    <td className="font-mono text-xs text-muted">
                      ({c.center[0].toFixed(0)}, {c.center[1].toFixed(0)})
                    </td>
                    <td className="font-mono text-[11px] text-faint">
                      [{c.bbox.map((v) => v.toFixed(0)).join(", ")}]
                    </td>
                  </tr>
                  {open === c.index && (
                    <tr key={`${c.index}-d`}>
                      <td colSpan={6} className="bg-[rgb(var(--surface-2))]">
                        <div className="space-y-2 px-2 py-3 text-sm">
                          <Detail label="Function" value={c.function} />
                          <Detail label="Estimated purpose" value={c.purpose} />
                          {c.manufacturer && <Detail label="Manufacturer" value={c.manufacturer} />}
                          {c.product_family && <Detail label="Product family" value={c.product_family} />}
                          {c.nameplate_text && <Detail label="Nameplate text" value={c.nameplate_text} mono />}
                          <Detail label="Mounting" value={c.mounting.join(", ")} />
                          {c.notes?.length > 0 && (
                            <div>
                              <p className="text-[11px] font-semibold uppercase tracking-wide text-muted">
                                Notes
                              </p>
                              <ul className="mt-0.5 space-y-0.5 text-xs text-muted">
                                {c.notes.map((n, i) => <li key={i}>• {n}</li>)}
                              </ul>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Detail({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  if (!value) return null;
  return (
    <div className="grid gap-0.5 sm:grid-cols-[160px_1fr] sm:gap-3">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-muted">{label}</p>
      <p className={cx("text-sm leading-relaxed", mono && "font-mono text-xs")}>{value}</p>
    </div>
  );
}

/* --------------------------- findings & caveats ------------------------- */

const SEVERITY_TONE: Record<string, "red" | "amber" | "gray"> = {
  important: "red", advisory: "amber", info: "gray",
};

function MissingCard({ items }: { items: MissingComponent[] }) {
  return (
    <Card className="p-5">
      <SectionTitle title="Possible missing components" />
      {!items?.length ? (
        <p className="text-sm text-muted">
          Every component expected for this panel type was detected.
        </p>
      ) : (
        <ul className="space-y-2.5">
          {items.map((m) => (
            <li key={m.class_id} className="flex items-start gap-2.5">
              <ShieldAlert className={cx("mt-0.5 h-4 w-4 shrink-0",
                m.severity === "important" ? "text-danger" : "text-signal-400")} />
              <div className="min-w-0">
                <p className="text-sm font-semibold">
                  {m.name}{" "}
                  <Badge tone={SEVERITY_TONE[m.severity] ?? "gray"}>{m.severity}</Badge>
                </p>
                <p className="text-xs leading-relaxed text-muted">{m.rationale}</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function MaintenanceCard({ items }: { items: MaintenanceNote[] }) {
  return (
    <Card className="p-5">
      <SectionTitle title="Potential maintenance notes" />
      {!items?.length ? (
        <p className="text-sm text-muted">
          No observations raised from the detected inventory.
        </p>
      ) : (
        <ul className="space-y-2.5">
          {items.map((n) => (
            <li key={n.code} className="flex items-start gap-2.5">
              <Wrench className={cx("mt-0.5 h-4 w-4 shrink-0",
                n.severity === "important" ? "text-danger"
                : n.severity === "advisory" ? "text-signal-400" : "text-muted")} />
              <div className="min-w-0">
                <Badge tone={SEVERITY_TONE[n.severity] ?? "gray"}>{n.severity}</Badge>
                <p className="mt-1 text-xs leading-relaxed text-muted">{n.message}</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function ConfidenceCard({ r }: { r: PanelResult }) {
  const c = r.confidence;
  const rows: [string, string][] = [
    ["Detections", fmt(c?.count)],
    ["Mean", c?.mean != null ? c.mean.toFixed(3) : "—"],
    ["Median", c?.median != null ? c.median.toFixed(3) : "—"],
    ["Range", c?.min != null && c?.max != null ? `${c.min.toFixed(2)} – ${c.max.toFixed(2)}` : "—"],
    ["Below 0.50", fmt(c?.below_0_5)],
    ["Unclassified", fmt(c?.unknown)],
  ];
  return (
    <Card className="p-5">
      <SectionTitle
        title="Confidence statistics"
        action={r.duration_ms != null
          ? <span className="text-xs text-muted">{r.duration_ms} ms</span> : undefined}
      />
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 sm:grid-cols-3">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt className="text-[11px] font-semibold uppercase tracking-wide text-muted">{k}</dt>
            <dd className="text-lg font-bold tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
      {(c?.unknown ?? 0) > 0 && (
        <p className="mt-3 flex items-start gap-2 rounded-lg surface-2 p-3 text-xs text-muted">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-signal-400" />
          {c.unknown} detection(s) are reported as <em>Unknown Industrial
          Component</em>: a device is present but the model is not confident
          enough to name it. These are reported honestly rather than guessed, and
          they are exactly the crops the detector should be retrained on.
        </p>
      )}
      {r.ocr && (
        <p className="mt-2 text-[11px] text-faint">
          Nameplate OCR engine: {r.ocr.engine ?? "none"}
          {r.ocr.item_count ? ` · ${r.ocr.item_count} text item(s)` : ""}
        </p>
      )}
    </Card>
  );
}

function GateCard({ r }: { r: PanelResult }) {
  const d = r.diagnostics;
  const reasons = Object.entries(d?.dropped_by_reason ?? {});
  return (
    <Card className="p-5">
      <SectionTitle title="Detection gate" />
      <p className="text-sm text-muted">
        <span className="font-bold text-[rgb(var(--text))] tabular-nums">
          {fmt(d?.input_count)}
        </span>{" "}raw candidate(s) →{" "}
        <span className="font-bold text-[rgb(var(--text))] tabular-nums">
          {fmt(d?.output_count)}
        </span>{" "}accepted
        {d?.suppression_rate != null && (
          <> · {Math.round(d.suppression_rate * 100)}% suppressed</>
        )}
      </p>
      {reasons.length > 0 ? (
        <ul className="mt-3 space-y-1.5">
          {reasons.map(([reason, n]) => (
            <li key={reason} className="flex items-center justify-between gap-3 text-xs">
              <span className="truncate font-mono text-muted">{reason.replace(/_/g, " ")}</span>
              <span className="shrink-0 tabular-nums text-faint">{n}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-2 text-xs text-faint">Nothing suppressed.</p>
      )}
      {(d?.relabelled_unknown ?? 0) > 0 && (
        <p className="mt-2 text-xs text-muted">
          {d.relabelled_unknown} detection(s) demoted to unknown rather than guessed.
        </p>
      )}
    </Card>
  );
}

function NotesCard({ notes, loaded }: { notes: string[]; loaded?: boolean }) {
  return (
    <Card className={cx("p-4", loaded === false && "border-signal-500/40")}>
      <div className="flex items-start gap-2.5">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-signal-400" />
        <div>
          <p className="text-sm font-semibold">
            {loaded === false ? "No trained component model is loaded" : "Inspection notes"}
          </p>
          <ul className="mt-1 space-y-1 text-xs leading-relaxed text-muted">
            {notes.map((n, i) => <li key={i}>• {n}</li>)}
          </ul>
          {loaded === false && (
            <p className="mt-2 text-xs text-muted">
              Zero components are reported rather than fabricated ones. Train and
              export a detector (see <span className="font-mono">training/electrical/README.md</span>),
              or select a zero-shot open-vocabulary backend on the AI Models page.
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}

function WireNotice({ r }: { r: PanelResult }) {
  const w = r.wire_analysis;
  if (!w || w.enabled) return null;
  return (
    <Card className="p-4">
      <div className="flex items-start gap-2.5">
        <ClipboardList className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
        <div>
          <p className="text-sm font-semibold">Wiring analysis — disabled by design</p>
          <p className="mt-1 text-xs leading-relaxed text-muted">{w.reason}</p>
        </div>
      </div>
    </Card>
  );
}
