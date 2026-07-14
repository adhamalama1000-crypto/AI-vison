import { useEffect, useRef, useState } from "react";
import { Maximize2, Minimize2, VideoOff, Loader2 } from "lucide-react";
import { streamUrl } from "../lib/api";
import { cx } from "../lib/format";
import { Dot } from "./ui/primitives";

export function CameraStream({
  cameraId, ai = false, showControls = true, badge, className, rounded = true,
}: {
  cameraId: string | null; ai?: boolean; showControls?: boolean;
  badge?: string; className?: string; rounded?: boolean;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [status, setStatus] = useState<"loading" | "playing" | "error">("loading");
  const [fs, setFs] = useState(false);

  useEffect(() => {
    if (!cameraId) return;
    setStatus("loading");
    const img = imgRef.current;
    if (img) img.src = streamUrl(cameraId, ai);
    return () => { if (img) img.src = ""; };
  }, [cameraId, ai]);

  useEffect(() => {
    const onFs = () => setFs(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  const toggleFs = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else wrapRef.current?.requestFullscreen?.();
  };

  return (
    <div ref={wrapRef} className={cx("group relative overflow-hidden bg-slate-950", rounded && "rounded-2xl", className)}>
      {!cameraId ? (
        <div className="flex aspect-video w-full flex-col items-center justify-center text-slate-500">
          <VideoOff className="mb-2 h-8 w-8" /><span className="text-sm">No camera selected</span>
        </div>
      ) : (
        <>
          <img
            ref={imgRef} alt="live camera"
            className={cx("aspect-video w-full object-contain transition-opacity", status === "playing" ? "opacity-100" : "opacity-0")}
            onLoad={() => setStatus("playing")} onError={() => setStatus("error")}
          />
          {status !== "playing" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400">
              {status === "loading"
                ? <><Loader2 className="mb-2 h-7 w-7 animate-spin" /><span className="text-sm">Connecting to stream…</span></>
                : <><VideoOff className="mb-2 h-7 w-7" /><span className="text-sm">Stream unavailable</span></>}
            </div>
          )}

          {/* top-left badges */}
          <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2">
            <span className="flex items-center gap-1.5 rounded-full bg-black/55 px-2.5 py-1 text-xs font-semibold text-white backdrop-blur">
              <Dot tone="red" pulse /> LIVE
            </span>
            {ai && <span className="rounded-full bg-brand-600/90 px-2.5 py-1 text-xs font-semibold text-white backdrop-blur">AI OVERLAY</span>}
            {badge && <span className="rounded-full bg-black/55 px-2.5 py-1 text-xs font-semibold text-white backdrop-blur">{badge}</span>}
          </div>

          {showControls && (
            <div className="absolute right-3 top-3 flex gap-2 opacity-0 transition-opacity group-hover:opacity-100">
              <button onClick={toggleFs} title="Fullscreen"
                className="grid h-9 w-9 place-items-center rounded-lg bg-black/55 text-white backdrop-blur hover:bg-black/70">
                {fs ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
