import { useEffect, type ReactNode } from "react";
import { X } from "lucide-react";
import { cx } from "../../lib/format";

export function Dialog({ open, onClose, title, subtitle, children, footer, size = "md" }: {
  open: boolean; onClose: () => void; title: ReactNode; subtitle?: ReactNode;
  children: ReactNode; footer?: ReactNode; size?: "sm" | "md" | "lg" | "xl";
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.removeEventListener("keydown", onKey); document.body.style.overflow = prev; };
  }, [open, onClose]);

  if (!open) return null;
  const width = { sm: "max-w-md", md: "max-w-2xl", lg: "max-w-4xl", xl: "max-w-6xl" }[size];

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4 sm:p-6">
      <div className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm animate-fade-in" onClick={onClose} />
      <div className={cx("card relative z-10 my-auto w-full animate-scale-in overflow-hidden", width)}>
        <div className="flex items-start justify-between gap-4 border-b p-5">
          <div>
            <h3 className="text-lg font-bold leading-tight">{title}</h3>
            {subtitle && <p className="mt-0.5 text-sm text-muted">{subtitle}</p>}
          </div>
          <button onClick={onClose} className="btn-icon btn-ghost -mr-1 -mt-1"><X className="h-5 w-5" /></button>
        </div>
        <div className="max-h-[calc(100vh-15rem)] overflow-y-auto p-5">{children}</div>
        {footer && <div className="flex items-center justify-end gap-2.5 border-t p-4 surface-2">{footer}</div>}
      </div>
    </div>
  );
}

export function ConfirmDialog({ open, onClose, onConfirm, title, message, confirmLabel = "Delete", danger = true, loading }: {
  open: boolean; onClose: () => void; onConfirm: () => void; title: string; message: ReactNode;
  confirmLabel?: string; danger?: boolean; loading?: boolean;
}) {
  return (
    <Dialog open={open} onClose={onClose} title={title} size="sm"
      footer={<>
        <button className="btn-ghost" onClick={onClose}>Cancel</button>
        <button className={danger ? "btn-danger" : "btn-primary"} onClick={onConfirm} disabled={loading}>{confirmLabel}</button>
      </>}>
      <p className="text-sm text-muted">{message}</p>
    </Dialog>
  );
}
