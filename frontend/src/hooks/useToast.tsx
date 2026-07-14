import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { CheckCircle2, AlertTriangle, XCircle, Info, X } from "lucide-react";
import { cx } from "../lib/format";

type Kind = "success" | "error" | "warning" | "info";
interface Toast { id: number; kind: Kind; msg: string; }
interface ToastCtx { push: (msg: string, kind?: Kind) => void; }
const Ctx = createContext<ToastCtx>({ push: () => {} });

let counter = 0;
const ICON: Record<Kind, typeof Info> = { success: CheckCircle2, error: XCircle, warning: AlertTriangle, info: Info };
const TONE: Record<Kind, string> = {
  success: "text-emerald-500", error: "text-rose-500", warning: "text-amber-500", info: "text-brand-400",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const remove = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const push = useCallback((msg: string, kind: Kind = "info") => {
    const id = ++counter;
    setToasts((t) => [...t, { id, kind, msg }]);
    setTimeout(() => remove(id), 4200);
  }, [remove]);

  return (
    <Ctx.Provider value={{ push }}>
      {children}
      <div className="fixed bottom-5 right-5 z-[100] flex flex-col gap-2.5 w-[min(92vw,380px)]">
        {toasts.map((t) => {
          const Icon = ICON[t.kind];
          return (
            <div key={t.id} className="card animate-slide-up flex items-start gap-3 p-3.5 pr-3 shadow-xl">
              <Icon className={cx("mt-0.5 h-5 w-5 shrink-0", TONE[t.kind])} />
              <p className="text-sm leading-snug flex-1">{t.msg}</p>
              <button onClick={() => remove(t.id)} className="text-faint hover:text-muted transition-colors">
                <X className="h-4 w-4" />
              </button>
            </div>
          );
        })}
      </div>
    </Ctx.Provider>
  );
}

export const useToast = () => useContext(Ctx);
