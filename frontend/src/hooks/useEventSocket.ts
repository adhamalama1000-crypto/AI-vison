import { useEffect, useRef, useState } from "react";
import type { WSMessage } from "../lib/types";

export type ConnState = "connecting" | "live" | "offline";

interface Options { onMessage?: (m: WSMessage) => void; }

export function useEventSocket({ onMessage }: Options = {}) {
  const [state, setState] = useState<ConnState>("connecting");
  const [last, setLast] = useState<WSMessage | null>(null);
  const cbRef = useRef(onMessage);
  cbRef.current = onMessage;

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const connect = () => {
      setState("connecting");
      const proto = location.protocol === "https:" ? "wss" : "ws";
      try { ws = new WebSocket(`${proto}://${location.host}/ws/events`); }
      catch { schedule(); return; }

      ws.onopen = () => setState("live");
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as WSMessage;
          setLast(msg);
          cbRef.current?.(msg);
        } catch { /* ignore malformed */ }
      };
      ws.onclose = () => { setState("offline"); if (!closed) schedule(); };
      ws.onerror = () => { try { ws?.close(); } catch { /* ignore */ } };
    };
    const schedule = () => { if (!closed) retry = setTimeout(connect, 2000); };

    connect();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      try { ws?.close(); } catch { /* ignore */ }
    };
  }, []);

  return { state, last };
}
