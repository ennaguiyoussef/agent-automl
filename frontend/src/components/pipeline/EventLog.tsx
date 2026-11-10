import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { clockTime } from "@/lib/format";
import type { AgentEvent } from "@/lib/types";

const STATUS_DOT: Record<string, string> = {
  running: "bg-signal-run",
  completed: "bg-signal-ok",
  failed: "bg-signal-err",
  waiting: "bg-signal-wait",
};

function dotClass(evt: AgentEvent): string {
  if (evt.type === "error") return "bg-signal-err";
  return STATUS_DOT[evt.status ?? ""] ?? "bg-muted-foreground";
}

export function EventLog({ events }: { events: AgentEvent[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  // Chat messages are surfaced in the chat panel, not the technical log.
  const rows = events.filter((e) => e.type !== "chat_message");

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [rows.length]);

  if (rows.length === 0) {
    return (
      <p className="px-1 py-6 text-center text-sm text-muted-foreground">
        Waiting for the first event…
      </p>
    );
  }

  return (
    <div className="max-h-[320px] overflow-y-auto rounded-lg border border-border bg-background/60 p-1">
      <ul className="space-y-0.5">
        {rows.map((evt, i) => (
          <li
            key={`${evt.timestamp}-${i}`}
            className="flex animate-fade-in-up items-start gap-2.5 rounded-md px-2.5 py-1.5 hover:bg-accent/40"
          >
            <span className={cn("mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full", dotClass(evt))} />
            <span className="mt-0.5 shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground">
              {clockTime(evt.timestamp)}
            </span>
            <div className="min-w-0 flex-1">
              {evt.agent && (
                <span className="mr-1.5 font-mono text-[10px] uppercase tracking-wider text-primary/80">
                  {evt.agent}
                </span>
              )}
              <span className="text-sm text-foreground/90">{evt.message}</span>
            </div>
          </li>
        ))}
      </ul>
      <div ref={endRef} />
    </div>
  );
}
