import { AlertCircle, Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { StageState, StageStatus } from "@/lib/pipeline";

const DOT: Record<StageStatus, string> = {
  pending: "border-border bg-card",
  running: "border-signal-run bg-signal-run/20 animate-pulse-ring",
  waiting: "border-signal-wait bg-signal-wait/20",
  done: "border-signal-ok bg-signal-ok",
  error: "border-signal-err bg-signal-err/20",
};

const LABEL: Record<StageStatus, string> = {
  pending: "text-muted-foreground",
  running: "text-foreground",
  waiting: "text-foreground",
  done: "text-foreground",
  error: "text-foreground",
};

function StageIcon({ status, index }: { status: StageStatus; index: number }) {
  if (status === "done") return <Check className="h-3.5 w-3.5 text-primary-foreground" strokeWidth={3} />;
  if (status === "running")
    return <Loader2 className="h-3.5 w-3.5 animate-spin text-signal-run" strokeWidth={2.5} />;
  if (status === "waiting")
    return <span className="h-2 w-2 animate-pulse rounded-full bg-signal-wait" />;
  if (status === "error") return <AlertCircle className="h-3.5 w-3.5 text-signal-err" />;
  return <span className="font-mono text-[10px] font-medium text-muted-foreground">{index + 1}</span>;
}

export function PipelineRail({ stages }: { stages: StageState[] }) {
  return (
    <ol className="relative">
      {stages.map((stage, i) => {
        const last = i === stages.length - 1;
        const connectorDone = stage.status === "done";
        return (
          <li key={stage.key} className="relative flex gap-3.5 pb-1">
            {/* Connector + dot column */}
            <div className="relative flex w-7 flex-col items-center">
              <span
                className={cn(
                  "z-10 flex h-7 w-7 items-center justify-center rounded-full border-2 transition-colors",
                  DOT[stage.status],
                )}
              >
                <StageIcon status={stage.status} index={i} />
              </span>
              {!last && (
                <span
                  className={cn(
                    "w-0.5 flex-1 transition-colors",
                    connectorDone ? "bg-signal-ok/60" : "bg-border",
                  )}
                  style={{ minHeight: 22 }}
                />
              )}
            </div>

            {/* Text column */}
            <div className={cn("min-w-0 flex-1 pb-4", last && "pb-0")}>
              <div className="flex items-center gap-2">
                <span className={cn("text-sm font-medium leading-tight", LABEL[stage.status])}>
                  {stage.label}
                </span>
                {stage.status === "running" && (
                  <span className="font-mono text-[10px] uppercase tracking-wider text-signal-run">
                    live
                  </span>
                )}
                {stage.status === "waiting" && (
                  <span className="font-mono text-[10px] uppercase tracking-wider text-signal-wait">
                    waiting
                  </span>
                )}
              </div>
              <p className="mt-0.5 text-xs text-muted-foreground">{stage.blurb}</p>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
