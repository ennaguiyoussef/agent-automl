import { Plus, Sparkles, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { statusMeta } from "@/lib/status";
import { timeAgo } from "@/lib/format";
import type { ExperimentSnapshot, HealthResponse } from "@/lib/types";
import { Logo } from "./Logo";

interface SidebarProps {
  experiments: ExperimentSnapshot[];
  activeId: string | null;
  health: HealthResponse | null;
  onNew: () => void;
  onSelect: (id: string) => void;
}

export function Sidebar({ experiments, activeId, health, onNew, onSelect }: SidebarProps) {
  return (
    <aside className="flex h-full w-full flex-col gap-5 border-r border-border bg-card/40 p-4">
      <div className="flex items-center gap-2.5 px-1 pt-1">
        <Logo />
        <div className="leading-tight">
          <div className="font-display text-[15px] font-semibold tracking-tight">Agentic AutoML</div>
          <div className="eyebrow">pipeline · v0.1</div>
        </div>
      </div>

      <Button onClick={onNew} className="w-full justify-start">
        <Plus />
        New experiment
      </Button>

      <div className="flex min-h-0 flex-1 flex-col">
        <div className="eyebrow mb-2 px-1">Experiments</div>
        {experiments.length === 0 ? (
          <p className="px-1 text-sm text-muted-foreground">
            No runs yet. Upload a dataset to start.
          </p>
        ) : (
          <nav className="-mx-1 flex-1 space-y-1 overflow-y-auto pr-1">
            {experiments.map((exp) => {
              const meta = statusMeta(exp.status);
              const active = exp.id === activeId;
              return (
                <button
                  key={exp.id}
                  onClick={() => onSelect(exp.id)}
                  className={cn(
                    "group w-full rounded-md border px-3 py-2 text-left transition-colors",
                    active
                      ? "border-primary/40 bg-accent"
                      : "border-transparent hover:border-border hover:bg-accent/50",
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-medium">
                      {exp.user_query || "Untitled run"}
                    </span>
                    <Badge variant={meta.variant} className="shrink-0 px-1.5 py-0 text-[10px]">
                      {meta.label}
                    </Badge>
                  </div>
                  <div className="mt-1 flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
                    <span>{exp.problem_type ?? "—"}</span>
                    <span aria-hidden>·</span>
                    <span>{timeAgo(exp.updated_at)}</span>
                  </div>
                </button>
              );
            })}
          </nav>
        )}
      </div>

      <div className="rounded-md border border-border bg-background/60 p-3">
        <div className="flex items-center gap-2 text-xs">
          {health?.llm_available ? (
            <>
              <Sparkles className="h-3.5 w-3.5 text-signal-ok" />
              <span className="text-muted-foreground">
                LLM online · <span className="font-mono text-foreground">{health.llm_provider}</span>
              </span>
            </>
          ) : (
            <>
              <Zap className="h-3.5 w-3.5 text-signal-wait" />
              <span className="text-muted-foreground">Heuristic mode (no LLM key)</span>
            </>
          )}
        </div>
      </div>
    </aside>
  );
}
