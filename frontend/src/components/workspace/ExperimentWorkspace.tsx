import { useEffect, useMemo, useState } from "react";
import { Activity, Database, LineChart } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { statusMeta } from "@/lib/status";
import { prettyMetric } from "@/lib/format";
import { deriveStages, stageProgress } from "@/lib/pipeline";
import type { AgentEvent, DatasetInfo, DatasetUploadResponse, ExperimentSnapshot, ResultsResponse } from "@/lib/types";
import { PipelineRail } from "@/components/pipeline/PipelineRail";
import { EventLog } from "@/components/pipeline/EventLog";
import { TargetConfirm } from "@/components/hitl/TargetConfirm";
import { DatasetProfile } from "@/components/dataset/DatasetProfile";
import { ResultsPanel } from "@/components/results/ResultsPanel";
import { ErrorState } from "@/components/states/ErrorState";

type Tab = "pipeline" | "dataset" | "results";

interface WorkspaceProps {
  snapshot: ExperimentSnapshot;
  results: ResultsResponse | null;
  events: AgentEvent[];
  connected: boolean;
  cachedDataset: DatasetUploadResponse | null;
  themeKey: string;
  submittingHuman: boolean;
  onConfirmTarget: (target: string) => void;
}

function useDatasetInfo(
  events: AgentEvent[],
  cached: DatasetUploadResponse | null,
): DatasetInfo | null {
  return useMemo(() => {
    const evt = [...events].reverse().find((e) => e.type === "dataset_analysis_completed");
    const fromEvent = evt?.data?.dataset_info as DatasetInfo | undefined;
    return fromEvent ?? cached?.info ?? null;
  }, [events, cached]);
}

export function ExperimentWorkspace(props: WorkspaceProps) {
  const { snapshot, results, events, connected, cachedDataset, themeKey, submittingHuman, onConfirmTarget } =
    props;

  const stages = useMemo(() => deriveStages(events, snapshot.status), [events, snapshot.status]);
  const progress = stageProgress(stages);
  const datasetInfo = useDatasetInfo(events, cachedDataset);
  const meta = statusMeta(snapshot.status);

  const [tab, setTab] = useState<Tab>("pipeline");
  const hasResults = Boolean(results && (results.best_model || results.evaluation));

  // Jump to results the moment they land.
  const [autojumped, setAutojumped] = useState(false);
  useEffect(() => {
    if (hasResults && !autojumped) {
      setTab("results");
      setAutojumped(true);
    }
  }, [hasResults, autojumped]);
  useEffect(() => {
    setAutojumped(false);
    setTab("pipeline");
  }, [snapshot.id]);

  const tabs: { id: Tab; label: string; icon: typeof Activity; disabled?: boolean }[] = [
    { id: "pipeline", label: "Pipeline", icon: Activity },
    { id: "dataset", label: "Dataset", icon: Database, disabled: !datasetInfo },
    { id: "results", label: "Results", icon: LineChart, disabled: !hasResults },
  ];

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-5 px-6 py-6">
      {/* Header */}
      <header className="space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="eyebrow mb-1">Experiment</div>
            <h1 className="font-display text-2xl font-semibold leading-tight tracking-tight">
              {snapshot.user_query}
            </h1>
          </div>
          <div className="flex items-center gap-2">
            {connected && snapshot.status === "running" && (
              <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-signal-run">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-signal-run" />
                streaming
              </span>
            )}
            <Badge variant={meta.variant}>{meta.label}</Badge>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 font-mono text-xs">
          <Chip label="target" value={snapshot.target_column} />
          <Chip label="problem" value={snapshot.problem_type} />
          <Chip label="metric" value={prettyMetric(snapshot.metric)} />
        </div>

        {snapshot.status !== "completed" && snapshot.status !== "failed" && (
          <Progress value={progress} />
        )}
      </header>

      {snapshot.status === "awaiting_human" && snapshot.pending_human && (
        <TargetConfirm
          request={snapshot.pending_human}
          submitting={submittingHuman}
          onConfirm={onConfirmTarget}
        />
      )}

      {snapshot.status === "failed" && (
        <ErrorState
          title="The pipeline stopped"
          message={snapshot.error ?? "An unexpected error occurred during the run."}
        />
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-border">
        {tabs.map((t) => (
          <button
            key={t.id}
            disabled={t.disabled}
            onClick={() => setTab(t.id)}
            className={cn(
              "flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors",
              tab === t.id
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground",
              t.disabled && "cursor-not-allowed opacity-40 hover:text-muted-foreground",
            )}
          >
            <t.icon className="h-4 w-4" />
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "pipeline" && (
        <div className="grid gap-5 md:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <div className="rounded-lg border border-border bg-card p-5">
            <div className="eyebrow mb-4">Agent pipeline</div>
            <PipelineRail stages={stages} />
          </div>
          <div>
            <div className="eyebrow mb-2">Activity</div>
            <EventLog events={events} />
          </div>
        </div>
      )}

      {tab === "dataset" &&
        (datasetInfo ? (
          <DatasetProfile info={datasetInfo} />
        ) : (
          <DatasetSkeleton />
        ))}

      {tab === "results" &&
        (results ? (
          <ResultsPanel results={results} themeKey={themeKey} />
        ) : (
          <DatasetSkeleton />
        ))}
    </div>
  );
}

function Chip({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-card px-2 py-1">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-foreground">{value || "—"}</span>
    </span>
  );
}

function DatasetSkeleton() {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-20" />
        ))}
      </div>
      <Skeleton className="h-64" />
    </div>
  );
}
