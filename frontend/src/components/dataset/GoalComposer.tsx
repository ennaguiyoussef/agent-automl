import { useState } from "react";
import { ArrowLeft, Rocket, Table2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { fmtInt, fmtPct } from "@/lib/format";
import type { CreateExperimentPayload } from "@/lib/api";
import type { DatasetUploadResponse } from "@/lib/types";

const METRICS = ["", "f1", "accuracy", "roc_auc", "precision", "recall", "r2", "rmse", "mae"];

interface GoalComposerProps {
  dataset: DatasetUploadResponse;
  submitting: boolean;
  onBack: () => void;
  onCreate: (payload: CreateExperimentPayload) => void;
}

export function GoalComposer({ dataset, submitting, onBack, onCreate }: GoalComposerProps) {
  const { info } = dataset;
  const [query, setQuery] = useState("");
  const [target, setTarget] = useState("");
  const [metric, setMetric] = useState("");

  const submit = () => {
    if (!query.trim()) return;
    onCreate({
      dataset_id: dataset.dataset_id,
      user_query: query.trim(),
      target_column: target || null,
      optimize_metric: metric || null,
    });
  };

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6 px-6 py-10">
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 self-start text-sm text-muted-foreground transition-colors hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        Choose a different file
      </button>

      <div>
        <div className="eyebrow mb-2">Step 2 · Describe the objective</div>
        <h2 className="font-display text-2xl font-semibold tracking-tight">
          What should the model predict?
        </h2>
      </div>

      <div className="flex items-center gap-3 rounded-lg border border-border bg-card p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
          <Table2 className="h-5 w-5" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate font-medium">{dataset.filename}</div>
          <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-xs text-muted-foreground">
            <span>{fmtInt(info.n_rows)} rows</span>
            <span>{fmtInt(info.n_columns)} cols</span>
            <span>{fmtPct(info.missing_pct)} missing</span>
            {info.duplicate_rows > 0 && <span>{fmtInt(info.duplicate_rows)} dupes</span>}
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <label htmlFor="goal" className="text-sm font-medium">
          Objective
        </label>
        <Textarea
          id="goal"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. Predict which customers are likely to churn next month"
          className="min-h-[96px]"
          autoFocus
        />
        <div className="flex flex-wrap gap-1.5">
          {[
            "Predict customer churn",
            "Estimate the sale price",
            "Classify the outcome",
          ].map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => setQuery(ex)}
              className="rounded-full border border-border px-2.5 py-0.5 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <label htmlFor="target" className="text-sm font-medium">
            Target column <span className="text-muted-foreground">· optional</span>
          </label>
          <select
            id="target"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <option value="">Let the agent decide</option>
            {info.column_names.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label htmlFor="metric" className="text-sm font-medium">
            Optimize for <span className="text-muted-foreground">· optional</span>
          </label>
          <select
            id="metric"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {METRICS.map((m) => (
              <option key={m} value={m}>
                {m === "" ? "Agent's recommendation" : m}
              </option>
            ))}
          </select>
        </div>
      </div>

      {target === "" && (
        <p className="-mt-2 text-xs text-muted-foreground">
          If the agent isn't confident about the target, it will pause and ask you to confirm.
        </p>
      )}

      <div className="flex items-center justify-between">
        <Badge variant="outline">{info.n_columns} columns available</Badge>
        <Button onClick={submit} disabled={!query.trim() || submitting} size="lg">
          <Rocket />
          {submitting ? "Launching…" : "Launch pipeline"}
        </Button>
      </div>
    </div>
  );
}
