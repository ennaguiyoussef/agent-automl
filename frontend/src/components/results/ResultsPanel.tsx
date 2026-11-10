import { api } from "@/lib/api";
import { fmtScore, fmtSeconds, prettyMetric } from "@/lib/format";
import type { ResultsResponse } from "@/lib/types";
import { BestModelCard } from "./BestModelCard";
import { Leaderboard } from "./Leaderboard";
import { ReviewCard } from "./ReviewCard";
import { Badge } from "@/components/ui/badge";

interface ResultsPanelProps {
  results: ResultsResponse;
  themeKey: string;
}

export function ResultsPanel({ results, themeKey }: ResultsPanelProps) {
  const { best_model, evaluation, review, experiment_id } = results;
  const metric = results.metric ?? evaluation?.metric ?? "";

  return (
    <div className="space-y-5">
      {best_model && (
        <BestModelCard
          best={best_model}
          modelUrl={api.downloadModelUrl(experiment_id)}
          reportUrl={api.downloadReportUrl(experiment_id)}
        />
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        {evaluation && (
          <div className="rounded-lg border border-border bg-card p-5">
            <Leaderboard
              entries={evaluation.leaderboard}
              metric={metric}
              bestKey={evaluation.best_model_key}
              themeKey={themeKey}
            />
          </div>
        )}
        {review && <ReviewCard review={review} />}
      </div>

      {evaluation && evaluation.results.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-border bg-card">
          <div className="border-b border-border px-4 py-2.5">
            <span className="text-sm font-medium">All candidates</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  <th className="py-2 pl-4 pr-2 font-medium">Model</th>
                  <th className="px-2 py-2 font-medium">Status</th>
                  <th className="px-2 py-2 text-right font-medium">{prettyMetric(metric)}</th>
                  <th className="px-2 py-2 text-right font-medium">CV mean</th>
                  <th className="py-2 pl-2 pr-4 text-right font-medium">Time</th>
                </tr>
              </thead>
              <tbody>
                {evaluation.results.map((r) => {
                  const isBest = r.model_key === evaluation.best_model_key;
                  return (
                    <tr key={r.model_key} className="border-t border-border/70 hover:bg-accent/40">
                      <td className="py-2 pl-4 pr-2 font-medium">
                        {r.model_name}
                        {isBest && (
                          <Badge variant="ok" className="ml-2 px-1.5 py-0 text-[10px]">
                            best
                          </Badge>
                        )}
                      </td>
                      <td className="px-2 py-2">
                        {r.status === "success" ? (
                          <span className="text-signal-ok">success</span>
                        ) : (
                          <span className="text-signal-err" title={r.error ?? ""}>
                            failed
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-2 text-right font-mono tabular-nums">
                        {fmtScore(r.primary_score)}
                      </td>
                      <td className="px-2 py-2 text-right font-mono tabular-nums text-muted-foreground">
                        {r.cv_mean?.[metric] !== undefined ? fmtScore(r.cv_mean[metric]) : "—"}
                      </td>
                      <td className="py-2 pl-2 pr-4 text-right font-mono tabular-nums text-muted-foreground">
                        {fmtSeconds(r.training_time_sec)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
