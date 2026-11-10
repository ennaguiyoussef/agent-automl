import { Award, Download, FileJson, Timer } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { fmtScore, fmtSeconds, prettyMetric } from "@/lib/format";
import type { BestModel } from "@/lib/types";

interface BestModelCardProps {
  best: BestModel;
  modelUrl: string;
  reportUrl: string;
}

export function BestModelCard({ best, modelUrl, reportUrl }: BestModelCardProps) {
  const primary = best.metric;
  const cvMean = best.cv_mean?.[primary];
  const cvStd = best.cv_std?.[primary];
  const secondary = Object.entries(best.metrics).filter(([k]) => k !== primary);

  return (
    <div className="overflow-hidden rounded-lg border border-primary/30 bg-gradient-to-br from-primary/[0.07] to-transparent">
      <div className="flex flex-col gap-5 p-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
            <Award className="h-5.5 w-5.5" />
          </div>
          <div>
            <div className="eyebrow !text-primary">Best model</div>
            <h3 className="font-display text-xl font-semibold tracking-tight">{best.model_name}</h3>
            <div className="mt-1 flex items-center gap-2 font-mono text-xs text-muted-foreground">
              <Timer className="h-3.5 w-3.5" />
              trained in {fmtSeconds(best.training_time_sec)}
            </div>
          </div>
        </div>

        <div className="text-right">
          <div className="font-display text-4xl font-semibold tabular-nums text-primary">
            {fmtScore(best.score)}
          </div>
          <div className="eyebrow mt-0.5">{prettyMetric(primary)}</div>
          {cvMean !== undefined && (
            <div className="mt-1 font-mono text-[11px] text-muted-foreground">
              CV {fmtScore(cvMean)}
              {cvStd !== undefined && ` ± ${fmtScore(cvStd)}`}
            </div>
          )}
        </div>
      </div>

      {secondary.length > 0 && (
        <div className="flex flex-wrap gap-2 border-t border-border/60 px-5 py-3">
          {secondary.map(([k, v]) => (
            <Badge key={k} variant="secondary" className="font-mono">
              <span className="text-muted-foreground">{prettyMetric(k)}</span>
              <span className={cn("ml-1 tabular-nums")}>{fmtScore(v)}</span>
            </Badge>
          ))}
        </div>
      )}

      <div className="flex flex-wrap gap-2 border-t border-border/60 bg-card/40 px-5 py-3">
        <Button asChild>
          <a href={modelUrl} download>
            <Download />
            Download model (.pkl)
          </a>
        </Button>
        <Button asChild variant="outline">
          <a href={reportUrl} download>
            <FileJson />
            Download report
          </a>
        </Button>
      </div>
    </div>
  );
}
