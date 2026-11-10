/** Formatting helpers shared across result and profile views. */

/** Format a metric score. Scores in [0,1] show as 3 decimals; others as 2. */
export function fmtScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (Math.abs(value) <= 1) return value.toFixed(3);
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return value.toFixed(2);
}

export function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value.toFixed(digits)}%`;
}

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.round(value).toLocaleString();
}

export function fmtSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value < 1) return `${(value * 1000).toFixed(0)} ms`;
  return `${value.toFixed(2)} s`;
}

/** "f1" -> "F1", "roc_auc" -> "ROC AUC", "r2" -> "R²". */
export function prettyMetric(metric: string | null | undefined): string {
  if (!metric) return "";
  const special: Record<string, string> = {
    r2: "R²",
    rmse: "RMSE",
    mae: "MAE",
    mse: "MSE",
    roc_auc: "ROC AUC",
    f1: "F1",
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
  };
  return special[metric] ?? metric.replace(/_/g, " ");
}

export function prettyModelName(name: string | null | undefined): string {
  return name ?? "—";
}

/** Short relative time like "3s ago", "5m ago". */
export function timeAgo(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const secs = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (secs < 5) return "just now";
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

/** Clock time HH:MM:SS for event log rows. */
export function clockTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString(undefined, { hour12: false });
}
