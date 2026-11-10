import { AlertTriangle, Hash, Sigma, Table2, ToggleLeft, Type as TypeIcon } from "lucide-react";
import type { ComponentType } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { fmtInt, fmtPct } from "@/lib/format";
import type { ColumnInfo, DatasetInfo, InferredType } from "@/lib/types";

const TYPE_ICON: Record<InferredType, ComponentType<{ className?: string }>> = {
  numeric: Sigma,
  categorical: TypeIcon,
  boolean: ToggleLeft,
  datetime: Hash,
  text: TypeIcon,
};

function StatTile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="eyebrow">{label}</div>
      <div className="mt-1 font-display text-2xl font-semibold tabular-nums">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}

function ColumnRow({ col }: { col: ColumnInfo }) {
  const Icon = TYPE_ICON[col.inferred_type] ?? TypeIcon;
  return (
    <tr className="border-t border-border/70 hover:bg-accent/40">
      <td className="max-w-[220px] truncate py-2 pl-3 pr-2 font-medium">{col.name}</td>
      <td className="px-2 py-2">
        <span className="inline-flex items-center gap-1.5 font-mono text-xs text-muted-foreground">
          <Icon className="h-3.5 w-3.5" />
          {col.inferred_type}
        </span>
      </td>
      <td className="px-2 py-2 text-right tabular-nums">
        <span className={cn(col.missing_pct > 20 && "text-signal-wait")}>
          {fmtPct(col.missing_pct)}
        </span>
      </td>
      <td className="px-2 py-2 text-right tabular-nums text-muted-foreground">{fmtInt(col.unique)}</td>
      <td className="py-2 pl-2 pr-3">
        <div className="flex flex-wrap justify-end gap-1">
          {col.is_constant && <Badge variant="wait" className="px-1.5 py-0 text-[10px]">constant</Badge>}
          {col.is_high_cardinality && (
            <Badge variant="outline" className="px-1.5 py-0 text-[10px]">high-card</Badge>
          )}
        </div>
      </td>
    </tr>
  );
}

export function DatasetProfile({ info }: { info: DatasetInfo }) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile label="Rows" value={fmtInt(info.n_rows)} />
        <StatTile
          label="Columns"
          value={fmtInt(info.n_columns)}
          hint={`${info.numeric_columns.length} num · ${info.categorical_columns.length} cat`}
        />
        <StatTile
          label="Missing"
          value={fmtPct(info.missing_pct)}
          hint={`${fmtInt(info.total_missing)} cells`}
        />
        <StatTile
          label="Duplicates"
          value={fmtInt(info.duplicate_rows)}
          hint={`${info.memory_usage_mb.toFixed(1)} MB in memory`}
        />
      </div>

      {(info.constant_columns.length > 0 || info.high_cardinality_columns.length > 0) && (
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-signal-wait/30 bg-signal-wait/5 px-3 py-2 text-sm">
          <AlertTriangle className="h-4 w-4 shrink-0 text-signal-wait" />
          <span className="text-muted-foreground">
            {info.constant_columns.length > 0 &&
              `${info.constant_columns.length} constant column(s) will be dropped. `}
            {info.high_cardinality_columns.length > 0 &&
              `${info.high_cardinality_columns.length} high-cardinality column(s) flagged.`}
          </span>
        </div>
      )}

      <div className="overflow-hidden rounded-lg border border-border bg-card">
        <div className="flex items-center gap-2 border-b border-border px-3 py-2.5">
          <Table2 className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Columns</span>
          <Badge variant="outline" className="ml-auto">{info.columns.length}</Badge>
        </div>
        <div className="max-h-[420px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card">
              <tr className="text-left font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                <th className="py-2 pl-3 pr-2 font-medium">Name</th>
                <th className="px-2 py-2 font-medium">Type</th>
                <th className="px-2 py-2 text-right font-medium">Missing</th>
                <th className="px-2 py-2 text-right font-medium">Unique</th>
                <th className="py-2 pl-2 pr-3 text-right font-medium">Flags</th>
              </tr>
            </thead>
            <tbody>
              {info.columns.map((c) => (
                <ColumnRow key={c.name} col={c} />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
