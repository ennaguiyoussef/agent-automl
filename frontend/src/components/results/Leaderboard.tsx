import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useChartColors } from "@/hooks/useChartColors";
import { fmtScore, fmtSeconds, prettyMetric } from "@/lib/format";
import type { LeaderboardEntry } from "@/lib/types";

interface LeaderboardProps {
  entries: LeaderboardEntry[];
  metric: string;
  bestKey: string | null;
  themeKey: string;
}

interface TipDatum {
  model_name: string;
  score: number;
  training_time_sec: number;
  model_key: string;
}

export function Leaderboard({ entries, metric, bestKey, themeKey }: LeaderboardProps) {
  const c = useChartColors(themeKey);
  const data: TipDatum[] = entries
    .filter((e) => e.primary_score !== null)
    .map((e) => ({
      model_name: e.model_name,
      score: e.primary_score as number,
      training_time_sec: e.training_time_sec,
      model_key: e.model_key,
    }));

  if (data.length === 0) {
    return <p className="text-sm text-muted-foreground">No comparable scores were produced.</p>;
  }

  const height = Math.max(120, data.length * 46);

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="eyebrow">Model comparison</span>
        <span className="font-mono text-xs text-muted-foreground">
          by {prettyMetric(metric)}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 44, bottom: 4, left: 8 }}
          barCategoryGap={12}
        >
          <XAxis
            type="number"
            domain={[0, "dataMax"]}
            tick={{ fill: c.axis, fontSize: 11 }}
            axisLine={{ stroke: c.grid }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="model_name"
            width={130}
            tick={{ fill: c.axis, fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: c.grid, opacity: 0.3 }}
            contentStyle={{
              background: c.card,
              border: `1px solid ${c.grid}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value: number, _name, item) => [
              `${fmtScore(value)}  ·  ${fmtSeconds((item?.payload as TipDatum).training_time_sec)}`,
              prettyMetric(metric),
            ]}
          />
          <Bar dataKey="score" radius={[0, 5, 5, 0]} maxBarSize={26} label={{
            position: "right",
            fill: c.axis,
            fontSize: 11,
            formatter: (v: number) => fmtScore(v),
          }}>
            {data.map((d) => (
              <Cell key={d.model_key} fill={d.model_key === bestKey ? c.primary : c.muted} fillOpacity={d.model_key === bestKey ? 1 : 0.45} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
