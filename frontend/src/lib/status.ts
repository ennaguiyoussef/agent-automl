import type { BadgeProps } from "@/components/ui/badge";
import type { ExperimentStatus } from "./types";

type Meta = { label: string; variant: NonNullable<BadgeProps["variant"]> };

const MAP: Record<ExperimentStatus, Meta> = {
  created: { label: "Queued", variant: "secondary" },
  running: { label: "Running", variant: "run" },
  awaiting_human: { label: "Needs input", variant: "wait" },
  completed: { label: "Completed", variant: "ok" },
  failed: { label: "Failed", variant: "err" },
};

export function statusMeta(status: ExperimentStatus): Meta {
  return MAP[status] ?? { label: status, variant: "secondary" };
}
