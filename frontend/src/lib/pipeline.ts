import type { AgentEvent, EventType, ExperimentStatus } from "./types";

export type StageStatus = "pending" | "running" | "waiting" | "done" | "error";

export interface StageDef {
  key: string;
  label: string;
  blurb: string;
  /** Events that indicate this stage has begun. */
  startEvents: EventType[];
  /** Events that indicate this stage is finished. */
  doneEvents: EventType[];
}

// The 10 LangGraph nodes, in execution order. Labels/blurbs are user-facing.
export const STAGES: StageDef[] = [
  {
    key: "planner",
    label: "Planner",
    blurb: "Reads your goal and drafts an ML plan",
    startEvents: ["planner_started"],
    doneEvents: ["planner_completed"],
  },
  {
    key: "dataset_analyzer",
    label: "Dataset analysis",
    blurb: "Profiles columns, types, and quality",
    startEvents: ["dataset_analysis_started"],
    doneEvents: ["dataset_analysis_completed"],
  },
  {
    key: "eda",
    label: "Exploratory analysis",
    blurb: "Distributions, correlations, class balance",
    startEvents: ["eda_started"],
    doneEvents: ["eda_completed"],
  },
  {
    key: "target_confirmation",
    label: "Target confirmation",
    blurb: "Confirms what to predict",
    startEvents: ["human_input_required"],
    doneEvents: ["human_input_received"],
  },
  {
    key: "preprocessing",
    label: "Preprocessing",
    blurb: "Builds a leakage-safe transform pipeline",
    startEvents: ["preprocessing_started"],
    doneEvents: ["preprocessing_completed"],
  },
  {
    key: "model_selection",
    label: "Model selection",
    blurb: "Picks candidate algorithms",
    startEvents: [],
    doneEvents: ["model_selection_completed"],
  },
  {
    key: "training",
    label: "Training",
    blurb: "Fits and cross-validates each candidate",
    startEvents: ["training_started"],
    doneEvents: [],
  },
  {
    key: "evaluation",
    label: "Evaluation",
    blurb: "Scores models on held-out data",
    startEvents: [],
    doneEvents: ["evaluation_completed"],
  },
  {
    key: "model_reviewer",
    label: "Model review",
    blurb: "Interprets metrics and caveats",
    startEvents: ["reviewer_started"],
    doneEvents: ["reviewer_completed"],
  },
  {
    key: "artifact_generation",
    label: "Artifacts",
    blurb: "Serializes the model and report",
    startEvents: [],
    doneEvents: ["artifact_created"],
  },
];

export interface StageState extends StageDef {
  status: StageStatus;
}

/**
 * Derive per-stage status from the event backlog + current experiment status.
 * "Done" propagates backward (a later completion implies earlier stages
 * finished), which lets start-less stages resolve correctly and keeps the rail
 * monotonic even if an event is missed.
 */
export function deriveStages(
  events: AgentEvent[],
  status: ExperimentStatus | undefined,
): StageState[] {
  const seen = new Set(events.map((e) => e.type));
  const n = STAGES.length;

  const doneMask = STAGES.map((s) => s.doneEvents.some((e) => seen.has(e)));
  const startMask = STAGES.map((s) => s.startEvents.some((e) => seen.has(e)));

  // If completed, every stage is done.
  if (status === "completed" || seen.has("experiment_completed")) {
    return STAGES.map((s) => ({ ...s, status: "done" as StageStatus }));
  }

  // Propagate "done" backward from the furthest completed stage.
  let maxDone = -1;
  for (let i = 0; i < n; i++) if (doneMask[i]) maxDone = i;
  for (let i = 0; i <= maxDone; i++) doneMask[i] = true;

  const frontier = maxDone + 1; // first not-yet-done stage
  const waitingTarget = seen.has("human_input_required") && !seen.has("human_input_received");
  const running = status === "running" || status === "awaiting_human";

  return STAGES.map((s, i) => {
    let st: StageStatus = "pending";
    if (doneMask[i]) st = "done";
    else if (s.key === "target_confirmation" && waitingTarget) st = "waiting";
    else if (i === frontier && running) st = "running";
    else if (startMask[i] && running) st = "running";

    // On failure, the frontier stage is where it broke.
    if (status === "failed" && i === frontier) st = "error";
    return { ...s, status: st };
  });
}

/** Overall completion fraction 0–100 for the header progress bar. */
export function stageProgress(stages: StageState[]): number {
  const done = stages.filter((s) => s.status === "done").length;
  const running = stages.filter((s) => s.status === "running" || s.status === "waiting").length;
  return ((done + running * 0.5) / stages.length) * 100;
}
