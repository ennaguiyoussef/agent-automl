// TypeScript mirror of the backend Pydantic schemas (app/schemas/*).
// Keep field names identical to the JSON the API returns.

export type ProblemType = "classification" | "regression";

export type ExperimentStatus =
  | "created"
  | "running"
  | "awaiting_human"
  | "completed"
  | "failed";

export type InferredType = "numeric" | "categorical" | "boolean" | "datetime" | "text";

export type EventType =
  | "experiment_started"
  | "dataset_loaded"
  | "planner_started"
  | "planner_completed"
  | "dataset_analysis_started"
  | "dataset_analysis_completed"
  | "eda_started"
  | "eda_completed"
  | "human_input_required"
  | "human_input_received"
  | "preprocessing_started"
  | "preprocessing_completed"
  | "model_selection_completed"
  | "training_started"
  | "model_training"
  | "model_completed"
  | "evaluation_completed"
  | "reviewer_started"
  | "reviewer_completed"
  | "artifact_created"
  | "experiment_completed"
  | "chat_message"
  | "log"
  | "error";

export interface AgentEvent {
  type: EventType;
  experiment_id: string;
  timestamp: string;
  agent: string | null;
  status: string | null;
  message: string | null;
  data: Record<string, unknown>;
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  inferred_type: InferredType;
  missing: number;
  missing_pct: number;
  unique: number;
  unique_pct: number;
  is_constant: boolean;
  is_high_cardinality: boolean;
  sample_values: unknown[];
}

export interface DatasetInfo {
  n_rows: number;
  n_columns: number;
  column_names: string[];
  columns: ColumnInfo[];
  numeric_columns: string[];
  categorical_columns: string[];
  datetime_columns: string[];
  boolean_columns: string[];
  constant_columns: string[];
  high_cardinality_columns: string[];
  duplicate_rows: number;
  total_missing: number;
  missing_pct: number;
  memory_usage_mb: number;
  dtypes: Record<string, string>;
  head: Record<string, unknown>[];
}

export interface DatasetUploadResponse {
  dataset_id: string;
  filename: string;
  info: DatasetInfo;
}

export interface PendingHumanRequest {
  reason: string;
  question: string;
  options: string[];
  suggested: string | null;
}

export interface ExperimentSnapshot {
  id: string;
  status: ExperimentStatus;
  user_query: string;
  dataset_id: string | null;
  target_column: string | null;
  problem_type: ProblemType | null;
  metric: string | null;
  created_at: string;
  updated_at: string;
  error: string | null;
  has_results: boolean;
  pending_human: PendingHumanRequest | null;
  plan: Record<string, unknown> | null;
}

export interface ChatResponse {
  reply: string;
  intent: string;
  actions: string[];
}

export interface ModelResult {
  model_key: string;
  model_name: string;
  status: string;
  metrics: Record<string, number>;
  train_metrics: Record<string, number>;
  cv_mean: Record<string, number>;
  cv_std: Record<string, number>;
  training_time_sec: number;
  primary_metric: string | null;
  primary_score: number | null;
  error: string | null;
}

export interface LeaderboardEntry {
  model_key: string;
  model_name: string;
  primary_score: number | null;
  training_time_sec: number;
  rank: number;
}

export interface EvaluationReport {
  problem_type: ProblemType;
  metric: string;
  best_model_key: string | null;
  best_model_name: string | null;
  best_score: number | null;
  results: ModelResult[];
  leaderboard: LeaderboardEntry[];
  explanation: string;
}

export interface ModelReview {
  recommendation: string;
  explanation: string;
  warnings: string[];
  limitations: string[];
}

export interface BestModel {
  model_key: string;
  model_name: string;
  metric: string;
  score: number | null;
  metrics: Record<string, number>;
  train_metrics: Record<string, number>;
  cv_mean: Record<string, number>;
  cv_std: Record<string, number>;
  training_time_sec: number;
}

export interface ResultsResponse {
  experiment_id: string;
  problem_type: ProblemType | null;
  target_column: string | null;
  metric: string | null;
  evaluation: EvaluationReport | null;
  review: ModelReview | null;
  best_model: BestModel | null;
  artifacts: Record<string, string>;
}

export interface HealthResponse {
  status: string;
  app: string;
  llm_available: boolean;
  llm_provider: string;
}
