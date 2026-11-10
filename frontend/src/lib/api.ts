import type {
  ChatResponse,
  DatasetUploadResponse,
  ExperimentSnapshot,
  HealthResponse,
  ResultsResponse,
} from "./types";

// In dev, Vite proxies "/api" to the backend, so a relative base works. In a
// production build, VITE_API_BASE_URL points at the deployed API.
const BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export interface CreateExperimentPayload {
  dataset_id: string;
  user_query: string;
  target_column?: string | null;
  optimize_metric?: string | null;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  uploadDataset: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DatasetUploadResponse>("/api/datasets/upload", {
      method: "POST",
      body: form,
    });
  },

  createExperiment: (payload: CreateExperimentPayload) =>
    request<ExperimentSnapshot>("/api/experiments", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listExperiments: () => request<ExperimentSnapshot[]>("/api/experiments"),

  getExperiment: (id: string) => request<ExperimentSnapshot>(`/api/experiments/${id}`),

  chat: (id: string, message: string) =>
    request<ChatResponse>(`/api/experiments/${id}/chat`, {
      method: "POST",
      body: JSON.stringify({ message }),
    }),

  humanResponse: (id: string, targetColumn: string) =>
    request<ExperimentSnapshot>(`/api/experiments/${id}/human-response`, {
      method: "POST",
      body: JSON.stringify({ target_column: targetColumn }),
    }),

  results: (id: string) => request<ResultsResponse>(`/api/experiments/${id}/results`),

  /** Absolute URLs for artifact downloads (used as <a href>). */
  downloadModelUrl: (id: string) => `${BASE}/api/experiments/${id}/download-model`,
  downloadReportUrl: (id: string) => `${BASE}/api/experiments/${id}/download-report`,

  /** SSE endpoint URL. */
  eventsUrl: (id: string) => `${BASE}/api/experiments/${id}/events`,
};
