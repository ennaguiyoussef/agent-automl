import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { ExperimentSnapshot, ResultsResponse } from "@/lib/types";

const ACTIVE = new Set(["created", "running", "awaiting_human"]);

/**
 * Poll an experiment snapshot until it reaches a terminal state, then fetch its
 * results once. SSE drives the live activity feed; this provides authoritative
 * status, the pending human request, and final results.
 */
export function useExperiment(experimentId: string | null) {
  const [snapshot, setSnapshot] = useState<ExperimentSnapshot | null>(null);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const resultsFetched = useRef<string | null>(null);

  const refresh = useCallback(async () => {
    if (!experimentId) return;
    try {
      const snap = await api.getExperiment(experimentId);
      setSnapshot(snap);
      setError(null);
      if (
        (snap.status === "completed" || snap.has_results) &&
        resultsFetched.current !== experimentId
      ) {
        resultsFetched.current = experimentId;
        try {
          setResults(await api.results(experimentId));
        } catch {
          resultsFetched.current = null; // allow retry on next poll
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load experiment");
    }
  }, [experimentId]);

  useEffect(() => {
    setSnapshot(null);
    setResults(null);
    setError(null);
    resultsFetched.current = null;
    if (!experimentId) return;

    let stop = false;
    void refresh();
    const tick = window.setInterval(() => {
      if (stop) return;
      setSnapshot((cur) => {
        // Stop polling once terminal; otherwise keep going.
        if (cur && !ACTIVE.has(cur.status)) {
          window.clearInterval(tick);
          return cur;
        }
        void refresh();
        return cur;
      });
    }, 1200);

    return () => {
      stop = true;
      window.clearInterval(tick);
    };
  }, [experimentId, refresh]);

  return { snapshot, results, error, refresh };
}
