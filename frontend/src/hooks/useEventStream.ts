import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { AgentEvent, EventType } from "@/lib/types";

// All named SSE event types the backend emits (matches EventType enum).
const EVENT_TYPES: EventType[] = [
  "experiment_started",
  "dataset_loaded",
  "planner_started",
  "planner_completed",
  "dataset_analysis_started",
  "dataset_analysis_completed",
  "eda_started",
  "eda_completed",
  "human_input_required",
  "human_input_received",
  "preprocessing_started",
  "preprocessing_completed",
  "model_selection_completed",
  "training_started",
  "model_training",
  "model_completed",
  "evaluation_completed",
  "reviewer_started",
  "reviewer_completed",
  "artifact_created",
  "experiment_completed",
  "chat_message",
  "log",
  "error",
];

const TERMINAL: EventType[] = ["experiment_completed"];

/**
 * Subscribe to an experiment's Server-Sent Events. The backend replays the full
 * backlog on connect, so on (transparent) reconnects we dedupe by a stable key
 * to avoid double-counting the replayed history.
 */
export function useEventStream(experimentId: string | null) {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const seenKeys = useRef<Set<string>>(new Set());

  useEffect(() => {
    setEvents([]);
    setConnected(false);
    seenKeys.current = new Set();
    if (!experimentId) return;

    const source = new EventSource(api.eventsUrl(experimentId));
    let closed = false;

    const push = (raw: string) => {
      try {
        const evt = JSON.parse(raw) as AgentEvent;
        const key = `${evt.type}|${evt.timestamp}|${evt.message ?? ""}|${JSON.stringify(evt.data)}`;
        if (seenKeys.current.has(key)) return;
        seenKeys.current.add(key);
        setEvents((prev) => [...prev, evt]);
        if (TERMINAL.includes(evt.type) && !closed) {
          closed = true;
          source.close();
          setConnected(false);
        }
      } catch {
        /* ignore malformed frame */
      }
    };

    source.onopen = () => setConnected(true);
    for (const type of EVENT_TYPES) {
      source.addEventListener(type, (e) => push((e as MessageEvent).data));
    }
    source.onmessage = (e) => push(e.data); // fallback for unnamed frames
    source.onerror = () => {
      setConnected(false);
      // Let EventSource auto-reconnect unless we've already closed on terminal.
      if (closed) source.close();
    };

    return () => {
      closed = true;
      source.close();
    };
  }, [experimentId]);

  return { events, connected };
}
