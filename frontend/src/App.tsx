import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type {
  CreateExperimentPayload,
} from "@/lib/api";
import type {
  DatasetUploadResponse,
  ExperimentSnapshot,
  HealthResponse,
} from "@/lib/types";
import { useTheme } from "@/hooks/useTheme";
import { useExperiment } from "@/hooks/useExperiment";
import { useEventStream } from "@/hooks/useEventStream";
import { Sidebar } from "@/components/layout/Sidebar";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { UploadHero } from "@/components/dataset/UploadHero";
import { GoalComposer } from "@/components/dataset/GoalComposer";
import { ExperimentWorkspace } from "@/components/workspace/ExperimentWorkspace";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { ErrorState } from "@/components/states/ErrorState";
import { Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";

type View = "upload" | "compose" | "experiment";

export default function App() {
  const { theme, toggle } = useTheme();

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSnapshot[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [composerDataset, setComposerDataset] = useState<DatasetUploadResponse | null>(null);
  const [creating, setCreating] = useState(false);
  const [submittingHuman, setSubmittingHuman] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Cache uploaded dataset profiles so the Dataset tab renders instantly and
  // survives even if the analysis event scrolls out of the backlog.
  const datasetCache = useRef<Map<string, DatasetUploadResponse>>(new Map());

  const { snapshot, results, error: experimentError, refresh } = useExperiment(activeId);
  const { events, connected } = useEventStream(activeId);

  // ---- Data loading --------------------------------------------------------
  const loadExperiments = useCallback(async () => {
    try {
      setExperiments(await api.listExperiments());
    } catch {
      /* transient; keep last-known list */
    }
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    void loadExperiments();
    const tick = window.setInterval(loadExperiments, 3000);
    return () => window.clearInterval(tick);
  }, [loadExperiments]);

  // ---- Handlers ------------------------------------------------------------
  const handleUpload = useCallback(async (file: File) => {
    const res = await api.uploadDataset(file);
    datasetCache.current.set(res.dataset_id, res);
    setComposerDataset(res);
    setActiveId(null);
    setSidebarOpen(false);
  }, []);

  const handleCreate = useCallback(
    async (payload: CreateExperimentPayload) => {
      setCreating(true);
      try {
        const snap = await api.createExperiment(payload);
        setComposerDataset(null);
        setActiveId(snap.id);
        setExperiments((prev) => [snap, ...prev.filter((e) => e.id !== snap.id)]);
        void loadExperiments();
      } catch (e) {
        // Surface as an alert; the composer stays put so the user can retry.
        alert(e instanceof Error ? e.message : "Could not start the experiment.");
      } finally {
        setCreating(false);
      }
    },
    [loadExperiments],
  );

  const handleNew = useCallback(() => {
    setActiveId(null);
    setComposerDataset(null);
    setSidebarOpen(false);
  }, []);

  const handleSelect = useCallback((id: string) => {
    setActiveId(id);
    setComposerDataset(null);
    setSidebarOpen(false);
  }, []);

  const handleConfirmTarget = useCallback(
    async (target: string) => {
      if (!activeId) return;
      setSubmittingHuman(true);
      try {
        await api.humanResponse(activeId, target);
        await refresh();
        void loadExperiments();
      } catch (e) {
        alert(e instanceof Error ? e.message : "Could not submit your choice.");
      } finally {
        setSubmittingHuman(false);
      }
    },
    [activeId, refresh, loadExperiments],
  );

  // ---- View selection ------------------------------------------------------
  const view: View = activeId ? "experiment" : composerDataset ? "compose" : "upload";

  const activeDataset =
    snapshot?.dataset_id != null ? datasetCache.current.get(snapshot.dataset_id) ?? null : null;

  const showChat = view === "experiment" && snapshot != null;

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground">
      {/* Sidebar — off-canvas on small screens, fixed on large */}
      <div
        className={`fixed inset-y-0 left-0 z-30 w-72 transform transition-transform lg:static lg:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Sidebar
          experiments={experiments}
          activeId={activeId}
          health={health}
          onNew={handleNew}
          onSelect={handleSelect}
        />
      </div>
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-background/70 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden
        />
      )}

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top bar */}
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setSidebarOpen((o) => !o)}
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? <X /> : <Menu />}
          </Button>
          <div className="hidden lg:block" />
          <ThemeToggle theme={theme} onToggle={toggle} />
        </header>

        {/* Body */}
        <div className="flex min-h-0 flex-1">
          <main className="min-w-0 flex-1 overflow-y-auto">
            {view === "upload" && <UploadHero onUpload={handleUpload} />}

            {view === "compose" && composerDataset && (
              <GoalComposer
                dataset={composerDataset}
                submitting={creating}
                onBack={handleNew}
                onCreate={handleCreate}
              />
            )}

            {view === "experiment" &&
              (snapshot ? (
                <ExperimentWorkspace
                  snapshot={snapshot}
                  results={results}
                  events={events}
                  connected={connected}
                  cachedDataset={activeDataset}
                  themeKey={theme}
                  submittingHuman={submittingHuman}
                  onConfirmTarget={handleConfirmTarget}
                />
              ) : experimentError ? (
                <div className="mx-auto max-w-2xl px-6 py-16">
                  <ErrorState
                    title="Couldn't load this experiment"
                    message={experimentError}
                    onRetry={refresh}
                  />
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                  Loading experiment…
                </div>
              ))}
          </main>

          {/* Chat rail */}
          {showChat && (
            <aside className="hidden w-[340px] shrink-0 border-l border-border bg-card/40 xl:block">
              <ChatPanel
                experimentId={snapshot.id}
                onActivity={refresh}
                seedGreeting="I'm running your pipeline. Ask me about the plan, the data, or the results — every number I quote is computed in Python."
              />
            </aside>
          )}
        </div>
      </div>
    </div>
  );
}
