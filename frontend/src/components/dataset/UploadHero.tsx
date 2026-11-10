import { useCallback, useRef, useState } from "react";
import { AlertCircle, FileUp, Loader2, UploadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Logo } from "@/components/layout/Logo";

interface UploadHeroProps {
  onUpload: (file: File) => Promise<void>;
}

export function UploadHero({ onUpload }: UploadHeroProps) {
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      if (!/\.csv$/i.test(file.name)) {
        setError("Please choose a .csv file.");
        return;
      }
      setError(null);
      setBusy(true);
      try {
        await onUpload(file);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed.");
      } finally {
        setBusy(false);
      }
    },
    [onUpload],
  );

  return (
    <div className="relative flex min-h-full flex-col items-center justify-center overflow-hidden px-6 py-16">
      <div className="grid-backdrop pointer-events-none absolute inset-0" aria-hidden />

      <div className="relative z-10 mx-auto flex max-w-2xl flex-col items-center text-center">
        <div className="mb-6 flex items-center gap-2 rounded-full border border-border bg-card/60 px-3 py-1 backdrop-blur">
          <Logo className="h-4 w-4" />
          <span className="eyebrow !tracking-[0.14em]">Agent-driven modeling</span>
        </div>

        <h1 className="font-display text-4xl font-semibold leading-[1.05] tracking-tight sm:text-5xl">
          Hand off a CSV.
          <br />
          <span className="text-primary">Watch an agent build the model.</span>
        </h1>
        <p className="mt-4 max-w-lg text-balance text-muted-foreground">
          Upload a dataset and a LangGraph agent will profile it, decide the ML problem, train and
          cross-validate candidates, then explain the winner — every metric computed in Python, never
          invented.
        </p>

        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void handleFile(e.dataTransfer.files?.[0]);
          }}
          className={cn(
            "group mt-9 flex w-full max-w-lg cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed border-border bg-card/50 px-6 py-10 text-center transition-colors",
            dragging && "border-primary bg-accent/50",
            busy && "pointer-events-none opacity-70",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            className="sr-only"
            onChange={(e) => void handleFile(e.target.files?.[0])}
            disabled={busy}
          />
          <div
            className={cn(
              "flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary transition-transform group-hover:scale-105",
            )}
          >
            {busy ? <Loader2 className="h-6 w-6 animate-spin" /> : <UploadCloud className="h-6 w-6" />}
          </div>
          <div>
            <div className="font-medium">
              {busy ? "Profiling dataset…" : "Drop a CSV here, or click to browse"}
            </div>
            <div className="mt-0.5 font-mono text-xs text-muted-foreground">
              up to 50&nbsp;MB · 200k rows
            </div>
          </div>
          {!busy && (
            <Button type="button" variant="outline" size="sm" className="mt-1 pointer-events-none">
              <FileUp />
              Choose file
            </Button>
          )}
        </label>

        {error && (
          <div className="mt-4 flex items-center gap-2 rounded-md border border-signal-err/30 bg-signal-err/10 px-3 py-2 text-sm text-signal-err">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
}
