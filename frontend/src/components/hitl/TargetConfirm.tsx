import { useState } from "react";
import { Check, HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { PendingHumanRequest } from "@/lib/types";

interface TargetConfirmProps {
  request: PendingHumanRequest;
  submitting: boolean;
  onConfirm: (target: string) => void;
}

export function TargetConfirm({ request, submitting, onConfirm }: TargetConfirmProps) {
  const [choice, setChoice] = useState<string>(request.suggested ?? "");

  return (
    <div className="animate-fade-in-up rounded-lg border border-signal-wait/40 bg-signal-wait/5 p-5">
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-signal-wait/15 text-signal-wait">
          <HelpCircle className="h-4.5 w-4.5" />
        </div>
        <div>
          <div className="eyebrow !text-signal-wait">Your input needed</div>
          <div className="font-display font-semibold">Confirm the target column</div>
        </div>
      </div>

      <p className="mt-3 text-sm text-muted-foreground">{request.question}</p>

      <div className="mt-4 flex flex-wrap gap-2">
        {request.options.map((opt) => {
          const selected = opt === choice;
          const suggested = opt === request.suggested;
          return (
            <button
              key={opt}
              onClick={() => setChoice(opt)}
              className={cn(
                "flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                selected
                  ? "border-primary bg-primary/10 text-foreground"
                  : "border-border bg-card hover:border-primary/40",
              )}
            >
              {selected && <Check className="h-3.5 w-3.5 text-primary" />}
              <span className="font-mono">{opt}</span>
              {suggested && !selected && (
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  suggested
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div className="mt-4 flex justify-end">
        <Button onClick={() => choice && onConfirm(choice)} disabled={!choice || submitting}>
          {submitting ? "Resuming…" : `Confirm "${choice || "…"}"`}
        </Button>
      </div>
    </div>
  );
}
