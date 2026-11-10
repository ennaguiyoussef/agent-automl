import { cn } from "@/lib/utils";

/** The pipeline-signal mark: three nodes joined by a routed path. */
export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={cn("h-7 w-7", className)} aria-hidden="true">
      <path
        d="M8 8 L24 16 L8 24"
        fill="none"
        stroke="hsl(var(--primary))"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.85"
      />
      <circle cx="8" cy="8" r="3.2" fill="hsl(var(--primary))" />
      <circle cx="8" cy="24" r="3.2" fill="hsl(var(--signal-ok))" />
      <circle cx="24" cy="16" r="3.2" fill="hsl(var(--signal-run))" />
    </svg>
  );
}
