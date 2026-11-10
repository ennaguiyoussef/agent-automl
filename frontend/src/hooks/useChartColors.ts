import { useEffect, useState } from "react";

function readVar(name: string): string {
  if (typeof window === "undefined") return "0 0% 50%";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "0 0% 50%";
}

export interface ChartColors {
  primary: string;
  muted: string;
  ok: string;
  grid: string;
  axis: string;
  card: string;
}

/**
 * Resolve theme CSS variables to concrete `hsl(...)` strings for Recharts
 * (SVG presentation attributes don't resolve `var()`). Recomputes whenever the
 * root theme class changes.
 */
export function useChartColors(themeKey: string): ChartColors {
  const [colors, setColors] = useState<ChartColors>(() => compute());

  function compute(): ChartColors {
    return {
      primary: `hsl(${readVar("--primary")})`,
      muted: `hsl(${readVar("--muted-foreground")})`,
      ok: `hsl(${readVar("--signal-ok")})`,
      grid: `hsl(${readVar("--border")})`,
      axis: `hsl(${readVar("--muted-foreground")})`,
      card: `hsl(${readVar("--card")})`,
    };
  }

  useEffect(() => {
    // Defer a tick so the new theme class is applied before reading.
    const id = window.setTimeout(() => setColors(compute()), 0);
    return () => window.clearTimeout(id);
  }, [themeKey]);

  return colors;
}
