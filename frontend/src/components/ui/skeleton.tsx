import { cn } from "@/lib/utils";

/** Shimmering placeholder block. Uses the `.skeleton` component class. */
export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("skeleton", className)} {...props} />;
}
