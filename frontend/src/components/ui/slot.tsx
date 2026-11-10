import * as React from "react";

/**
 * Minimal `asChild` slot: merges the component's props onto its single child
 * element. Enough for our Button; avoids a Radix dependency.
 */
export const Slot = React.forwardRef<HTMLElement, React.HTMLAttributes<HTMLElement>>(
  ({ children, ...slotProps }, ref) => {
    if (!React.isValidElement(children)) return null;
    const child = children as React.ReactElement<Record<string, unknown>>;
    return React.cloneElement(child, {
      ...slotProps,
      ...child.props,
      className: [
        (slotProps as { className?: string }).className,
        (child.props as { className?: string }).className,
      ]
        .filter(Boolean)
        .join(" "),
      ref,
    });
  },
);
Slot.displayName = "Slot";
