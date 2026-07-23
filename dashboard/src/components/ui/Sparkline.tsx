import type { JSX } from "solid-js";

export function Sparkline(props: { values?: number[]; label: string }): JSX.Element {
  const points = () => {
    const values = props.values || [];
    if (values.length < 2) return "";
    const low = Math.min(...values);
    const high = Math.max(...values);
    const span = high - low || 1;
    return values
      .map((value, index) => {
        const x = (index / (values.length - 1)) * 100;
        const y = 28 - ((value - low) / span) * 24;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };
  return (
    <svg class="spark" viewBox="0 0 100 32" role="img" aria-label={props.label}>
      <polyline points={points()} fill="none" vector-effect="non-scaling-stroke" />
    </svg>
  );
}
