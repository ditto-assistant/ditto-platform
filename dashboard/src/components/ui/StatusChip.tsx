import type { JSX } from "solid-js";

const STATUS_LABELS: Record<string, string> = {
  waiting_screening: "Waiting for screening",
  screening: "Screening",
  waiting_validation: "Waiting for validator",
  validating: "Benchmarking",
  scored: "Scored",
  rejected: "Rejected",
  under_review: "Under review",
  not_queued: "Not queued",
};

export function StatusChip(props: { status?: string | null }): JSX.Element {
  const status = () => props.status || "unknown";
  return (
    <span class={`status-chip status-${status()}`}>
      {STATUS_LABELS[status()] || status().replaceAll("_", " ")}
    </span>
  );
}
