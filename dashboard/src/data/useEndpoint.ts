import { createResource } from "solid-js";
import type { Accessor } from "solid-js";

import { getJSON } from "../lib/api";

export interface ResourceState<T> {
  data: Accessor<T | undefined>;
  loading: Accessor<boolean>;
  error: Accessor<unknown>;
  refresh: () => void;
}

export function useEndpoint<T>(path: Accessor<string> | string): ResourceState<T> {
  const source = typeof path === "string" ? () => path : path;
  const [data, { refetch }] = createResource(source, (next) => getJSON<T>(next));
  return {
    data,
    loading: () => data.loading,
    error: () => data.error,
    refresh: () => void refetch(),
  };
}
