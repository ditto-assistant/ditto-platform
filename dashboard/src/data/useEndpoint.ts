import { createResource, getOwner, onCleanup } from "solid-js";
import type { Accessor } from "solid-js";

import { getJSON } from "../lib/api";

export interface ResourceState<T> {
  data: Accessor<T | undefined>;
  loading: Accessor<boolean>;
  error: Accessor<unknown>;
  refresh: () => void;
}

export interface UseEndpointOptions {
  /** Refresh cadence. Hidden tabs skip network refreshes entirely and catch
   * up once on return, so idle dashboards stop polling the API (the
   * monolith's document.hidden rule around its load() tick). */
  pollMs?: number;
}

// Every live endpoint's refresh, so the shell's manual refresh reaches data
// owned by pages and module-scope stores, not just the resources App holds —
// the monolith's single load() had this property for free.
const liveRefreshes = new Set<() => void>();

export function refreshAllEndpoints(): void {
  liveRefreshes.forEach((refresh) => refresh());
}

export function useEndpoint<T>(
  path: Accessor<string> | string,
  options?: UseEndpointOptions,
): ResourceState<T> {
  const source = typeof path === "string" ? () => path : path;
  const [data, { refetch }] = createResource(source, (next) => getJSON<T>(next));
  // The page-level refresh and a faster page-local poll can land together.
  // Share one in-flight request so a slower duplicate cannot fail after a
  // successful response and replace live data with an error state
  // (loadOperations 9789–9793). A refresh that arrives mid-flight is not
  // dropped — one trailing refetch runs after the current request settles,
  // so "refresh now" always eventually observes the live API.
  let inFlight = false;
  let trailing = false;
  const refresh = (): void => {
    if (inFlight) {
      trailing = true;
      return;
    }
    inFlight = true;
    Promise.resolve(refetch())
      .catch(() => undefined)
      .finally(() => {
        inFlight = false;
        if (trailing) {
          trailing = false;
          refresh();
        }
      });
  };
  liveRefreshes.add(refresh);

  let timer: ReturnType<typeof setInterval> | undefined;
  let onVisibility: (() => void) | undefined;
  if (options?.pollMs) {
    let refreshStale = false;
    timer = setInterval(() => {
      if (document.hidden) {
        refreshStale = true;
        return;
      }
      refresh();
    }, options.pollMs);
    onVisibility = () => {
      if (!document.hidden && refreshStale) {
        refreshStale = false;
        refresh();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
  }

  // Component-owned endpoints unregister on unmount; module-scope stores
  // (created under createRoot) live for the page lifetime, like the monolith's
  // globals, so their lack of an owner is fine.
  if (getOwner()) {
    onCleanup(() => {
      liveRefreshes.delete(refresh);
      if (timer !== undefined) clearInterval(timer);
      if (onVisibility) document.removeEventListener("visibilitychange", onVisibility);
    });
  }

  return {
    data,
    loading: () => data.loading,
    error: () => data.error,
    refresh,
  };
}
