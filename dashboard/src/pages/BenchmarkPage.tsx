// Placeholder: the benchmark page port replaces this file. The site footer
// already renders here because the monolith mounts it inside this page's
// section (2995–3007) — keep it when porting the rest of the page.
import type { JSX } from "solid-js";

import { SiteFooter } from "../components/shell/Sidebar";

export function BenchmarkPage(): JSX.Element {
  return (
    <section class="page active" data-page="benchmark">
      <h2 class="visually-hidden">Benchmark</h2>
      <SiteFooter />
    </section>
  );
}
