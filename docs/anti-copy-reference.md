# Reference-aware anti-copy comparison

The platform compares only the submission-specific residue left after removing a
canonical public starter-kit corpus. The packaged corpus contains every unique blob
reachable from the official starter-kit `main` history at revision
`959cd69a1a8d3b0defbfb8296518adb7d4f17c14`; its commit-set identity is
`21dc06cd72aafefb56d0e89e8b3127280dda249ae26cb649ee855185121e9ce6`.
`scripts/build_reference_fingerprints.py` regenerates the deterministic bundles and
records both the requested ref and its resolved immutable revision.

## Operator baseline bundle (text side)

The corpus above is one-way: it answers "was this window ever in the kit?" and
cannot yield a file, a path, or a line. Operator review needs the opposite, so
`scripts/build_starter_kit_baseline.py` packages the kit's *text* as
`ditto/anticopy/starter_kit_baseline_v1.json.gz`, read through
`ditto.api_server.starter_kit`. Like the shingle bundles it is committed, so
request handling never needs network access.

It carries the tip tree's `path -> text` (the readable diff baseline) plus
content digests — exact and normalized — of every text blob across mainline
history. The historical set is what makes the subtraction honest: miners fork at
different commits, so a file can be untouched kit code while still differing
from the tip. Matching the whole lineage keeps those files marked `stock_kit`
instead of inflating the reviewer's delta.

`GET /admin/screening-submissions/{agent_id}/baseline-diff` and
`.../baseline-diff/file` serve this, reusing the copy-review diff engine. The
headline `custom_added_lines` counts only lines that are neither baseline nor
kit code at any revision.

Note the two bundles are versioned and regenerated independently, so their
pinned revisions can drift apart. The corpus identity is load-bearing for stored
fingerprints (`nsh2:{corpus_id}:…`) and must not be regenerated casually; the
baseline bundle is review-only and safe to refresh whenever the kit moves.

## Active comparison policy

- Exact-byte routing remains a separate defense-in-depth path.
- Normalized-source equality, lexical similarity, and size fallback compare a
  candidate only with chronologically earlier submissions from another miner.
  Submission time is authoritative; UUID order breaks equal-time ties.
- Lexical v2 subtracts the canonical corpus before sketching. The public Jaccard
  and containment thresholds remain `0.75` and `0.95`.
- The lexical residual floor is eight shingles. Measured calibration showed that a
  floor of 16 missed a copied 12-shingle custom block, while eight retained exact
  containment for that copy. Residues below eight remain versioned empty sketches
  and cannot fall through to structural or size evidence.
- Normalized-source v2 values include the algorithm and corpus identity before the
  digest. Prompt p2 and lexical v2 sketches also carry the corpus identity.
- Structural AST overlap remains advisory. The current whole-crate structural
  sketch is not reference-aware; aggregate review evidence found threshold-level
  saturation among starter-kit derivatives. Demoting it avoids re-holding an
  independent fork after lexical reference subtraction while preserving the
  measured structural value in operator evidence. Prompt overlap is advisory for
  the same convergence reason.
- A lexical version or corpus mismatch is an explicit inconclusive review result.
  It never silently falls through to structural or archive-size heuristics.
- Size similarity is a legacy fallback only when both rows lack lexical and
  structural fingerprints. A valid negative, empty, or incompatible fingerprint
  is not overridden by similar size.
- Existing holds and their original attribution are never changed by this policy.

## Read-only comparison adapter

`compare_anti_copy_pair(candidate=..., reference=...)` delegates its decision to
the authoritative score-path gate and returns aggregate metrics and public
provenance only. It has no database or storage dependency and returns no hashes,
sketch members, source, paths, artifact locations, or credentials.

The durable review endpoint constructs both ledger values from canonical agent and
median finalized-score data. It always passes the held agent as the candidate and
the review row's immutable `original_duplicate_of` agent as the reference, using
each agent's upload time as `first_seen`. A successful read returns only
`comparison.to_wire()`. Missing score/reference data is unavailable and fails
closed.

The wire marks bulk eligibility only when the corrected decision is clear, the
reference is chronologically eligible and from another miner, and both lexical
fingerprints are canonical v2 values from this exact corpus. Missing or incompatible
fingerprints, reversed chronology, same-miner pairs, and every hold or inconclusive
decision remain bulk-ineligible. The platform still exposes no bulk mutation.

## Metadata transition order

The metadata backfill tool ships with the algorithm but must not be applied as part
of this change. Rollout order is:

1. Land and deploy the durable ATH review migration/API so existing holds are
   snapshotted with legacy/original evidence (`d84b3a91f620`).
2. Deploy the reference-aware algorithm and endpoint integration with the backfill
   tool present but unused. Newly opened holds snapshot the v2 algorithm, revision,
   corpus, and exclusion mode; legacy rows remain unchanged.
3. Run the fingerprint backfill without `--apply`, review aggregate counts, then
   separately authorize the metadata-only apply and a catch-up pass. The tool uses
   bounded batches, is idempotent on algorithm plus corpus identity, and updates
   only lexical, normalized-source, and prompt fingerprint metadata.
4. Verify provenance-bearing current comparison output before enabling downstream
   bulk-eligibility presentation.

The backfill never changes agent status, duplicate attribution, review reason,
scores, public mirrors, or verdicts. This repository provides no bulk re-review,
release, rejection, or ban operation for this transition.
