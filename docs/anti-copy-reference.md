# Reference-aware anti-copy comparison

The platform compares only the submission-specific residue left after removing a
canonical public starter-kit corpus. The packaged corpus contains every unique blob
reachable from the official starter-kit `main` history. The committed
`reference_manifest_v2.json` is the authoritative immutable revision and commit-set
identity; `scripts/build_reference_fingerprints.py` regenerates the deterministic
bundles and that manifest together.

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
- An algorithm-version mismatch is an explicit inconclusive review result. A
  same-version corpus mismatch is skipped: it is expected while an official
  starter-kit refresh is being backfilled and is not evidence of copying. Exact
  bytes remain enforced before that check, and the gate continues checking other
  same-corpus references instead of falling through to structural or archive-size
  heuristics.
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
of a code change. Rollout order for each reference refresh is:

1. Land and deploy the durable ATH review migration/API so existing holds are
   snapshotted with legacy/original evidence (`d84b3a91f620`).
2. Deploy the refreshed reference bundle with the backfill tool present but unused.
   Newly uploaded artifacts carry the new corpus identity; a mixed-corpus pair does
   not enter ATH review solely because the rollout is in progress.
3. Run the fingerprint backfill without `--apply`, review aggregate counts, then
   separately authorize the metadata-only apply and a catch-up pass. The tool uses
   bounded batches, is idempotent on algorithm plus corpus identity, and updates
   only lexical, normalized-source, and prompt fingerprint metadata.
4. Verify provenance-bearing current comparison output before enabling downstream
   bulk-eligibility presentation.

The backfill never changes agent status, duplicate attribution, review reason,
scores, public mirrors, or verdicts. This repository provides no bulk re-review,
release, rejection, or ban operation for this transition.

## Keeping the public reference current

`.github/workflows/refresh-anti-copy-reference.yml` checks the official
starter-kit `main` branch every day and on demand. When its deterministic bundles
change, it updates the single `automation/starter-kit-reference` branch and opens
or refreshes a reviewable platform PR. It does not merge or deploy automatically.
This keeps ordinary starter-kit evolution visible before stale public scaffolding
can dominate a miner-to-miner comparison again.
