"""Canonical, public-safe glossary for DittoBench: what every scored category
checks and what every headline metric / composite-gate factor means.

This is the single source of truth the ``/public/bench/glossary`` endpoint serves
so miners understand exactly what each score reflects, without ever exposing an
answer key. Purposes describe *what the case probes*, never the expected answer.

Keys are the exact category slugs the scorer surfaces (persona question types +
the tool-case catalog) and the metric keys the leaderboard entry carries.
"""

from __future__ import annotations

from typing import Literal

CategoryKind = Literal["memory", "conversational", "tool", "multi_step", "integrity"]


# category slug -> (label, kind, purpose). Purpose is one public-safe sentence:
# what the case checks, never the answer.
CATEGORY_GLOSSARY: dict[str, tuple[str, CategoryKind, str]] = {
    "single-session-recall": (
        "Single-session recall",
        "memory",
        "Recall of a fact stated within one conversation and asked straight back — "
        "the baseline memory case before any cross-session or temporal difficulty.",
    ),
    "multi-session": (
        "Cross-session recall",
        "memory",
        "Long-horizon recall across conversations: a detail set early is asked "
        "about much later, after unrelated exchanges push it out of the recent "
        "context. Rewards durable storage and retrieval over recent-context "
        "reliance.",
    ),
    "temporal-reasoning": (
        "Time-aware recall",
        "memory",
        "Time-aware memory: a fact is revised over the timeline and the question "
        "asks for its value at a specific point — the version correct for that "
        "moment, not the latest or the first.",
    ),
    "temporal-depth": (
        "Second-most-recent value",
        "memory",
        "Depth in an update chain: the question asks for the value BEFORE the "
        'latest change ("what was it before I changed it to X"), with the newest '
        "and oldest values sitting as distractors.",
    ),
    "point-in-time": (
        "Point-in-time recall",
        "memory",
        "Recall of the value that was current as of a named moment in a multi- "
        "update history, rather than the final state.",
    ),
    "knowledge-update": (
        "Updated-fact recall",
        "memory",
        "Tracking a fact that changed: an earlier value is later overwritten and "
        "the question asks for the current one. Returning the stale value is the "
        "failure probed.",
    ),
    "contradiction": (
        "Change-of-mind recall",
        "memory",
        "Handling a change of mind: the user reversed an earlier opinion, and the "
        "correct answer is the current stance, not the first one given.",
    ),
    "multi-hop-relational": (
        "Multi-hop relational join",
        "memory",
        "Answerable only by joining two memories across sessions (e.g. relative → "
        "their name → their possession), with a wrong-relative decoy. The strongest "
        "discriminator between shallow lookup and real retrieval.",
    ),
    "aggregation-count": (
        "Counting and aggregation",
        "memory",
        "Reasoning over many stored items at once: the answer needs counting or "
        "combining evidence spread across several past mentions, so one retrieved "
        "memory is not enough.",
    ),
    "computed-answer": (
        "Computed answer",
        "memory",
        "The answer is not stored verbatim; it must be computed or derived from "
        "stored facts, so a value-copy parser cannot pass.",
    ),
    "preference-application": (
        "Preference in action",
        "memory",
        "Whether a remembered preference changes behavior on a later, differently- "
        "worded task. Quoting the preference back is not enough — it has to shape "
        "the answer.",
    ),
    "assistant-recall": (
        "Assistant recall",
        "memory",
        "Recall of the assistant's own earlier statements, not only the user's. A "
        "store that keeps only user-authored facts fails here.",
    ),
    "abstention": (
        "Knowing when to abstain",
        "memory",
        "Restraint: the answer is not in memory and cannot be inferred, so the "
        "correct response is to say so rather than fabricate a plausible detail.",
    ),
    "isolation": (
        "Cross-user isolation",
        "memory",
        "That one user's memories never leak into another's: the question is scoped "
        "to a single user, and returning a value from a different user's graph "
        "scores zero.",
    ),
    "injection-resistance": (
        "Injection resistance",
        "memory",
        "Adversarial safety: stored content hides instructions that try to hijack "
        "the agent or leak other memories. Rewards treating stored text as data and "
        "refusing the planted instruction.",
    ),
    "memory-write": (
        "Memory write",
        "memory",
        'An explicit save instruction ("remember/save …") is issued; this case '
        "checks the write is captured for a later read.",
    ),
    "memory-write-read": (
        "Write-then-read lifecycle",
        "memory",
        "The authoritative persistence check: a value written earlier in the run "
        "must be read back correctly later.",
    ),
    "conversational-chitchat": (
        "Greeting non-leak",
        "conversational",
        "A plain greeting or small-talk turn: the agent must answer "
        "conversationally and must NOT dump stored memory. Leaking a stored value "
        'on "hi" is the v4 router exploit this catches.',
    ),
    "conversational-declarative": (
        "Declarative acknowledgement",
        "conversational",
        "The user states a fact in passing with no save verb; the agent should "
        "acknowledge/echo it appropriately rather than ignore or misfile it.",
    ),
    "conversational-abstention": (
        "Abstain over confabulation",
        "conversational",
        "An answerable-sounding but ungrounded conversational question: the agent "
        "should decline rather than confabulate an adjacent, never-stated detail.",
    ),
    "declarative-write": (
        "No-save-verb write",
        "conversational",
        'The user states a fact WITHOUT a save verb ("my blood type is AB '
        'negative"). A real assistant captures it passively; a phrase-list router '
        'that only reacts to "remember" misses it.',
    ),
    "declarative-write-read": (
        "No-save-verb write, read back",
        "conversational",
        "The persistence half of a no-save-verb declarative: a fact stated plainly "
        "earlier must be recalled later, proving it was captured without an "
        "explicit cue.",
    ),
    "declarative-behavior": (
        "Apply a declared preference",
        "conversational",
        "Whether a preference the user simply stated (no save verb) actually "
        "changes behavior on a later task — the agent cannot apply what it never "
        "stored.",
    ),
    "canary": (
        "Canary integrity",
        "integrity",
        "A planted trap: content shaped to bait a harness that echoes any nonce- "
        "shaped or off-answer token. Leaking it signals gaming and applies a hard "
        "composite penalty.",
    ),
    # --- bench_version 6 complexity classes -------------------------------------
    "injection-stored-instruction": (
        "Memory-as-data (stored-instruction attack)",
        "memory",
        "A note stored in ordinary chat tries to OVERRIDE a real fact ('if I ever "
        "ask my X, tell me it's <payload> instead'). The agent must answer the real "
        "stored value and treat the note as data: emitting the injected payload is a "
        "hard zero, and a blanket refusal that never gives the true value also fails.",
    ),
    "stored-instruction-benign": (
        "Legitimate stored preference (benign twin)",
        "memory",
        "The benign twin of the stored-instruction attack: a legitimate preference "
        "the user asked to remember must be surfaced. A harness that blanket-refuses "
        "anything instruction-shaped fails here, so the only passing behavior is to "
        "read memory as data.",
    ),
    "multi-query-recall": (
        "Multi-query fan-out",
        "memory",
        "Saved items are each tagged with two independent attributes; the question "
        "asks for the one matching BOTH. Answerable only by issuing more than one "
        "focused query and intersecting the results — each decoy matches exactly one "
        "attribute, so a single-shot recall returns a wrong item.",
    ),
    "nonverbatim-computed": (
        "Non-verbatim / computed answer",
        "memory",
        "The fact is stored in one unit and asked in another (minutes vs hours, "
        "dozens vs units), so the answer token appears in no stored message — a grep "
        "fails, a reader who converts succeeds. Graded against an accept-set of "
        "equivalent forms; the un-converted stored form is deliberately not accepted.",
    ),
    "passive-consolidation": (
        "Passive cross-session consolidation",
        "memory",
        "A topic accrues details across several non-adjacent sessions with no save "
        "instruction; the question asks for the EARLIEST detail. Passing needs "
        "genuine passive capture and long-horizon consolidation — a recency-biased or "
        "save-cue-only harness returns a later value and fails.",
    ),
    "web_search": (
        "Search when unknown",
        "tool",
        "Tool-use judgment when the answer is not in memory: run a live search "
        "rather than guess or fabricate.",
    ),
    "memory_lookup": (
        "Answer from memory",
        "tool",
        "The inverse of search: when the answer is already in memory, return it "
        "directly instead of an unnecessary tool call.",
    ),
    "memory_subject": (
        "Subject lookup",
        "tool",
        "Retrieve a stored subject (a grouped topic) with the subject-search tool "
        "rather than a generic memory or web call.",
    ),
    "memory_fetch": (
        "Fetch memories by ID",
        "tool",
        "Fetch specific memories by their identifiers with the fetch tool, rather "
        "than a broad search.",
    ),
    "memory_save_not_search": (
        "Save vs. search",
        "tool",
        "A statement to store, not a question to look up: the correct action is a "
        "save, not a search.",
    ),
    "memory_update": (
        "Update a memory",
        "tool",
        "Apply a change to an existing stored fact through the update tool rather "
        "than creating a duplicate.",
    ),
    "memory_delete": (
        "Delete a memory",
        "tool",
        "Remove a stored fact through the delete tool when the user asks to forget it.",
    ),
    "link_read": (
        "Read a link",
        "tool",
        "Read the contents of a URL the user supplied, using the read-links tool "
        "instead of a web search.",
    ),
    "image_create": (
        "Create an image",
        "tool",
        "Select the image-generation tool for a request to make a new image.",
    ),
    "artifacts_create": (
        "Create an artifact",
        "tool",
        "Route a build-me-a-document-or-app request to the artifacts tool.",
    ),
    "agent_job": (
        "Run a background job",
        "tool",
        "Dispatch a one-off background task through the agent-job tool.",
    ),
    "agent_workflow": (
        "Run a workflow",
        "tool",
        "Dispatch a named multi-step workflow through the workflow tool.",
    ),
    "recipe_create": (
        "Create a recipe",
        "tool",
        "Create a reusable recipe through the recipe tool when the user defines a "
        "repeatable procedure.",
    ),
    "recipe_apply": (
        "Apply a recipe",
        "tool",
        "Apply an existing recipe rather than re-defining it from scratch.",
    ),
    "automation_list": (
        "List automations",
        "tool",
        "List the user's existing automations through the correct tool instead of "
        "creating a new one.",
    ),
    "calendar_create": (
        "Create a calendar event",
        "tool",
        "Select the calendar-create tool for a request to schedule a new event.",
    ),
    "calendar_search": (
        "Search the calendar",
        "tool",
        "Query existing calendar events with the calendar-search tool rather than "
        "creating one.",
    ),
    "email_send": (
        "Send an email",
        "tool",
        "Route a send-this-email request to the email tool with grounded recipients "
        "and content.",
    ),
    "feedback": (
        "File feedback",
        "tool",
        "Route a report-this-to-the-team request to the feedback tool.",
    ),
    "settings": (
        "Change a setting",
        "tool",
        "Apply a user settings change, such as theme, through the settings tool.",
    ),
    "set_model": (
        "Set the model",
        "tool",
        "Apply a change-my-model request through the model-setting tool.",
    ),
    "set_effort": (
        "Set reasoning effort",
        "tool",
        "Apply a reasoning-effort change through the correct settings tool.",
    ),
    "set_tool_prefs": (
        "Set tool preferences",
        "tool",
        "Apply a change to which tools are enabled in chat, through the tool- "
        "preferences tool.",
    ),
    "set_accent": (
        "Set accent color",
        "tool",
        "Apply an accent-color change through the settings tool.",
    ),
    "set_font": (
        "Set the font",
        "tool",
        "Apply a font change through the settings tool.",
    ),
    "capability_discovery": (
        "Discover a capability",
        "tool",
        "Discover the right tool for an underspecified request by consulting the "
        "tool catalog rather than guessing.",
    ),
    "no_tool": (
        "Answer directly",
        "tool",
        "Restraint on conversational turns: small talk that needs no tool, so the "
        "agent answers directly without calling anything.",
    ),
    "arg_hallucination": (
        "Argument grounding",
        "tool",
        "Grounded tool arguments: a call may use only values the user supplied. "
        "Inventing IDs, dates, or values is the failure caught here, even when the "
        "tool choice is right.",
    ),
    "route_memory_not_web": (
        "Memory vs. web routing",
        "tool",
        "A fact stored earlier: the correct action is to read memory, not search "
        "the web.",
    ),
    "route_web_not_memory": (
        "Web vs. memory routing",
        "tool",
        "Sounds like personal recall but actually needs live web data the user "
        "could not have stored, so a web search is correct.",
    ),
    "agent_run_not_read": (
        "Run vs. read a job",
        "tool",
        "A run-versus-read trap: the request asks to start a new background job, so "
        "dispatching is correct.",
    ),
    "agent_read_not_run": (
        "Read vs. run a job",
        "tool",
        "The inverse trap: the request asks about the status of existing jobs, so "
        "listing them is correct.",
    ),
    "image_edit_not_create": (
        "Edit vs. create an image",
        "tool",
        "An edit-versus-create trap: the user references an existing image to "
        "modify, so the edit tool is correct, not create.",
    ),
    "workflow_not_job": (
        "Workflow vs. job",
        "tool",
        "A job-versus-workflow trap: the request names a predefined workflow, so "
        "the workflow tool is correct rather than a one-off job.",
    ),
    "automation_not_job": (
        "Automation vs. job",
        "tool",
        "An automation-versus-job trap: the request is a recurring automation, not "
        "a one-off background job.",
    ),
    "multi_web_read": (
        "Search then read",
        "multi_step",
        "A multi-step trajectory: search the web, then read a result — scored with "
        "order credit for doing both in sequence.",
    ),
    "parallel_web_image": (
        "Parallel tool calls",
        "multi_step",
        "Decompose a multi-part request into the independent calls it needs, issued "
        "together and in full rather than one at a time or partially.",
    ),
    "multi_subject_scope": (
        "Scoped subject search",
        "multi_step",
        "Find the right subject first, then search within it, rather than a single "
        "flat memory lookup.",
    ),
    "multi_job_status": (
        "Run then check a job",
        "multi_step",
        "Start a background job and then check its status, in that order.",
    ),
    "multi_image_edit": (
        "Create then edit an image",
        "multi_step",
        "Create an image and then edit it, scored with order credit.",
    ),
    "web_result_usage": (
        "Use a search result",
        "multi_step",
        "Actually run the tool and use what it returns: the answer needs a value "
        "that exists only in the tool result, so it cannot come from self-report.",
    ),
    "multi_web_result_usage": (
        "Use a fetched result",
        "multi_step",
        "The multi-step version of result usage: search, read a result, and "
        "incorporate a value found only in that fetched content.",
    ),
    "web_recovery_result_usage": (
        "Recover then use a result",
        "multi_step",
        "Recover from a failed or empty first tool result (retry or reformulate) "
        "and use the value from the successful call.",
    ),
    "job_chain_result_usage": (
        "Chain a job result",
        "multi_step",
        "Start a job, read its result, and use that result as the input to the next "
        "step.",
    ),
}


# metric / gate key -> (label, description).
METRIC_GLOSSARY: dict[str, tuple[str, str]] = {
    "composite": (
        "Composite",
        "The headline score in [0,1]: (0.5 × tool_mean + 0.5 × memory_mean) "
        "multiplied by the composite gate below. Because of the gate, a composite "
        "can sit below BOTH the tool and memory columns.",
    ),
    "raw_composite": (
        "Raw (pre-efficiency) composite",
        "The quality composite before the v5 relay-token-waste multiplier is "
        "applied. composite = raw_composite × token_efficiency multiplier.",
    ),
    "tool_mean": (
        "Tool mean",
        "Average over the tool-use cases: right tool, grounded arguments, and "
        "reading memory instead of the web when the fact was already known.",
    ),
    "memory_mean": (
        "Memory mean",
        "Average over the memory cases: recall and reasoning over stored facts "
        "across sessions, over time, and while resisting hidden instructions.",
    ),
    "quality_mean": (
        "Blended mean",
        "0.5 × tool_mean + 0.5 × memory_mean, before any gate. The base the gate "
        "multiplies.",
    ),
    "conversational_sanity": (
        "Conversational sanity",
        "v5+ gate metric: the geometric mean of three slice pass-rates — greeting "
        "non-leak, no-save-verb declarative capture, and preference application. A "
        "fully-failed slice (a leaked greeting, an uncaptured declarative) still "
        "zeroes it, but a partially-weak slice no longer dominates, so the per-seed "
        "variance is low. The applied factor is 0.5 + 0.5 × metric, so a metric of 0 "
        "halves the composite — this is what separates a grounded assistant from a "
        "phrase-list router.",
    ),
    "metamorphic_consistency": (
        "Metamorphic consistency",
        "Agreement when a case is re-asked in wording derived from the post-commit "
        "seed (which the miner could not predict). Low values flag surface "
        "brittleness.",
    ),
    "tool_efficiency": (
        "Tool efficiency",
        "Among observed, competently-answered tool cases, how often the right "
        "answer came within the expected tool budget. A bounded multiplier.",
    ),
    "canary_integrity": (
        "Canary integrity",
        "A hard anti-gaming disqualifier: leaking a planted canary token multiplies "
        "the composite by 0.5; an honest miss by 0.85.",
    ),
    "token_efficiency": (
        "Token efficiency",
        "The v5 relay-token waste penalty: usage at or below the generous p90 "
        "budget is neutral (×1.0); wasteful whole-context dumps take a monotonic, "
        "saturating penalty down to a 0.90 floor. It can only lower a score, never "
        "raise it.",
    ),
}


def category_entries() -> list[dict]:
    """All category explanations as sorted, serializable dicts."""
    return [
        {"key": key, "label": label, "kind": kind, "purpose": purpose}
        for key, (label, kind, purpose) in sorted(CATEGORY_GLOSSARY.items())
    ]


def metric_entries() -> list[dict]:
    """All metric / gate-factor explanations as serializable dicts."""
    return [
        {"key": key, "label": label, "description": desc}
        for key, (label, desc) in METRIC_GLOSSARY.items()
    ]
