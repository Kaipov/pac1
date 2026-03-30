# PAC Practical Insights

This is a practical summary based on the public `bitgn/challenges/pac` docs and the current sample agent in this repository. It is not a leak of hidden tasks. It is a working note with the insights most likely to help us improve the agent for the competition.

## What Matters Most

- `bitgn/challenges/pac` is a rules and mechanics repository, not a hidden-task repository.
- The canonical document there is `handbook.md`.
- The competition is about solving tasks safely, deterministically, and with the right side effects.
- The main event date is `2026-04-11`, and the schedule is anchored to `Europe/Vienna`.

## What Is Actually Scored

BitGN scores observable outcomes, not "smart-looking prose":

- required side effects happened;
- forbidden side effects did not happen;
- required `refs` or `flags` are present;
- output protocol is correct;
- unsafe, unnecessary, or destructive actions are avoided.

Practical implication:

- optimize for exact outcomes, not just nice answers;
- final answers must be supported by what the agent actually observed and did in the same run;
- unnecessary actions can cost points even when the final answer looks correct.

## Main Failure Modes

- prompt injection;
- secret exfiltration;
- destructive actions without explicit need;
- constraint violations;
- malformed output;
- missing grounding refs;
- tool spam and loops.

Practical implication:

- safe refusal must be a first-class path, not an accident;
- "do not do extra things" matters as much as "do the required thing";
- the agent must know when to stop, retry differently, or refuse.

## Fair Play And Blind Window

- Human-in-the-loop is not allowed inside a run.
- The agent may be changed between runs.
- During the blind competition window, detailed feedback is suppressed.
- Hall of Fame uses one completed session; if none is selected manually, the last completed session before cutoff is auto-submitted.

Practical implication:

- we need local diagnostics before the blind window starts;
- we cannot rely on platform feedback during the final event window;
- traces and failure analysis must be built into our local workflow.

## Performance Constraints

- Public docs mention a fairness cap around `~1000 API calls per task`.
- Quickstart gives an operational target around `~30 minutes for ~100 tasks with limited parallelization`.

Practical implication:

- step count and tool count matter;
- budgets per tool and repeated-call detection are worth implementing;
- a disciplined agent can beat a more verbose and expensive one.

## New Signals From The Organizer Post

The organizer shared an additional public note about the current Sandbox and the expected runtime for the main event on `2026-04-11`.

Direct signals from that post:

- Sandbox has passed the smoke-test phase.
- Engineers are already connecting to Harness and receiving action-based evaluation.
- The current Sandbox is intentionally simple: an Obsidian-like folder with markdown files and typed records.
- Sandbox currently has only `7` tasks and does not require authorization.
- The main competition runtime is expected to simulate more tools than the current Sandbox.
- The organizer explicitly mentioned chats, email, remote server access, and destructive commands as likely runtime elements.
- Leaderboards, profiles, keys, debug mode, and related platform features are expected to be enabled soon.

Practical implication:

- do not overfit to Sandbox because it is a smoke-test environment, not a faithful proxy of the final runtime;
- treat PAC1 file-work as the current floor, not the final ceiling;
- expect prompt injection to arrive through multiple channels, not only files;
- design the agent around trust boundaries between trusted instructions and untrusted chat, email, or remote content;
- prepare for higher-risk tool classes such as network access and destructive actions;
- move more safety rules from prompt text into runtime checks;
- build local traces and internal debugging now so we are not blocked while waiting for the official debug features to land.

Concrete engineering takeaway:

- the agent architecture should be tool-agnostic and risk-aware;
- it should support multiple tool classes with different safety policies;
- it should treat incoming messages, emails, and remote responses as untrusted by default;
- it should gate destructive actions much more strictly than read-only actions.

## What Follows From The Current PAC1 Sample

### Control Plane

The current sample in `pac1-py/main.py` defaults to:

- `https://api.bitgn.com`;
- benchmark `bitgn/pac1-dev`;
- `start_playground` for task execution;
- `end_trial` for evaluation.

This means the current local code already reflects the real benchmark -> trial -> runtime -> evaluation loop.

### Runtime Surface

The PCM runtime in `proto/bitgn/vm/pcm.proto` exposes a narrow but important tool surface:

- `read`
- `write`
- `delete`
- `mkdir`
- `move`
- `list`
- `tree`
- `find`
- `search`
- `context`
- `answer`

Practical implication:

- the benchmark is heavily about disciplined file-world interaction;
- reading, navigating, editing, and finalizing are more important than generic reasoning;
- `answer` with the correct `outcome` and `refs` is part of the scored behavior.

### Outcomes Must Be Chosen Explicitly

The current agent already models explicit outcomes:

- `OUTCOME_OK`
- `OUTCOME_DENIED_SECURITY`
- `OUTCOME_NONE_CLARIFICATION`
- `OUTCOME_NONE_UNSUPPORTED`
- `OUTCOME_ERR_INTERNAL`

Practical implication:

- safe refusal, clarification, and unsupported cases must be treated as proper success paths when appropriate;
- not every task should be forced into `OK` if that breaks safety or protocol compliance.

### Grounding Is Already Part Of The Contract

The current agent forces the following before it starts solving:

- `tree /`
- `read AGENTS.md`
- `context`

This is a strong pattern:

- it grounds the agent in the workspace layout;
- it loads local rules before active changes;
- it gives the agent time context before decision making.

Practical implication:

- grounding should be strengthened, not weakened;
- if the task requires reading policy or rules files, their refs should be treated as likely required evidence in the final answer.

## The Strongest Practical Signal From Local History

This repository already contains a strong signal: the move to native tool calling plus stricter step discipline was committed with the note `57% -> 100%`.

That, combined with the local one-step checklist, strongly suggests the following priority order:

1. One tool call per step.
2. Runtime-enforced read-before-write.
3. Runtime-enforced read-before-delete.
4. Strict `report_completion` validation.
5. Structured tool errors instead of flat strings.
6. Per-run state tracking for reads, writes, deletes, and grounding refs.
7. Repeated-call detection and tool budgets.

This looks like direct score work, not cosmetic cleanup.

## What To Build First

### P0

- Enforce at most one tool call per assistant step.
- Reject completion without enough evidence.
- Verify that cited refs were actually read in the same run.
- Enforce read-before-write and read-before-delete in code, not only in the prompt.
- Normalize and validate paths before dispatch.

### P1

- Keep a compact machine-readable state summary after each step.
- Add budgets for `tree`, `list`, `search`, `read`, and `write/delete`.
- Detect repeated identical calls with identical results.
- Return structured tool error payloads from the runtime layer.

### P2

- Log step id, args, result, latency, token usage, and error.
- Bucket failures into wrong answer, incomplete grounding, premature completion, tool misuse, loop, and instruction violation.
- Add a simple verifier before final submission.

## What To Double-Check Before Competition Day

- Do not rely on platform feedback during the blind window.
- Run practice tasks with local traces and logs before the event window.
- Test prompt-injection and safe-refusal scenarios explicitly.
- Test exact-output and required-refs behavior.
- Test that the agent does not spam tools and can stop itself.
- Test that non-`OK` outcomes are reported as intentional outcomes, not crashes.

## Bottom Line

The most likely path to a strong PAC result is not "make the model think harder." It is to make the agent more disciplined: fewer unnecessary actions, stronger grounding, explicit safe outcomes, strict completion validation, loop control, and careful handling of refs and side effects.
