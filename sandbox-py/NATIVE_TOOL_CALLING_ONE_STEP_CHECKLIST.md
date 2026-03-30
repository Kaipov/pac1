# Native Tool Calling One-Step Checklist

This checklist is tailored to the current `agent_tc.py` loop.

The current branch already has a ReAct-like loop:

- model sees the task and tool history
- model calls tool(s)
- runtime executes them
- tool output goes back into history
- model decides the next step

The main change for a stricter "one-step" agent is this:

- one assistant turn should produce at most one tool call
- every tool result should force a fresh re-evaluation before the next action

## P0: Core reliability changes

- Enforce one tool call per step.
  If the model returns multiple `tool_calls`, reject the extra calls and feed back a tool error or policy message so the model retries with one action only.

- Treat `report_completion` as a gated terminal action.
  Allow completion only after the agent has gathered enough evidence for the answer and has non-empty `grounding_refs` when the task required reading files.

- Add explicit completion checks in the runtime.
  Validate exact-answer tasks, required refs, and obvious policy requirements before accepting `report_completion`.

- Make tool execution stateful and explicit.
  Track:
  `files_read`, `files_written`, `files_deleted`, `grounding_refs`, `steps_taken`, `tool_errors`.

- Enforce read-before-write and read-before-delete in code, not only in prompt.
  The current prompt says this, but runtime checks are more reliable than prompt-only rules.

- Normalize and validate paths before dispatch.
  Reject malformed paths, duplicated slashes, and any path outside the allowed workspace root.

## P1: Better agent behavior

- Replace unlimited free-form thinking turns with a stricter step contract.
  A non-terminal step should usually do exactly one of:
  call one tool,
  or call `report_completion`.

- Keep a short machine-readable state summary in context.
  After each step, append a compact summary such as:
  `Read: AGENTS.MD, docs/a.md | Pending: verify amount | Errors: none`
  This reduces drift and helps the model re-plan from observations instead of raw transcript alone.

- Add a per-tool budget.
  Example:
  `tree`: 2
  `list`: 4
  `search`: 6
  `read`: 12
  `write/delete`: 3
  This prevents loops and encourages deliberate tool use.

- Add a repeated-action detector.
  If the agent calls the same tool with the same args repeatedly and gets the same result, stop and ask it to choose a different action.

- Turn tool errors into structured observations.
  Instead of plain `ERROR: ...`, return a JSON payload with fields like `code`, `message`, `retryable`, `tool_name`, `args`.

## P1: Safer completion behavior

- Add a final verifier pass before accepting `report_completion`.
  The verifier can be simple at first:
  check exact-match instructions,
  check that cited refs were actually read,
  check that the answer does not add unsupported claims.

- Separate "I am done" from "submit final answer".
  One pattern is:
  `finalize_candidate_answer`
  followed by verifier approval,
  then `report_completion`.
  This avoids premature finish calls.

- Require the agent to cite evidence gathered in the same run.
  Do not allow refs that were never observed in this trajectory.

## P2: Observability and evals

- Log every step in a structured trace.
  At minimum store:
  `step_id`, `assistant_text`, `tool_name`, `tool_args`, `tool_result`, `latency_ms`, `token_usage`, `error`.

- Add benchmark-side counters.
  Useful metrics:
  completion rate,
  exact-answer rate,
  average steps,
  repeated-call rate,
  premature-completion rate,
  tool-error recovery rate.

- Save failure cases by type.
  Good buckets:
  wrong final answer,
  incomplete grounding,
  premature completion,
  tool misuse,
  loop,
  instruction violation.

## P2: Prompt and interface cleanup

- Move policy from prose into runtime where possible.
  Prompts should guide reasoning, but hard invariants should live in code.

- Make tool descriptions more operational.
  Example:
  `report_completion` should explicitly say when it is forbidden:
  "Do not call until all required files are read and the answer is fully supported."

- Consider renaming `tree` to `outline` if you want the tool name to match the VM method and reduce ambiguity in traces.

## Suggested implementation order

1. Enforce one tool call per step.
2. Add runtime state tracking.
3. Gate `report_completion` with validator checks.
4. Return structured tool errors.
5. Add repeated-action detection and budgets.
6. Add verifier before final submission.
7. Add structured traces and benchmark metrics.

## Minimal acceptance criteria for this branch

- One tool call max per assistant turn
- Runtime-enforced read-before-write/delete
- Runtime-enforced completion validation
- Structured tool error payloads
- Basic per-run state tracking

If you want, the next change in this branch can be implementing items 1-4 directly in `agent_tc.py`.
