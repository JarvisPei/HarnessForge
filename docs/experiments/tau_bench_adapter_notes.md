# Tau-Bench Adapter Notes

Date: 2026-06-03

These notes summarize the first adapter survey for connecting HarnessForge to
tau2-bench text-mode evaluation.

## Sources Checked

- `docs/cli-reference.md`
- `src/tau2/runner/README.md`
- `src/tau2/registry.py`
- `src/tau2/runner/build.py`
- `src/tau2/agent/llm_agent.py`
- `src/tau2/agent/base_agent.py`
- `src/tau2/orchestrator/orchestrator.py`
- split metadata under `data/tau2/domains/*/split_tasks.json`

## Split Metadata

The starter text domains have official splits.

| domain | official splits |
| --- | --- |
| `airline` | `train: 30`, `test: 20`, `base: 50` |
| `retail` | `train: 74`, `test: 40`, `base: 114` |
| `telecom` | `small: 20`, `train: 74`, `test: 40`, `base: 114`, `full: 2285` |

Recommended mapping:

```text
official train -> HarnessForge teacher-visible probe/dev/repair
official test  -> HarnessForge blind final evaluation
```

Do not use official `test` for teacher diagnosis, repair, prompt revision, or
adapter debugging. For smoke testing, use `small` when available or a small
subset of official `train`.

## Train-Internal Evaluation Discipline

Small task slices are mechanism probes, not benchmark claims. A three-task
slice is useful when it reveals whether a new harness permission has any
runtime effect, such as no-tool fallback, pre-tool-call guarding, or candidate
state. It is not sufficient evidence that the harness generalizes.

Recommended tau-bench development protocol:

```text
diagnosis slice       official train tasks visible to teacher
train-heldout dev     official train tasks not shown to teacher, used for iteration signals
official test         final blind evaluation only after freezing the harness
```

Use fixed task ids for each slice instead of always taking the first N tasks.
This prevents the same examples from silently serving as both teacher-visible
diagnosis and dev evaluation. If a mechanism cannot produce runtime effect on
the diagnosis slice, it is usually not worth scaling. If it does produce effect,
the next question must be train-heldout transfer, not more local tuning on the
same examples.

For airline cancellation/candidate-selection work, the initial convention is:

```text
diagnosis examples: task ids seen in teacher traces, currently task 1
sanity examples: task ids 0 and 3 for regression checks around known workflows
train-heldout dev: later official train task ids selected by failure family
```

The teacher must not receive official test traces or train-heldout dev traces
before a harness is frozen for that evaluation round.

## Runner Structure

tau2-bench has a layered runner:

```text
simulation.py -> pure run_simulation(orchestrator)
build.py      -> build environment, agent, user, orchestrator
batch.py      -> run_domain/run_tasks/run_single_task with saving, retries, concurrency
```

The recommended API for new code is the modular `tau2.runner` package, not the
legacy monolithic `tau2.run` module.

For HarnessForge, programmatic runner use is preferable to shelling out to the
CLI because we need direct access to trajectories, reward info, traces, and
patch metadata.

## Agent Registration

The registry supports custom agent factories:

```text
registry.register_agent_factory(factory, name)
```

The factory signature is:

```text
factory(tools, domain_policy, **kwargs) -> agent instance
```

`build_agent()` obtains the official environment tools and policy, then calls
the selected factory with:

```text
tools=environment.get_tools()
domain_policy=environment.get_policy()
llm=config.llm_agent
llm_args=config.llm_args_agent
task=task
```

This is a clean integration point for HarnessForge. The first adapter should
register a `harnessforge_agent` factory after importing tau2-bench, without
patching tau2-bench source files.

## Agent Interface

For text-mode runs, a custom agent should implement the `HalfDuplexAgent`
interface:

```text
get_init_state(message_history=None)
generate_next_message(message, state) -> (AssistantMessage, state)
```

The default `llm_agent` builds a system prompt from:

```text
agent instruction + domain policy
```

and calls the model with the official tool list and message history.

HarnessForge can replace this with:

```text
domain policy
+ teacher-generated skills/guidelines
+ compact state summaries
+ runtime policy decisions
+ weak model call
```

The adapter should preserve the official communication rule: each assistant
message is either a user-facing text message or tool call(s), never both.

## Tool Boundary

Tool execution is owned by the tau2 environment. The orchestrator routes agent
tool calls to:

```text
environment.get_response(tool_call)
```

This means teacher-generated helper tools cannot be exposed naively as extra
LLM tools unless the tau2 environment can also execute them.

First adapter rule:

```text
teacher-generated runtime policies may force official tau2 tool calls only
```

Teacher-generated helper tools should remain disabled for tau-bench until one
of these is implemented:

1. an environment wrapper that can execute both official tau2 tools and
   HarnessForge helper tools, or
2. an internal agent-side helper mechanism that never emits helper tool calls to
   the tau2 orchestrator.

This prevents unknown-tool errors and keeps benchmark environment behavior
controlled.

## Recommended First Adapter

Implement a custom half-duplex agent wrapper.

```text
HarnessForgeTauAgent(HalfDuplexAgent)
  - receives official tau2 tools and domain policy
  - loads current HarnessForge harness snapshot
  - builds weak-model messages from tau2 messages
  - adds teacher-generated skills/guidelines/state summaries
  - optionally applies runtime policy before weak-model generation
  - returns tau2 AssistantMessage
```

Reasons:

- It gives HarnessForge enough control to inject harness artifacts.
- It preserves tau2's official environment, user simulator, task loading, and
  evaluator.
- It avoids modifying tau2-bench source.
- It keeps text-mode support separate from voice/full-duplex support.

Avoid using only the CLI for the main loop because CLI output is too indirect
for teacher diagnosis and repair. Avoid wrapping `llm_agent` too tightly because
that would mostly restrict HarnessForge to prompt injection.

## Initial Implementation Steps

1. Add an optional tau-bench integration module under HarnessForge. Done:
   `agentdistill/tau_bench.py`.
2. Register `harnessforge_agent` with tau2's registry at runtime. Done in the
   smoke adapter.
3. Run a weak baseline on official `airline` train tasks.
4. Convert tau2 `SimulationRun` objects into HarnessForge traces. Done in the
   smoke adapter.
5. Run teacher diagnosis only on official `train` traces.
6. Apply accepted prompt/skill/runtime-policy patches.
7. Re-run held-out official `train` tasks for repair/dev.
8. Run official `test` only after the harness is frozen.

Smoke command:

```bash
TAU2_DATA_DIR=$HOME/projects/tau2-bench/data \
python -m agentdistill.tau_bench \
  --domain airline \
  --split train \
  --num-tasks 2 \
  --output-dir outputs/tau_bench_smoke/airline_train_2
```

Set `TAU_USER_LLM` or pass `--user-llm` for the tau2 user simulator. The
HarnessForge agent uses the normal `WEAK_*` environment variables.

Runtime setup note:

- `pip install git+https://github.com/sierra-research/tau2-bench.git` installs
  the Python package.
- The wheel does not provide the benchmark data directory in the expected place.
  Keep a source checkout and set `TAU2_DATA_DIR=$HOME/projects/tau2-bench/data`
  for smoke/evaluation runs.

## First Cloud Smoke

Date: 2026-06-03

Before the canonical smoke, the cloud checkout was cleaned with:

```bash
git clean -fd harness
git clean -fdX harness
```

This removed old generated harness artifacts so the tau-bench adapter smoke was
not influenced by prior synthetic-benchmark harness files.

Cloud VM command shape:

```bash
TAU2_DATA_DIR=$HOME/projects/tau2-bench/data \
python -m agentdistill.tau_bench \
  --domain airline \
  --split train \
  --num-tasks 1 \
  --user-llm gpt-5.5 \
  --output-dir outputs/tau_bench_smoke/airline_train_1_clean \
  --max-steps 40 \
  --max-errors 3 \
  --timeout 600
```

Result:

```text
task_id = 0
termination_reason = user_stop
reward = 1.0
messages = 12
official tool call observed = get_reservation_details({"reservation_id": "EHGLP3"})
official tool call observed = transfer_to_human_agents({...})
trace path = outputs/tau_bench_smoke/airline_train_1_clean/0.json
```

Interpretation:

- the custom `harnessforge_agent` was accepted by tau2's half-duplex
  orchestrator
- the weak model produced a parseable JSON tool call
- tau2 executed the official environment tool and returned the result
- the adapter exported a HarnessForge trace with messages and reward info
- no teacher diagnosis or harness evolution was used in this smoke

## First Supported Harness Artifact Types

Supported in first adapter:

- prompt guideline
- skill
- state summary format
- runtime policy that selects official tau2 tools
- validator over final answer or trajectory metadata

Deferred:

- teacher-generated executable helper tools
- tau2 environment wrappers
- voice/full-duplex agents
- banking knowledge retrieval harnesses

## First Experiment Shape

Start with `airline` for integration smoke:

```text
domain: airline
split: train
num_tasks: 2 to 5
teacher: disabled
goal: weak baseline traces and result parsing
```

Then move to focused `retail`:

```text
domain: retail
teacher-visible train tasks: 10
repair/dev train tasks: 10
blind test tasks: 10
artifact types: prompt/skill/runtime-policy only
```

Only after this produces a stable positive or diagnostic negative result should
the full `retail` test split be used.

## Open Questions

- What is the cleanest way to serialize tau2 messages into the existing
  HarnessForge trace schema?
- Should state summaries be generated deterministically by HarnessForge or by
  the teacher as a harness artifact?
- How much domain policy should be passed to the teacher during diagnosis?
- Should the weak model see full policy text every turn, or a teacher-generated
  policy index/summary?
- When should teacher-generated helper tools be enabled through an environment
  wrapper?

## Decision

Proceed with a custom text-mode `harnessforge_agent` using tau2's programmatic
runner and official train/test splits. Keep teacher-generated helper tools out
of the first tau-bench adapter. The next engineering task is to run a slightly
larger weak baseline slice on official `airline` or `retail` train tasks and
convert those traces into teacher diagnosis inputs.

## First Weak Baseline Slice

Date: 2026-06-03

Cloud VM output:

```text
outputs/tau_bench_baseline/airline_train_3_step20_v1
```

Command shape:

```bash
TAU2_DATA_DIR=$HOME/projects/tau2-bench/data \
python -m agentdistill.tau_bench \
  --domain airline \
  --split train \
  --num-tasks 3 \
  --user-llm gpt-5.5 \
  --output-dir outputs/tau_bench_baseline/airline_train_3_step20_v1 \
  --max-steps 20 \
  --max-errors 3 \
  --timeout 300
```

Result:

| task_id | termination | reward | tool calls | observed pattern |
| --- | --- | ---: | ---: | --- |
| `0` | `max_steps` | `0.0` | `0` | Repeatedly asked how to help after the user provided reservation code `EHGLP3`; later repeated the slot request after user id `emma_kim_9957` was provided. |
| `1` | `max_steps` | `0.0` | `0` | Repeated generic help prompts after the user provided cancellation/refund intent and user id `raj_sanchez_7340`. |
| `3` | `max_steps` | `0.0` | `0` | Repeatedly requested user id and intent after the user provided user id `anya_garcia_5901` and modify intent. |

Interpretation:

- the adapter remained stable on multiple official train tasks
- all observed failures reached the tau2 orchestrator and evaluator cleanly
- the weak agent did not emit official tau2 tool calls in any failed trace
- the recurring behavior is a real harness gap around conversation-state
  tracking and official-tool activation, not a final-answer formatting issue

This is the right failure shape for the next HarnessForge step. The teacher
should receive compact train-trace evidence and decide whether to improve the
weak harness through a skill, state representation, validator, or a runtime
policy that triggers official tau2 tools only. Do not manually write the tau
harness content from this note, and do not use official test traces for repair.

## Teacher Probe And Repair

Date: 2026-06-03

The train digest was passed to the teacher with the real tau2 weak system
prompt and official tool specs. The teacher proposed a minimal executable
bundle:

- `harness/runtime_policies/tau_airline_force_initial_lookup.py`
- `harness/tests/tau_airline_force_initial_lookup.json`

The first validation rejected the bundle because the reservation-code extractor
was too permissive and misread ordinary workflow words like `REFUND`,
`MODIFY`, and `FLIGHT` as reservation codes.

After feeding that rejection back through `benchmark_context.patch_feedback`
and `repair_scope`, the teacher repaired the same bundle. The repaired version
passed contract validation in a clean staging repo.

Interpretation:

- the teacher can now operate as an architect over a real benchmark digest
- contract feedback is strong enough to repair a rejected runtime-policy bundle
- the validation gate caught a bad extraction rule before it could land
- the next cloud step should decide whether this policy is worth landing in
  the canonical harness or whether a broader state-representation layer is a
  better first repair for tau-bench

## Runtime Hook Impact Probe

Date: 2026-06-03

Before running the teacher-generated runtime policy, the tau adapter was fixed
to append incoming user and tool messages into the weak agent state. The older
`0/3` baseline was therefore not a valid behavior baseline: the weak model had
not been seeing the user messages correctly.

After the adapter fix and with a clean harness, the same `airline train`
three-task slice produced:

| condition | task 0 | task 1 | task 3 | aggregate |
| --- | ---: | ---: | ---: | ---: |
| clean harness | `1.0` | `0.0` | `1.0` | `2/3` |
| teacher runtime policy installed | `1.0` | `0.0` | `1.0` | `2/3` |

The installed policy passed validation but did not fire in these corrected
traces. The weak model now calls official tau2 tools by itself. The remaining
failure on task `1` is higher level: after `get_user_details`, the weak agent
iterates over multiple reservations and fails to efficiently match the user's
route description, Philadelphia to LaGuardia, before hitting `max_steps`.

Interpretation:

- adapter state handling was an infrastructure bug and is now fixed
- the first teacher policy solved the old no-tool-activation failure, but that
  failure disappeared once user context was wired correctly
- the next useful teacher target is not initial tool activation; it is a
  state/selection harness that helps the weak model compare user-described
  constraints against a set of official reservation records
- this should be generated from the corrected train failure, not manually
  encoded from task `1`

## Candidate Selection Skill Probe

Date: 2026-06-03

The corrected task `1` failure was passed to the teacher with the full train
trajectory and the note that the previous activation policy did not fire. The
teacher generated a prompt-only skill:

```text
harness/skills/tau_airline_candidate_reservation_selection.md
```

The skill taught the weak model to maintain a private candidate checklist after
`get_user_details`, inspect candidate reservations one by one with official
tools, compare route constraints against official reservation details, and avoid
showing internal tool-call JSON to the user.

Impact on the same three official train tasks:

| condition | task 0 | task 1 | task 3 | aggregate |
| --- | ---: | ---: | ---: | ---: |
| clean harness after adapter fix | `1.0` | `0.0` | `1.0` | `2/3` |
| activation runtime policy | `1.0` | `0.0` | `1.0` | `2/3` |
| candidate-selection skill v1 | `1.0` | `1.0` | `0.0` | `2/3` |
| repaired candidate-selection skill | `1.0` | `0.0` | `1.0` | `2/3` |

Interpretation:

- the teacher found the right high-level axis: candidate reservation selection
  and private state tracking
- a text-only skill was enough to flip the original task `1` failure from
  `0.0` to `1.0`
- the first skill regressed task `3` by changing an exact baggage workflow into
  a generic profile/cabin answer without first identifying the reservation
- teacher feedback repaired task `3`, but task `1` fell back to `max_steps`
- prompt-only skill is therefore too unstable as the sole mechanism for this
  benchmark family

Next direction:

The next tau-bench harness should give the teacher a stronger state/selection
mechanism than free-form text. A useful target is an agent-side candidate-state
runtime layer that tracks:

- user constraints
- candidate reservation ids
- checked and rejected reservation ids
- mismatch reasons
- currently selected reservation
- which final action or exact answer is waiting on the selected reservation

This should still only call official tau2 tools, but it should reduce reliance
on the weak model remembering the candidate checklist in natural language.

## Candidate-State Runtime Policy Probe

Date: 2026-06-03

The next teacher attempt used the same corrected train evidence and generated
an executable candidate-state runtime policy:

```text
harness/runtime_policies/tau_airline_candidate_state.py
harness/tests/tau_airline_candidate_state.json
```

The bundle passed manifest validation, runtime policy tests with conversation
metadata, and the teacher policy audit. It was installed into the cloud harness
and evaluated on the same three official `airline train` tasks.

Impact:

| condition | task 0 | task 1 | task 3 | aggregate |
| --- | ---: | ---: | ---: | ---: |
| clean harness after adapter fix | `1.0` | `0.0` | `1.0` | `2/3` |
| candidate-state runtime policy | `1.0` | `0.0` | `1.0` | `2/3` |

The task `1` failure changed shape. Instead of hitting `max_steps`, the weak
agent ended with `user_stop` and reward `0.0` after directly calling
`cancel_reservation` on the wrong candidate reservation (`MZDDS4`). The policy
did not get a chance to intervene because the current adapter only evaluated
runtime policies when the weak model produced no parseable tool call.

Interpretation:

- the teacher-generated policy was valid but attached to the wrong runtime
  interception point
- no-tool fallback is insufficient for tau-bench failures where the weak model
  emits a parseable but premature or wrong official tool call
- the next adapter capability should let runtime policies inspect the weak
  model's proposed tool call before execution and replace it with another
  official tau2 tool when conversation/tool-result state shows the call is not
  justified

This is a harness permission upgrade rather than a hand-written domain fix:
the teacher should still generate the tau-specific guard policy from train
evidence, while the adapter only supplies the generic pre-tool-call hook and
preserves the official tau2 tool boundary.

## Pre-Tool Guard Runtime Effect Probe

Date: 2026-06-03

After adding the generic pre-tool-call runtime policy hook, the task `1`
failure trace was passed back to the teacher. The teacher generated and the
framework accepted:

```text
harness/runtime_policies/tau_airline_cancel_route_guard.py
harness/tests/tau_airline_cancel_route_guard.json
```

The policy acts as a guard over proposed `cancel_reservation` calls. If the
conversation and prior tool results do not prove that the proposed reservation
matches the user's requested route, it replaces the destructive cancel call
with an official `get_reservation_details` call.

Impact on the same three official `airline train` tasks:

| condition | task 0 | task 1 | task 3 | aggregate |
| --- | ---: | ---: | ---: | ---: |
| clean harness after adapter fix | `1.0` | `0.0` | `1.0` | `2/3` |
| pre-tool cancel guard | `1.0` | `0.0` | `1.0` | `2/3` |

The aggregate did not improve, but the mechanism did have runtime effect. In
task `1`, the guard intercepted repeated proposed
`cancel_reservation({"reservation_id": "MZDDS4"})` calls and replaced them with
`get_reservation_details({"reservation_id": "MZDDS4"})`. This prevented the
wrong destructive action from reaching the tau2 environment, but it did not
advance candidate search to the next unchecked reservation. The weak model
therefore looped over the same mismatching reservation until `max_steps`.

Additional adapter observation:

One weak response mixed user-facing text with embedded tool JSON:

```text
I’m cancelling your reservation now.{"tool_call":{"name":"cancel_reservation",...}}
```

The current parser treats this as user-facing text because parseable tool calls
must occupy the full response. That preserves tau2's half-duplex protocol, but
it means malformed mixed responses bypass proposed-tool-call guard logic unless
a separate protocol-normalization layer extracts or rejects embedded tool JSON.

Interpretation:

- pre-tool-call guard is a real new teacher permission: it changed the executed
  tau2 tool sequence without manually coding the domain policy
- the first guard is too conservative because it rechecks the same bad
  candidate rather than selecting the next unchecked candidate from user
  details
- the next useful harness capability is either a stronger teacher-generated
  candidate-state policy that tracks checked/rejected reservation ids and
  advances to the next candidate, or a generic adapter protocol-normalization
  hook that handles mixed text/tool JSON before policy evaluation

## Train-Heldout First Attempt

Date: 2026-06-04

After adding fixed task-id selection, the first train-heldout attempt targeted
official `airline train` task ids `4`, `5`, and `7`, which were not used as the
teacher-visible task `1` diagnosis trace. The clean run used an empty runtime
policy directory so the generated cancel guard could not affect the baseline.

The first task, `4`, already exposed a related but distinct candidate-search
family: the user supplied a user id but no reservation id and asked about
compensation for a canceled business flight. The weak model retrieved the user
details and began scanning reservations, but the task timed out:

```text
task 4 clean heldout: reward=0.0, termination=timeout, messages=12
```

The attempted online slice was stopped during task `5` because the run was too
slow for interactive iteration. Task `5` also produced another mixed
text-plus-tool response:

```text
I’m locating your reservation ID now.{"tool_call": {"name": "get_user_details", ...}}
```

Interpretation:

- train-heldout evaluation is the right next protocol, but a three-task online
  slice is already slow enough that future heldout sweeps should run in the
  background or use a smaller per-turn/task budget
- the heldout task `4` supports the broader diagnosis that candidate-state
  search, not only cancellation guarding, is a recurring airline failure family
- mixed text/tool JSON is a generic adapter-protocol issue and should be
  normalized before proposed-tool-call guard evaluation

Follow-up adapter support:

Runtime policy payloads now include compact structured state under
`metadata.messages` in addition to the plain `metadata.conversation` transcript.
This gives teacher-generated policies a stable way to inspect prior tool calls
and tool results, track checked/rejected candidates, and decide whether a
proposed destructive action is justified without parsing the entire transcript
as free text.

## Structured Candidate-State Policy Probe

Date: 2026-06-04

After adding `metadata.messages`, the teacher was given official-train
diagnosis evidence from task `1` and task `4`. Task `5` and task `7` were held
out from the teacher prompt for this round. The teacher generated:

```text
harness/runtime_policies/tau_airline_candidate_state.py
harness/tests/tau_airline_candidate_state.json
```

The bundle passed manifest validation, runtime policy tests, and teacher policy
audit. It uses structured runtime metadata to force official tau tool calls for
candidate-state search:

- force `get_user_details` when the user supplies a user id but no candidate
  list is present
- force `get_reservation_details` for the next unchecked reservation id from
  the user record
- avoid repeating already checked candidates when metadata shows prior
  reservation-detail results

Bounded task `1` diagnosis replay:

```text
output: outputs/tau_bench_policy/airline_train_1_candidate_state_structured_v1/1.json
reward: 0.0
termination: timeout
messages: 10
```

Runtime effect:

```text
forced get_user_details({"user_id": "raj_sanchez_7340"})
checked MZDDS4
forced get_reservation_details({"reservation_id": "60RX9E"})
```

This is better than the previous cancel guard, which could recheck the same bad
candidate. The new policy advanced to the next unchecked candidate, but the run
timed out before completing the full candidate list.

Heldout train task `5`:

```text
output: outputs/tau_bench_policy/airline_train_5_candidate_state_structured_v1/5.json
reward: 0.0
termination: max_steps
messages: 15
```

Runtime effect:

```text
get_user_details(mei_brown_7075)
get_reservation_details(DB1Y70)
get_reservation_details(MUGYUB)
get_reservation_details(3JA7XV)
get_reservation_details(CYPIDV)
```

The heldout transfer signal is meaningful: the policy advanced through the
candidate list on a task the teacher did not see. It found the matching
reservation `3JA7XV`, whose tool result contained flight `HAT045` from `PHX` to
`SEA`, matching the user's request. However, the policy still forced the next
unchecked candidate `CYPIDV` instead of stopping candidate search and allowing
the weak model to finalize from the matched evidence.

Interpretation:

- structured metadata gave teacher-generated harness code enough state to
  produce real train-heldout runtime effect
- the remaining failure is no longer "cannot search candidates"; it is
  "does not recognize matched-candidate evidence as a stop condition"
- the next repair should be focused on the same policy/test: when
  `metadata.messages` shows a checked reservation matching the user's requested
  flight/route, return `requires_tool: false` instead of forcing the next
  unchecked candidate
- a compact focused repair prompt was attempted, but teacher API calls timed
  out repeatedly; future repair runs should use a background job, shorter
  provider timeout, or an even more compressed repair payload
