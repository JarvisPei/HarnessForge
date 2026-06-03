# Tool Audit Inventory Boundary

Date: 2026-06-03

This records the first run after adding teacher-owned `tool_audit_cases`. The goal was to test whether teacher-generated semantic coverage checks could catch the inventory tool's operation-generalization failures before blind evaluation.

## Mechanism Change

The diagnosis schema now accepts:

```text
tool_audit_cases: {tool_name: [case, ...]}
```

Each case follows the existing tool test case schema:

```json
{
  "input": {"task": "..."},
  "expected": {"ok": true, "result": 123}
}
```

During atomic patch application, these teacher-owned audit cases run against the staged tool before the bundle can be accepted. This is intentionally meta-level: the framework does not hard-code inventory verbs or benchmark-specific cases. The teacher must propose the semantic coverage checks.

## Run

Cloud command:

```bash
git clean -fd harness
git clean -fdX harness

REQUEST_TIMEOUT_SECONDS=1200 TEACHER_TIMEOUT_SECONDS=1200 WEAK_TIMEOUT_SECONDS=1200 \
REQUEST_MAX_RETRIES=2 REQUEST_RETRY_BACKOFF_SECONDS=5 \
TEACHER_MAX_RETRIES=5 TEACHER_RETRY_BACKOFF_SECONDS=10 \
  python -m agentdistill.benchmark \
  --config configs/benchmark_inventory_focused.yaml \
  --run-id inventory_focused_tool_audit_v2
```

The run also confirmed that invalid JSON relay responses are now retried:

```text
[model-retry] retrying role=weak model=gpt-5.4-mini attempt=1/3 wait_seconds=5 error=invalid_json_response
```

## Result

```text
dev improved = 1 / 2
dev regressed = 0 / 2
blind improved = 0 / 2
blind regressed = 2 / 2
blind runtime effects = 2 / 2
blind runtime wins = 0 / 2
```

Patch/repair telemetry:

```text
accepted = 2
rejected = 1
inner_repair_attempts = 2
inner_repair_accepted = 1
contract_failures = 4
teacher_call_proxy = 5
```

## What Worked

The mechanism itself worked:

```text
teacher emits tool_audit_cases
-> staged tool runs those cases
-> first bundle is rejected when ordinary tests and teacher tool audit fail
-> focused repair receives the contract failures
-> repaired executable bundle is accepted
```

The first rejected bundle failed both the ordinary tool tests and the new teacher tool audit. One teacher audit case caught a missed subtraction cue:

```text
input: "printed 10 sheets..., discarded 35..., shipped 400..."
expected result: 685
actual result: 720
missing operation: subtract discarded 35
```

This confirms that `tool_audit_cases` are connected to the contract gate and can force repair.

## What Failed

The accepted repaired tool still under-covered blind-only subtraction paraphrases:

```text
blind_inventory_badges
expected = 3323
tool_result = 4529
missing operation = subtract handed out 1206

blind_inventory_vouchers
expected = 1809
tool_result = 2909
missing operations = subtract voided 96, subtract redeemed 1004
```

The blind failures were regressions because the weak baseline happened to answer both blind tasks correctly before harness evolution. After harness evolution, the weak model/tool path produced wrong answers.

## Interpretation

This is a negative but useful result.

Teacher-owned tool audit is necessary, but not sufficient. Letting the same teacher generate both the tool and its audit cases improved the contract gate, but the audit distribution was still too close to observed train/dev wording. It caught `discarded` and similar cues, but did not require broader subtractive paraphrase coverage.

The next mechanism should separate implementation and semantic stress generation more strongly. Options:

```text
tool developer teacher -> writes tool
semantic audit teacher/critic -> generates adversarial operation paraphrases
contract gate -> validates tool against both ordinary tests and semantic audit tests
focused repair -> repairs failures while preserving the same audit set
```

This does not require hard-coding inventory verbs. The important change is giving the frontier model a distinct role: generate adversarial semantic coverage tests for claimed operation classes before accepting a tool.
