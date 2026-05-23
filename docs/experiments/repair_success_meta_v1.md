# Meta-Skill Harness Repair Run

Date: 2026-05-15

This records the first inventory benchmark run after removing domain-specific parser scaffolds from the teacher diagnosis prompt and replacing them with domain-agnostic meta-skills.

## Run

Server workspace:

```text
clean server workspace for the meta-inventory run
```

Command:

```bash
REQUEST_TIMEOUT_SECONDS=120 TEACHER_TIMEOUT_SECONDS=300 WEAK_TIMEOUT_SECONDS=120 \
  python -m agentdistill.benchmark \
  --config configs/benchmark_inventory.yaml \
  --run-id repair_success_meta_v1
```

The workspace was a clean clone at commit `572beef`.

## Result

Patch summary:

```text
train_steps = 3
accepted = 2
rejected = 1
accepted_code_manifest_bundles = 2
accepted_tool_test_policy_bundles = 1
contract_failures = 2
```

Transfer:

```text
dev_transfer.improved = 2
dev_transfer.regressed = 0
blind_transfer.improved = 0
blind_transfer.regressed = 0
```

## Repair Sequence

Iteration 1 proposed a full code bundle:

```text
bundle_id = inventory_arithmetic_tool_policy
status = rejected
```

The teacher generated:

```text
harness/tools/inventory_arithmetic.py
harness/tests/inventory_arithmetic.json
harness/runtime_policies/force_inventory_arithmetic.py
harness/tests/force_inventory_arithmetic.json
```

The contract gate rejected the bundle because the parser mishandled a comma-formatted later quantity, producing `2919` instead of `1813` on a heldout-style tool test.

Iteration 2 repaired the same bundle id:

```text
bundle_id = inventory_arithmetic_tool_policy
status = accepted
```

All contracts passed:

```text
manifest matches patch bundle
tool test file exists and all cases passed
forced tool call succeeded
all policy tests passed
```

Iteration 3 added a tool repair:

```text
bundle_id = inventory_arithmetic_subtract_verbs
status = accepted
```

This expanded subtractive verb handling in the generated tool and passed tool tests.

## Interpretation

This run confirms the main mechanism still works without a hard-coded inventory parser scaffold:

```text
teacher infers a code harness
-> contract gate rejects a bad parser
-> patch_feedback reaches the next teacher iteration
-> teacher repairs the bundle
-> accepted tool/runtime policy improves weak-model dev transfer
```

The blind tasks did not improve because the accepted runtime policy only triggered for a narrow inventory noun set. It recognized labels, tags, stickers, items, cards, and tickets, but not blind nouns such as badges and vouchers. The generated tool was closer to transferable than the policy trigger.

## Next Signal

The next useful goal is not to add more inventory-specific nouns by hand. The better direction is to give the teacher a meta-level harness audit step or adversarial policy-test generation capability that asks whether a runtime policy trigger is overfit to observed surface nouns, labels, or entities before the bundle is accepted.
