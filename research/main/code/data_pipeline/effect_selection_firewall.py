"""Fail-closed checks separating model effects from dataset selection.

This module implements the result-selection boundary in report 1082. It does
not decide semantic quality. It only verifies that confirmatory manifests were
not selected using experimental-model outcomes, and that any effect-filtered
training version declares its unfiltered parent and frozen rule.
"""

from __future__ import annotations

from collections.abc import Mapping
import re


CONFIRMATORY_ROLES = frozenset(
    {
        "G0-locked-method-selection",
        "G1-locked-mechanism",
        "Stage1-validation",
        "Stage1-internal-test",
        "external-benchmark",
    }
)

TRAIN_ROLE = "Stage1-train"

EFFECT_FIELD_PREFIXES = (
    "teacher_gap",
    "model_utilization",
    "condition_effect",
    "evaluation_effect",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _has_effect_value(record: Mapping[str, object]) -> list[str]:
    fields: list[str] = []
    for key, value in record.items():
        if value is None or value == "":
            continue
        if any(str(key).startswith(prefix) for prefix in EFFECT_FIELD_PREFIXES):
            fields.append(str(key))
    return sorted(fields)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def validate_effect_selection_firewall(record: Mapping[str, object]) -> list[str]:
    """Return selection-firewall violations for one manifest record.

    Confirmatory roles must state ``selection_uses_model_effect=false`` and
    must not carry non-null effect-derived fields. Stage1 training may declare
    a secondary effect-filtered version, but only with an unfiltered parent
    manifest SHA and a frozen selection-rule SHA.
    """

    errors: list[str] = []
    role = record.get("dataset_role")
    uses_effect = record.get("selection_uses_model_effect")
    effect_fields = _has_effect_value(record)

    if role in CONFIRMATORY_ROLES:
        if uses_effect is not False:
            errors.append("confirmatory_role_requires_selection_uses_model_effect_false")
        if effect_fields:
            errors.append(
                "confirmatory_role_contains_effect_fields:" + ",".join(effect_fields)
            )

    if uses_effect is True:
        if role != TRAIN_ROLE:
            errors.append("effect_filtered_selection_only_allowed_for_stage1_train")
        if not _valid_sha256(record.get("parent_unfiltered_manifest_sha256")):
            errors.append("missing_or_invalid_parent_unfiltered_manifest_sha256")
        if not _valid_sha256(record.get("effect_selection_rule_sha256")):
            errors.append("missing_or_invalid_effect_selection_rule_sha256")

    if role == TRAIN_ROLE and uses_effect not in (True, False):
        errors.append("stage1_train_requires_explicit_selection_uses_model_effect")

    return errors
