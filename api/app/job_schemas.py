"""Validated job admission HTTP schemas shared by job and batch routes."""
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .gpu import loras as lora_registry

_NEW_POLICY_FIELDS = {
    "allocation",
    "account_id",
    "instance_id",
    "lifecycle",
    "retain_idle_minutes",
}


def _legacy_lifecycle(policy: "Policy") -> str:
    """Translate explicitly supplied v0.1 lifecycle flags without guessing defaults."""
    if not policy.shutdown_after_job:
        return "retain"
    if policy.idle_grace_minutes:
        return "smart"
    return "save"


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["mock", "colab", "kaggle"] = "mock"
    # Kept for wire compatibility with v0.1 clients.
    shutdown_after_job: bool = True
    max_runtime_minutes: int = Field(default=60, ge=1, le=1440)
    idle_grace_minutes: int = Field(default=0, ge=0, le=30)

    allocation: Literal["auto", "existing"] = "auto"
    account_id: str | None = None
    instance_id: str | None = None
    lifecycle: Literal["save", "batch", "smart", "retain"] | None = None
    retain_idle_minutes: int = Field(default=15, ge=1, le=30)

    @field_validator("account_id", "instance_id")
    @classmethod
    def _uuid_or_none(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if info.field_name == "account_id" and info.data.get("provider") == "mock" and value.startswith("mock-"):
            uuid.UUID(value[5:])
            return value
        try:
            parsed = uuid.UUID(str(value))
        except (AttributeError, ValueError, TypeError) as exc:
            raise ValueError("account_id/instance_id必须是UUID或null") from exc
        return str(parsed)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "Policy":
        if self.allocation == "existing" and self.instance_id is None:
            raise ValueError("allocation=existing必须提供instance_id")
        # Supplying an instance is an unambiguous request to reuse it.  This is
        # intentionally normalized before admission so every caller sees the
        # same allocation invariant.
        if self.instance_id is not None and self.allocation == "auto":
            self.allocation = "existing"

        if "lifecycle" in self.model_fields_set and self.lifecycle is not None:
            legacy_fields = self.model_fields_set & {"shutdown_after_job", "idle_grace_minutes"}
            if legacy_fields and self.lifecycle != _legacy_lifecycle(self):
                raise ValueError("新旧生命周期字段冲突")
        return self

    def model_dump(self, *args, **kwargs):  # noqa: ANN002, ANN003
        """Preserve wire compatibility without reintroducing default conflicts."""
        data = super().model_dump(*args, **kwargs)
        for field in _NEW_POLICY_FIELDS - self.model_fields_set:
            data.pop(field, None)
        # Pydantic materializes legacy defaults during model_dump().  When a
        # v0.2 lifecycle was the only lifecycle input, those defaults were not
        # part of the request and must not become explicit on a second parse.
        if "lifecycle" in self.model_fields_set:
            for field in {"shutdown_after_job", "idle_grace_minutes"} - self.model_fields_set:
                data.pop(field, None)
        return data


def normalize_policy(policy: Policy | dict) -> dict:
    """Return the complete internal policy while preserving the raw request separately."""
    parsed = policy if isinstance(policy, Policy) else Policy.model_validate(policy)
    data = parsed.model_dump()
    lifecycle = parsed.lifecycle if parsed.lifecycle is not None else _legacy_lifecycle(parsed)
    data.update(
        allocation="existing" if parsed.instance_id is not None else parsed.allocation,
        account_id=parsed.account_id,
        instance_id=parsed.instance_id,
        lifecycle=lifecycle,
        retain_idle_minutes=parsed.retain_idle_minutes,
    )
    return data


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str
    model_id: str
    workflow_key: str
    workflow_version: int | str
    params: dict = Field(default_factory=dict)
    asset_ids: list[str] = Field(default_factory=list, max_length=10)
    loras: list[dict] = Field(default_factory=list, max_length=lora_registry.MAX_LORAS)
    preset_id: str | None = None
    execution_policy: Policy = Field(default_factory=Policy)

    def model_dump(self, *args, **kwargs):  # noqa: ANN002, ANN003
        """Use Policy's compatibility serializer for nested HTTP payloads."""
        data = super().model_dump(*args, **kwargs)
        policy_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "by_alias",
                "exclude_unset",
                "exclude_defaults",
                "exclude_none",
                "round_trip",
                "warnings",
                "fallback",
                "serialize_as_any",
            }
        }
        if "execution_policy" in data:
            data["execution_policy"] = self.execution_policy.model_dump(*args, **policy_kwargs)
        return data
