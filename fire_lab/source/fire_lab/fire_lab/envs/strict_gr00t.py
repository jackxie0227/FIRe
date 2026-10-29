"""Strict GR00T pose-chunk validation for opt-in FIRe environments."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def validate_pose_chunk(response: object, *, num_envs: int, chunk_size: int, device: str) -> torch.Tensor:
    """Return an exact (B, H, 6) chunk, refusing ambiguous server output."""
    keys = ("action.eef_position_delta", "action.eef_rotation_delta")
    if not isinstance(response, Mapping):
        raise ValueError(f"GR00T response must be a mapping, got {type(response).__name__}")
    missing = [key for key in keys if key not in response]
    if missing:
        raise ValueError(f"GR00T response missing action keys: {missing}; present={list(response)}")
    parts = []
    for key in keys:
        try:
            value = torch.as_tensor(response[key], dtype=torch.float32, device=device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"GR00T {key} cannot be converted to a tensor: {exc}") from exc
        expected = (num_envs, chunk_size, 3)
        if tuple(value.shape) != expected:
            raise ValueError(f"GR00T {key} shape {tuple(value.shape)}; expected {expected}")
        if not torch.isfinite(value).all():
            raise ValueError(f"GR00T {key} contains NaN or Inf")
        parts.append(value)
    return torch.cat(parts, dim=-1)


def refresh_episode_chunk(client, observations, parse, *, pending: bool):
    """Drain an old async request before requesting the new episode's first chunk."""
    if pending:
        client.get_result()  # Discard the response associated with the old episode.
    response = client.get_action_sync(observations)
    return response, parse(response)


def require_synchronized_reset(reset_ids, num_envs: int) -> None:
    """The current chunk scheduler indexes all environments from env 0."""
    if len(reset_ids) != num_envs:
        raise RuntimeError(
            f"GearMesh partial reset {reset_ids.tolist()} of {num_envs} environments "
            "cannot safely share a global VLA chunk"
        )
