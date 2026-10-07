"""Pure helpers for composing VLA and residual actions."""

from __future__ import annotations

import torch


def compose_vla_residual_actions(
    vla_action: torch.Tensor,
    residual_action: torch.Tensor,
    residual_limit: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip only the residual, then form the unscaled, unclipped direct sum."""

    if vla_action.shape != residual_action.shape:
        raise ValueError(
            "VLA and residual action shapes must match, got "
            f"{tuple(vla_action.shape)} and {tuple(residual_action.shape)}"
        )
    if residual_limit is None:
        clipped_residual = residual_action
    else:
        if residual_limit <= 0:
            raise ValueError("residual_limit must be positive")
        clipped_residual = torch.clamp(residual_action, -residual_limit, residual_limit)
    return clipped_residual, vla_action + clipped_residual
