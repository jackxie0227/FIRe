"""Pure FIRe-official action, reward, and success helpers.

These helpers intentionally have no Isaac Sim dependency.  Keeping the equations
in one small module makes the website-aligned semantics directly unit testable.
"""

from __future__ import annotations

import torch


def force_reward_terms(
    contact_force: torch.Tensor,
    force_reference: torch.Tensor,
    previous_potential: torch.Tensor,
    *,
    alpha: float = 1.0e-5,
    shaping_lambda: float = 1.0e-5,
    gamma: float = 0.995,
    potential_c: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return over-force, penalty, shaping, and the current potential."""

    contact_over = torch.clamp_min(contact_force - force_reference, 0.0)
    potential = -torch.log(potential_c * contact_over + 1.0)
    penalty = -alpha * contact_over
    shaping = shaping_lambda * (gamma * potential - previous_potential)
    return contact_over, penalty, shaping, potential


def peg_insert_success_mask(
    lateral_error: torch.Tensor,
    z_displacement: torch.Tensor,
    *,
    lateral_tolerance_m: float = 0.005,
    depth_tolerance_m: float = 0.025,
) -> torch.Tensor:
    """FIRe project-page PegInsert success tolerances (inclusive)."""

    return torch.logical_and(
        lateral_error <= lateral_tolerance_m,
        z_displacement <= depth_tolerance_m,
    )
