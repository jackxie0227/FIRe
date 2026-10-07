"""RL-Games observer for FIRe complete-episode metrics and success checkpoints."""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import torch
from rl_games.common.algo_observer import IsaacAlgoObserver


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return repr(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FireSuccessObserver(IsaacAlgoObserver):
    """Track the latest completed episodes and save the best 100-episode policy."""

    STATE_KEY = "fire_success_resume_state"

    def __init__(self, agent_cfg: dict):
        super().__init__()
        self.agent_cfg = agent_cfg
        config = agent_cfg["params"]["config"].get("success_checkpoint", {})
        self.enabled = bool(config.get("enabled", False))
        self.window_size = int(config.get("window_size", 100))
        self.checkpoint_stem = str(config.get("checkpoint_stem", "Forge_best_success"))
        self.rolling_successes: deque[float] = deque(maxlen=self.window_size)
        self.completed_episodes = 0
        self.best_success_rate = float("-inf")
        self.best_epoch: int | None = None
        self.best_frame: int | None = None
        self.fire_ep_infos: list[dict[str, Any]] = []

    def after_init(self, algo):
        super().after_init(algo)
        self._install_resume_state_hooks()

    def _state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "window_size": self.window_size,
            "rolling_successes": list(self.rolling_successes),
            "completed_episodes": self.completed_episodes,
            "best_success_rate": self.best_success_rate,
            "best_epoch": self.best_epoch,
            "best_frame": self.best_frame,
            "last_lr": float(getattr(self.algo, "last_lr", 0.0)),
            "entropy_coef": float(getattr(self.algo, "entropy_coef", 0.0)),
            "scheduler": {
                "class": type(getattr(self.algo, "scheduler", None)).__name__,
                "attributes": _json_safe(
                    vars(getattr(self.algo, "scheduler", None))
                    if getattr(self.algo, "scheduler", None) is not None
                    else {}
                ),
            },
        }

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        values = [float(item) for item in state.get("rolling_successes", [])]
        self.rolling_successes = deque(values[-self.window_size :], maxlen=self.window_size)
        self.completed_episodes = int(state.get("completed_episodes", len(values)))
        self.best_success_rate = float(state.get("best_success_rate", float("-inf")))
        self.best_epoch = state.get("best_epoch")
        self.best_frame = state.get("best_frame")

        if "last_lr" in state:
            self.algo.last_lr = float(state["last_lr"])
            self.algo.update_lr(self.algo.last_lr)
        if "entropy_coef" in state:
            self.algo.entropy_coef = float(state["entropy_coef"])
        scheduler_attrs = state.get("scheduler", {}).get("attributes", {})
        scheduler = getattr(self.algo, "scheduler", None)
        if scheduler is not None:
            for key, value in scheduler_attrs.items():
                if isinstance(value, (bool, int, float, str)):
                    setattr(scheduler, key, value)

    def _install_resume_state_hooks(self) -> None:
        original_get = self.algo.get_full_state_weights
        original_set = self.algo.set_full_state_weights

        def get_full_state_weights():
            weights = original_get()
            weights[self.STATE_KEY] = self._state_dict()
            return weights

        def set_full_state_weights(weights, set_epoch=True):
            original_set(weights, set_epoch=set_epoch)
            self._load_state_dict(weights.get(self.STATE_KEY, {}))

        self.algo.get_full_state_weights = get_full_state_weights
        self.algo.set_full_state_weights = set_full_state_weights

    def process_infos(self, infos, done_indices):
        super().process_infos(infos, done_indices)
        if not isinstance(infos, dict):
            return
        if "fire_episode" in infos:
            self.fire_ep_infos.append(infos["fire_episode"])
        episode = infos.get("episode")
        if not episode or "success_rate" not in episode:
            return
        successes = torch.as_tensor(episode["success_rate"]).detach().cpu().flatten().tolist()
        self.rolling_successes.extend(float(value) for value in successes)
        self.completed_episodes += len(successes)

    def _log_exact_episode_metrics(self, epoch_num: int) -> None:
        if not self.fire_ep_infos:
            return
        keys = self.fire_ep_infos[0].keys()
        for key in keys:
            values = [torch.as_tensor(info[key]).detach().cpu().flatten() for info in self.fire_ep_infos]
            self.writer.add_scalar(key, torch.cat(values).float().mean().item(), epoch_num)
        self.fire_ep_infos.clear()

    def _git_commit(self) -> str | None:
        repo_root = Path(__file__).resolve().parents[5]
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def _metadata(self, epoch_num: int, frame: int, checkpoint: Path | None = None) -> dict[str, Any]:
        config = self.agent_cfg["params"]["config"]
        metadata = {
            "schema_version": 1,
            "epoch": int(epoch_num),
            "frames": int(frame),
            "window_success_rate": (
                sum(self.rolling_successes) / len(self.rolling_successes)
                if self.rolling_successes
                else None
            ),
            "window_episode_count": len(self.rolling_successes),
            "completed_episodes": self.completed_episodes,
            "configuration": {
                "task": config.get("fire_task_id"),
                "seed": self.agent_cfg["params"].get("seed"),
                "num_actors": config.get("num_actors"),
                "horizon_length": config.get("horizon_length"),
                "minibatch_size": config.get("minibatch_size"),
                "mini_epochs": config.get("mini_epochs"),
                "gamma": config.get("gamma"),
                "tau": config.get("tau"),
                "learning_rate": config.get("learning_rate"),
                "e_clip": config.get("e_clip"),
                "entropy_coef": config.get("entropy_coef"),
            },
            "git_commit": self._git_commit(),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "sha256": sha256_file(checkpoint) if checkpoint and checkpoint.exists() else None,
            "resume_state": self._state_dict(),
        }
        return _json_safe(metadata)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        self._log_exact_episode_metrics(epoch_num)
        if not self.enabled:
            return

        if self.rolling_successes:
            rolling_rate = sum(self.rolling_successes) / len(self.rolling_successes)
            self.writer.add_scalar("Success/rolling_100", rolling_rate, epoch_num)
        else:
            rolling_rate = None

        nn_dir = Path(self.algo.nn_dir)
        nn_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            nn_dir / f"fire_success_state_ep_{int(epoch_num)}.json",
            self._metadata(epoch_num, frame),
        )

        if len(self.rolling_successes) < self.window_size:
            return
        if rolling_rate is None or rolling_rate <= self.best_success_rate:
            return

        self.best_success_rate = rolling_rate
        self.best_epoch = int(epoch_num)
        self.best_frame = int(frame)
        checkpoint_stem = nn_dir / self.checkpoint_stem
        self.algo.save(str(checkpoint_stem))
        checkpoint = checkpoint_stem.with_suffix(".pth")
        self._write_json(
            checkpoint_stem.with_suffix(".json"),
            self._metadata(epoch_num, frame, checkpoint),
        )
        print(
            f"[FIRe] New best {self.window_size}-episode success rate "
            f"{rolling_rate:.4f}; saved {checkpoint}"
        )
