"""GRPO (Group Relative Policy Optimization) training.

Proper implementation with:
- Reference model for KL penalty
- Clipped importance ratio (PPO-style)
- Tournament collection: multi-round ratcheting with reward backpropagation
- Domain-agnostic: uses Domain protocol for evaluation and query handling
"""
from __future__ import annotations

import math
import os
import random
import time
from typing import Any, TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .base import Domain, Trajectory
from .generation import generate_with_queries
from .model import (
    get_logits, logits_at_last, save_checkpoint, clone_model, param_count,
)

if TYPE_CHECKING:
    pass


def _build_prompt(domain: Domain, task: Any, current_expr: str,
                  feedback_type: str, feedback_acc: float) -> list[int]:
    """Build a prompt for generation from the current refinement state."""
    prompt = [domain.bos_id, domain.sep_task_id]
    prompt.extend(domain.encode_task_prefix(task))
    prompt.append(domain.sep_expr_id)
    prompt.extend(domain.tokenize_program(current_expr))
    prompt.append(domain.sep_feedback_id)
    fb_id = domain.feedback_token_ids.get(feedback_type)
    if fb_id is not None:
        prompt.append(fb_id)
    if feedback_type == "WRONG":
        prompt.extend(domain.accuracy_tokens(feedback_acc))
    prompt.append(domain.sep_edit_id)
    return prompt


def collect_trajectory(
    model: nn.Module,
    domain: Domain,
    task: Any,
    max_steps: int = 4,
    temperature: float = 0.8,
    start_expr: str = "_HOLE_",
) -> Trajectory:
    """Collect one trajectory for GRPO.

    Multi-step iterative refinement: evaluate → generate edit → repeat.
    """
    current_expr = start_expr
    steps = []

    for step in range(max_steps):
        fb_type, fb_acc = domain.evaluate(current_expr, task)
        if fb_type == "CORRECT":
            break

        prompt = _build_prompt(domain, task, current_expr, fb_type, fb_acc)
        gen_ids, log_probs = generate_with_queries(
            model, prompt, domain, task, temperature=temperature,
        )
        steps.append((prompt, gen_ids, log_probs))

        # Extract edit: find SEP_EDIT in generated tokens
        edit_ids = gen_ids
        if domain.sep_edit_id in gen_ids:
            edit_start = gen_ids.index(domain.sep_edit_id) + 1
            edit_ids = gen_ids[edit_start:]

        edit_expr = domain.detokenize_program(edit_ids)
        if not edit_expr or edit_expr == "NO_OP" or edit_expr == "_HOLE_":
            break
        current_expr = edit_expr

    # Final reward (pass program and task for context-aware shaping)
    fb_type, fb_acc = domain.evaluate(current_expr, task)
    reward = domain.shaped_reward(fb_type, fb_acc, program=current_expr, task=task)

    return Trajectory(task=task, steps=steps, reward=reward, final_expr=current_expr)


def tournament_collect_trajectories(
    model: nn.Module,
    domain: Domain,
    task: Any,
    group_size: int = 4,
    n_rounds: int = 3,
    max_steps: int = 4,
    temperature: float = 0.8,
    verbose: bool = False,
) -> list[list[Trajectory]]:
    """Tournament GRPO: multi-round with ratcheting and reward backpropagation.

    Each round generates group_size trajectories:
    - Round 0: all start from _HOLE_
    - Round N: half exploit (start from best), half explore (reward-weighted pick)

    When a later round achieves a perfect solve, reward is backpropagated
    to earlier rounds' best trajectories.
    """
    best_expr = "_HOLE_"
    best_reward = 0.0
    rounds: list[list[Trajectory]] = []
    round_best_indices: list[int] = []

    # Domain-specific threshold for "perfect" — ARC uses 5.0, KB uses 1.0
    # We detect by checking the reward for a CORRECT evaluation
    perfect_threshold = domain.shaped_reward("CORRECT", 1.0)

    for rnd in range(n_rounds):
        # Determine starting points
        if rnd == 0 or not rounds:
            starts = [best_expr] * group_size
        else:
            prev_trajs = rounds[-1]
            prev_rewards = [t.reward for t in prev_trajs]
            total_r = sum(max(r, 0.01) for r in prev_rewards)
            weights = [max(r, 0.01) / total_r for r in prev_rewards]
            explore_traj = random.choices(prev_trajs, weights=weights, k=1)[0]
            explore_expr = explore_traj.final_expr
            if not explore_expr or explore_expr in ("NO_OP", "_HOLE_"):
                explore_expr = best_expr

            n_exploit = group_size // 2
            starts = [best_expr] * n_exploit + [explore_expr] * (group_size - n_exploit)

        round_trajs = []
        for start in starts:
            traj = collect_trajectory(
                model, domain, task,
                max_steps=max_steps,
                temperature=temperature,
                start_expr=start,
            )
            round_trajs.append(traj)

        rounds.append(round_trajs)

        # Track best
        best_idx = max(range(len(round_trajs)), key=lambda i: round_trajs[i].reward)
        best_traj = round_trajs[best_idx]
        round_best_indices.append(best_idx)

        if verbose:
            print(f"    [round {rnd+1}/{n_rounds}] best_reward={best_traj.reward:.3f} "
                  f"start={best_expr[:40]}", flush=True)

        if best_traj.reward >= perfect_threshold:
            # Backpropagate reward to earlier rounds' best trajectories
            for prev_rnd in range(len(rounds) - 1):
                prev_best_idx = round_best_indices[prev_rnd]
                prev_traj = rounds[prev_rnd][prev_best_idx]
                distance = len(rounds) - 1 - prev_rnd
                backprop_reward = perfect_threshold * (0.5 ** distance)
                if backprop_reward > prev_traj.reward:
                    rounds[prev_rnd][prev_best_idx] = Trajectory(
                        task=prev_traj.task,
                        steps=prev_traj.steps,
                        reward=backprop_reward,
                        final_expr=prev_traj.final_expr,
                    )
            break

        # Ratchet: next round starts from best if it improved
        next_expr = best_traj.final_expr
        if not next_expr or next_expr in ("_HOLE_", "NO_OP"):
            break
        if best_traj.reward <= best_reward and rnd > 0:
            break

        best_reward = best_traj.reward
        best_expr = next_expr

    return rounds


def grpo_train(
    model: nn.Module,
    domain: Domain,
    tasks: list[Any],
    epochs: int = 30,
    group_size: int = 4,
    n_rounds: int = 3,
    tasks_per_epoch: int = 64,
    max_steps: int = 4,
    temperature: float = 0.8,
    clip_eps: float = 0.2,
    kl_coef: float = 0.05,
    lr: float = 1e-5,
    max_seq_len: int = 192,
    verbose: bool = True,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 5,
) -> dict[str, tuple[float, str]]:
    """Tournament GRPO training for iterative refinement.

    Proper GRPO with reference model, clipped ratio, and KL penalty.
    Tournament collection with multi-round ratcheting.

    Returns dict of best solutions: {task_id: (reward, program)}.
    """
    # Freeze reference model
    ref_model = clone_model(model, domain.vocab_size)

    optimizer = optim.Adam(learning_rate=lr)
    best_solutions: dict[str, tuple[float, str]] = {}

    if verbose:
        n_params = param_count(model)
        print(f"GRPO (tournament): {n_params:,} params, {epochs} epochs, "
              f"group={group_size}, rounds={n_rounds}, "
              f"{tasks_per_epoch} tasks/epoch", flush=True)

    t0 = time.time()
    perfect_threshold = domain.shaped_reward("CORRECT", 1.0)

    for epoch in range(epochs):
        random.shuffle(tasks)
        batch = tasks[:tasks_per_epoch]
        epoch_reward = 0.0
        epoch_total = 0
        epoch_perfect = 0

        for task in batch:
            rounds = tournament_collect_trajectories(
                model, domain, task,
                group_size=group_size,
                n_rounds=n_rounds,
                max_steps=max_steps,
                temperature=temperature,
            )

            # Track stats
            task_id = getattr(task, "task_id", str(id(task)))
            for round_trajs in rounds:
                for traj in round_trajs:
                    epoch_reward += traj.reward
                    epoch_total += 1
                    if traj.reward >= perfect_threshold:
                        epoch_perfect += 1
                    if traj.reward > best_solutions.get(task_id, (0.0, ""))[0]:
                        best_solutions[task_id] = (traj.reward, traj.final_expr)

            # GRPO loss: per-round advantages with proper ratio + KL
            def grpo_loss(m):
                total_loss = mx.array(0.0)
                n_tokens = 0

                for round_trajs in rounds:
                    rewards = [t.reward for t in round_trajs]
                    mean_r = sum(rewards) / len(rewards)
                    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards))
                    if std_r < 1e-8:
                        continue

                    advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

                    for traj, adv in zip(round_trajs, advantages):
                        if abs(adv) < 1e-8:
                            continue
                        for prompt, gen_ids, old_lps in traj.steps:
                            if not gen_ids:
                                continue
                            full_ids = prompt + gen_ids
                            if len(full_ids) > max_seq_len:
                                continue

                            x = mx.array([full_ids[:-1]], dtype=mx.int32)
                            logits = get_logits(m, x, domain.vocab_size)
                            seq_len = len(full_ids) - 1
                            logits = logits.reshape(seq_len, domain.vocab_size)

                            gen_start = len(prompt)
                            for i, (token_id, old_lp) in enumerate(zip(gen_ids, old_lps)):
                                pos = gen_start + i
                                if pos >= logits.shape[0]:
                                    break

                                new_lp = mx.log(mx.softmax(logits[pos]) + 1e-10)[token_id]

                                ref_logits = logits_at_last(ref_model, full_ids[:pos + 1])
                                ref_lp = mx.log(mx.softmax(ref_logits) + 1e-10)[token_id]

                                ratio = mx.exp(new_lp - old_lp)
                                clipped = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                                pg_loss = -mx.minimum(ratio * adv, clipped * adv)
                                kl = new_lp - ref_lp

                                total_loss = total_loss + pg_loss + kl_coef * kl
                                n_tokens += 1

                return total_loss / max(n_tokens, 1)

            loss, grads = nn.value_and_grad(model, grpo_loss)(model)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

        avg_reward = epoch_reward / max(epoch_total, 1)
        n_perfect = sum(1 for r, _ in best_solutions.values() if r >= perfect_threshold)
        elapsed = time.time() - t0

        if verbose:
            print(f"GRPO {epoch+1}/{epochs}: reward={avg_reward:.3f} "
                  f"perfect={epoch_perfect}/{epoch_total} "
                  f"best: {n_perfect} perfect "
                  f"elapsed={elapsed:.0f}s", flush=True)

            if (epoch + 1) % 10 == 0:
                top = sorted(best_solutions.items(), key=lambda x: -x[1][0])[:5]
                for tid, (r, prog) in top:
                    print(f"    {tid}: r={r:.3f} {prog[:60]}", flush=True)

        # Checkpoint
        if checkpoint_dir and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"grpo_epoch{epoch+1}.npz")
            save_checkpoint(model, path)
            if verbose:
                print(f"  checkpoint: {path}", flush=True)

    # Save final
    if checkpoint_dir:
        save_checkpoint(model, os.path.join(checkpoint_dir, "grpo_final.npz"))
        if verbose:
            print(f"  final checkpoint saved", flush=True)

    return best_solutions
