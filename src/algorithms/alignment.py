"""Alignment math primitives for preference-optimization experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=0.1,
):
    """Direct Preference Optimization loss with inspectable reward diagnostics.

    Inputs are sequence-level log-probabilities for chosen and rejected
    completions. The primitive keeps the policy/reference margin explicit so
    visualization code can show how preference pairs shape the scalar loss.
    """
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_logratios - reference_logratios)
    losses = -F.logsigmoid(logits)

    rewards_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
    rewards_rejected = beta * (policy_rejected_logps - reference_rejected_logps)
    diagnostics = {
        "losses": losses,
        "logits": logits,
        "policy_logratios": policy_logratios,
        "reference_logratios": reference_logratios,
        "rewards_chosen": rewards_chosen,
        "rewards_rejected": rewards_rejected,
        "reward_margins": rewards_chosen - rewards_rejected,
        "preference_accuracy": (logits > 0).to(logits.dtype).mean(),
    }
    return losses.mean(), diagnostics


def group_relative_advantages(rewards, group_size, eps=1e-8):
    """Normalize rewards within prompt groups, matching GRPO-style intuition.

    GRPO-style training compares multiple sampled completions for the same
    prompt. This helper exposes the group-centered, group-scaled advantages
    without launching any rollout or optimizer machinery.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if rewards.numel() % group_size != 0:
        raise ValueError("rewards length must be divisible by group_size")

    original_shape = rewards.shape
    grouped = rewards.reshape(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, unbiased=False, keepdim=True)
    advantages = (grouped - means) / (stds + eps)
    return advantages.reshape(original_shape)
