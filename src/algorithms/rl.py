"""Reinforcement-learning primitives for small controlled environments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def discounted_returns(rewards, gamma):
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    running = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def advantages(returns, values, normalize=True, eps=1e-8):
    adv = returns - values
    if normalize and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + eps)
    return adv


def ppo_clipped_objective(log_probs, old_log_probs, adv, clip_epsilon=0.2):
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    return -torch.minimum(unclipped, clipped).mean()


@dataclass(frozen=True)
class GridWorld:
    width: int = 5
    height: int = 5
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (4, 4)
    step_reward: float = -0.01
    goal_reward: float = 1.0

    @property
    def action_count(self):
        return 4

    def step(self, state, action):
        row, col = state
        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1
        else:
            raise ValueError("Action must be in [0, 3].")

        row = min(max(row, 0), self.height - 1)
        col = min(max(col, 0), self.width - 1)
        next_state = (row, col)
        done = next_state == self.goal
        reward = self.goal_reward if done else self.step_reward
        return next_state, reward, done


def value_iteration_payload(iterations=20, gamma=0.95):
    env = GridWorld()
    values = torch.zeros(env.height, env.width)
    for _ in range(iterations):
        updated = values.clone()
        for row in range(env.height):
            for col in range(env.width):
                if (row, col) == env.goal:
                    updated[row, col] = env.goal_reward
                    continue
                q_values = []
                for action in range(env.action_count):
                    (next_row, next_col), reward, _ = env.step((row, col), action)
                    q_values.append(reward + gamma * values[next_row, next_col])
                updated[row, col] = torch.tensor(q_values).max()
        values = updated
    return {"value_map": values.tolist(), "goal": env.goal, "start": env.start}


if __name__ == "__main__":
    print(value_iteration_payload())
