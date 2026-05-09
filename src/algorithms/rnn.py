"""Recurrent neural-network cells implemented with explicit tensor operations."""

from __future__ import annotations

import math

import torch
from torch import nn


class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        scale = 1.0 / math.sqrt(hidden_size)
        self.w_ih = nn.Parameter(torch.empty(input_size, hidden_size).uniform_(-scale, scale))
        self.w_hh = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(-scale, scale))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_t):
        return torch.tanh(x_t @ self.w_ih + h_t @ self.w_hh + self.bias)


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x_proj = nn.Linear(input_size, 3 * hidden_size)
        self.h_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x_t, h_t):
        x_reset, x_update, x_new = self.x_proj(x_t).chunk(3, dim=-1)
        h_reset, h_update, h_new = self.h_proj(h_t).chunk(3, dim=-1)
        reset = torch.sigmoid(x_reset + h_reset)
        update = torch.sigmoid(x_update + h_update)
        candidate = torch.tanh(x_new + reset * h_new)
        return (1.0 - update) * candidate + update * h_t


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x_proj = nn.Linear(input_size, 4 * hidden_size)
        self.h_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x_t, state):
        h_t, c_t = state
        i, f, g, o = (self.x_proj(x_t) + self.h_proj(h_t)).chunk(4, dim=-1)
        input_gate = torch.sigmoid(i)
        forget_gate = torch.sigmoid(f)
        candidate = torch.tanh(g)
        output_gate = torch.sigmoid(o)
        next_c = forget_gate * c_t + input_gate * candidate
        next_h = output_gate * torch.tanh(next_c)
        return next_h, next_c


def run_vanilla_sequence(sequence, hidden_size):
    cell = VanillaRNNCell(sequence.shape[-1], hidden_size)
    h_t = sequence.new_zeros(sequence.shape[1], hidden_size)
    states = []
    for x_t in sequence:
        h_t = cell(x_t, h_t)
        states.append(h_t)
    return torch.stack(states)


def gradient_flow_payload(sequence_length=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    sequence = torch.randn(sequence_length, 1, 3, generator=generator)
    cell = VanillaRNNCell(sequence.shape[-1], hidden_size=4)
    h_t = sequence.new_zeros(sequence.shape[1], 4)
    state_history = []
    for x_t in sequence:
        h_t = cell(x_t, h_t)
        h_t.retain_grad()
        state_history.append(h_t)
    states = torch.stack(state_history)
    loss = states[-1].pow(2).mean()
    loss.backward()
    norms = [
        0.0 if state.grad is None else float(state.grad.norm())
        for state in state_history
    ]
    return {"hidden_states": states.detach().squeeze(1).tolist(), "gradient_norms": norms}


if __name__ == "__main__":
    print(gradient_flow_payload())
