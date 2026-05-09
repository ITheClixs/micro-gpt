"""Deterministic payloads used by the Streamlit research lab and tests."""

from __future__ import annotations

import torch

from src.algorithms import backprop, cnn, optimizers, rl, rnn
from src.micro_gpt.config import MicroGPTConfig
from src.micro_gpt.model import MicroGPT, attention_probe


def backprop_payload():
    return backprop.demo_loss_curve(steps=20, seed=1)


def cnn_payload():
    return cnn.feature_map_payload(seed=2)


def rnn_payload():
    return rnn.gradient_flow_payload(sequence_length=10, seed=3)


def rl_payload():
    return rl.value_iteration_payload(iterations=15)


def optimizer_payload():
    return optimizers.optimizer_payload()


def micro_gpt_payload():
    torch.manual_seed(4)
    config = MicroGPTConfig(
        vocab_size=24,
        block_size=6,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = MicroGPT(config)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    attention = attention_probe(model, input_ids)
    logits = model(input_ids).logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)[0]
    top_probs, top_tokens = torch.topk(probs, k=6)
    return {
        "attention": attention.tolist(),
        "token_probabilities": [
            {"token": int(token), "probability": float(prob)}
            for token, prob in zip(top_tokens, top_probs)
        ],
    }


def all_payloads():
    return {
        "backprop": backprop_payload(),
        "cnn": cnn_payload(),
        "rnn": rnn_payload(),
        "rl": rl_payload(),
        "optimizers": optimizer_payload(),
        "micro_gpt": micro_gpt_payload(),
    }
