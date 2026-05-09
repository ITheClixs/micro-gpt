"""Streamlit front end for the deep-learning research lab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.research_lab import demo_data


st.set_page_config(page_title="Micro-GPT Research Lab", layout="wide")
st.title("Micro-GPT Research Lab")
st.caption("From-scratch deep-learning algorithms, visual probes, and micro-GPT internals.")

section = st.sidebar.radio(
    "Track",
    ["Backpropagation", "CNNs", "RNNs", "Reinforcement Learning", "Optimizers", "Micro-GPT"],
)

if section == "Backpropagation":
    payload = demo_data.backprop_payload()
    st.subheader("Manual Backpropagation")
    st.line_chart(pd.DataFrame({"loss": payload["loss_curve"]}))
    st.line_chart(pd.DataFrame({"gradient_norm": payload["gradient_norms"]}))

elif section == "CNNs":
    payload = demo_data.cnn_payload()
    st.subheader("Convolutional Feature Maps")
    col1, col2, col3 = st.columns(3)
    col1.write("Input")
    col1.dataframe(pd.DataFrame(payload["image"]))
    col2.write("Kernel")
    col2.dataframe(pd.DataFrame(payload["kernel"]))
    col3.write("Activation")
    col3.dataframe(pd.DataFrame(payload["activation"]))

elif section == "RNNs":
    payload = demo_data.rnn_payload()
    st.subheader("Recurrent Hidden-State Dynamics")
    st.line_chart(pd.DataFrame(payload["hidden_states"]))
    st.line_chart(pd.DataFrame({"gradient_norm": payload["gradient_norms"]}))

elif section == "Reinforcement Learning":
    payload = demo_data.rl_payload()
    st.subheader("GridWorld Value Iteration")
    st.dataframe(pd.DataFrame(payload["value_map"]))
    st.write({"start": payload["start"], "goal": payload["goal"]})

elif section == "Optimizers":
    payload = demo_data.optimizer_payload()
    st.subheader("Optimizer Update Geometry")
    st.line_chart(
        pd.DataFrame(
            {
                "parameter": payload["parameter"],
                "loss": payload["loss_surface"],
                "gradient": payload["gradient"],
                "adamw_update": payload["adamw_update"],
            }
        )
    )

else:
    payload = demo_data.micro_gpt_payload()
    st.subheader("Micro-GPT Attention and Token Probabilities")
    st.dataframe(pd.DataFrame(payload["attention"]))
    st.bar_chart(pd.DataFrame(payload["token_probabilities"]).set_index("token"))
