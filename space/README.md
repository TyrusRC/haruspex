---
title: Haruspex
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - TyrusRC/haruspex
  - fastino/GLiNER2.5-Decide
short_description: Calibrated, abstaining decisions for security operations
---

Demo of [Haruspex](https://github.com/TyrusRC/haruspex): typed, calibrated security
decisions (boolean / choice / score) that abstain when not confident enough. Adapters load
from [`TyrusRC/haruspex`](https://huggingface.co/TyrusRC/haruspex) on first use.
