# Stage 69 deviations from research/llm-gpu-scale-protocol.json

Recorded 22 November 2026, after the run finished and before the analysis report was written.

1. **GPU type.** The protocol names one NVIDIA A100. The journal's environment records show `NVIDIA L4` (23,034 MiB) for both the original session (2026-11-21T22:55Z) and the resumed session that wrote the summary (2026-11-22T21:35Z). Every run therefore used an L4. Step costs, update budgets and r values are L4 measurements. The decision rules, data, seeds and budget definition were unchanged.
2. **Software.** Colab provided torch 2.14.1+cu130, CUDA 13.0, Python 3.13.16; transformers 4.46.3 was installed as specified. The protocol recorded rather than pinned these versions.
3. **Sessions.** The journal contains two environment records. The second resumed with all 96 runs already complete and only rebuilt the summary; no run was repeated or replaced.
4. **Notebook edits.** The owner saved an executed copy of the notebook (commit ea39b53) and edited the upload cell to ask for a token interactively. Neither affects the experiment cells.

Not a deviation, but needed for interpretation: on the Arabic Wikipedia target, random LoRA training on the PADT/EWT candidate pool raised target loss above its starting value (for example Qwen2.5-0.5B, ρ = 0: 2.8138 to 3.0018 after 400 updates). The protocol did not include a no-update baseline; comparisons with the starting loss are reported as post-hoc context only.
