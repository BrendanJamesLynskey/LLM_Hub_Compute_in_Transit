# Compute in Transit

Moving data has become the bottleneck in LLM training and serving. *Compute in transit* means doing useful work on data while it crosses a link, rather than after it arrives. Optalysys has demonstrated a photonic link module that does this: a QSFP56 pluggable with an integrated optical Fourier transform device, running FFT, NTT, MAC, and modular and floating-point arithmetic on data in the link. This series asks which LLM data movements such a module could work on, and how much work each needs per byte. It covers training as well as inference, and its strongest candidate is in training: compressing gradient or momentum updates sent between data centres.

**Live index:** https://brendanjameslynskey.github.io/LLM_Hub_Compute_in_Transit/

## Presentations in this series

| # | Title | Status | What it covers |
|---|-------|--------|----------------|
| 01 | [Compute in Transit: the Idea and the Budget](https://brendanjameslynskey.github.io/CiT_01_Compute_in_Transit/) | live | Where computation can sit on a data path (endpoint, link, switch); Optalysys's photonic link module and what its paper shows; the published operation set and what it lacks; the operations-per-byte budget (0.40 to 3.96 on a QSFP56 link, if the claim is read as 10 to 99 GFLOP/s); whether the optical transform counts against it; precision, energy, and five tests every candidate must pass. |
| 02 | [Data Movement in LLM Training](https://brendanjameslynskey.github.io/CiT_02_Data_Movement_in_LLM_Training/) | live | The traffic each kind of parallelism creates and how big it is; ring all-reduce; in-network aggregation (SwitchML); low-communication training (DiLoCo, Streaming DiLoCo, MuLoCo, SparseLoCo); frequency-domain gradient compression (FFT sparsification, DeMo); mapping DeMo onto the module, and the planned fixed-frequency-band experiment. |
| 03 | [Data Movement in LLM Inference](https://brendanjameslynskey.github.io/CiT_03_Data_Movement_in_LLM_Inference/) | live | The KV hand-off and when it binds; quantised and transform-domain KV compression; KV tiering and prefix caching; query-aware KV selection by dot products; ring-attention prefill; MoE all-to-all; weight streaming; confidential AI; the candidate table. |
| 04 | The Fixed-Band Experiment | planned | DeMo with its top-k replaced by a fixed frequency band, with the optics emulated (noise, native transform size): loss against bytes communicated, then a discrete-event model of the wall-clock gain across data centres. |

## Analysis and code

| Path | What's inside |
|------|---------------|
| [analysis/cit_budget.py](analysis/cit_budget.py) | The budget analysis: operations per byte at line rate; work per byte of candidate kernels with the transform charged or free; training sync sizes and link times for Llama-3-8B and -70B; KV hand-off and weight-load sizes; a fixed frequency band against top-k on synthetic signals |
| [analysis/results.md](analysis/results.md) | Its output: the source of every number in the decks |
| [analysis/test_cit_budget.py](analysis/test_cit_budget.py) | pytest: the budget, the FFT cost convention, KV bytes against the Fourier Optics series, ring all-reduce volume, the DCT, and top-k against a fixed band |

Reproduce: `pip install numpy pytest` and `pip install -e` a checkout of [Disaggregated_Inference_Sim](https://github.com/BrendanJamesLynskey/Disaggregated_Inference_Sim) (model shapes and link presets come from it), then `python analysis/cit_budget.py` and `pytest analysis`.

## What is illustrative

The module's compute rate is the authors' claim ("tens of GFLOPs" per link), bracketed here by 10 and 99 GFLOP/s; a reviewer notes it is asserted rather than measured. Link energies and latencies are the simulator's illustrative presets. Compression ratios (2x, 8x, 85x) are the cited papers' figures, used only to size transfers. The fixed band against top-k comparison uses synthetic signals; real momentum tensors are what the planned experiment measures. The placement of candidates is this series' judgement.

> Optalysys and this series. Optalysys publicly describes Compute-in-Transit and lists AI infrastructure among its targets, but its only published evaluation is FHE (CKKS bootstrapping). Everything here about applying it to LLM training or inference traffic is this series' own analysis, attributed to no company.

## How to read this series

Deck 01 sets out the idea, the hardware as published, and the budget every candidate must fit. Deck 02 goes through training traffic and ends with the experiment; deck 03 does the same for inference. The [glossary](https://brendanjameslynskey.github.io/LLM_Hub_Compute_in_Transit/#glossary) explains every concept and links to the slide that covers it.

Sister series: [Fourier Optics for Inference](https://github.com/BrendanJamesLynskey/LLM_Hub_Fourier_Optics_Inference) (the optics and its precision and energy bill; optical prefill pools), [LLM Inference Simulators](https://github.com/BrendanJamesLynskey/LLM_Hub_Inference_Simulators), [FHE Accelerator Simulators](https://github.com/BrendanJamesLynskey/FHE_Hub_Accelerator_Simulators).

Licences: slides and text CC BY 4.0 ([LICENSE](LICENSE)); code MIT ([LICENSE-CODE](LICENSE-CODE)).
