# Where the Recording Ends: a boundary shortcut in the HMS EEG benchmark

Code, executed notebooks and result tables for the paper *"Where the Recording Ends: A Boundary Shortcut in the HMS EEG Benchmark Inflates Long-Context Gains and Leaderboard Scores"* (submitted to IEEE JBHI).

**Finding.** HMS expert labels were propagated across stationary periods, so how much raw EEG surrounds a labelled window depends on the annotation round, and the rounds differ in label mix. Every model we trained with 200 s windows (Mamba-2, GRU, CNN, Transformer, pretrained CBraMod) learned to read where the recording ends instead of the EEG. The same cue enters standard 50 s pipelines through ±20 s window-shift augmentation with zero padding. Mirror filling with random truncation removes it for models trained from scratch; the genuine long-context benefit is 2–3%.

## Notebooks

All notebooks ran on Kaggle (GPU T4 unless noted) with the HMS competition data attached. They are stored **with outputs**, so every number can be checked without rerunning. Model checkpoints and out-of-fold predictions are in the outputs of the Kaggle versions linked below.

| # | Notebook | What it does | Kaggle version |
|---|---|---|---|
| 01 | `01_train_base.ipynb` | 50 s and 200 s models for all four mixers (5 folds, seed 42); GRU at 100/200 s; causal vs bidirectional Mamba; pooled out-of-fold statistics | [notebook86c832275b](https://www.kaggle.com/code/beakedjungle90/notebook86c832275b), Version 5 |
| 01b | `01b_train_base_v3_run.ipynb` | The run that trained the Mamba, causal Mamba, Transformer and CNN models reused by 01 (6 h 15 min) | notebook86c832275b, Version 4 ("v3") |
| 01c | `01c_efficiency_benchmark.ipynb` | Latency and memory vs context length; streaming benchmark to 72 h (KV cache vs recurrent state) | notebook86c832275b, early version ("v2") |
| 02 | `02_shortcut_check.ipynb` | Reflect / truncate interventions on the original 200 s models; boundary-only predictor | [notebookd13836b6e5](https://www.kaggle.com/code/beakedjungle90/notebookd13836b6e5) |
| 03 | `03_ba_mitigation.ipynb` | Boundary-agnostic (BA) training at 200 s, reweighting ablation, 10 min spectrogram audit | [notebookbb6726eb71](https://www.kaggle.com/code/beakedjungle90/notebookbb6726eb71) |
| 04 | `04_seeds_ba.ipynb` | BA without reweighting, seeds 42–3042 | [notebook22e15f591d](https://www.kaggle.com/code/beakedjungle90/notebook22e15f591d) |
| 05 | `05_seeds_50s.ipynb` | Extra seeds for the 50 s benchmark | [notebook92c60ba812](https://www.kaggle.com/code/beakedjungle90/notebook92c60ba812) |
| 06 | `06_cbramod.ipynb` | Pretrained CBraMod at 50 s, 200 s original and 200 s BA | [notebookdcb82b4ee1](https://www.kaggle.com/code/beakedjungle90/notebookdcb82b4ee1) |
| 07 | `07_shift_augmentation.ipynb` | 50 s training with no shift, ±20 s zero-filled shift and ±20 s mirror-filled shift; edge probe; shift TTA | [notebookda41823fab](https://www.kaggle.com/code/beakedjungle90/notebookda41823fab) |
| 08 | `08_hidden_test_submission.ipynb` | Hidden-test inference with temperature calibration and prior shift (one of 37 submissions; set `SUBMIT`, `PRIOR_SHIFT`, `CONSTANT`) | [notebookc56ce56913](https://www.kaggle.com/code/beakedjungle90/notebookc56ce56913) |
| 09 | `09_final_stats.ipynb` | CPU only. Recomputes every table and figure from saved out-of-fold predictions (cross-fitted temperatures, shared patient-level cluster bootstrap, Holm correction), including the extra seeds from 11 and 12 | [notebookc1c5d38c84](https://www.kaggle.com/code/beakedjungle90/notebookc1c5d38c84) |
| 10 | `10_prior_correction.ipynb` | CPU only. Compares five label-prior corrections on test-like training recordings and selects one by a rule fixed in advance; per-class KL contributions; fold assignments | [notebook30aefe105c](https://www.kaggle.com/code/beakedjungle90/notebook30aefe105c) |
| 11 | `11_seeds_original_200s.ipynb` | Original 200 s models, seeds 1042/2042 with reflect/truncate views; causal GRU; Transformer learning-rate check | [notebookaf63d6a691](https://www.kaggle.com/code/beakedjungle90/notebookaf63d6a691) |
| 12 | `12_shift_augmentation_seeds.ipynb` | Window-shift experiment (07) with seeds 1042/2042; seed-averaged tests | [notebook27e9b723d5](https://www.kaggle.com/code/beakedjungle90/notebook27e9b723d5) |
| 13 | `13_mechanism_probe_fill.ipynb` | Linear probe of the learned representations; BA training with spliced-EEG and noise fills | [notebook5da7aea52b](https://www.kaggle.com/code/beakedjungle90/notebook5da7aea52b) |
| 14 | `14_cbramod_ba_seeds.ipynb` | CBraMod boundary-agnostic 200 s model, training seed 1042; residual shortcut per seed and seed-averaged | [notebook86ca90702e](https://www.kaggle.com/code/beakedjungle90/notebook86ca90702e) |

Checkpoints for the v6 seeds used by 08 are also in the Kaggle dataset [hms-v6-ckpts](https://www.kaggle.com/datasets/beakedjungle90/hms-v6-ckpts).

**Run order.** 01b → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 11 → 12 → 13 → 14, then 10 (prior correction) and 08 (hidden-test submissions), and finally 09 for the statistics. Each notebook's first cell lists the outputs it needs as inputs (Kaggle: Add Input → Your Work → Notebooks). 01c is independent.

## Paper ↔ source

| Paper | Source |
|---|---|
| Fig. 1 (mechanism schematic) | `scripts/fig1_mechanism.py` |
| Table II, shortcut signatures (three seeds) | 09 → `T4_shortcut_signatures.csv`, `T5_matched_input_asymmetry.csv`, `T5b_truncation_shift_per_class.csv` (seeds from 11) |
| Context availability and label mix (Sec. IV-A) | 09 → `T1a_context_by_votes.csv`, `T1b_label_priors.csv` |
| Boundary-only predictor (Sec. IV-A) | 02 |
| Embedding probe (Sec. IV-A) | 13 |
| Spectrogram audit (Sec. IV-B) | 03 |
| Table III and Fig. 2, hidden test | 08 (each Kaggle version = one row) → `results/hidden_test_scores.csv`; 09 → `T9a`–`T9c`, `fig_hidden_test` |
| Prior-correction proxy (Sec. IV-C) | 10 → `T10a`–`T10c`, `primary_method.json` |
| Table IV, window-shift augmentation (three seeds) | 12 (seed 42 also in 07) |
| Fig. 3, matched-input asymmetry | 09 → `fig_asymmetry` |
| Fig. 4, class-probability shift | 09 → `fig_truncation_shift` |
| BA residual checks, CBraMod, fill types (Sec. IV-E) | 09 → `T5`; 06; 14 (CBraMod seeds); 13 |
| Table V, real context effect; per-class effects | 09 → `T6_real_context_effect.csv`; 10 → `T11a`, `T11b` |
| Table VI and Fig. 5, architectures and seeds | 09 → `T2`, `T3`, `fig_architecture_seeds`; Transformer learning rates in 11 |
| Reweighting, calibration (Sec. IV-H) | 09 → `T7`, `T8` |
| Causal Mamba and causal GRU (Sec. IV-H) | 11 (pooled out-of-fold table) |
| Streaming memory and latency (Sec. IV-H) | 01c |
| Fold assignment of every training row | `results/folds/fold_assignments.csv` (from 10) |

## Data

HMS – Harmful Brain Activity Classification (Kaggle, 2024): https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification. The data are not redistributed here; accept the competition rules on Kaggle to use them.

## Licence

MIT (see `LICENSE`). `06_cbramod.ipynb` adapts the CBraMod module from braindecode under BSD-3-Clause (see `THIRD_PARTY_NOTICES.txt`).

## Citation

To be added on publication.
