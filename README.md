# Selective Correction — experiment code

Companion code for the paper

> **Identification Limits and a Protected Correction for LLM-Derived Guidance**

The repository contains the experiment code behind the paper: reference selectors, reference-uncertainty
routing, the LLM pairwise-query pipeline, the correction objective and its λ selection, the two controls
(direction-shuffled and random-probability), the protected-region audit, the cross-model replication, and
the appendix studies (comparison with recent LLM feature-selection methods, the Breast component ablation,
and direct LLM-only selection).

---

## 1. Method naming used throughout this repository

| In this repo | In the paper |
|---|---|
| `reference` | **Reference** — the reference selector alone, no external guidance |
| `global` | **Global Correction** — the correction objective on a broad comparison graph fixed independently of reference uncertainty (weights without certainty) |
| `global_certainty` | Global correction with the certainty factor applied to the weights |
| `selective` | **Selective Correction** — the method of Section 3 |
| `selective_no_certainty` | the `Q+B` arm of the component ablation |
| `lam` (λ) | correction strength; the grid always contains `0`, so the reference is a selectable outcome |

Per-comparison quantities follow the paper: `Q` substitution probability, `B` pairwise balance,
`U = Q · B` actionability, `C = 1 − Ent(p)/log 2` certainty, `w = λ · U · C`.

Note on scale: the objective is implemented as
`0.5 * mean((s - a)**2) + lam * sum(w * CE(p, sigmoid(s_i - s_j))) / sum(w)`,
i.e. the quadratic anchor and the pairwise term are normalised by the number of items `P` and by the total
weight `sum(w)`. The reported λ therefore corresponds to the paper's λ rescaled by `sum(w)/P`, and the local
correction budget is `rho_j = P * lam * sum_{e ∋ j} w_e / sum(w)`.

## 2. Path handling

Every script that reads or writes files resolves them against the package root, which is located by the
`.repro_root` marker in this directory (or by the `REPRO_ROOT` environment variable). Input data are placed
under the package root in the layout the scripts expect; shell scripts declare
`REPRO_ROOT="${REPRO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"`.

The LLM measurement scripts expect the provider credential in an external local file; no credential is
stored in this repository.

## 3. Layout

| Paper location | Directory |
|---|---|
| Sec. 4.2, Table 1 (Renal TCMR) | `01_renal_tcmr/` |
| Sec. 4.2, Table 1 (GSE272769 sepsis) | `02_gse272769_sepsis/` |
| Sec. 4.2, Table 1 (Breast pCR) | `03_breast_pcr/` |
| Sec. 4.2, Table 1 (CREDIT-G) | `04_credit_g/` |
| Sec. 4.2, Table 1 (Hospital osteoporosis) | `05_hospital_osteoporosis/` |
| Sec. 4.2, Table 1 (Darmanis GBM) | `06_darmanis_gbm/` |
| Sec. 4.2–4.3, Table 1 / Table 5 (JKP153) | `07_jkp153/` |
| Q2–Q5 diagnostics (matched global, controls, protection, fallback) | `08_diagnostics/` |
| Sec. 4.3, Table 4 (cross-model replication) | `09_cross_model_qwen/` |
| App. D.2 / D.4 / D.5 | `10_appendix/` |
| Query routing + correction engine shared by sepsis and breast | `shared_engine/` |

---

## 4. Experiments (main text)

### 4.1 `01_renal_tcmr/` — Renal TCMR (GSE36059 → GSE48581, elastic net + SIS, k = 50)

`formal_handoff/01_GSE36059_to_GSE48581_TCMR/scripts/`

| File | Role |
|---|---|
| `run_frozen_preprocessing_and_screen.py` | probe annotation, probe→gene collapse by median, gene-level screen on the frozen platform mapping |
| `download_gse36059_individual_metadata.py` | per-sample metadata of the development cohort |
| `build_pre_llm_freeze_v3_2.py` | freezes the candidate universe, the routed comparison sets and the evidence-gate coverage before any query |
| `finalize_freeze_v3_2.py`, `build_clean_final_freeze_v3_2.py` | seal the frozen input manifest |
| `finalize_pre_external_freeze_v3_2.py` | seals the state immediately before the external cohort is loaded |
| `finalize_recipient_id_audit.py` | sample-identifier audit for the external cohort |
| `finalize_runtime_gate_v3_2.py` | runtime precondition gate |
| `deepseek_generic_runtime_smoke_v3_2.py` | provider client smoke test |
| `run_formal_deepseek_v3_2.py` | runs the global and selective query sets (AB/BA both presentation orders) |
| `run_formal_r200.py` | R = 200 resampling that produces the reference score/rank distribution |
| `aggregate_formal_measurements_simple_abba_v3_2.py` | AB/BA aggregation into preference `p_e` and certainty `C_e` |
| `select_lambda_foldlocal_v3_2.py` | selection routing, the correction objective, and λ selected fold-locally on development |
| `evaluate_gse48581_external_v3_2.py` | external evaluation on GSE48581 |
| `run_semantic_shuffle_v3_2.py`, `run_random_llm_probability_null_v3_2.py` | the two controls |
| `build_final_report_v3_2.py` | result tables |
| `geo_utils.py`, `screen_utils.py`, `formal_screen_utils.py`, `src/geo_utils.py` | shared helpers |

### 4.2 `02_gse272769_sepsis/` — sepsis 30-day outcome (elastic net, k = 50, strict nested outer CV)

| File | Role |
|---|---|
| `gse272769/audit/build_gene_matrix.py` | transcript-cluster matrix → gene-level features |
| `gse272769/audit/clinical_confounder_audit.py`, `audit/summarize_variance.py` | cohort and variance summaries |
| `gse272769/reference/run_reference_p1500.py` | reference selector on the 1500-gene candidate universe |
| `gse272769/reference/run_formal_nested_reference.py` | reference under the nested outer CV |
| `gse272769/pre_llm/build_broad_graphs.py`, `pre_llm/build_broad_queries.py` | the broad comparison graph used by the global arm |
| `gse272769/selective_internal/run_nested272_frozen.py` | selective routing and correction, nested |
| `gse272769/selective_shuffle/run_nested272_shufflebase.py`, `run_shuffle_worker.py` | direction-shuffled control |
| `gse272769/tmp_k40_90_llm/…`, `tmp_k40_90_shuffle/…` | k-sweep runners and their shuffle counterpart |
| `package_code/run_top30_top50_nested.py` | the reported k = 50 nested run (correction objective + λ selection) |
| `package_code/run_strict_nested_selective_routing.py` | routing tables: resampling → substitution, balance, actionability |
| `package_code/run_worker.py`, `run_shuffle_worker.py`, `run_nested_shufflebase.py`, `run_nested_k40_90.py` | workers for the control and k-sweep runs |
| `package_code/BUILD_PACKAGE_FROM_WORKSPACE.py` | assembles the frozen package |

### 4.3 `03_breast_pcr/` — pathological complete response (GSE25055 → GSE25065, SIS, k = 20)

`reproduce/`

| File | Role |
|---|---|
| `reproduce_selective_measurement_postprocess.py` | aggregation of the pair-source measurements into `p_e`, `C_e` |
| `reproduce_final_sealed.py` | sealed-cohort evaluation of the selected set |
| `reproduce_selective_semantic_shuffle_control.py` | direction-shuffled control |
| `reproduce_global_bt.py` | the global comparison arm |
| `verify_package.py`, `run_quick_reproduction.sh` | input verification and one-shot driver |

The full pipeline for this dataset (candidate universe, routing, global graph, sealed validation) is in
`shared_engine/`.

### 4.4 `04_credit_g/` — German credit risk (GBM-permutation k = 10, elastic net k = 10)

| File | Role |
|---|---|
| `03_CODE/01_TASK_FREEZE/build_phase1_split.py` | development / locked-holdout split |
| `03_CODE/02_DATA_ONLY/run_reference.py`, `run_reference_formal_fixed.py` | reference selectors and resampling |
| `03_CODE/02_DATA_ONLY/tune_reference_hyperparameters.py`, `postprocess_reference.py` | tuning and aggregation of the reference runs |
| `03_CODE/04_METHOD_FREEZE/build_selective_graph.py`, `build_broad_graph.py` | the selective and the broad comparison graphs |
| `03_CODE/04_METHOD_FREEZE/build_query_manifests.py`, `run_deepseek_measurement.py` | query manifest and measurement call |
| `03_CODE/04_METHOD_FREEZE/deepseek_*` | client smoke tests and a raw log-probability probe |
| `03_CODE/05_CONTROLS/build_post_final_benchmark_control.py`, `analyze_post_final_benchmark_control.py` | literature-benchmark control |
| `03_CODE/06_RESULTS/analyze_llm_measurement.py` | `p_e` and certainty extraction |
| `03_CODE/06_RESULTS/select_lambda_development.py` | λ selection on development |
| `03_CODE/06_RESULTS/build_final_selected_sets.py` | final selected sets |
| `03_CODE/06_RESULTS/evaluate_final_holdout_once.py` | locked-holdout evaluation |
| `03_CODE/06_RESULTS/build_final_report.py` | result tables |
| `09_REPRO_CHECK/verify_reproduction.py` | package verification |

### 4.5 `05_hospital_osteoporosis/` — low T-score risk prediction (L1 logistic rank, k = 10 / k = 5)

`code/`

| File | Role |
|---|---|
| `run_selector_selective_portability_v10.py` | routing + correction, carried across the selector family |
| `run_selector_robustness_extension_v10.py` | k = 5 / k = 10 extension of the same protocol |
| `run_pairwise_measurement_v181.py` | pairwise measurement calls |
| `run_batch1_downstream_v19.py` | development-batch downstream run (objective + λ) |
| `freeze_final_predictors_pre_batch2_v20.py`, `finalize_pre_batch2_freeze_v20.py` | freeze the predictors before the temporal batch is opened |
| `evaluate_batch2_final_v201.py` | temporal external evaluation on Batch 2 |

`dataonly_scripts/` — the data-only preparation chain of the cohort
(`p0_inputs.py` → `p1_task_freeze.py` → `p2_eligibility.py` → `p3_leakage.py` → `p4_baseline.py` →
`p5_stability.py` → `p6_confusion.py` → `p7_assertions.py` → `p8_finalize.py`, plus the later
`p9`–`p16` revisions, with `common.py`, `modeling.py`, `v2_core.py`, `pilot_plots.py` and
`check_reproducibility.py` as shared modules). Variable names in this directory are indexed
(`f01`, `f02`, …) rather than printed in full.

### 4.6 `06_darmanis_gbm/` — GBM core vs periphery (GSE84465; SIS-10/20, EN-10)

`reproduce/SCRIPTS/`

| File | Role |
|---|---|
| `build_and_run_gbm_selective_v29.py` | routing, correction objective, λ selection, full-development selected sets |
| `run_gbm_reference_global_downstream_v29.py` | the reference and global arms |
| `gbm_downstream_fold_consistency_v29.py` | fold-consistency check of the corrected rankings |
| `analyze_gbm_downstream_results_v29.py`, `audit_and_summarize_gbm_downstream_v29.py` | result tables and summaries |
| `final_measurement_audit_and_aggregate_v28.py` | aggregation of the pair-source measurements |
| `postprocess_completed_ivy_v28.py`, `postprocess_completed_g116_v28.py`, `postprocess_completed_g132_v28.py` | per-source measurement post-processing |
| `RUN_OFFLINE_REPRODUCTION.sh` | one-shot driver |

`gbm_canonical_scripts/` — construction of the analysis matrix: `s1_recon.py`,
`s2_cell_fingerprint.py`, `s2b_global_rematch.py`, `s2c_gene_stability.py`, `s3_patient_mapping.py`,
`s3b_plate_audit.py`, `s4_candidate_universe.py`, `s5_canonical_align.py`, `s5b_x_only_builder.py`,
`s6_stats.py`, `s7_verify.py`, `s8_package.py`, `s9_review_bundle.py`, `s10_package_v2.py`,
`s11_hard_checks_v2.py`.

### 4.7 `07_jkp153/` — factor ranking, 153 JKP factors (2024 validation, 2025 held-out)

| File | Role |
|---|---|
| `phaseB/build_query_graphs.py`, `build_broad_graphs.py`, `build_measurement_union.py` | selective and broad query graphs and their union |
| `phaseB/render_selective_prompts.py`, `render_union_prompts.py` | prompt rendering for both arms |
| `phaseB/run_parallel_measurement.py`, `run_logprob_measurement.py` | measurement calls |
| `phaseB/aggregate_edge_probabilities.py`, `aggregate_union_probabilities.py` | aggregation into preference probabilities and certainty |
| `phaseB/evaluate_selective_fullrank.py` | selective ranking evaluation over the correction-strength grid |
| `phaseB/evaluate_matched_reference_global_selective.py` | matched reference / global / selective comparison |
| `phaseB/evaluate_routing_ablations.py` | routing ablation |
| `phaseB/fit_broad_bt_rankfusion.py` | the global rank-fusion comparator |
| `phaseB/pair_resolution_diagnostic.py`, `measurement_diagnostics_phaseB.py`, `posthoc_order_consistent_sensitivity.py`, `final_diagnostics.py` | measurement diagnostics and sensitivity |
| `qwen25/run_local_qwen_measurement.py`, `run_local_qwen_measurement_2025.py`, `aggregate_local_qwen.py` | local-checkpoint measurement for the 2024/2025 periods |
| `qwen25/dl_extension/*` | extension of the same protocol to the additional factor block |
| `corrected_downstream/run_calendar_aligned_corrected_downstream.py` | calendar-aligned target construction and the global ranking evaluation |

The full-ranking task is evaluated on the cross-sectional rank scale: the correction enters as a
rank-scale guidance shift with a strength grid that contains `0` (so the reference ranking is a
selectable outcome), rather than as the anchored score objective used by the six prediction datasets.

---

## 5. `shared_engine/` — routing and correction engine (sepsis + breast)

`multi_dataset_expansion/scripts/`

| File | Role |
|---|---|
| `freeze_p2000_datasets.py` | freeze the candidate universes |
| `build_breast_candidate_review.py`, `build_sepsis_candidate_review.py`, `build_primary_candidate_review_v2.py` | candidate-universe review |
| `run_reference_nested_feasibility.py`, `run_reference_gse272769.py` | reference selector runs |
| `run_strict_nested_selective_routing.py`, `run_strict_nested_selective_routing_gse272769.py` | resampling → `Q`, `B`, `U` routing tables |
| `selective_routing_fold_worker.py` | per-fold routing worker |
| `finalize_selective_routing_from_caches.py` | routing consolidation |
| `run_nested_selective_internal.py` | **the correction objective** (`optimize_selective`) used by all prediction datasets in this engine |
| `run_selective_production.py`, `run_frozen_selective_queries.py` | production run of the selective query set |
| `selective_en_chunk_worker.py`, `selective_en_p0_alt_worker.py`, `merge_selective_en_chunks.py` | chunked execution of the large query sets |
| `postprocess_full_selective_measurements.py` | aggregation into `p_e` and `C_e` |
| `finalize_development_selective_tuning.py` | λ selected on development, with the exact `λ = 0` fallback gate |
| `build_pre_llm_query_tables.py`, `preflight_llm_execution_gate.py` | evidence gate and execution gate |
| `build_global_broad_graphs.py`, `build_global_broad_queries.py`, `build_global_broad_master_manifest.py` | global comparison graph |
| `run_global_internal_and_final.py`, `postprocess_global_broad.py`, `preflight_global_broad_gate.py` | global arm evaluation |
| `estimate_global_broad_budget.py`, `estimate_global_d10_runtime.py` | budget estimation for the global arm |
| `run_selective_semantic_shuffle_control.py`, `summarize_selective_semantic_shuffle_control.py` | direction-shuffled control |
| `run_one_time_sealed_validation.py`, `build_pre_sealed_manifest.py` | sealed evaluation |
| `build_full_selective_execution_table.py`, `build_final_synthesis.py` | execution table and synthesis |
| `audit_pre_llm_freeze.py`, `check_lambda0_reference_reproduction.py`, `run_generic_deepseek_smoke.py` | freeze audit, `λ = 0` reproduction check, client smoke test |

## 6. `08_diagnostics/` — Q2–Q5

`q3_q4/scripts/` — `run_q3_gbm_nulls.py`, `run_q3_breast_random_null.py`, `run_q3_credit_nulls.py`,
`run_q3_hospital_nulls.py`, `run_q3_hospital_nulls_fast.py` (controls), `run_q4_protection.py`
(displacement audit of the protected region), `run_q4_gse272769_en50.py`,
`run_q4_breast_positive_cells.py`.

`q3_q4/06_q2_q5_completion/scripts/` — the completed Q2–Q5 pass: matched-global constructions
(`breast_q2_matched.py`, `breast_q2_source_matched.py`, `gbm_q2_matched.py`, `hospital_q2_matched.py`,
`gse272769_q2_build_queries.py`, `gse272769_q2_run_worker.py`, `gse272769_q2_evaluate.py`,
`credit_q2_q4_migration.py`), controls (`gse272769_random_null.py`, `hospital_temporal_null.py`,
`breast_sis20_random_macroap.py`, `run_q3_hospital_k5_nulls.py`, `renal_semantic_external.py`),
protection (`breast_q4_formal.py`, `gbm_q4.py`, `hospital_q4.py`), and the aggregation
(`build_final_outputs.py`, `build_final_q2_q5_report.py`, `build_safe_borrowing_audit.py`,
`build_initial_inventory.py`).

## 7. `09_cross_model_qwen/` — cross-model replication (Table 4)

`multi_dataset_swap/scripts/` — per dataset: `renal_*`, `sepsis_*`, `breast_*`, `credit_*`, `hospital_*`,
`darmanis_*` variants of the routing, measurement, control and evaluation steps, run with the local
checkpoint in place of the API model (`*_swap.py`). Plus the shared pieces:
`materialize_frozen_prompts.py`, `build_local_llm_portable_bundle.py`, `aggregate_q2_swap_measurements.py`,
`aggregate_q2_qwen_local.py`, `aggregate_breast_q2_source.py`, `audit_and_convert_qwen_results.py`,
`build_qwen_completion_audit.py`, `build_integrity_audit.py`, `build_trimodel_agreement.py`,
`build_qwen_trimodel_final.py`, `build_external_q3_pvalues.py`, `build_cross_model_comparison.py`,
`build_final_modelswap_report.py`, and the driver `run_main_downstream_all.sh`.

`local_inference/run_qwen3_32b_measurements.py` — the local measurement runner: first-step
full-vocabulary logits over the response options, A-vs-B normalisation, binary entropy and certainty.

## 8. `10_appendix/`

**`d02_recent_llm_fs/`** (Table 3) — `cross_dataset/`: `generate_pointwise_llm_scores.py`,
`eval_highdim_llm_score.py`, `run_highdim_llmlasso.py`, `run_highdim_datacentric.py`,
`run_highdim_freeform_pyramid.py`, `run_highdim_llm_rank.py`,
`OSTEOPOROSIS/run_osteoporosis_recent_suite.py`, `build_cross_dataset_summary.py`.
`credit_g_methods/`: the credit-risk variants of the same comparison
(`run_credit_g_llmselect_deepseek.py`, `run_datacentric_creditg.py`, `run_freeform_creditg.py`,
`run_icesearch_creditg.py`, `run_llmlasso_creditg.py`, plus the prompt builders
`build_credit_prompts.py`, `build_renal_rank_prompt.py`, `build_sepsis_rank_prompt.py`,
`build_osteoporosis_rank_prompt.py`, the direct-LLM driver `direct_llm_driver.py`,
`run_credit_g_seq_orchestrator.py`, `run_credit_g_deepseek_calls.py` and the rank evaluators
`eval_credit_g_score_rank.py`, `eval_renal_llm_rank.py`, `eval_sepsis_llm_rank.py`,
`eval_osteoporosis_llm_rank.py`).

**`d04_selective_ablation/`** (Table 6) — `04_RUNNERS/run_selective_ablation_offline.py` runs the four
weight arms (`Q+B+C`, `B+C`, `Q+C`, `Q+B`) on Breast SIS-20 with λ selected on development;
`build_selective_ablation_manifests.py`, `build_missing_query_manifest.py`,
`run_deepseek_missing_selective_queries.py`, `postprocess_ablation_measurements.py` complete the chain.
`d04_selective_ablation_design/01_AUDIT/audit_route_universe.py` audits the routing component universe.

**`d05_direct_llm_only/`** (Table 7) — `generate_pointwise_llm_scores.py`,
`evaluate_standardized_pure_llm.py`, `build_comparison_report.py`, and the per-dataset direct-selection
runners in `existing_method_baselines/` (`run_creditg_llmselect_baselines.py`,
`run_darmanis_pure_llm_score.py`, `run_osteoporosis_pure_llm.py`, `run_renal_pure_llm_score.py`).
