# Automated Test Report

Generated: 2026-11-08T07:39:07.345045+00:00

**Result:** FAILED

- Passed: 5
- Failed: 15
- Skipped: 0
- Total: 20

## Failed tests

- `tests.test_api::test_health_and_prediction_log_only_metadata`
- `tests.test_api::test_api_rejects_unknown_input_fields`
- `tests.test_api::test_api_rejects_missing_and_out_of_range_features`
- `tests.test_inference_monitoring::test_summary_reports_request_prediction_probability_and_latency_metrics`
- `tests.test_inference_monitoring::test_summary_accepts_legacy_records_without_latency`
- `tests.test_inference_monitoring::test_summary_rejects_invalid_log_records[{"prediction": 1, "late_probability": 1.2}-Invalid late_probability]`
- `tests.test_inference_monitoring::test_summary_rejects_invalid_log_records[{"prediction": 1, "late_probability": 0.5, "latency_ms": -1}-Invalid latency_ms]`
- `tests.test_inference_monitoring::test_summary_rejects_invalid_log_records[not-json-Invalid JSON]`
- `tests.test_inference_monitoring::test_summary_rejects_missing_and_empty_logs`
- `tests.test_monitoring::test_compare_batches_reports_and_saves_feature_drift`
- `tests.test_monitoring::test_drift_monitor_rejects_missing_columns_and_nonfinite_values`
- `tests.test_validation_and_model_loader::test_load_model_reads_valid_pipeline_artifact`
- `tests.test_validation_and_model_loader::test_load_model_rejects_missing_or_invalid_artifact`
- `tests.test_validation_and_model_loader::test_load_training_data_rejects_missing_columns`
- `tests.test_validation_and_model_loader::test_load_training_data_rejects_non_numeric_feature_and_bad_labels`

## Passed tests

- `tests.test_monitoring::test_psi_is_zero_for_identical_distributions`
- `tests.test_monitoring::test_psi_detects_shift_and_status_thresholds`
- `tests.test_pipeline::test_training_data_schema_and_labels`
- `tests.test_pipeline::test_pipeline_fits_and_predicts_without_id`
- `tests.test_validation_and_model_loader::test_training_input_schema_contains_all_shipment_fields`
