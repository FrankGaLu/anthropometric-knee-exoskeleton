# Data scope

The included results describe five-input prediction of 25 excluded linear dimensions in a 1,774-row military-labeled working table, split into 1,241 training, 266 validation and 267 test rows. All linear error metrics use mm. The selected model is Ridge, based on validation MAE.

- `model_metrics.csv`: model-level validation/test metrics and recorded confidence intervals.
- `target_metrics.csv`: test MAE and RMSE for each target and model.
- `training_history.csv`: aggregate training history.
- `variable_dictionary.csv`: field names, roles and units.
- `author_definitions_20261010.csv`: author-reported case definitions and unresolved source equivalences; no participant measurements.
- `clustering.csv` and `clustering_config.json`: aggregate cluster scores and settings.

The source population tables, row-level predictions, saved models and participant material are excluded. This package provides aggregate consistency checks, not independent reproduction of fitting or verification of record-level provenance.

## External source archive for optional fitting

Filename: `Información_Antropometría.zip`.
SHA-256: `215be3f30308fc8a977a7eba23f50a961fa06d5e0b29d40d640f1072feea33cb`.

The optional analysis reads `dataset_maestro_3.xlsx`, `predicciones_exoesqueleto_28_30 dimensiones.xlsx` and `Original_NHANES_civilians 5 dimensions_Database.csv` from the original ZIP member structure. No download or access-on-request arrangement is implied.

## Interpretation limits

The author reports OPEN Design Lab as the acquisition route and male-only selection. Exact military row lineage, civilian survey cycle and participant-level joining of separately downloaded variables are not established. The 2,000 civilian working rows include 1,236 repeated full vectors and 26 constant-filled fields; they are excluded from predictive fitting. The consolidation code's military-mean filling, 1700/1770 mm stature groups and pooled rank percentiles agree with the working master.

Bimalleolar breadth versus height and calf length versus height are not established as equivalent between population and case fields. Existing numerical outputs are retained with those qualifications. The unused `Medida2` trial is excluded from the analyses.
