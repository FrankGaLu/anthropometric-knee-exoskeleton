# Anthropometric prediction: code and aggregate results

Supporting package for “Anthropometric Data Integration and Held-Out Prediction for Preliminary Knee Exoskeleton Sizing”.

Documentation, package filenames and execution messages are in English. Exact source identifiers are retained for compatibility; their English labels and interpretation limits are listed in [Field names](docs/FIELD_NAMES.md).

This package contains original analysis code, aggregate model and target-level metrics, training history, a variable dictionary and an aggregate summary plot. It is prepared locally and has not been deposited publicly. It is not a release of the source population tables or participant-level dataset.

## Run without source data

```powershell
python -m pip install -r scripts/requirements.txt
python scripts/verify_results.py
python scripts/plot_summary.py
```

The checker compares aggregate MAE and RMSE with the 25 target-level summaries and checks validation-based model selection. The figure script exports `figures/model_summary.pdf` using only included CSVs. Neither step trains models or verifies participant independence, input provenance or training-only preprocessing.

## Optional full analysis with existing source files

```powershell
python scripts/reproduce_analysis.py --archive "PATH/Información_Antropometría.zip" --output "local_results"
```

This requires an authorized copy of the original archive, which is not included. Without `--archive`, the analysis entry point only checks existing aggregate exports. Full fitting cannot be independently reproduced from this package alone. The original archive structure and input filenames are documented in `docs/DATA.md`.

## Scope and licenses

Original code: MIT (`LICENSE-CODE`). Original aggregate results and documentation: CC BY 4.0 (`LICENSE-DATA`). Source population data, individual predictions, saved models, case measurements, acquisition images and code containing case values are excluded from this package. Third-party dependency licenses remain unchanged. See `docs/LICENSING.md`.

The reported dataset labels do not establish verified survey lineage. Some anatomical field correspondences remain ambiguous. These are limitations of the existing results, not additional tasks performed by these programs.

## Authors

Raymundo Macías González; Javier Antonio Campos Notario; David Luviano Cruz; Alma Guadalupe Rodríguez Ramírez; Francesco García-Luna. Universidad Autónoma de Ciudad Juárez, Mexico.

Correspondence: alma.rodriguez.ram@uacj.mx; francesco.garcia@uacj.mx.

`manifest.csv` records the package files and SHA-256 hashes. `CITATION.cff` supplies author and version metadata; `docs/MDPI_DATA_SCOPE.md` explains the scope represented to the journal. No public URL, DOI or accepted-publication status is asserted. Citation of the associated article and any future deposited version is requested as scholarly practice, without adding license conditions.
