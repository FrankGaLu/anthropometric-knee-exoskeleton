# Execution modes

Python 3.11 and the pinned packages in `scripts/requirements.txt` are used. Run from this folder.

## Included aggregate workflow

```powershell
python scripts/verify_results.py
python scripts/plot_summary.py
```

This checks consistency between exported aggregate and per-target metrics, checks validation-based selection and regenerates the summary PDF. It requires no source ZIP, author reply, consent document, repository link or additional measurement. Passing these checks does not verify the original data, split isolation or fitted models.

## Optional source-based workflow

`reproduce_analysis.py --archive PATH --output PATH` fits the original protocol only when a real source archive is explicitly provided. It does not invent missing data. Running it without `--archive` checks existing aggregate exports instead.

`verify_results.py --archive PATH` additionally checks record-level splits, training scalers and saved model predictions when the corresponding complete local result files are present. Those files are not distributed in this aggregate package, so this mode belongs to the full local project.

The manuscript build and the eleven article figures remain in the full local project. This smaller package contains only the summary figure generated from aggregate results. No unverified source or participant material is required by its default workflow.
