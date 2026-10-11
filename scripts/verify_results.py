"""Check split isolation, train-only preprocessing, saved model, and metric exports."""
import argparse
import io
import json
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,help='Optional source archive for full record-level checks')
args=parser.parse_args()
r=Path(__file__).resolve().parents[1]/'results'
metrics_table = pd.read_csv(r/'model_metrics.csv').set_index('model')
by_target = pd.read_csv(r/'target_metrics.csv')
assert not by_target.duplicated(['model', 'target']).any()
assert set(by_target.model) == set(metrics_table.index)
for model, values in by_target.groupby('model'):
    assert len(values) == 25 and values.target.nunique() == 25
    assert np.isfinite(values[['mae_mm', 'rmse_mm']].to_numpy()).all()
    assert (values.mae_mm >= 0).all() and (values.rmse_mm >= values.mae_mm - 1e-8).all()
    assert np.isclose(values.mae_mm.mean(), metrics_table.loc[model, 'mae_mm'], atol=1e-7)
    assert np.isclose(np.sqrt(np.mean(values.rmse_mm**2)), metrics_table.loc[model, 'rmse_mm'], atol=1e-7)
selected = metrics_table.index[metrics_table.selected_on_validation.astype(str).str.lower().eq('true')]
assert len(selected) == 1 and selected[0] == metrics_table.validation_mae_mm.idxmin()
assert (metrics_table.test_mae_ci_low <= metrics_table.mae_mm).all()
assert (metrics_table.test_mae_ci_high >= metrics_table.mae_mm).all()
print('PASS: aggregate metrics agree with the 25 target-level summaries; selection matches validation MAE.')
if args.archive is None:
    print('Scope: exported aggregate consistency only; no claim of source-data, split, scaler or model verification.')
    raise SystemExit(0)
p=json.loads((r/'provenance.json').read_text(encoding='utf-8'))
parts=pd.read_csv(r/'partitions.csv')
assert parts.master_row.is_unique and len(parts)==1774
sets={k:set(v.input_group) for k,v in parts.groupby('partition')}
assert not(sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
assert not set(p['inputs'])&set(p['targets'])
with zipfile.ZipFile(args.archive) as z:
    name=next(n for n in z.namelist() if n.endswith('/dataset_maestro_3.xlsx'))
    master=pd.read_excel(io.BytesIO(z.read(name)))
master.columns=master.columns.str.strip()
train=master.iloc[parts.loc[parts.partition.eq('train'),'master_row'].to_numpy()-1]
bundle=joblib.load(r/'models.joblib')
assert np.allclose(bundle['scaler_x'].mean_,train[p['inputs']].mean())
assert np.allclose(bundle['scaler_y'].mean_,train[p['targets']].mean())
assert bundle['scaler_x'].n_samples_seen_==1241
pred=pd.read_csv(r/'test_predictions.csv')
metric=pd.read_csv(r/'model_metrics.csv').set_index('model')
assert len(pred)==3*267*25
assert set(pred.master_row)==set(parts.loc[parts.partition.eq('test'),'master_row'])
for model,group in pred.groupby('model'):
    err=group.prediction_mm-group.reference_mm
    assert np.isclose(abs(err).mean(),metric.loc[model,'mae_mm'],atol=1e-7)
    assert np.isclose(np.sqrt(np.mean(err**2)),metric.loc[model,'rmse_mm'],atol=1e-7)
assert metric.validation_mae_mm.idxmin()==p['selected_model']
test=master.iloc[parts.loc[parts.partition.eq('test'),'master_row'].to_numpy()-1]
actual=bundle['scaler_y'].inverse_transform(bundle['ridge'].predict(bundle['scaler_x'].transform(test[p['inputs']].to_numpy())))
expected=pred[pred.model.eq('Ridge')].pivot(index='master_row',columns='target',values='prediction_mm').loc[test.index+1,p['targets']].to_numpy()
assert np.allclose(actual,expected,atol=1e-6)
print('PASS: disjoint inputs/targets, grouped split, train-only scalers, all test metrics and serialized Ridge predictions.')
