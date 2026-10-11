"""Reanalysis of the supplied archive; no extraction or source-file mutation.

Usage: python scripts/reproduce_analysis.py --archive PATH_TO_ZIP
Outputs: results/ (tables, split, fitted models and provenance).
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

INPUTS = ['Estatura', 'Trocanter_altura', 'Rodilla_altura_sentado',
          'Pie_longitud', 'Nalga_rodilla_longitud']
META = ['cluster', 'poblacion', 'percentil_estatura']
SEED = 42


def metrics(reference, output):
    delta = np.asarray(output) - np.asarray(reference)
    return {'mae_mm': float(np.abs(delta).mean()),
            'rmse_mm': float(np.sqrt(np.square(delta).mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Source ZIP for model fitting; omitted: check existing aggregate exports')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    args = parser.parse_args()
    if args.archive is None:
        import subprocess
        import sys
        subprocess.run([sys.executable, str(Path(__file__).with_name('verify_results.py'))], check=True)
        print('Existing aggregate exports checked. No training or source-data validation was performed.')
        return
    archive = args.archive.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with zipfile.ZipFile(archive) as z:
        def read_xlsx(name, **kwargs):
            paths = [n for n in z.namelist() if n.endswith('/' + name)]
            if len(paths) != 1:
                raise ValueError(f'Expected exactly one {name}, got {len(paths)}')
            return pd.read_excel(io.BytesIO(z.read(paths[0])), **kwargs)
        master = read_xlsx('dataset_maestro_3.xlsx')
        case = read_xlsx('predicciones_exoesqueleto_28_30 dimensiones.xlsx', sheet_name='Resultados_1')
        # Correct the mislabeled processed civilian table using source column names.
        raw = [n for n in z.namelist() if n.endswith('Original_NHANES_civilians 5 dimensions_Database.csv')]
        civilian = pd.read_csv(io.BytesIO(z.read(raw[0])), encoding='latin1')
    master.columns = master.columns.str.strip()
    targets = [c for c in master if c not in INPUTS + META + ['Peso']]
    assert len(targets) == 25 and not set(INPUTS) & set(targets)
    military = master.loc[master.poblacion.eq('Militar')].copy()
    if military[INPUTS + targets].isna().any().any():
        raise ValueError('Missing values require an explicitly documented imputation protocol.')
    X, y = military[INPUTS].to_numpy(float), military[targets].to_numpy(float)
    # Equal input vectors are inseparable groups, regardless of output differences.
    groups = pd.factorize(pd.MultiIndex.from_frame(military[INPUTS]))[0]
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEED).split(X, y, groups))
    ti, vi = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=SEED + 1).split(X[trainval], y[trainval], groups[trainval]))
    train, val = trainval[ti], trainval[vi]
    for a, b in [(train,val), (train,test), (val,test)]:
        assert not set(groups[a]) & set(groups[b])
    sx, sy = StandardScaler().fit(X[train]), StandardScaler().fit(y[train])
    xt = torch.tensor(sx.transform(X[train]), dtype=torch.float32)
    yt = torch.tensor(sy.transform(y[train]), dtype=torch.float32)
    xv = torch.tensor(sx.transform(X[val]), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(5,128), torch.nn.ReLU(),
                              torch.nn.Linear(128,64), torch.nn.ReLU(), torch.nn.Linear(64,25))
    optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=1e-4)
    best, best_epoch, best_state, stale = float('inf'), 0, None, 0
    history = []
    # Checkpoints selected by native-unit validation MAE, never by test error.
    for epoch in range(1, 1001):
        net.train(); optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward(); optimizer.step()
        net.eval()
        with torch.no_grad():
            pv = sy.inverse_transform(net(xv).numpy())
        vm = metrics(y[val], pv)['mae_mm']
        history.append({'epoch': epoch, 'train_scaled_mse': float(loss.detach()), 'validation_mae_mm': vm})
        if vm < best - 1e-5:
            best, best_epoch, best_state, stale = vm, epoch, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
        if stale >= 75:
            break
    net.load_state_dict(best_state); net.eval()
    ridge_candidates = []
    for alpha in [.01, .1, 1., 10., 100.]:
        model = Ridge(alpha=alpha).fit(sx.transform(X[train]), sy.transform(y[train]))
        score = metrics(y[val], sy.inverse_transform(model.predict(sx.transform(X[val]))))['mae_mm']
        ridge_candidates.append((score, alpha, model))
    _, alpha, ridge = min(ridge_candidates, key=lambda t:t[0])
    mean = y[train].mean(axis=0)
    def predict(name, values):
        if name == 'Training mean': return np.tile(mean, (len(values),1))
        if name == 'Ridge': return sy.inverse_transform(ridge.predict(sx.transform(values)))
        with torch.no_grad():
            return sy.inverse_transform(net(torch.tensor(sx.transform(values),dtype=torch.float32)).numpy())
    models = ['Training mean','Ridge','MLP']
    validation = {name:metrics(y[val],predict(name,X[val])) for name in models}
    selected = min(models, key=lambda name:validation[name]['mae_mm'])
    records=[]; per_dimension=[]; predictions=[]
    for name in models:
        pt=predict(name,X[test]); mt=metrics(y[test],pt)
        # Bootstrap input groups, preserving within-group dependence.
        residual=np.abs(pt-y[test]).mean(axis=1)
        test_groups=np.unique(groups[test]); rng=np.random.default_rng(SEED+100)
        sums=np.array([residual[groups[test]==g].sum() for g in test_groups])
        counts=np.array([(groups[test]==g).sum() for g in test_groups])
        draws=rng.integers(0,len(test_groups),size=(2000,len(test_groups)))
        ci=np.percentile(sums[draws].sum(axis=1)/counts[draws].sum(axis=1),[2.5,97.5])
        records.append({'model':name,'validation_mae_mm':validation[name]['mae_mm'],**mt,
                        'test_mae_ci_low':float(ci[0]),'test_mae_ci_high':float(ci[1]),'selected_on_validation':name==selected})
        for j,target in enumerate(targets):
            per_dimension.append({'model':name,'target':target,**metrics(y[test,j],pt[:,j])})
        for k,idx in enumerate(test):
            for j,target in enumerate(targets):
                predictions.append({'model':name,'master_row':int(military.index[idx])+1,'input_group':int(groups[idx]),
                                    'target':target,'reference_mm':float(y[idx,j]),'prediction_mm':float(pt[k,j])})
    # The case is evaluated only after validation-based selection.
    refs=case.assign(key=case['Característica'].str.strip().str.lower()).set_index('key')
    case_x=np.array([[float(refs.loc[c.lower(),'Valor_Digital']) for c in INPUTS]])
    manual=np.array([[float(refs.loc[c.lower(),'Medida física']) for c in targets]])
    digital=np.array([[float(refs.loc[c.lower(),'Valor_Digital']) for c in targets]])
    case_rows=[]; case_metrics=[]
    for name in models:
        pc=predict(name,case_x)
        case_metrics.append({'model':name,**{f'manual_{k}':v for k,v in metrics(manual,pc).items()},
                              **{f'digital_{k}':v for k,v in metrics(digital,pc).items()}})
        for j,c in enumerate(targets):
            case_rows.append({'model':name,'target':c,'manual_mm':float(manual[0,j]),'digital_mm':float(digital[0,j]),'prediction_mm':float(pc[0,j])})
    partition=pd.DataFrame({'master_row':military.index.to_numpy()+1,'input_group':groups,'partition':''})
    for label,indices in [('train',train),('validation',val),('test',test)]: partition.loc[indices,'partition']=label
    bundle={'input_columns':INPUTS,'target_columns':targets,'scaler_x':sx,'scaler_y':sy,'ridge':ridge,
            'training_mean':mean,'mlp_state':{k:v.numpy() for k,v in best_state.items()},'selected':selected}
    joblib.dump(bundle,out/'models.joblib')
    loaded=joblib.load(out/'models.joblib')
    assert np.allclose(loaded['ridge'].predict(sx.transform(X[test])),ridge.predict(sx.transform(X[test])))
    restored=copy.deepcopy(net)
    restored.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in loaded['mlp_state'].items()})
    with torch.no_grad():
        assert torch.equal(restored(xv),net(xv)), 'Serialized MLP must reproduce its predictions.'
    exports={'model_metrics.csv':pd.DataFrame(records),'target_metrics.csv':pd.DataFrame(per_dimension),
             'test_predictions.csv':pd.DataFrame(predictions),'partitions.csv':partition,
             'training_history.csv':pd.DataFrame(history),'case_predictions.csv':pd.DataFrame(case_rows),
             'case_metrics.csv':pd.DataFrame(case_metrics)}
    civilian=civilian.rename(columns={'subjectid':'Local_ID','Stature':'Estatura','trochanterionheight':'Trocanter_altura',
        'kneeheightsitting':'Rodilla_altura_sentado','footlength':'Pie_longitud','buttockkneelength':'Nalga_rodilla_longitud'})
    exports['corrected_civilian_table.csv']=civilian[['Local_ID']+INPUTS]
    geometry=[]
    for name in models+['Manual reference','Digital reference']:
        if name in models:
            row={r['target']:r['prediction_mm'] for r in case_rows if r['model']==name}
            tro=float(case_x[0,INPUTS.index('Trocanter_altura')])
        else:
            col='Medida física' if name=='Manual reference' else 'Valor_Digital'
            row={c:float(refs.loc[c.lower(),col]) for c in targets}
            tro=float(refs.loc['trocanter_altura',col])
        patella=row['Rotula_medial_altura']; malleolus=row['Maleolo_lateral_altura']
        distal=patella-35-malleolus; proximal=tro-patella+70
        assert distal>0 and proximal>0
        geometry.append({'model':name,'trochanter_mm':tro,'patella_mm':patella,'malleolus_mm':malleolus,
                         'distal_bar_mm':distal,'proximal_bar_mm':proximal,'distal_proximal_percent':100*distal/proximal})
    exports['predicted_geometry.csv']=pd.DataFrame(geometry)
    dictionary=[]
    for c in master:
        role='metadata' if c in META else 'input' if c in INPUTS else 'target' if c in targets else 'excluded_mass'
        dictionary.append({'field':c,'role_predictive_protocol':role,'unit':'hg' if c=='Peso' else 'mm' if c not in META else 'metadata',
                           'civilian_constant_filled':bool(c not in META+INPUTS),
                           'source_definition_status':'Original landmark/source mapping requires author confirmation' if c not in META else 'Working-table metadata'})
    exports['variable_dictionary.csv']=pd.DataFrame(dictionary)
    for name,df in exports.items(): df.to_csv(out/name,index=False,float_format='%.10g')
    provenance={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':SEED,
        'rows':len(military),'groups':len(np.unique(groups)),'split_rows':{n:len(i) for n,i in [('train',train),('validation',val),('test',test)]},
        'inputs':INPUTS,'targets':targets,'selection':'minimum validation MAE; test not used for tuning',
        'selected_model':selected,'ridge_alpha':alpha,'mlp_best_epoch':best_epoch,'mlp_completed_epochs':epoch,
        'mlp':{'layers':[5,128,64,25],'learning_rate':.001,'weight_decay':1e-4,'patience':75,'max_epochs':1000},
        'bootstrap':{'unit':'identical-input group','replicates':2000,'seed':SEED+100,'interval':'percentile 95%'},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},
        'cautions':['Military label has unverified survey lineage.','No independent participant identifiers.','One retrospective case; no clinical validation.'],
        'digital_manual_baseline_25_targets':metrics(manual,digital)}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'protocol':provenance,'test_results':records,'case_results':case_metrics},indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
