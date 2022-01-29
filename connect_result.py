#%%
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

score_path = Path('./experiment/matches_scores')
norm_score = score_path / 'normal'
zero_score = score_path / 'zero_h0'
mat_path = Path('./Dataset/matches_csv')
save_norm = Path('./experiment/scores_with_matdata/norm')
save_zero = Path('./experiment/scores_with_matdata/zero_h0')
if not save_norm.exists(): save_norm.mkdir(parents=True)
if not save_zero.exists(): save_zero.mkdir(parents=True)

#%%
matches_norm = list(norm_score.glob('*'))
for idx, match in enumerate(matches_norm):
    print(f'[Normal results] ({idx:5d}/{len(matches_norm)}) {idx/len(matches_norm)*100:3.1f}% progressing...', end='\r')
    mat_csv = mat_path / f'{match.name[:-4]}.csv'

    mat_csv = pd.read_csv(mat_csv)
    mat_csv.insert(len(mat_csv.columns), 'score', np.nan)

    with match.open('rb') as file:
        mat_sco = pickle.load(file)

    for player in range(10):
        if (player+1) not in list(mat_csv.player): continue
        scos = mat_sco[player].squeeze().tolist()
        if isinstance(scos, list): scos.reverse()
        mat_csv.loc[(mat_csv.player==player+1), 'score'] = scos
    mat_csv.to_feather(save_norm / f'{match.name[:-4]}.ftr')

#%%
matches_zero = list(zero_score.glob('*'))
for idx, match in enumerate(matches_zero):
    print(f'[Zero_h0 results] ({idx:5d}/{len(matches_zero)}) {idx/len(matches_zero)*100:3.1f}% progressing...', end='\r')
    mat_csv = mat_path / f'{match.name[:-4]}.csv'

    mat_csv = pd.read_csv(mat_csv)
    mat_csv.insert(len(mat_csv.columns), 'score', np.nan)

    with match.open('rb') as file:
        mat_sco = pickle.load(file)

    for player in range(10):
        if (player+1) not in list(mat_csv.player): continue
        scos = mat_sco[player].squeeze().tolist()
        if isinstance(scos, list): scos.reverse()
        mat_csv.loc[(mat_csv.player==player+1), 'score'] = scos
    mat_csv.to_feather(save_zero / f'{match.name[:-4]}.ftr')
# %%
