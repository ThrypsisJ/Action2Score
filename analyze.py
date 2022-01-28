#%%
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

norm_path = Path('./experiment/matches_scores/normal')
norm_list = list(norm_path.glob('*'))

zero_path = Path('./experiment/matches_scores/zero_h0')
zero_list = list(zero_path.glob('*'))

with open('./Dataset/meta_analysis.ftr', 'rb') as file:
    meta = pd.read_feather(file)

#%%
meta['score_norm'] = np.nan
for idx, norm_file in enumerate(norm_list):
    print(f'[Norm] ({idx:5d}/{len(norm_list)}) {idx/len(norm_list)*100:3.1f}% progressing...', end='\r')
    mat_name = norm_file.name[:-4]

    with norm_file.open('rb') as file:
        scores = pickle.load(file)
    scores = [scores[idx].sum().item() for idx in range(10)]

    meta.loc[(meta.mat_no==mat_name), 'score_norm'] = scores

#%%
meta['score_zero'] = np.nan
for idx, zero_file in enumerate(zero_list):
    print(f'[Zero] ({idx:5d}/{len(zero_list)}) {idx/len(zero_list)*100:3.1f}% progressing...', end='\r')
    mat_name = zero_file.name[:-4]

    with zero_file.open('rb') as file:
        scores = pickle.load(file)
    scores = [scores[idx].sum().item() for idx in range(10)]

    meta.loc[(meta.mat_no==mat_name), 'score_zero'] = scores

#%%
meta.to_feather('./Dataset/meta_analysis_with_score.ftr')
# %%
