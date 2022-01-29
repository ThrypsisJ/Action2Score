#%%
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

#%%
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
with open('./Dataset/meta_analysis_with_score.ftr', 'rb') as file:
    meta = pd.read_feather(file)
meta.rename(columns={'gold_earned':'gold', 'score_norm':'score_model1', 'score_zero':'score_model2'}, inplace=True)
columns = ['match', 'player', 'kda', 'gold', 'creep', 'score_model1', 'score_model2']

block_num = int(len(meta) / 10)
ar = np.zeros([len(meta), len(columns)])
df = pd.DataFrame(ar, columns=columns)

for idx in tqdm(range(block_num)):
    # print(f'[{idx/block_num:3.1f}%] ({idx:5d}/{block_num:5d}) progressing...', end='\r')
    sdx = idx * 10
    edx = sdx + 9
    tmp = meta.loc[sdx:edx]
    df.loc[sdx:edx, 'match'] = tmp['mat_no']
    df.loc[sdx:edx, 'player'] = tmp['player']

    for col in columns[2:]:
        tmp.sort_values(by=col, ascending=True, ignore_index=True)
        data = [tmp.index[tmp.player==pi].item() for pi in range(10)]
        df.loc[sdx:edx, col] = data

meta.to_feather('./Dataset/analysis_result2.ftr')
# %%
