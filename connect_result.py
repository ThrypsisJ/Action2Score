#%%
import pandas as pd
import pickle
import numpy as np
from os import listdir, makedirs
from os.path import exists
from tqdm import tqdm

score_path = './experiment/matches_scores'
norm_score = f'{score_path}/normal'
zero_score = f'{score_path}/zero_h0'
matches_norm = listdir(norm_score)
matches_zero = listdir(zero_score)

mat_path = './Dataset/matches_csv/legacy'

save_path = './experiment/scores_with_matdata'
if not exists(save_path): makedirs(save_path)
if not exists(f'{save_path}/norm'): makedirs(f'{save_path}/norm')
if not exists(f'{save_path}/zero_h0'): makedirs(f'{save_path}/zero_h0')

#%%
for match in tqdm(matches_norm, ncols=80, ascii=True):
    for serv in ['kr', 'euw1', 'eun1', 'na1', 'jp1']:
        if serv in match:
            mat_csv = f'{mat_path}/{serv}/{match[:-4]}.csv'
            break

    mat_csv = pd.read_csv(mat_csv)
    mat_csv.insert(len(mat_csv.columns), 'score', np.nan)
    with open(f'{norm_score}/{match}', 'rb') as file:
        mat_sco = pickle.load(file)
    
    for player in range(10):
        if len(mat_csv.loc[mat_csv.player==player+1]) == 0: continue
        scos = mat_sco[player].squeeze().tolist()
        scos.reverse()
        mat_csv.loc[(mat_csv.player==player+1), 'score'] = scos

    mat_csv.to_feather(f'{save_path}/norm/{match[:-4]}.ftr')

#%%
for match in tqdm(matches_zero, ncols=80, ascii=True):
    for serv in ['kr', 'euw1', 'eun1', 'na1', 'jp1']:
        if serv in match:
            mat_csv = f'{mat_path}/{serv}/{match[:-4]}.csv'
            break

    mat_csv = pd.read_csv(mat_csv)
    mat_csv.insert(len(mat_csv.columns), 'score', np.nan)
    with open(f'{zero_score}/{match}', 'rb') as file:
        mat_sco = pickle.load(file)
    
    for player in range(10):
        if len(mat_csv.loc[mat_csv.player==player+1]) == 0: continue
        scos = mat_sco[player].squeeze().tolist()
        scos.reverse()
        mat_csv.loc[(mat_csv.player==player+1), 'score'] = scos
        
    mat_csv.to_feather(f'{save_path}/zero_h0/{match[:-4]}.ftr')