#%%
import pandas as pd
import pickle
from os import listdir
from tqdm import tqdm

score_path = './experiment/matches_scores'
norm_score = f'{score_path}/normal'
# zero_score = f'{score_path}/zero_h0'
matches_norm = listdir(norm_score)
# matches_zero = listdir(zero_score)

mat_path = './Dataset/matches_csv/legacy'

#%%
for match in tqdm(matches_norm, ncols=80, ascii=True):
    for serv in ['kr', 'euw1', 'eun1', 'na1', 'jp1']:
        if serv in match:
            mat_csv = f'{mat_path}/{serv}/{match[:-4]}.csv'
            break

    mat_csv = pd.read_csv(mat_csv)
    mat_csv = mat_csv[mat_csv['is_valid']==True]
    with open(f'{norm_score}/{match}', 'rb') as file:
        mat_sco = pickle.load(file)
    
    for player in range(10):
        tmp = mat_csv[mat_csv['player']==player+1]
        if len(tmp) == 0: continue

        



# %%
with open('./experiment/matches_scores/normal/210510_eun1_2825473113.pkl', 'rb') as file:
    tmp = pickle.load(file)

tmp[4].squeeze().tolist()
# %%
