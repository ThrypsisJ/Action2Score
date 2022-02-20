#%%
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

base_path = Path('./experiment/matches_scores')
csv_path = Path('./Dataset/matches_csv')
save_path = Path('./experiment/feature_scores.csv')
result_path = Path('./Dataset/results.csv')
models = [model.name for model in list(base_path.glob('*'))]
match_list = [match.name[:-4] for match in list((base_path / models[0]).glob('*'))]
match_result = pd.read_csv(result_path)

first = True
for match in tqdm(match_list):
    features = pd.read_csv((csv_path / f'{match}.csv'))
    features.sort_values(['player', 'time'], ascending=[True, False]).reset_index(drop=True)
    features.drop(['player_level', 'skill_level', 'tower_diff', 'is_valid'], axis='columns', inplace=True)
    features.insert(0, 'match_no', match)
    winner = [1,2,3,4,5] if (match_result[match_result['match']==match]['win'].item()=='blue') else [6,7,8,9,10]
    for model in models:
        features[model] = np.nan
        reverse = True if 'reverse' in model else False
        score_file = base_path / model / f'{match}.pkl'
        with score_file.open('rb') as file: scores = pickle.load(file)
        for player, p_scores in enumerate(scores):
            p_scores = p_scores.squeeze().tolist()
            if not isinstance(p_scores, list): p_scores = [p_scores]
            if reverse: p_scores.reverse()
            features.loc[features['player']==(player+1), model] = p_scores
            features.loc[features['player']==(player+1), 'win'] = True if (player+1) in winner else False
    if first: features.to_csv('./experiment/scores_with_features.csv', index=False, mode='w'); first=False
    else: features.to_csv('./experiment/scores_with_features.csv', index=False, mode='a')
# %%
convert = pd.read_csv('./experiment/scores_with_features.csv')
convert.to_feather('./experiment/scores_with_features.ftr')
# %%
