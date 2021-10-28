import pandas as pd

# match_id = '210510_eun1_2825546524'
match_id = '210510_euw1_5253755356' # too many WARD_PLACED
# match_id = '210510_kr_5179842525'

data = pd.read_feather(f'../processed_ftr/scores_spaghettini.ftr')
data = data[data['match_id'] == match_id]

sum_scores = []
for player in range(10):
    player_scores = data[data['player'] == player]['score'].tolist()
    sum_scores.append(sum(player_scores))

data = pd.read_csv(f'../processed_csvs/{match_id}_analyze.csv')
data.insert(len(data.columns), 'total score', sum_scores, True)

data.to_csv(f'../processed_csvs/{match_id}_analyze.csv', index=False)