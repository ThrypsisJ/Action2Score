import pandas as pd
from os import listdir

def concat(g3=False):
    # match_id = '210510_eun1_2825546524'
    # match_id = '210510_euw1_5253755356' # too many WARD_PLACED
    # match_id = '210510_kr_5179842525'

    # data = pd.read_feather(f'../processed_ftr/scores_spaghettini.ftr')
    # data = data[data['match_id'] == match_id]

    path = './processed_csvs/challengers'
    score_path = './processed_ftr/challenger_g3_score.csv' if g3 else './processed_ftr/scores_challenger.csv'

    tmp_matches = listdir(path)
    matches = []
    for match in tmp_matches:
        if '_meta' in match : continue
        else                : matches.append(match)

    for match in matches:
        data = pd.read_csv(score_path)
        data = data[data['match_id'] == match[:-12]]


        sum_scores = []
        for player in range(10):
            player_scores = data[data['player'] == player+1]['score'].tolist()
            sum_scores.append(sum(player_scores))

        data = pd.read_csv(f'./processed_csvs/challengers/{match}')
        data.insert(len(data.columns), 'total score', sum_scores, True)

        data.to_csv(f'./processed_csvs/challengers/{match}', index=False)