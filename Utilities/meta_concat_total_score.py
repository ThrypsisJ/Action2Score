import pandas as pd
from os import listdir

def concat(g3=False):
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