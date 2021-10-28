import pandas as pd

csv_path = '../processed_csvs/'

# match_id = '210510_eun1_2825546524'
match_id = '210510_euw1_5253755356' # too many WARD_PLACED
# match_id = '210510_kr_5179842525'

data = pd.read_csv(f'{csv_path}{match_id}_meta.csv')

KDAs = []
GoldESs = []
players = data['Unnamed: 0'].tolist()

for i in range(10):
    KDA = (data.loc[i, 'kills'] + data.loc[i, 'assists']) / data.loc[i, 'deaths']
    KDAs.append(KDA)
    GoldES = data.loc[i, 'goldEarned'] + data.loc[i, 'goldSpent']
    GoldESs.append(GoldES)

analyzeData = data[['totalDamageDealtToChampions', 'damageDealtToBuildings', 'totalDamageTaken', 'totalHeal', 'totalMinionsKilled', 'totalTimeCCDealt', 'totalTimeSpentDead', 'win']]
analyzeData.insert(0, 'Gold Earn & Spend', GoldESs, True)
analyzeData.insert(0, 'KDA', KDAs, True)
analyzeData.insert(0, 'Player', players, True)

rank_score = {
    'player': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
for column in analyzeData.columns[1:-1]:
    sortedData = analyzeData.sort_values(column, axis=0, ascending=False, inplace=False, ignore_index=True)
    rank_of_player = [0 for _ in range(10)]
    for rank in range(10):
        player = sortedData.loc[rank, 'Player']
        rank_of_player[player] = rank
    rank_score[f'{column}_rank'] = rank_of_player

rank_score = pd.DataFrame(rank_score)

sum_ranks = []
for i in range(10):
    ranks = rank_score.loc[i].tolist()
    sum_ranks.append(sum(ranks))
rank_score.insert(len(rank_score.columns), 'sum of ranks', sum_ranks, True)

rank_score.to_csv(f'{csv_path}{match_id}_analyze.csv', index=False)