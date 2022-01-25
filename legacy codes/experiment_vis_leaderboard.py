import pandas as pd

def load_leaderboard(match_id, challenger=False):
    if challenger   : data = pd.read_csv(f'./processed_csvs/challengers/{match_id}_analyze.csv')
    else            : data = pd.read_csv(f'./processed_csvs/{match_id}_analyze.csv')
    index = [f'Rank {i+1}' for i in range(10)]

    leaderboard = {}
    for column in data.columns[1:-2]:
        rank_of_column = []
        for rank in range(10):
            player = data[data[column] == rank]['player'].iloc[0]
            rank_of_column.append('Player %2d'%player)
        leaderboard[column[:-5]] = rank_of_column

    sorted_by_sum_of_rank = data.sort_values(by='sum of ranks', axis=0, ascending=True, ignore_index=True)['player'].tolist()
    for i in range(len(sorted_by_sum_of_rank)): sorted_by_sum_of_rank[i] = 'Player %2d'%sorted_by_sum_of_rank[i]
    leaderboard['sum of ranks'] = sorted_by_sum_of_rank

    sorted_by_score = data.sort_values(by='total score', axis=0, ascending=False, ignore_index=True)['player'].tolist()
    for i in range(len(sorted_by_score)): sorted_by_score[i] = 'Player %2d'%sorted_by_score[i]
    leaderboard['total score'] = sorted_by_score

    leaderboard = pd.DataFrame(leaderboard, index=index)
    return leaderboard

def load_leaderboard_graph(match_id, challenger=False):
    if challenger:
        data = pd.read_csv(f'./processed_csvs/challengers/{match_id}_analyze.csv')
        result = pd.read_csv(f'processed_csvs/challenger_result.csv')
    else:
        data = pd.read_csv(f'./processed_csvs/{match_id}_analyze.csv')
        result = pd.read_csv(f'./processed_csvs/match_result_test.csv')
        
    result = result[result['match_no'] == match_id]['win'].item()
    
    isWin = []
    for i in range(5):
        if result == 'blue': isWin.append(True)
        else: isWin.append(False)
    for i in range(5, 10):
        if result == 'blue': isWin.append(False)
        else: isWin.append(True)
    index = [f'Player {i+1}' for i in range(10)]
    
    leaderboard = {}
    lboard_columns = ['KDA', 'Gold', 'Creep Score', 'PASTA score']
    columns = ['KDA_rank', 'Gold Earn & Spend_rank', 'totalMinionsKilled_rank', 'total score']
    for lcolumn, column in zip(lboard_columns, columns):
        if lcolumn != 'PASTA score':
            tmp = data[column].tolist()
            for i in range(0, len(tmp)): tmp[i] += 1
            leaderboard[lcolumn] = tmp
        else:
            tmp = data.sort_values(by='total score', axis=0, ascending=False, ignore_index=True)['player'].tolist()
            rank = [0 for i in range(10)]
            for i in range(10): rank[tmp[i]-1] = i+1
            leaderboard[lcolumn] = rank
    
    leaderboard = pd.DataFrame(leaderboard, index=index)
    return leaderboard, isWin