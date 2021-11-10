import os
import json
import pandas as pd

def extract():
    keys = (
        'kills', 'assists', 'deaths', 'goldEarned', 'goldSpent',
        'champLevel', 'consumablesPurchased',
        'damageDealtToBuildings', 'damageDealtToObjectives', 'damageDealtToTurrets', 'damageSelfMitigated',
        'magicDamageDealt', 'magicDamageDealtToChampions', 'magicDamageTaken',
        'physicalDamageDealt', 'physicalDamageDealtToChampions', 'physicalDamageTaken',
        'totalDamageDealt', 'totalDamageDealtToChampions', 'totalDamageShieldedOnTeammates', 'totalDamageTaken', 'totalHeal', 'totalHealsOnTeammates',
        'totalMinionsKilled', 'totalTimeCCDealt', 'totalTimeSpentDead',
        'trueDamageDealt', 'trueDamageDealtToChampions', 'trueDamageTaken',
        'win'
    )

    # if 'kr' in match: region = 'kr'
    # if 'euw1' in match: region = 'euw1'
    # if 'eun1' in match: region = 'eun1'
    # if 'na1' in match: region = 'na1'
    # if 'jp1' in match: region = 'jp1'

    route = '../../Dataset/League_of_Legends/challenger_raw/'

    tmp_matches = os.listdir(route)
    matches = []
    for match in tmp_matches:
        if 'timeline' not in match:
            matches.append(match)

    # match = "/210510_euw1_5253755356.json"

    for match in matches:
        with open(route+match, encoding="utf-8") as loaded_file:
            json_file = json.load(loaded_file)

            participants = json_file['info']['participants']
            players = []

            for i in range(0, 10):
                player = {key:participants[i][key] for key in keys}
                players.append(player)

            players = pd.DataFrame(players)
            players.to_csv(f'./processed_csvs/challengers/{match[:-5]}_meta.csv', index=False)