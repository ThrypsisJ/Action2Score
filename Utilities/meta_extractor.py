import os
import json
import pandas as pd

match = "/210510_euw1_5253755356.json"
if 'kr' in match: region = 'kr'
if 'euw1' in match: region = 'euw1'
if 'eun1' in match: region = 'eun1'
if 'na1' in match: region = 'na1'
if 'jp1' in match: region = 'jp1'

route = f"/mnt/devStorage/Dataset/League_of_Legends/raw/{region}"

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

with open(route+match, encoding="utf-8") as loaded_file:
    json_file = json.load(loaded_file)

participants = json_file['info']['participants']
players = []

for i in range(0, 10):
    player = {key:participants[i][key] for key in keys}
    players.append(player)

players = pd.DataFrame(players)
players.to_csv(f'../processed_csvs{match[:-5]}_meta.csv', index=False)