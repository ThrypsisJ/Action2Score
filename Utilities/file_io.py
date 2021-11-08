import json
import csv
import torch
from os.path import exists
from os import listdir
import pandas as pd

#################################################
############### File IO functions ###############
#################################################
def timeline_file_opener(file_name, server, train=True):
    if server != None:
        route = f"./mat_datas_train/{server}" if train else f"./mat_datas_test/{server}"
    else:
        route = '../../Dataset/League_of_Legends/challenger_raw'

    with open(f"{route}/{file_name}", encoding="utf-8") as loaded_file:
        json_file = json.load(loaded_file)
    return json_file

def csv_opener(filename, mode="r", encode="utf-8"):
    csv_file = None
    if exists(filename):
        csv_file = open(filename, mode, encoding=encode, newline='')
    return csv_file

def save_to_csv(pd_data, file_name="Unnamed", sort_by=None):
    pd_data = pd_data.sort_values(by=sort_by)
    pd_data.to_csv(file_name, index=False, encoding="utf-8")

def item_dict_opener():
    item_csv = csv_opener("./processed_csvs/items.csv", "r", "cp949")
    item_reader = csv.reader(item_csv)
    item_dict = {}
    for item in item_reader:
        item_dict[str(item[0])] = {"buy": item[1], "sell": item[2]}
    return item_dict

def champion_dict_opener():
    file = f"./DataDragon/11.21.1/data/ko_KR/championFull.json"
    champ_cat_csv = csv_opener("./processed_csvs/champs_dummy_variable.csv", "r", "utf-8")
    champ_cat_reader = csv.reader(champ_cat_csv)
    champ_cat_dic = {}
    for champ in champ_cat_reader:
        champ_cat_dic[str(champ[0])] = {
            "mage": 1 if champ[1] == "1" else 0,
            "fighter": 1 if champ[2] == "1" else 0,
            "support": 1 if champ[3] == "1" else 0,
            "tank": 1 if champ[4] == "1" else 0,
            "assassin": 1 if champ[5] == "1" else 0,
            "marksman": 1 if champ[6] == "1" else 0
        }

    with open(file, encoding="utf-8") as loaded_file:
        champion_file = json.load(loaded_file)

    champion_dict = {}
    for c_name, c_data in champion_file["data"].items():
        cham_id = c_data["key"]

        spells = {}
        for idx, spell in enumerate(c_data["spells"]):
            spells[str(idx+1)] = spell["maxrank"]
        
        champ_cat = champ_cat_dic[c_name]

        champion_dict[cham_id] = {
            "spells": spells,
            "category": champ_cat
        }

    return champion_dict

def save_prediction(match_name, prediction, file_name, winner):
    predict_file = pd.DataFrame({
            "match_id": [match_name],
            "prediction": [prediction],
            "winner": [winner]
        })
    if not exists(file_name):
        predict_file.to_csv(file_name, mode="w", encoding="utf-8", index=False)
    else:
        predict_file.to_csv(file_name, mode="a", encoding="utf-8", index=False, header=False)

def save_feature(match_name, features, scores, file_name, winner):
    labels = [
        "match_id", "player", "time", "mage", "fighter", "support", "tank", "assassin", "marksman", "TOP", "MIDDLE", "BOTTOM", "UTILITY", "JUNGLE", "x_position", "y_position",
        "deviation", "ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "SKILL_LEVEL_UP", "LEVEL_UP", "WARD_PLACED", "WARD_KILL", "CHAMPION_KILL", "CHAMPION_KILL_ASSIST",
        "CHAMPION_KILL_VICTIM", "BUILDING_KILL", "BUILDING_KILL_ASSIST", "ELITE_MONSTER_KILL", "ELITE_MONSTER_KILL_ASSIST", "event_weight", "score", "win"
        ]

    score_file = csv_opener(file_name, mode="a")
    if score_file != None:
        writer = csv.writer(score_file)
        for player in range(0, 10):
            temp_feature = features[player].cpu().detach().numpy().tolist()
            temp_score = scores[player].cpu().detach().numpy().tolist()
            win = (player < 5 and winner=="blue") or (player >= 5 and winner=="red")
            for event_idx in range(0, len(temp_feature[0])):
                save_features = [match_name, player] + temp_feature[0][event_idx] + [temp_score[0][event_idx][0], win]
                writer.writerow(save_features)
    else:
        score_file = open(file_name, mode="w", encoding="utf-8")
        writer = csv.writer(score_file)
        writer.writerow(labels)
        for player in range(0, 10):
            temp_feature = features[player].cpu().detach().numpy().tolist()
            temp_score = scores[player].cpu().detach().numpy().tolist()
            win = (player < 5 and winner=="blue") or (player >= 5 and winner=="red")
            for event_idx in range(0, len(temp_feature[0])):
                save_features = [match_name, player] + temp_feature[0][event_idx] + [temp_score[0][event_idx][0], win]
                writer.writerow(save_features)
    score_file.close()
    
def g3_save_features(match_name, features, scores, file_name, winner):
    labels = [
        "match_id", "player", "time", "mage", "fighter", "support", "tank", "assassin", "marksman", "TOP", "MIDDLE", "BOTTOM", "UTILITY", "JUNGLE", "x_position", "y_position",
        "deviation", "ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "SKILL_LEVEL_UP", "LEVEL_UP", "WARD_PLACED", "WARD_KILL", "CHAMPION_KILL", "CHAMPION_KILL_ASSIST",
        "CHAMPION_KILL_VICTIM", "BUILDING_KILL", "BUILDING_KILL_ASSIST", "ELITE_MONSTER_KILL", "ELITE_MONSTER_KILL_ASSIST", "event_weight", "score", "win"
        ]
    
    data = pd.read_feather('../../Dataset/League_of_Legends/features_ftr_test/%s.ftr'%match_name)
    data.sort_values(by='time', axis=0, ascending=False, inplace=True)
    team_seq = []
    team_seq.append(data[data['player']<=5]['player'].tolist())
    team_seq.append(data[data['player']>=6]['player'].tolist())
    
    score_file = csv_opener(file_name, mode="a")
    if score_file != None:
        writer = csv.writer(score_file)
        for team in range(2):
            temp_feature = features[team][0].cpu().detach().numpy().tolist()
            temp_score = scores[team][0].cpu().detach().numpy().tolist()
            win = (team < 1 and winner=="blue") or (team >= 1 and winner=="red")
            for player, event, score in zip(team_seq[team], temp_feature, temp_score):
                save_features = [match_name, player] + event + [score[0], win]
                writer.writerow(save_features)
    
    
def feature_names(train=True):
    list = []
    postfix = "train" if train else "test"
    for filename in listdir("./processed_csvs/"):
        if f"features_{postfix}" in filename:
            list.append(filename)
    return list

def open_feature(filename, train=True):
    route = f"f:/Dataset/League_of_Legends/features_ftr_train/{filename}" if train else f"f:/Dataset/League_of_Legends/features_ftr_test/{filename}"
    features = pd.read_feather(route)
    return features