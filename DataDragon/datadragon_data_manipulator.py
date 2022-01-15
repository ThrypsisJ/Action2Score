import json
import pandas as pd

def item_structure():
    i_data_structure = {
        "id": [],
        "gold_purchase": [],
        "gold_sell": [],
        }
    return i_data_structure

def champion_structure():
    c_data_structure = {
        "name": [],
        "key": [],
        "tags": [],
    }
    return c_data_structure

def file_opener(data_type, version):
    path = f"./{version}/{version}/data/ko_KR"

    with open(f"{path}/{data_type}.json", encoding="utf-8") as loaded_file:
        json_file = json.load(loaded_file)
    return json_file["data"]

def item_to_csv(items, version):
    item_datas = item_structure()

    for item_id in items.keys():
        item = items[item_id]
        item_datas['id'].append(item_id)
        item_datas['gold_purchase'].append(item['gold']['base'])
        item_datas['gold_sell'].append(item['gold']['sell'])
    
    pd_data = pd.DataFrame(item_datas)
    pd_data = pd_data.sort_values(by=["id"])
    pd_data.to_csv("../processed_csvs/items.csv", index=False) #, encoding="cp949")

def champ_to_csv(champs):
    champ_datas = champion_structure()

    for champ_name in champs.keys():
        champ = champs[champ_name]

        champ_datas["name"].append(champ_name)
        for key in keys:
            champ_datas[key].append(champ[key])
    
    pd_data = pd.DataFrame(champ_datas)
    pd_data = pd_data.sort_values(by=["key"])
    pd_data.to_csv("../processed_csvs/champs.csv", index=False, encoding="utf-8")

def champ_to_dummy():
    dummies = ['Mage', 'Fighter', 'Support', 'Tank', 'Assassin', 'Marksman']
    champs = pd.read_csv('../processed_csvs/champs.csv')
    tags = champs['tags'].tolist()

    cham_dummies = {}
    for dummy in dummies:
        cham_dummies[dummy] = []

    for tag in tags:
        for dummy in dummies:
            if dummy in tag : cham_dummies[dummy].append(1)
            else            : cham_dummies[dummy].append(0)

    cham_dummies = pd.DataFrame(cham_dummies)
    cham_dummies

    champs.drop(columns=['tags'], inplace=True)
    champs = pd.concat([champs, cham_dummies], axis=1)

    champs.to_csv('../processed_csvs/champs_dummy_variable.csv', index=False)