import json
from pandas import DataFrame

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

def file_opener(data_type):
    route = "./DataDragon/dragontail-11.10.1/11.10.1/data/ko_KR"

    with open(f"{route}/{data_type}.json", encoding="utf-8") as loaded_file:
        json_file = json.load(loaded_file)
    return json_file["data"]

def item_to_csv(items):
    item_datas = item_structure()
    keys = list(item_datas.keys())
    keys.remove("id")

    for item_id in items:
        item = items[item_id]

        item_datas["id"].append(item_id)
        for key in keys:
            if key == "gold_purchase":
                item_datas[key].append(item["gold"]["base"])
            elif key == "gold_sell":
                item_datas[key].append(item["gold"]["sell"])
            else:
                item_datas[key].append(item[key])
    
    pd_data = DataFrame(item_datas)
    pd_data = pd_data.sort_values(by=["id"])
    pd_data.to_csv("items.csv", index=False, encoding="cp949")

def champ_to_csv(champs):
    champ_datas = champion_structure()
    keys = list(champ_datas.keys())
    keys.remove("name")

    for champ_name in champs:
        champ = champs[champ_name]

        champ_datas["name"].append(champ_name)
        for key in keys:
            champ_datas[key].append(champ[key])
    
    pd_data = DataFrame(champ_datas)
    pd_data = pd_data.sort_values(by=["key"])
    pd_data.to_csv("champs.csv", index=False, encoding="utf-8")