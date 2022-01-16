import pandas as pd
import sys, json

def champion_structure():
    c_data_structure = {
        "name": [],
        "key": [],
        "tags": [],
    }
    return c_data_structure

def file_opener(data_type, version):
    path = f"./{version}/data/ko_KR"

    with open(f"{path}/{data_type}.json", encoding="utf-8") as loaded_file:
        json_file = json.load(loaded_file)
    return json_file["data"]

def item_to_csv(items, version):
    item_datas = { 'id': [], 'gold_purchase': [], 'gold_sell': [] }

    for item_id in items.keys():
        item = items[item_id]
        item_datas['id'].append(item_id)
        item_datas['gold_purchase'].append(item['gold']['base'])
        item_datas['gold_sell'].append(item['gold']['sell'])
    
    pd_data = pd.DataFrame(item_datas)
    pd_data = pd_data.sort_values(by=["id"])
    pd_data.to_csv(f'./{version}/items.csv', index=False) #, encoding="cp949")

def champ_to_csv(champs, version):
    champ_datas = champion_structure()

    for champ_name in champs.keys():
        champ = champs[champ_name]

        champ_datas['name'].append(champ_name)
        champ_datas['key'].append(champ['key'])
        champ_datas['tags'].append(champ['tags'])
    
    pd_data = pd.DataFrame(champ_datas)
    pd_data = pd_data.sort_values(by=["key"])
    pd_data.to_csv(f'./{version}/champs.csv', index=False) #, encoding="utf-8")

def champ_to_dummy(version):
    dummies = ['Mage', 'Fighter', 'Support', 'Tank', 'Assassin', 'Marksman']
    champs = pd.read_csv(f'./{version}/champs.csv')
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

    champs.to_csv(f'./{version}/champs_dummy_variable.csv', index=False)

if __name__ == '__main__':
    version = sys.argv[1]

    items = file_opener('item', version)
    champions = file_opener('champion', version)

    item_to_csv(items, version)
    champ_to_csv(champions, version)
    champ_to_dummy(version)