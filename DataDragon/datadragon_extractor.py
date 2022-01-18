import pandas as pd
import sys, json
from os.path import exists

def file_opener(data_type, version):
    path = f"./{version}/data/ko_KR"

    with open(f"{path}/{data_type}.json", encoding="utf-8") as loaded_file:
        json_file = json.load(loaded_file)
    return json_file["data"]

def item_to_csv(items, version):
    if exists(f'./{version}/items.csv'): return

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
    if exists('./{version}/champs.csv'): return

    champ_datas = { 'name': [], 'key': [], 'tags': [] }
    for champ_name in champs.keys():
        champ = champs[champ_name]
        champ_datas['name'].append(champ_name)
        champ_datas['key'].append(champ['key'])
        champ_datas['tags'].append(champ['tags'])
    
    pd_data = pd.DataFrame(champ_datas)
    pd_data = pd_data.sort_values(by=["key"])
    pd_data.to_csv(f'./{version}/champs.csv', index=False) #, encoding="utf-8")

def champ_to_vector_by_tag(version):
    if exists(f'./{version}/champ_vector_by_tag.csv'): return

    roles = ['Mage', 'Fighter', 'Support', 'Tank', 'Assassin', 'Marksman']
    champs = pd.read_csv(f'./{version}/champs.csv')
    champ_names = champs['name'].to_list()
    tags = champs['tags'].to_list()

    champ_vectors = []
    for tag in tags:
        tag = tag.replace('\'', '').strip('][').split(', ')
        champ_vectors.append([1 if (role in tag) else 0 for role in roles])

    champ_vectors = pd.DataFrame(champ_vectors, columns=roles, index=champ_names)
    champ_vectors.to_csv(f'./{version}/champ_vector_by_tag.csv')

def champ_to_vector_by_name(version):
    if exists(f'./{version}/champ_vector_by_name.csv'): return

    champs = pd.read_csv(f'./{version}/champs.csv')
    champ_names = champs['name'].to_list()
    champs.drop(['key', 'tags'], axis='columns', inplace=True)

    champs = pd.get_dummies(champs, prefix='', prefix_sep='')
    champs.index = champ_names
    champs.to_csv(f'./{version}/champ_vector_by_name.csv')

if __name__ == '__main__':
    version = sys.argv[1]

    items = file_opener('item', version)
    champions = file_opener('champion', version)

    item_to_csv(items, version)
    champ_to_csv(champions, version)
    champ_to_vector_by_tag(version)
    champ_to_vector_by_name(version)