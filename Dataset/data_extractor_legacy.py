import pandas as pd
import numpy as np
import csv, json
from os.path import exists
from os import listdir, makedirs
from tqdm import tqdm

def match_meta_json_loader(server):
    path = f'./matches_raw/legacy/{server}/'
    for file in listdir(path):
        if 'timeline' in file: continue
        with open(f'{path}{file}', encoding='utf-8') as json_file:
            yield json.load(json_file)

def match_timeline_json_loader(server):
    path = f'./matches_raw/legacy/{server}/'
    for file in listdir(path):
        if not 'timeline' in file: continue
        with open(f'{path}{file}', encoding='utf-8') as json_file:
            yield json.load(json_file)

def match_info_legacy(server):
    pathes = ('./matches_meta/', './matches_meta/legacy/')
    for path in pathes:
        if not exists(path): makedirs(path)

    keys = pd.read_csv('meta_info_columns.csv')['columns'].to_list()
    with open(f'./matches_meta/legacy/meta_{server}.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['version'] + keys)
        for idx, meta in enumerate(match_meta_json_loader(server)):
            print(f'File no.{idx:5d}')
            version = meta['info']['gameVersion'][:-9]
            participants = meta['info']['participants']
            for player in range(0, 10):
                player_info = []
                player_info.append(version)
                for key in keys: player_info.append(participants[player][key])
                writer.writerow(player_info)

def item_dictionary(game_version):
    version = f'{game_version}.1'
    with open(f'../DataDragon/{version}/data/ko_KR/item.json') as json_file:
        item_dict = json.load(json_file)
        return item_dict

def champ_vector_by_tag(game_version):
    version = f'{game_version}.1'
    champ_vector = pd.read_csv(f'../DataDragon/{version}/champ_vector_by_tag.csv', index_col=0)
    return champ_vector

def match_timeline_legacy(server):
    features = pd.read_csv('./timeline_features.csv', names=['features'], squeeze=True).to_list()

    path = 'matches_raw/legacy/'
    for region in ['kr', 'na1', 'eun1', 'euw1', 'jp1']:
        tline_list = [tline for tline in listdir(f'{path}{region}/') if 'timeline' in tline]

        for f_name in tqdm(tline_list, ncols=80, ascii=True, desc=f'[{region}] '):
            mf_name = f'{f_name[:-14]}.json' # Match file name

            mat_file = open(f'{path}{region}/{mf_name}')
            mat_file = json.load(mat_file)
            duration = mat_file['info']['gameDuration']
            version = mat_file['info']['gameVersion'][:-9]
            players = [player for player in mat_file['info']['participants']]

            champ_vec = pd.read_csv(f'../DataDragon/{version}.1/champ_vector_by_tag.csv', index_col=0)
            item_vec = pd.read_csv(f'../DataDragon/{version}.1/items_vector.csv', index_col=0)

            t_file = open(f'{path}{region}/{f_name}')
            frames = json.load(t_file)['info']['frames']

            for idx, frame in enumerate(frames):
                if idx == 0: continue
                for event in frame['events']:
                    process_event(event, duration, champ_vec, item_vec)
            
def process_event(event, players, duration, champ_vec, item_vec):
    time = event['timestamp']
    player = event['participantId']
    champ = champ_vec.loc[players[player]['championName']].to_list()
    position = position_to_vec(players[player]['individualPosition'])
    location = location_to_vec(event)

    if event['type'] == ['ITEM_PURCHASED', 'ITEM_DESTROYED']:
        location = [0.0, 0.0] if player <= 5 else [1, 1]
    if event['type'] == 'LEVEL_UP':
        location = [np.nan, np.nan]
    if event['type'] == 'SKILL_LEVEL_UP':
        location = [np.nan, np.nan]
    if event['type'] in ['WARD_PLACED', 'WARD_KILL']:
        location = [np.nan, np.nan]

    
    return [time, player] + champ + position

def position_to_vec(position):
    if position == 'TOP': return [1, 0, 0, 0, 0]
    if position == 'MIDDLE': return [0, 1, 0, 0, 0]
    if position == 'BOTTOM': return [0, 0, 1, 0, 0]
    if position == 'UTILITY': return [0, 0, 0, 1, 0]
    if position == 'JUNGLE': return [0, 0, 0, 0, 1]

def location_to_vec(event):
    if event['type'] in ['ITEM_PURCHASED', 'ITEM_DESTROYED']:
        location = [0.0, 0.0] if event['participantId'] <= 5 else [1, 1]
    elif 'position' in event.keys():
        location = [event['position']['x']/15000, event['position']['y']/15000]
    else:
        location = [np.nan, np.nan]
    

if __name__ == '__main__':
    match_info_legacy('kr')