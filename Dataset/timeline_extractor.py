import timeline_feature_extractor as tfe
import pandas as pd
import json
from os import listdir, makedirs
from os.path import exists
import sys

def feature_reset():
    new_feature = {
        "time": [], "player": [],
        # champion
        "mage": [], "fighter": [], "support": [], "tank": [], "assassin": [], "marksman": [],
        # player position
        "TOP": [], "MIDDLE": [], "BOTTOM": [], "UTILITY": [], "JUNGLE": [],
        # location
        "x_location": [], "y_location": [], "distance": [],
        # event
        "ITEM_PURCHASED": [], "ITEM_SOLD": [], "ITEM_DESTROYED": [],
        "SKILL_LEVEL_UP": [], "LEVEL_UP": [],
        "WARD_PLACED": [], "WARD_KILL": [],
        "CHAMPION_KILL": [], "CHAMPION_KILL_ASSIST": [], "CHAMPION_KILL_VICTIM": [],
        "BUILDING_KILL": [], "BUILDING_KILL_ASSIST": [],
        "ELITE_MONSTER_KILL": [], "ELITE_MONSTER_KILL_ASSIST": [],
        "event_weight": [],
        # other info (it is not features)
        "player_level": [], "skill_level": [], "tower_diff": [], "is_valid": []
    }
    return new_feature

def extract(server):
    raw_path = f'./matches_raw/legacy/{server}/'

    save_path = ''
    for path in ['./matches_csv/', 'legacy/', f'{server}/']:
        save_path = save_path + path
        if not exists(save_path): makedirs(save_path)

    tline_json_list = [file for file in listdir(raw_path) if '_timeline' in file]

    total = len(tline_json_list)
    for idx, tline_json in enumerate(tline_json_list):
        print(f'[{idx+1:5d}/{total:5d}] extracting {tline_json}')

        feature = feature_reset()
        save_name = f'{save_path}{tline_json[:-14]}.csv'
        if exists(save_name): continue

        mat_json = open(f'{raw_path}{tline_json[:-14]}.json')
        mat_json = json.load(mat_json)
        
        tline_json = open(f'{raw_path}{tline_json}')
        tline_json = json.load(tline_json)
        parser = tfe.data_parser(tline_json, mat_json)

        for idx in range(10):
            for timestamp in parser.player_timelines[idx].values():
                for key in feature.keys():
                    if key == 'player':
                        feature[key].append(idx+1)
                    else:
                        feature[key].append(timestamp[key])
        
        feature = pd.DataFrame(feature)
        feature = feature[feature['is_valid']==True]
        feature.sort_values(by='time', inplace=True)
        feature.to_csv(save_name, index=False, encoding='utf-8')

if __name__ == '__main__':
    server = sys.argv[1]
    extract(server)