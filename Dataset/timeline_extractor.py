import timeline_feature_extractor as tfe
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def feature_reset():
    new_feature = {
        "time": [], "player": [],
        # champion
        "Mage": [], "Fighter": [], "Support": [], "Tank": [], "Assassin": [], "Marksman": [],
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

def extract():
    tline_path = Path('./matches_raw/timeline')
    meta_path = Path('./matches_raw/meta')
    save_path = Path('./matches_csv/')
    if not save_path.exists(): save_path.mkdir(parents=True)

    tline_json_list = sorted(list(tline_path.glob('*')))
    meta_json_list = sorted(list(meta_path.glob('*')))
    total = len(tline_json_list)
    for idx, (tline, meta) in enumerate(zip(tline_json_list, meta_json_list)):
        progress = (idx+1)/(total) * 100
        print(f'[{idx+1:6d}/{total:6d}, {progress:4.2f}%] extracting {tline}')

        feature = feature_reset()
        save_name = save_path / f'{tline.name[:-14]}.csv'
        if save_name.exists() : continue

        meta_json = json.load(meta.open('r'))
        tline_json = json.load(tline.open('r'))
        parser = tfe.data_parser(tline_json, meta_json)

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
    extract()