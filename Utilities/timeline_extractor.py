import timeline_feature_extractor as tfe
import file_io
import pandas as pd
import os

def feature_reset():
    new_feature = {
        "time": [],
        "player": [],
        # champion
        "mage": [],
        "fighter": [],
        "support": [],
        "tank": [],
        "assassin": [],
        "marksman": [],
        # player role
        "TOP": [],
        "MIDDLE": [],
        "BOTTOM": [],
        "UTILITY": [],
        "JUNGLE": [],
        # position
        "x_position": [],
        "y_position": [],
        "deviation": [],
        # event
        "ITEM_PURCHASED": [],
        "ITEM_SOLD": [],
        "ITEM_DESTROYED": [],
        "SKILL_LEVEL_UP": [],
        "LEVEL_UP": [],
        "WARD_PLACED": [],
        "WARD_KILL": [],
        "CHAMPION_KILL": [],
        "CHAMPION_KILL_ASSIST": [],
        "CHAMPION_KILL_VICTIM": [],
        "BUILDING_KILL": [],
        "BUILDING_KILL_ASSIST": [],
        "ELITE_MONSTER_KILL": [],
        "ELITE_MONSTER_KILL_ASSIST": [],
        "event_weight": [],
        # other info
        "player_level": [],
        "skill_level": [],
        "tower_diff": [],
        "is_valid": []
    }
    return new_feature

feature = {
    "time": [],
    "player": [],
    # champion
    "mage": [],
    "fighter": [],
    "support": [],
    "tank": [],
    "assassin": [],
    "marksman": [],
    # player role
    "TOP": [],
    "MIDDLE": [],
    "BOTTOM": [],
    "UTILITY": [],
    "JUNGLE": [],
    # position
    "x_position": [],
    "y_position": [],
    "deviation": [],
    # event
    "ITEM_PURCHASED": [],
    "ITEM_SOLD": [],
    "ITEM_DESTROYED": [],
    "SKILL_LEVEL_UP": [],
    "LEVEL_UP": [],
    "WARD_PLACED": [],
    "WARD_KILL": [],
    "CHAMPION_KILL": [],
    "CHAMPION_KILL_ASSIST": [],
    "CHAMPION_KILL_VICTIM": [],
    "BUILDING_KILL": [],
    "BUILDING_KILL_ASSIST": [],
    "ELITE_MONSTER_KILL": [],
    "ELITE_MONSTER_KILL_ASSIST": [],
    "event_weight": [],
    # other info
    "player_level": [],
    "skill_level": [],
    "tower_diff": [],
    "is_valid": []
}

tfe.data_parser.item_dict = file_io.item_dict_opener()
tfe.data_parser.champion_dict = file_io.champion_dict_opener()

dataroot = '../../Dataset/League_of_Legends'
raw = dataroot + '/challenger_raw/'
csv = dataroot + '/challenger_csv/'

total_count = 0
# for region in ["kr", "jp1", "na1", "euw1", "eun1"]:
#     total_count += len(os.listdir(f"./mat_datas/{region}"))
total_count = len(os.listdir(raw))

count = 0
# for region in ["kr", "jp1", "na1", "euw1", "eun1"]:
    # for files in os.listdir(f"./mat_datas/{region}"):
for files in os.listdir(raw):
    count += 1
    print(f"Parsing file no. {count}/{total_count}...")

    # save_name = "./features/" + files[:-5] + ".csv"
    save_name = csv + files[:-5] + ".csv"
    region = None
    if os.path.exists(save_name):
        continue

    if "_timeline" not in files:
        match_file = file_io.timeline_file_opener(files, region)
        timeline_file_name = files[:-5] + "_timeline.json"
        timeline_file = file_io.timeline_file_opener(timeline_file_name, region)
        parsed_data = tfe.data_parser(timeline_file, match_file)

        for key, timeline in parsed_data.player_timelines.items():
            for timestamp in timeline.values():
                feature["time"].append(timestamp["time"])
                feature["player"].append(key)
                # champion
                feature["mage"].append(timestamp["champion"]["mage"])
                feature["fighter"].append(timestamp["champion"]["fighter"])
                feature["support"].append(timestamp["champion"]["support"])
                feature["tank"].append(timestamp["champion"]["tank"])
                feature["assassin"].append(timestamp["champion"]["assassin"])
                feature["marksman"].append(timestamp["champion"]["marksman"])
                # player role
                feature["TOP"].append(timestamp["role"]["TOP"])
                feature["MIDDLE"].append(timestamp["role"]["MIDDLE"])
                feature["BOTTOM"].append(timestamp["role"]["BOTTOM"])
                feature["UTILITY"].append(timestamp["role"]["UTILITY"])
                feature["JUNGLE"].append(timestamp["role"]["JUNGLE"])
                # position
                feature["x_position"].append(timestamp["x_position"])
                feature["y_position"].append(timestamp["y_position"])
                feature["deviation"].append(timestamp["deviation"])
                # event
                feature["ITEM_PURCHASED"].append(timestamp["event"]["ITEM_PURCHASED"])
                feature["ITEM_SOLD"].append(timestamp["event"]["ITEM_SOLD"])
                feature["ITEM_DESTROYED"].append(timestamp["event"]["ITEM_DESTROYED"])
                feature["SKILL_LEVEL_UP"].append(timestamp["event"]["SKILL_LEVEL_UP"])
                feature["LEVEL_UP"].append(timestamp["event"]["LEVEL_UP"])
                feature["WARD_PLACED"].append(timestamp["event"]["WARD_PLACED"])
                feature["WARD_KILL"].append(timestamp["event"]["WARD_KILL"])
                feature["CHAMPION_KILL"].append(timestamp["event"]["CHAMPION_KILL"])
                feature["CHAMPION_KILL_ASSIST"].append(timestamp["event"]["CHAMPION_KILL_ASSIST"])
                feature["CHAMPION_KILL_VICTIM"].append(timestamp["event"]["CHAMPION_KILL_VICTIM"])
                feature["BUILDING_KILL"].append(timestamp["event"]["BUILDING_KILL"])
                feature["BUILDING_KILL_ASSIST"].append(timestamp["event"]["BUILDING_KILL_ASSIST"])
                feature["ELITE_MONSTER_KILL"].append(timestamp["event"]["ELITE_MONSTER_KILL"])
                feature["ELITE_MONSTER_KILL_ASSIST"].append(timestamp["event"]["ELITE_MONSTER_KILL_ASSIST"])
                feature["event_weight"].append(timestamp["event_weight"])
                # misc
                feature["player_level"].append(timestamp["player_level"])
                feature["skill_level"].append(timestamp["skill_level"])
                feature["tower_diff"].append(timestamp["tower_diff"])
                feature["is_valid"].append(timestamp["is_valid"])
        pdData = pd.DataFrame(feature)
        
        file_io.save_to_csv(pdData, file_name=save_name, sort_by="time")
        feature = feature_reset()

    else: continue # skip _timeline file (already processed)