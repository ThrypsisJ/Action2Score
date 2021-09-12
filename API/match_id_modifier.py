#%%
import json
import os
from tqdm import tqdm

path = "F:/Dataset/League_of_Legends/matchlists/"

for file in tqdm(os.listdir(path), desc="progress", ncols=80):
    region = file[10:-5].upper()
    with open(path+file, "r") as json_file:
        match_list = json.load(json_file)
        temp_match_list = []
        for match_id in tqdm(match_list, desc=region, ncols=80):
            temp_match_list.append(f"{region}_{match_id}")
        json_file.close()

    with open(path+file, "w") as json_file:
        json.dump(temp_match_list, json_file)
        json_file.close()
# %%
