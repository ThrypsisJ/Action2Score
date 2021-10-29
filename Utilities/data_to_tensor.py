#%%
import pandas as pd
import os
from tqdm import tqdm
from time import sleep
from torch import tensor, zeros

pickle_route = "../../Dataset/League_of_Legends/features_tensor"

#%%
print("Processing normal train features...")
sleep(0.5)
ftr_route = "f:/Dataset/League_of_Legends/features_ftr_train/"
feat_list = os.listdir(ftr_route)

for file_name in tqdm(feat_list):
    file = pd.read_feather(f"{ftr_route}{file_name}")
    file = file.loc[:,"time":"event_weight"]
    players = []
    for idx in range(0, 10):
        player = file[file["player"] == idx+1].drop(columns=["player"])
        temp_feature = tensor([player.values]) if len(player) != 0 else zeros([1, 1, 30])
        players.append(temp_feature)
    pd.to_pickle(players, f"{pickle_route}Normal_Train/{file_name[0:-4]}.pkl")
print("")

#%%
print("Processing normal test features...")
sleep(0.5)
ftr_route = "f:/Dataset/League_of_Legends/features_ftr_test/"
feat_list = os.listdir(ftr_route)

for file_name in tqdm(feat_list):
    file = pd.read_feather(f"{ftr_route}{file_name}")
    file = file.loc[:,"time":"event_weight"]
    players = []
    for idx in range(0, 10):
        player = file[file["player"] == idx+1].drop(columns=["player"])
        temp_feature = tensor([player.values]) if len(player) != 0 else zeros([1, 1, 30])
        players.append(temp_feature)
    pd.to_pickle(players, f"{pickle_route}Normal_Test/{file_name[0:-4]}.pkl")
print("")

# %%
print("Processing flipped train features...")
sleep(0.5)
ftr_route = "f:/Dataset/League_of_Legends/features_ftr_train/"
feat_list = os.listdir(ftr_route)

for file_name in tqdm(feat_list):
    file = pd.read_feather(f"{ftr_route}{file_name}")
    file = file.loc[:,"time":"event_weight"]
    players = []
    for idx in range(0, 10):
        player = file[file["player"] == idx+1].drop(columns=["player"])
        temp_feature = tensor([player.values]) if len(player) != 0 else zeros([1, 1, 30])
        if temp_feature.size()[1] > 1: temp_feature = temp_feature.flip(1)
        players.append(temp_feature)
    pd.to_pickle(players, f"{pickle_route}Flipped_Train/{file_name[0:-4]}.pkl")
print("")

# %%
print("Processing flipped test features...")
sleep(0.5)
ftr_route = "f:/Dataset/League_of_Legends/features_ftr_test/"
feat_list = os.listdir(ftr_route)

for file_name in tqdm(feat_list):
    file = pd.read_feather(f"{ftr_route}{file_name}")
    file = file.loc[:,"time":"event_weight"]
    players = []
    for idx in range(0, 10):
        player = file[file["player"] == idx+1].drop(columns=["player"])
        temp_feature = tensor([player.values]) if len(player) != 0 else zeros([1, 1, 30])
        if temp_feature.size()[1] > 1: temp_feature = temp_feature.flip(1)
        players.append(temp_feature)
    pd.to_pickle(players, f"{pickle_route}Flipped_Test/{file_name[0:-4]}.pkl")
print("")

# %%
import pandas as pd
import os
from tqdm import tqdm
from time import sleep
from torch import tensor, zeros

limit_time = 600000
print(f"Processing train features under {limit_time}...")
sleep(0.5)

ftr_route = "f:/Dataset/League_of_Legends/features_ftr_train/"
test_match_result_ftr = pd.read_feather("e:/Documents/projects/PASTA/processed_ftr/match_result_train.ftr")
pickle_route = f"f:/Dataset/League_of_Legends/features_tensor/Under{limit_time}_Flipped_Train/"

if not os.path.exists(pickle_route): os.mkdir(pickle_route)

for idx, row in tqdm(test_match_result_ftr.iterrows(), total=test_match_result_ftr.shape[0]):
    match_id = row["match_no"]
    duration = row["duration"]
    file = pd.read_feather(f"{ftr_route}{match_id}.ftr")
    border = limit_time / duration
    file = file[file["time"] <= border]
    file = file.loc[:,"time":"event_weight"]
    players = []
    for idx in range(0, 10):
        player = file[file["player"] == idx+1].drop(columns=["player"])
        temp_feature = tensor([player.values]) if len(player) != 0 else zeros([1, 1, 30])
        if temp_feature.size()[1] > 1: temp_feature = temp_feature.flip(1)
        players.append(temp_feature)
    pd.to_pickle(players, f"{pickle_route}{match_id}.pkl")
print("")
# %%
import pandas as pd
import os
from tqdm import tqdm
from time import sleep
from torch import tensor, zeros

limit_time = 900000
print(f"Processing test features under {limit_time}...")
sleep(0.5)

ftr_route = "f:/Dataset/League_of_Legends/features_ftr_test/"
test_match_result_ftr = pd.read_feather("e:/Documents/projects/PASTA/processed_ftr/match_result_test.ftr")
pickle_route = f"f:/Dataset/League_of_Legends/features_tensor/Under{limit_time}_Flipped_Test/"

if not os.path.exists(pickle_route): os.mkdir(pickle_route)

for idx, row in tqdm(test_match_result_ftr.iterrows(), total=test_match_result_ftr.shape[0]):
    match_id = row["match_no"]
    duration = row["duration"]
    file = pd.read_feather(f"{ftr_route}{match_id}.ftr")
    if duration < limit_time: continue
    border = limit_time / duration
    file = file[file["time"] <= border]
    file = file.loc[:,"time":"event_weight"]
    players = []
    for idx in range(0, 10):
        player = file[file["player"] == idx+1].drop(columns=["player"])
        temp_feature = tensor([player.values]) if len(player) != 0 else zeros([1, 1, 30])
        if temp_feature.size()[1] > 1: temp_feature = temp_feature.flip(1)
        players.append(temp_feature)
    pd.to_pickle(players, f"{pickle_route}{match_id}.pkl")
print("")
# %%
