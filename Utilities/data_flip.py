import pandas as pd
import os
from tqdm import tqdm
from time import sleep

train_route = "f:/dataset/League_of_Legends/features_ftr_train/"
test_route = "f:/dataset/League_of_Legends/features_ftr_test/"

train_route_flip = "f:/dataset/League_of_Legends/features_ftr_train_flipped/"
test_route_flip = "f:/dataset/League_of_Legends/features_ftr_test_flipped/"

trains = pd.read_csv("e:/documents/projects/pasta/utilities/train_ftr_list.csv")
tests = pd.read_csv("e:/documents/projects/pasta/utilities/test_ftr_list.csv")

print("Processing Normal Train Data")


print("\nProcessing Train Data")
sleep(0.5)
for idx, file_name in tqdm(trains.iterrows(), ncols=80, total=200000):
    file_name = file_name["match_id"][0:-4]
    if os.path.exists(f"{train_route_flip}{file_name}.ftr"): pass
    else:
        train_feat = pd.read_feather(f"{train_route}{file_name}.ftr")
        train_feat.sort_values(by=["player", "time"], ascending=False, inplace=True)
        train_feat.reset_index(inplace=True)
        train_feat.to_feather(f"{train_route_flip}{file_name}.ftr")
        del train_feat
    del [file_name, idx]

print("\nProcessing Test Data")
sleep(0.5)
for idx, file_name in tqdm(tests.iterrows(), ncols=80, total=45575):
    file_name = file_name["match_id"][0:-4]
    if os.path.exists(f"{test_route_flip}{file_name}.ftr"): pass
    else:
        test_feat = pd.read_feather(f"{test_route}{file_name}.ftr")
        test_feat.sort_values(by=["player", "time"], ascending=False, inplace=True)
        test_feat.reset_index(inplace=True)
        test_feat.to_feather(f"{test_route_flip}{file_name}.ftr")
        del test_feat
    del [file_name, idx]