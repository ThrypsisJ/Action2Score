from Utilities import file_io
import os
import pandas as pd
from tqdm import tqdm

def normal():
    train_match_results = {
        "match_no": [],
        "win": [],
        "duration": [],
    }

    test_match_results = {
        "match_no": [],
        "win": [],
        "duration": [],
    }

    train_total_count = 0
    test_total_count = 0
    for region in ["kr", "jp1", "na1", "euw1", "eun1"]:
        train_total_count += len(os.listdir(f"./mat_datas_train/{region}")) / 2
        test_total_count += len(os.listdir(f"./mat_datas_test/{region}")) / 2

    count_train = 0
    count_test = 0

    for region in ["kr", "jp1", "na1", "euw1", "eun1"]:
        for file in os.listdir(f"./mat_datas_train/{region}"):
            if "_timeline" in file:
                continue
            else:
                count_train += 1
                print(f"Train Data: Checking Result of {file[:-5]}. {count_train}/{train_total_count}...")
                match_file = file_io.timeline_file_opener(file, region, train=True)
                train_match_results["match_no"].append(file[:-5])

                win_team = match_file["info"]["participants"][0]["win"]
                result = "blue" if win_team else "red"
                train_match_results["win"].append(result)

                duration = match_file["info"]["gameDuration"]
                train_match_results["duration"].append(duration)

        for file in os.listdir(f"./mat_datas_test/{region}"):
            if "_timeline" in file:
                continue
            else:
                count_test += 1
                print(f"Test Data: Checking Result of {file[:-5]}. {count_test}/{test_total_count}...")
                match_file = file_io.timeline_file_opener(file, region, train=False)
                test_match_results["match_no"].append(file[:-5])

                win_team = match_file["info"]["participants"][0]["win"]
                result = "blue" if win_team else "red"
                test_match_results["win"].append(result)

                duration = match_file["info"]["gameDuration"]
                test_match_results["duration"].append(duration)

    train_pdData = pd.DataFrame(train_match_results)
    test_pdData = pd.DataFrame(test_match_results)

    train_save_name = "./match_result_train.csv"
    test_save_name = "./match_result_test.csv"

    file_io.save_to_csv(train_pdData, train_save_name, sort_by="match_no")
    file_io.save_to_csv(test_pdData, test_save_name, sort_by="match_no")
    test_match_results["duration"].append(duration)
    train_pdData = pd.DataFrame(train_match_results)
    test_pdData = pd.DataFrame(test_match_results)
    train_save_name = "./match_result_train.csv"
    test_save_name = "./match_result_test.csv"
    file_io.save_to_csv(train_pdData, train_save_name, sort_by="match_no")
    file_io.save_to_csv(test_pdData, test_save_name, sort_by="match_no")

def challengers():
    challenger_match_result = {
        "match_no": [],
        "win": [],
        "duration": [],
    }

    challenger_route = '../../Dataset/League_of_Legends/challenger_raw/'
    challenger_total_count = len(os.listdir(challenger_route)) / 2

    count = 0

    region = None
    for file in tqdm(os.listdir(challenger_route)):
        if '_timeline' in file: continue
        else:
            count += 1
            match_file = file_io.timeline_file_opener(file, region, train=False)
            challenger_match_result['match_no'].append(file[:-5])

            win_team = match_file['info']['participants'][0]['win']
            result = 'blue' if win_team else 'red'
            challenger_match_result['win'].append(result)

            duration = match_file['info']['gameDuration']
            challenger_match_result['duration'].append(duration)

    challenger_data = pd.DataFrame(challenger_match_result)
    challenger_save_name = './processed_csvs/challenger_result.csv'
    file_io.save_to_csv(challenger_data, challenger_save_name, sort_by='match_no')