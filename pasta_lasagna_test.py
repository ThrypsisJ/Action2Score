import pasta_lasagna as dGRU
import pandas as pd
from os.path import exists
from time import sleep

def test(predict_time=1200000, limited_duration=False, proportional_time=False, no_time_feature=False):
    hp = {"epochs": 1, "learning_rate": 0.01, "num_layers": 1, "hidden_size": 8, "input_size": 29 if no_time_feature else 30}
    PASTA = dGRU.PASTA(hp["input_size"], hp["hidden_size"], hp["num_layers"], hp["learning_rate"])

    route = "../../Dataset/League_of_Legends/features_tensor/"

    test_match_result_ftr = pd.read_feather("./processed_ftr/match_result_test.ftr")
    total_rows = test_match_result_ftr.shape[0]

    if (not limited_duration) & (not no_time_feature) & (not proportional_time):
        PASTA.load_parameter(postfix="_lasagna")
        confusion_matrix, c_label = [0, 0, 0, 0], ["TN", "FP", "FN", "TP"]

        print("--test start--")
        for idx, row in test_match_result_ftr.iterrows():
            print("[%6d/%6d] "%(idx+1, total_rows), end="")
            match_id, win = row["match_no"], row["win"]
            features = pd.read_pickle(f"{route}Normal_Test/{match_id}.pkl")
            winner, predict = PASTA.test(features, win, match_id, postfix="lasagna")
            correct = 2*winner + predict
            confusion_matrix[correct] += 1

            tn, fp, fn, tp = confusion_matrix[0], confusion_matrix[1], confusion_matrix[2], confusion_matrix[3]
            acc = (tp+tn)/(tn+fp+fn+tp) * 100
            pre = tp/(tp+fp) * 100 if (tp+fp) != 0 else 0
            rec = tp/(tp+fn) * 100 if (tp+fn) != 0 else 0
            print(" / Acc: %5.2f / Pre: %5.2f / Rec: %5.2f"%(acc, pre, rec))

        for idx in range(0, 4):
            print(f"{c_label[idx]}: {confusion_matrix[idx]} / ", end="")

    elif limited_duration & (not no_time_feature) & (not proportional_time):
        PASTA.load_parameter(postfix="_lasagna")
        confusion_matrix, c_label = [0, 0, 0, 0], ["TN", "FP", "FN", "TP"]

        print("--test start--")
        for idx, row in test_match_result_ftr.iterrows():
            print("[%6d/%6d] "%(idx+1, total_rows), end="")
            match_id, win = row["match_no"], row["win"]
            if not exists(f"{route}Normal_Under_{predict_time}/{match_id}.pkl"):
                print(f"Match skipped because the duration is shorter than {predict_time}")
                continue
            features = pd.read_pickle(f"{route}Normal_Under_{predict_time}/{match_id}.pkl")
            winner, predict = PASTA.test(features, win, match_id, postfix=f"lasagna_{predict_time}")
            correct = 2*winner + predict
            confusion_matrix[correct] += 1

            tn, fp, fn, tp = confusion_matrix[0], confusion_matrix[1], confusion_matrix[2], confusion_matrix[3]
            acc = (tp+tn)/(tn+fp+fn+tp) * 100
            pre = tp/(tp+fp) * 100 if (tp+fp) != 0 else 0
            rec = tp/(tp+fn) * 100 if (tp+fn) != 0 else 0
            print(" / Acc: %5.2f / Pre: %5.2f / Rec: %5.2f"%(acc, pre, rec))

        for idx in range(0, 4):
            print(f"{c_label[idx]}: {confusion_matrix[idx]} / ", end="")
        sleep(0.5)

    elif proportional_time & (not no_time_feature) & (not limited_duration):
        PASTA.load_parameter(postfix="_lasagna")
        confusion_matrix, c_label = [0, 0, 0, 0], ["TN", "FP", "FN", "TP"]

        print("--test start--")
        for idx, row in test_match_result_ftr.iterrows():
            print("[%6d/%6d] "%(idx+1, total_rows), end="")
            match_id, win = row["match_no"], row["win"]
            features = pd.read_pickle(f"{route}Under{predict_time}_Flipped_Test/{match_id}.pkl")
            winner, predict = PASTA.test(features, win, match_id, postfix=f"lasagna_{predict_time}")
            correct = 2*winner + predict
            confusion_matrix[correct] += 1

            tn, fp, fn, tp = confusion_matrix[0], confusion_matrix[1], confusion_matrix[2], confusion_matrix[3]
            acc = (tp+tn)/(tn+fp+fn+tp) * 100
            pre = tp/(tp+fp) * 100 if (tp+fp) != 0 else 0
            rec = tp/(tp+fn) * 100 if (tp+fn) != 0 else 0
            print(" / Acc: %5.2f / Pre: %5.2f / Rec: %5.2f"%(acc, pre, rec))

        for idx in range(0, 4):
            print(f"{c_label[idx]}: {confusion_matrix[idx]} / ", end="")
        sleep(0.5)

    else: pass
