import pasta_lasagna as dGRU
import pandas as pd
from time import sleep

limited_duration = False
no_time_feature = False
predict_time = 1200000

hp = {"epochs": 5, "learning_rate": 0.001, "input_size": 29 if no_time_feature else 30}
PASTA = dGRU.PASTA(hp["input_size"], hp["learning_rate"])

route = "../../Dataset/League_of_Legends/features_tensor/"

train_match_result_ftr = pd.read_feather("./processed_ftr/match_result_train.ftr")
total_rows = train_match_result_ftr.shape[0]

if (not limited_duration) & (not no_time_feature):
    for epoch in range(0, hp["epochs"]):
        current_iter = 0
        print(f"[Epoch {epoch}]: Loading parameters...")
        PASTA.load_parameter(postfix="_lasagna_g2")
        confusion_matrix, c_label = [0, 0, 0, 0], ["TN", "FP", "FN", "TP"]

        print("--Train start--")
        for idx, row in train_match_result_ftr.iterrows():
            print("[Epoch %1d][%6d/%6d] "%(epoch+1, idx+1, total_rows), end="")
            match_id, win = row["match_no"], row["win"]
            features = pd.read_pickle(f"{route}Normal_Train/{match_id}.pkl")
            
            winner, predict = PASTA.train(features, win)
            correct = 2*winner + predict
            confusion_matrix[correct] += 1

            tn, fp, fn, tp = confusion_matrix[0], confusion_matrix[1], confusion_matrix[2], confusion_matrix[3]
            acc = (tp+tn)/(tn+fp+fn+tp) * 100
            pre = tp/(tp+fp) * 100 if (tp+fp) != 0 else 0
            rec = tp/(tp+fn) * 100 if (tp+fn) != 0 else 0
            print(" / Acc: %5.2f / Pre: %5.2f / Rec: %5.2f"%(acc, pre, rec))

        for idx in range(0, 4):
            print(f"{c_label[idx]}: {confusion_matrix[idx]} / ", end="")
        print(f"Saving parameters...\n")
        PASTA.save_parameter(postfix="_lasagna_g2")

elif limited_duration & (not no_time_feature):
    for epoch in range(0, hp["epochs"]):
        print(f"[Epoch {epoch+1}]: Loading parameters...")
        PASTA.load_parameter(postfix=f"_{predict_time}_new_garnish")
        confusion_matrix, c_label = [0, 0, 0, 0], ["TN", "FP", "FN", "TP"]

        print("--Train start--")
        for idx, row in train_match_result_ftr.iterrows():
            print("[Epoch %1d][%6d/%6d] "%(epoch+1, idx+1, total_rows), end="")
            match_id, win, duration = row["match_no"], row["win"], row["duration"]
            features = pd.read_pickle(f"{route}Flipped_Train/{match_id}.pkl")
            winner, predict = PASTA.train_shorttime_prediction(features, win, duration, predict_time)
            correct = 2*winner + predict
            confusion_matrix[correct] += 1

            tn, fp, fn, tp = confusion_matrix[0], confusion_matrix[1], confusion_matrix[2], confusion_matrix[3]
            acc = (tp+tn)/(tn+fp+fn+tp) * 100
            pre = tp/(tp+fp) * 100 if (tp+fp) != 0 else 0
            rec = tp/(tp+fn) * 100 if (tp+fn) != 0 else 0
            print(" / Acc: %5.2f / Pre: %5.2f / Rec: %5.2f"%(acc, pre, rec))

        for idx in range(0, 4):
            print(f"{c_label[idx]}: {confusion_matrix[idx]} / ", end="")
        print(f"Saving parameters...\n")
        PASTA.save_parameter(postfix=f"_{predict_time}_new_garnish")

else: pass
