import pasta_spaghettini as spa
import pasta_spaghettini_g2 as spa_g2
import pandas as pd

hp = {"epochs": 3, "learning_rate": 0.001, "num_layers": 2, "hidden_size": 8, "input_size": 30}
PASTA = spa_g2.PASTA(hp["input_size"], hp["hidden_size"], hp["num_layers"], hp["learning_rate"])

route = "../../Dataset/League_of_Legends/features_tensor/"

train_match_result_ftr = pd.read_feather("./processed_ftr/match_result_train.ftr")
total_rows = train_match_result_ftr.shape[0]

for epoch in range(0, hp["epochs"]):
    print(f"[SPA G2: Epoch {epoch+1}]: Loading parameters...")
    PASTA.load_parameter(postfix="_spaghettini_nf_g2_lr0001_2layer")
    confusion_matrix, c_label = [0, 0, 0, 0], ["TN", "FP", "FN", "TP"]

    print("--Train start--")
    for idx, row in train_match_result_ftr.iterrows():
        print("[SPA G2: Epoch %1d][%6d/%6d] "%(epoch+1, idx+1, total_rows), end="")
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
    PASTA.save_parameter(postfix="_spaghettini_nf_g2_lr0001_2layer")

##############################################################################################################
PASTA = spa.PASTA(hp["input_size"], hp["hidden_size"], hp["num_layers"], hp["learning_rate"])
train_match_result_ftr = pd.read_feather("./processed_ftr/match_result_train.ftr")
total_rows = train_match_result_ftr.shape[0]

for epoch in range(0, hp["epochs"]):
    print(f"[SPA: Epoch {epoch+1}]: Loading parameters...")
    PASTA.load_parameter(postfix="_spaghettini_nf_lr0001_2layer")
    confusion_matrix, c_label = [0, 0, 0, 0], ["TN", "FP", "FN", "TP"]

    print("--Train start--")
    for idx, row in train_match_result_ftr.iterrows():
        print("[SPA: Epoch %1d][%6d/%6d] "%(epoch+1, idx+1, total_rows), end="")
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
    PASTA.save_parameter(postfix="_spaghettini_nf_lr0001_2layer")