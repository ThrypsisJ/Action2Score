import pasta_spaghettini as spa
from tqdm import tqdm
import pandas as pd

hp = {"epochs": 1, "learning_rate": 0.01, "num_layers": 1, "hidden_size": 8, "input_size": 30}
PASTA = spa.PASTA(hp["input_size"], hp["hidden_size"], hp["num_layers"], hp["learning_rate"])

route = "../../Dataset/League_of_Legends/features_tensor/challenger"

test_match_result_ftr = pd.read_csv("./processed_csvs/challenger_result.csv")
total_rows = test_match_result_ftr.shape[0]

PASTA.load_parameter(postfix="_spaghettini")
confusion_matrix, c_label = [0, 0, 0, 0], ["TN", "FP", "FN", "TP"]

print("--test start--")
for idx, row in tqdm(test_match_result_ftr.iterrows(), ncols=50, total=total_rows):
    match_id, win = row["match_no"], row["win"]
    features = pd.read_pickle(f"{route}/{match_id}.pkl")
    winner, predict = PASTA.test(features, win, match_id, postfix="challenger")
    correct = 2*winner + predict
    confusion_matrix[correct] += 1

for idx in range(0, 4):
    print(f"{c_label[idx]}: {confusion_matrix[idx]} / ", end="")