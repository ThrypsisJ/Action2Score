import pasta_spaghettini_g3 as spa_g3
import pandas as pd
from tqdm import tqdm

PASTA = spa_g3.PASTA(input_size=30, hidden_size=8, num_layers=1, learning_rate=0.01)

data_route = '../../Dataset/League_of_Legends/features_tensor/'
match_list = pd.read_feather('./processed_ftr/match_result_test.ftr')
total_rows = match_list.shape[0]

print("--Test start--")
PASTA.load_parameter()
confusion_matrix, c_label = [0, 0, 0, 0], ["TN", "FP", "FN", "TP"]

for idx in tqdm(range(total_rows), desc='[Epoch %1d] '%(epoch+1)):
    row = match_list.iloc[idx]
    match_id, win = row['match_no'], row['win']
    features = pd.read_pickle(f"{data_route}by_team_train/{match_id}.pkl")

    winner, predict = PASTA.test(features, win)
    correct = 2*winner + predict
    confusion_matrix[correct] += 1

tn, fp, fn, tp = confusion_matrix[0], confusion_matrix[1], confusion_matrix[2], confusion_matrix[3]
acc = (tp+tn)/(tn+fp+fn+tp) * 100
pre = tp/(tp+fp) * 100 if (tp+fp) != 0 else 0
rec = tp/(tp+fn) * 100 if (tp+fn) != 0 else 0
f1 = 2 * (pre * rec) / (pre + rec)

for idx in range(0, 4):
    print(f'{c_label[idx]}: {confusion_matrix[idx]} / ', end='')
print('\n' + 'Accuracy: %4.2f / Precision: %.4f / Recall: %.4f / f1 score: %.4f'%(acc, pre, rec, f1))