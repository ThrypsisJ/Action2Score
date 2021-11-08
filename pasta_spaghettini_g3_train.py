import pasta_spaghettini_g3 as spa_g3
import pandas as pd
from tqdm import tqdm

PASTA = spa_g3.PASTA(input_size=30, hidden_size=8, num_layers=1, learning_rate=0.001)
PASTA.load_parameter()

data_route = '../../Dataset/League_of_Legends/features_tensor/'
match_list = pd.read_feather('./processed_ftr/match_result_train.ftr')
total_rows = match_list.shape[0]
highest_acc = 0.9935 

print("--Train start--")
for epoch in range(10):
    confusion_matrix, c_label = [0, 0, 0, 0], ["TN", "FP", "FN", "TP"]
    match_list = match_list.sample(frac=1).reset_index(drop=True)
    
    for idx in tqdm(range(total_rows), desc='[Epoch %1d] '%(epoch+1)):
        row = match_list.iloc[idx]
        match_id, win = row['match_no'], row['win']
        features = pd.read_pickle(f"{data_route}by_team_train/{match_id}.pkl")

        winner, predict = PASTA.train(features, win)
        correct = 2*winner + predict
        confusion_matrix[correct] += 1

    tn, fp, fn, tp = confusion_matrix[0], confusion_matrix[1], confusion_matrix[2], confusion_matrix[3]
    acc = (tp+tn)/(tn+fp+fn+tp)
    pre = tp/(tp+fp) if (tp+fp) != 0 else 0
    rec = tp/(tp+fn) if (tp+fn) != 0 else 0
    f1 = 2 * (pre * rec) / (pre + rec)

    for idx in range(0, 4):
        print(f'{c_label[idx]}: {confusion_matrix[idx]} / ', end='')
    print('\n' + 'Accuracy: %.6f / Precision: %.6f / Recall: %.6f / f1 score: %.6f'%(acc, pre, rec, f1))

    if acc > highest_acc:
        print(f"New record. Saving parameters...\n")
        PASTA.save_parameter()
        highest_acc = acc
    else:
        print('Save temporary parameters...\n')
        PASTA.save_parameter('./parameters/tmp')
