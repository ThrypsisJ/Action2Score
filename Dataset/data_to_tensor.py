import pandas as pd
import sys, pickle
from torch import tensor, float32
from os import listdir, makedirs
from os.path import exists

if __name__ == '__main__':
    server = sys.argv[1]
    csv_path = f'./matches_csv/legacy/{server}/'
    save_path = './matches_tensor/'

    for path in ['', 'legacy/', f'{server}/']:
        save_path += path
        if not exists(save_path): makedirs(save_path)

    total = len(listdir(csv_path))
    for idx, file in enumerate(listdir(csv_path)):
        print(f'[{idx+1:5d}/{total:5d} tensorizing {file}]')
        full_path = csv_path + file
        print(full_path)
        save_name = save_path + file[:-4] + '.pkl'

        match_df = pd.read_csv(full_path, header=0)
        match_tensors = []
        for idx in range(10):
            player_df = match_df[(match_df['is_valid'] == True) & (match_df['player'] == idx+1)]
            player_df.drop(['player', 'player_level', 'skill_level', 'tower_diff', 'is_valid'], axis='columns', inplace=True)
            player_df.sort_values(by='time', ascending=False, inplace=True)
            match_tensors.append(tensor(player_df.to_numpy(), dtype=float32, requires_grad=True))

        with open(save_name, 'wb') as pickle_file: pickle.dump(match_tensors)