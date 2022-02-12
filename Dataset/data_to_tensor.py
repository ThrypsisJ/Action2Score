#%%
import pandas as pd
import pickle
from torch import tensor, float32
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    csv_path = Path('./matches_csv/')
    save_path = Path('./matches_tensor/')
    if not save_path.exists(): save_path.mkdir(parents=True)

    mat_list = list(csv_path.glob('*'))
    total = len(mat_list)
    for file in tqdm(mat_list):
        save_name = save_path / f'{file.name[:-4]}.pkl'
        match_df = pd.read_csv(file, header=0)
        match_tensors = []
        for idx in range(10):
            player_df = match_df[(match_df['is_valid'] == True) & (match_df['player'] == idx+1)].copy()
            player_df.drop(['player', 'player_level', 'skill_level', 'tower_diff', 'is_valid'], axis='columns', inplace=True)
            player_df.sort_values(by='time', ascending=False, inplace=True)
            match_tensors.append(tensor(player_df.to_numpy(), dtype=float32, requires_grad=False))

        with save_name.open('wb') as pickle_file: pickle.dump(match_tensors, pickle_file)
# %%
