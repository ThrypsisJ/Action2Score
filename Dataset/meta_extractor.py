from os import listdir, makedirs
from os.path import exists
import sys, json
import pandas as pd

if __name__ == '__main__':
    server = sys.argv[1]
    match_path = f'./matches_raw/legacy/{server}/'
    match_datas = [data for data in listdir(match_path) if 'timeline' not in data]

    results = {'win': [], 'duration': []}
    match_names = []
    total = len(match_datas)
    for idx, match in enumerate(match_datas):
        print(f'[{idx:5d}/{total:5d}] Processing {match}')
        file = open(f'{match_path}{match}', 'r')
        match_file = json.load(file)['info']

        match_names.append(match[:-5])
        results['win'].append('blue' if match_file['participants'][0]['win'] else 'red')
        results['duration'].append(match_file['gameDuration'])

    results = pd.DataFrame(results, index=match_names)
    save_path = f'./matches_result/'
    if not exists(save_path): makedirs(save_path)
    results.to_csv(save_path + f'{server}_results.csv')