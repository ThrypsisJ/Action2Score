from os import listdir, makedirs
from os.path import exists
import sys, json
import pandas as pd
from pathlib import Path

class meta_analyzer():
    def __init__(self):
        self.meta_datas = []
        self.columns = ['mat_no', 'player', 'puuid', 'lane', 'kda', 'gold_earned', 'gold_spent', 'creep', 'teamid', 'win']
    
    def meta_extractor(self):
        meta_path:Path = Path('./matches_raw/meta/')
        sco_norm_path:Path = Path('../experiment/matches_scores/normal/')
        mat_list:list = list(sco_norm_path.glob('*'))
        for idx, match in enumerate(mat_list):
            print(f'({idx:6d} / {len(mat_list)}) {idx/len(mat_list)*100:3.1f}% progressing...', end='\r')
            mat_name:str = match.name[:-4]
            meta_file:Path = meta_path / f'{mat_name}.json'

            with meta_file.open('r') as file:
                metainfo = json.load(file)['info']

            for player, info in enumerate(metainfo['participants']):
                puuid = info['puuid']
                lane = info['individualPosition']
                if info['deaths'] == 0  : kda = (info['kills'] + info['assists']) * 1.2
                else                    : kda = (info['kills'] + info['assists']) / info['deaths']
                gold_earned = info['goldEarned']
                gold_spent = info['goldSpent']
                creep = info['totalMinionsKilled']
                teamid = info['teamId']
                win = info['win']
                row = [mat_name, player, puuid, lane, kda, gold_earned, gold_spent, creep, teamid, win]
                self.meta_datas.append(row)

    def meta_saver(self, path:str, name:str):
        if path.endswith('/'): path = path[:-1]
        df = pd.DataFrame(self.meta_datas, columns=self.columns)
        df.to_feather(f'{path}/{name}.ftr')

    def outcome_extractor(self):
        results = { 'match': [], 'win': [] }
        meta_path:Path = Path('./matches_raw/meta/')
        mat_list:list = list(meta_path.glob('*'))

        for idx, match in enumerate(mat_list):
            print(f'({idx:6d} / {len(mat_list)}) {idx/len(mat_list)*100:3.1f}% progressing...', end='\r')
            mat_name:str = match.name[:-5]
            meta_file:Path = meta_path / f'{mat_name}.json'

            with meta_file.open('r') as file:
                metainfo = json.load(file)['info']
            winner = 'blue' if metainfo['teams'][0]['win'] else 'red'
            results['match'].append(mat_name)
            results['win'].append(winner)
        
        df = pd.DataFrame(results['win'], index=results['match'])
        save_path:Path = Path('./matches_result')
        if not save_path.exists(): save_path.mkdir(parents=True)
        df.to_csv(save_path / 'results.csv', index_label='match')

if __name__=='__main__':
    analyzer = meta_analyzer()
    analyzer.outcome_extractor()
    analyzer.meta_extractor()
    analyzer.meta_saver('.', 'meta_analysis')