import os, sys, json
import send_request
import pandas as pd

# get parameter
server = sys.argv[1]
api_key = sys.argv[2]

# define pathes￣
path = f'../Dataset/match_list/{server}/'
save_path = f'../Dataset/matches_raw/'
if not os.path.exists(save_path): os.makedirs(save_path)
save_path = f'{save_path}{server}/'
if not os.path.exists(save_path): os.makedirs(save_path)

sender = send_request.match_req_sender(server, api_key)

# load matchlist file
for league in os.listdir(path):
    league_path = f'{save_path}{league[:-4]}/'
    if not os.path.exists(league_path): os.makedirs(league_path)

    mat_list = pd.read_csv(f'{path}{league}', header=None, names=['match'])['match'].to_list()
    mat_list = list(set(mat_list)) # remove duplicates

    for idx, match_id in enumerate(mat_list):
        if idx == 3000: break
        mat_file = f'{league_path}/{match_id}.json'
        tm_file = f'{league_path}/{match_id}_timeline.json'

        if os.path.exists(mat_file):
            if os.path.exists(tm_file):
                print(f'[{league[:-4]} - {idx+1:04d}/3000] {match_id} data already exists.')
                continue
            else: os.remove(mat_file)

        res_match = sender.req_match(match_id)
        tm_match = sender.req_timeline(match_id)
        if (res_match == None) or (tm_match == None): continue

        res_match = res_match.json()
        tm_match = tm_match.json()

        with open(mat_file, 'w') as file: json.dump(res_match, file, indent=4)
        with open(tm_file, 'w') as file : json.dump(tm_match, file, indent=4)

        print(f'[{league[:-4]} - {idx+1:04d}/3000] {match_id} data has downloaded.')