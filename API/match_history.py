import os
import json
import send_request
import sys
from datetime import datetime
from tqdm import tqdm

# get parameter
server = str(sys.argv[1])

# define pathes￣
path = "F:/Dataset/League_of_Legends/matchlists/"
save_path = f"F:/Dataset/League_of_Legends/mat_datas/{server}/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load matchlist file
for matchid_file in os.listdir(path):
    if matchid_file.endswith(f"{server}.json"):
        with open(path + matchid_file) as json_file:
            match_list = json.load(json_file)
    else: continue

# date info.
year = str(datetime.today().year)[-2:].zfill(2)
month = str(datetime.today().month).zfill(2)
day = str(datetime.today().day).zfill(2)

# Use Riot API
mrs = send_request.match_req_sender(server)
for match_id in tqdm(match_list):
    if os.path.exists(f"{year+month+day}_{save_path}{match_id}_timeline.json"): continue
    res_match = mrs.req_match(f"{match_id}")
    if res_match is None: continue

    res_timeline = mrs.req_timeline(f"{match_id}")
    if res_timeline is None: continue

    js_match = res_match.json()
    js_timeline = res_timeline.json()

    # save json file
    with open(save_path+f"{year+month+day}_{match_id}.json", "w") as file_match:
        json.dump(js_match, file_match, indent=4)

    with open(save_path+f"{year+month+day}_{match_id}_timeline.json", "w") as file_timeline:
        json.dump(js_timeline, file_timeline, indent=4)