import sys
import send_request
import csv
from os import makedirs
from os import remove
from os.path import exists
from shutil import move

def over_dia(writer, sender, league):
    response = sender.req_user_sumids(league=league)
    if response == None: return
    response = response.json()
    entries = response['entries']
    for entry in entries:
        writer.writerow([entry['summonerId']])

def under_dia(writer, sender, league):
    for division in ['I', 'II', 'III', 'IV']:
        for page in range(0, 100):
            response = sender.req_user_sumids(league=league, division=division, page=page)
            if response == None: continue

            entries = response.json()
            for entry in entries: writer.writerow([entry['summonerId']])

if __name__ == '__main__':
    server = str(sys.argv[1])
    api_key = str(sys.argv[2])
    leagues = ['challenger', 'grandmaster', 'master', 'DIAMOND', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'IRON']
    over_dias = ['challenger', 'grandmaster', 'master']

    path = '../Dataset/summoner_ids/'
    if not exists(path): makedirs(path)

    path = f'{path}{server}/'
    if not exists(path): makedirs(path)

    sender = send_request.match_req_sender(server, api_key)

    for league in leagues:
        fname = f'{path}{league}.csv'
        if exists(fname): continue
        if exists(f'{fname[:-4]}_tmp.csv'): remove(f'{fname[:-4]}_tmp.csv')

        file = open(f'{fname[:-4]}_tmp.csv', 'w', encoding="utf-8", newline='')
        writer = csv.writer(file)

        if league in over_dias: over_dia(writer, sender, league)
        else                  : under_dia(writer, sender, league)

        file.close()
        move(f'{fname[:-4]}_tmp.csv', f'{fname}')