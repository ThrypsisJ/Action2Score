import sys, csv
import send_request
from os import makedirs, remove, listdir
from os.path import exists
from random import shuffle
from shutil import move

if __name__ == '__main__':
    server = str(sys.argv[1])
    api_key = str(sys.argv[2])

    puuid_path = f'../Dataset/puuids/{server}/'
    path = f'../Dataset/match_list/'
    if not exists(path): makedirs(path)

    path = f'{path}{server}/'
    if not exists(path): makedirs(path)

    sender = send_request.match_req_sender(server, api_key)

    for puuid_f in listdir(puuid_path):
        fname = path + puuid_f
        puuid_f = puuid_path + puuid_f
        if exists(fname): continue
        if exists(f'{fname[:-4]}_tmp.csv'): remove(f'{fname[:-4]}_tmp.csv')

        puuid_file = open(puuid_f, 'r', encoding='utf-8', newline='')
        reader = csv.reader(puuid_file)

        file = open(f'{fname[:-4]}_tmp.csv', 'w', encoding="utf-8", newline='')
        writer = csv.writer(file)

        puuids = []
        for row in enumerate(reader):
            puuids.append(row[1][0])
        shuffle(puuids)

        for idx, puuid in enumerate(puuids):
            if idx == 500: break

            print(f'[{puuid_f[:-4]} - {idx+1}/500] ', end='')
            response = sender.match_list_from_puuid(puuid)
            if response == None: continue
            response = response.json()
            for match in response: writer.writerow([match])

        file.close()
        move(f'{fname[:-4]}_tmp.csv', f'{fname}')