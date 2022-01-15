import sys
import send_request
import csv
from os import makedirs, remove, listdir
from os.path import exists
from shutil import move

if __name__ == '__main__':
    server = str(sys.argv[1])
    api_key = str(sys.argv[2])

    summ_ids_path = f'../Dataset/summoner_ids/{server}/'
    path = f'../Dataset/puuids/'
    if not exists(path): makedirs(path)

    pth = f'{path}{server}/'
    if not exists(path): makedirs(path)

    sender = send_request.match_req_sender(server, api_key)

    for summ_f in listdir(summ_ids_path):
        fname = path + summ_f
        summ_f = summ_ids_path + summ_f
        if exists(fname): continue
        if exists(f'{fname[:-4]}_tmp.csv'): remove(f'{fname[:-4]}_tmp.csv')

        summ_file = open(summ_f, 'r', encoding='utf-8', newline='')
        reader = csv.reader(summ_file)

        file = open(f'{fname[:-4]}_tmp.csv', 'w', encoding="utf-8", newline='')
        writer = csv.writer(file)

        for idx, row in enumerate(reader):
            response = sender.req_puuids(row[0])
            if response == None: continue
            response = response.json()
            writer.writerow(response['puuid'])
            if idx >= 1000: break

        file.close()
        move(f'{fname[:-4]}_tmp.csv', f'{fname}')