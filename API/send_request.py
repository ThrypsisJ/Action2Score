import time
import requests

class match_req_sender():
    def __init__(self, server, api_key):
        self.server = server
        self.url = f"https://{self.server}.api.riotgames.com/"
        self.req_header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Whale/2.11.126.19 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
            "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://developer.riotgames.com",
            "X-Riot-Token": api_key
        }
        self.res_stats = {
            "NoData"    : 404,
            "Retry"     : (400, 429, 500, 502, 503, 504),
            "Forbidden" : 403
        }

    def req_user_sumids(self, **kwargs):
        over_dia = kwargs['league'] not in ['DIAMOND', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'IRON']

        if over_dia: api_league = self.url + f'lol/league/v4/{kwargs["league"]}leagues/by-queue/RANKED_SOLO_5x5'
        else        : api_league = self.url + f'lol/league/v4/entries/RANKED_SOLO_5x5/{kwargs["league"]}/{kwargs["division"]}?page={kwargs["page"]+1}'

        res_league = requests.get(api_league, headers=self.req_header)
        while True:
            status = self.error_msg(res_league.status_code)
            if status == 'Return': return None
            if status == 'Proceed': break
            time.sleep(1.2)
            res_league = requests.get(api_league, headers=self.req_header)
        
        if over_dia: print(f'Got summoner\'s ID list on {kwargs["league"]} league')
        else        : print(f'Got summoner\'s ID list on {kwargs["league"]}-{kwargs["division"]} page {kwargs["page"]+1}')
        time.sleep(1.2)
        return res_league

    def req_puuids(self, somm_id):
        api_summ = self.url + f'lol/summoner/v4/summoners/{somm_id}'
        res_summ = requests.get(api_summ, headers=self.req_header)

        while True:
            status = self.error_msg(res_summ.status_code)
            if status == 'Return': return None
            if status == 'Proceed': break
            time.sleep(1.2)
            res_summ = requests.get(api_summ, headers=self.req_header)
        
        print(f'Got puuid of summoner id {{{somm_id}}}')
        time.sleep(1.2)
        return res_summ

    def match_list_from_puuid(self, puuid):
        api_mat_list = self.url + f'lol/match/v5/matches/by-puuid/{puuid}/ids'
        res_mat_list = requests.get(api_mat_list, headers=self.req_header)

        while True:
            status = self.error_msg(res_mat_list.status_code)
            if status == 'Return': return None
            if status == 'Proceed': break
            time.sleep(1.2)
            res_mat_list = requests.get(api_mat_list, headers=self.req_header)

        print(f'Got match list of puuid {{{puuid}}}')
        time.sleep(1.2)
        return res_mat_list

    def req_match(self, match_id):
        api_match = self.url + f"lol/match/v5/matches/{match_id}"

        res_match = requests.get(api_match, headers=self.req_header)
        while True:
            status = self.error_msg(res_match.status_code)
            if status == 'Return': return None
            if status == 'Proceed': break
            time.sleep(1.1)
            res_match = requests.get(api_match, headers=self.req_header)
        time.sleep(1.1)
        return res_match

    def req_timeline(self, match_id):
        api_timeline = self.url + f"lol/match/v5/matches/{match_id}/timeline"

        res_timeline = requests.get(api_timeline, headers=self.req_header)
        while True:
            status = self.error_msg(res_timeline.status_code)
            if status == 'Return': return None
            if status == 'Proceed': break
            time.sleep(1.1)
            res_timeline = requests.get(api_timeline, headers=self.req_header)
        time.sleep(1.1)
        return res_timeline

    def error_msg(self, response):
        if response == self.res_stats['NoData']:
            print('There was no data')
            time.sleep(1.2)
            return 'Return'

        if response == self.res_stats['Forbidden']:
            print('Unauthorized access!')
            time.sleep(1.2)
            return 'Return'

        if response in self.res_stats['Retry']:
            print('Resend request')
            time.sleep(1.2)
            return 'Retry'

        return 'Proceed' # No error