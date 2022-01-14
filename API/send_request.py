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
            "Retry"     : (429, 500, 502, 503, 504),
            "Forbidden" : 403
        }

    def req_user_sumids(self, league, division, page):
        api_league = self.url + f'lol/league/v4/entries/RANKED_SOLO_5x5/{league}/{division}?page={page+1}'

        res_league = requests.get(api_league, headers=self.req_header)
        while True:
            status = self.error_msg(res_league.status_code)
            if status == 'Return': return None
            if status == 'Proceed': break
            time.sleep(1.2)
            res_league = requests.get(api_league, headers=self.req_header)
            
        print(f'Got summoner\'s ID list on {league}-{division} page {page+1}')
        time.sleep(1.2)
        return res_league

    def req_match(self, match_id):
        api_match = self.url + f"lol/match/v5/matches/{match_id}"

        res_match = requests.get(api_match, headers=self.req_header)
        while True:
            status = self.error_msg(res_match.status_code)
            if status == 'Return': return None
            if status == 'Proceed': break
            time.sleep(1.2)
            res_match = requests.get(api_match, headers=self.req_header)

        print(f"Got match data with match id={match_id} successfuly.")
        time.sleep(1.2)
        return res_match

    def req_timeline(self, match_id):
        api_timeline = self.url + f"lol/match/v5/matches/{match_id}/timeline"

        res_timeline = requests.get(api_timeline, headers=self.req_header)
        while True:
            status = self.error_msg(res_timeline.status_code)
            if status == 'Return': return None
            if status == 'Proceed': break
            time.sleep(1.2)
            res_timeline = requests.get(api_timeline, headers=self.req_header)
        
        print(f"Got timeline data with match id={match_id} successfuly.")
        time.sleep(1.2)
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