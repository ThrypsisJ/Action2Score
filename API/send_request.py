import time
import requests

class match_req_sender():
    def __init__(self, server):
        self.api_key = "RGAPI-0cc27bc8-d3e9-48c4-8b44-e34eb1a594cb"
        self.server = str(server)
        self.url = f"https://{self.server}.api.riotgames.com/"
        self.req_header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://developer.riotgames.com",
            "X-Riot-Token": self.api_key
        }
        self.res_stats = {
            "NoData"    : 404,
            "Retry"     : (429, 500, 502, 503, 504)
        }

    def req_wait(self, interval):
        print("Wait for ", end="")
        for i in range(0, interval):
            print(f"{interval-i}...", end="")
            time.sleep(1)
        print("")

    def req_match(self, match_id):
        api_match = self.url + f"/lol/match/v5/matches/{match_id}"
        res_match = requests.get(api_match, headers=self.req_header)

        status_code = res_match.status_code

        if status_code == self.res_stats["NoData"]:
            # print(f"No data with match id = {match_id}")
            time.sleep(1)
            return None

        while status_code in self.res_stats["Retry"]:
            # print(f"Retry match id = {match_id}")
            time.sleep(1)
            res_match = requests.get(api_match, headers=self.req_header)
            status_code = res_match.status_code

        # print(f"Got match data with match id={match_id} successfuly.")
        time.sleep(1)
        return res_match

    def req_timeline(self, match_id):
        api_timeline = self.url + f"/lol/match/v5/matches/{match_id}/timeline"
        res_timeline = requests.get(api_timeline, headers=self.req_header)

        status_code = res_timeline.status_code
        if status_code == self.res_stats["NoData"]:
            # print(f"No data with match id = {match_id}")
            time.sleep(1)
            return None

        while status_code in self.res_stats["Retry"]:
            # print(f"Retry timeline of match id = {match_id}")
            time.sleep(1)
            res_timeline = requests.get(api_timeline, headers=self.req_header)
            status_code = res_timeline.status_code
        
        # print(f"Got timeline data with match id={match_id} successfuly.")
        time.sleep(1)
        return res_timeline