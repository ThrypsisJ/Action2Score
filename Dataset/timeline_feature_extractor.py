import math, copy
import json
import pandas as pd
from itertools import combinations

class data_parser:
    ward_dict = {
        "CONTROL_WARD": 30.0,
        "SIGHT_WARD": 30.0,
        "BLUE_TRINKET": 15.0,
        "YELLOW_TRINKET": 10.0,
        "UNDEFINED": 0.0
    }
    ##############################################
    ############### basic functions ##############
    ##############################################
    def __init__(self, timeline_json, mat_json):
        version = mat_json['info']['gameVersion'][:-9] + '.1'
        self.champ_vec = pd.read_csv(f'../DataDragon/{version}/champ_vector_by_tag.csv', index_col=0)
        self.item_vec = pd.read_csv(f'../DataDragon/{version}/items_vector.csv', index_col=0)

        self.skill_infos = self.get_skill_infos(mat_json, version)
        self.max_timestamp = timeline_json['info']['frames'][-1]['timestamp']
        self.player_timelines = [self.timestamps_creator(timeline_json) for _ in range(0, 10)]
        self.player_info_to_timeline()

        self.remain_turret = self.turret_remain(timeline_json)
        self.empty_turret_filler(self.remain_turret)

        self.level_calculator(timeline_json["info"]["frames"])
        self.empty_level_filler()

        self.position_calculator(timeline_json["info"]["frames"])
        self.empty_position_filler()
        self.cal_distance()
        self.position_normalize()

        self.skill_level_calculator(timeline_json["info"]["frames"], mat_json['info']['participants'])
        self.empty_skill_level_filler()

        self.score_events(timeline_json)
        self.time_normalizer()

    def get_skill_infos(self, mat_json, version):
        path = f'../DataDragon/{version}/data/ko_KR/champion/'
        skill_infos = []
        for player in mat_json['info']['participants']:
            champ_name = player["championName"]
            file = open(f'{path}{champ_name}.json', 'r')
            spells = json.load(file)['data'][champ_name]['spells']
            skill_info = {}
            for idx, spell in enumerate(spells):
                skill_info[idx+1] = spell['maxrank']
            skill_infos.append(skill_info)
        return skill_infos

    def player_info_to_timeline(self, mat_json):
        for player in mat_json['info']['participants']:
            for timestamp in self.player_timelines[player-1].values():
                champ_name = player['championName']
                for tag in self.champ_vec.columns: timestamp[tag] = int(self.champ_vec.loc[champ_name, tag])
                timestamp[player['individualPosition']] = 1

    def new_feature():
        feature = {
            'time': None,
            # champion
            'mage': 0, 'fighter': 0, 'support': 0, 'tank': 0, 'assassin': 0, 'marksman': 0,
            # player position
            'TOP': 0, 'MIDDLE': 0, 'BOTTOM': 0, 'UTILITY': 0, 'JUNGLE': 0,
            # location
            'x_location': None, 'y_location': None, 'distance': None,
            # event
            'ITEM_PURCHASED': 0, 'ITEM_SOLD': 0, 'ITEM_DESTROYED': 0,
            'SKILL_LEVEL_UP': 0, 'LEVEL_UP': 0,
            'WARD_PLACED': 0, 'WARD_KILL': 0,
            'CHAMPION_KILL': 0, 'CHAMPION_KILL_ASSIST': 0, 'CHAMPION_KILL_VICTIM': 0,
            'BUILDING_KILL': 0, 'BUILDING_KILL_ASSIST': 0,
            'ELITE_MONSTER_KILL': 0, 'ELITE_MONSTER_KILL_ASSIST': 0,
            'event_weight': None,
            # other info
            'player_level': None, 'skill_level': None, 'tower_diff': None, 'is_valid': False
        }
        return feature

    ####################################
    ############### time ###############
    ####################################
    def timestamps_creator(self, timeline_json):
        timestamps = {}
        for frame in timeline_json['info']['frames']:
            for event in frame['events']:
                if event['timestamp'] in timestamps.keys():
                    continue
                else:
                    timestamps[event['timestamp']] = self.new_feature()
            timestamps[frame['timestamp']] = self.new_feature()
        return timestamps

    def time_normalizer(self):
        for idx in range(0, 10):
            for key, timestamp in self.player_timelines[idx].items():
                timestamp['time'] = float(key) / self.max_timestamp

    ########################################
    ############### position ###############
    ########################################
    def position_calculator(self, frames):
        for frame in frames:
            for event in frame['events']:
                timestamp = event['timestamp']
                player = event['killerId'] if 'killerId' in event.keys() else event['participantId']
                if player == 0: continue

                if 'position' in event.keys():
                    x_location = event['position']['x']
                    y_location = event['position']['y']
                elif event['type'] in ['ITEM_PURCHASED', 'ITEM_SOLD']:
                    x_location = 0 if player <= 5 else 15000
                    y_location = 0 if player <= 5 else 15000
                self.player_timelines[player-1][timestamp]['x_location'] = x_location
                self.player_timelines[player-1][timestamp]['y_location'] = y_location

                if 'assistingParticipantIds' in event.keys():
                    for assist in event['assistingParticipantIds']:
                        self.player_timelines[assist-1][timestamp]['x_location'] = x_location
                        self.player_timelines[assist-1][timestamp]['y_location'] = y_location

                if 'victimId' in event.keys():
                    self.player_timelines[event['victimId']-1][timestamp]['x_location'] = x_location
                    self.player_timelines[event['victimId']-1][timestamp]['y_location'] = y_location

            timestamp = frame["timestamp"]
            for player, regular_frame in frame["participantFrames"].items():
                x_location = regular_frame['position']['x']
                y_location = regular_frame['position']['y']
                self.player_timelines[player-1][timestamp]['x_location'] = x_location
                self.player_timelines[player-1][timestamp]['y_location'] = y_location

    def empty_position_filler(self):
        for player, timeline in self.player_timelines.items():
            for key, timestamp in timeline.items():
                if timestamp['x_location'] != None:
                    prev_x = timestamp['x_location']
                    prev_y = timestamp['y_location']
                    x_chunk, y_chunk = self.cal_end_position_chunk(player-1, key)
                    prev_timestamp = key
                else:
                    time_diff = key - prev_timestamp
                    prev_timestamp = key
                    if x_chunk != None:
                        prev_x += x_chunk * time_diff
                        prev_y += y_chunk * time_diff
                    timestamp['x_location'] = prev_x
                    timestamp['y_location'] = prev_y

    def cal_end_position_chunk(self, player, start_stamp):
        timestamp_list = list(self.player_timelines[player].keys())
        start_idx = timestamp_list.index(start_stamp)
        x_chunk, y_chunk = None, None

        found_end = False
        for timestamp in timestamp_list[start_idx+1:-1]:
            if self.player_timelines[player][timestamp]['x_location'] != None:
                end_stamp = timestamp
                end_x_position = self.player_timelines[player][timestamp]['x_location']
                end_y_position = self.player_timelines[player][timestamp]['y_location']
                found_end = True
                break
            
        if found_end:
            time_diff = end_stamp - start_stamp
            x_diff = end_x_position - self.player_timelines[player][start_stamp]['x_location']
            y_diff = end_y_position - self.player_timelines[player][start_stamp]['y_location']

            x_chunk = float(x_diff) / float(time_diff)
            y_chunk = float(y_diff) / float(time_diff)

        return x_chunk, y_chunk

    def cal_distance(self):
        for timestamp in self.timestamps.keys():
            players = []
            for timeline in self.player_timelines.values():
                players.append((timeline[timestamp]['x_location'], timeline[timestamp]['y_location']))
            
            # mean distance
            pairs = list(combinations([0, 1, 2, 3, 4]))
            blue_distance, red_distance = [], []
            player_distances = [[] for _ in range(10)]
            for pair in pairs:
                b_dist = math.dist(players[pair[0]], players[pair[1]])
                blue_distance.append(b_dist)
                player_distances[pair[0]].append(b_dist)
                player_distances[pair[1]].append(b_dist)

                r_dist = math.dist(players[pair[0]+5], players[pair[1]+5])
                red_distance.append(r_dist)
                player_distances[pair[0]+5].append(r_dist)
                player_distances[pair[1]+5].append(r_dist)

            for idx in range(10):
                if idx < 5:
                    player_distances[idx] = sum(player_distances[idx]) / sum(blue_distance)
                else:
                    player_distances[idx] = sum(player_distances[idx]) / sum(red_distance)

            for idx, timeline in enumerate(self.player_timelines):
                timeline[timestamp]['distance'] = player_distances[idx]

    def position_normalize(self):
        for timeline in self.player_timelines:
            for timestamp in timeline.values():
                timestamp['x_location'] = timestamp['x_location'] / 15000.0
                timestamp['y_location'] = timestamp['y_location'] / 15000.0

    ###########################################
    ############### Tower Score ###############
    ###########################################
    def turret_remain(self, timeline_json):
        remain_turrets = {
            "0": {"blue":11, "red":11}
        }

        blue, red = 11, 11
        for frame in timeline_json["info"]["frames"]:
            for event in frame["events"]:
                if event["type"] == "BUILDING_KILL":
                    timestamp = event["timestamp"]

                    if event["teamId"] == 100:
                        blue -= 1
                    else:
                        red -= 1
                    remain_turrets[str(timestamp)] = {"blue":blue, "red":red}
        return remain_turrets
    
    def empty_turret_filler(self, remain_turrets):
        for player, timeline in self.player_timelines.items():
            diff_score = 0
            for key, timestamp in timeline.items():
                if key in remain_turrets.keys():
                    team = int(int(player)/6)
                    diff_blue = (remain_turrets[key]["blue"] - remain_turrets[key]["red"]) / 11.0
                    diff_red = (remain_turrets[key]["red"] - remain_turrets[key]["blue"]) / 11.0
                    diff_score = diff_blue if team == 0 else diff_red
                    timestamp["tower_diff"] = diff_score
                else:
                    timestamp["tower_diff"] = diff_score

    #####################################
    ############### Level ###############
    #####################################
    def level_calculator(self, frames):
        for timeline in self.player_timelines.values():
            timeline[0]["player_level"] = 1
        for frame in frames:
            for event in frame["events"]:
                if event["type"] == "LEVEL_UP":
                    player = event["participantId"]
                    timestamp = event["timestamp"]
                    level = event["level"]
                    self.player_timelines[player-1][timestamp]["player_level"] = level

    def empty_level_filler(self):
        for timeline in self.player_timelines.values():
            prev_level = 1
            for timestamp in timeline.values():
                if timestamp["player_level"] == None:
                    timestamp["player_level"] = prev_level
                else:
                    prev_level = timestamp["player_level"]

    def skill_level_calculator(self, frames):
        recent_level = {}
        for idx, timeline in enumerate(self.player_timelines):
            timeline[0]["skill_level"] = { 1: 0, 2: 0, 3: 0, 4: 0 }
            recent_level[idx] = copy.deepcopy(timeline[0]["skill_level"])

        for frame in frames:
            for event in frame["events"]:
                if event["type"] == "SKILL_LEVEL_UP":
                    player = event["participantId"]
                    timestamp = event["timestamp"]
                    slot = event["skillSlot"]
                    recent_level[player][slot] += 1
                    self.player_timelines[player-1][timestamp]["skill_level"] = recent_level[player].copy()

    def empty_skill_level_filler(self):
        for timeline in self.player_timelines.values():
            prev_level = { 1: 0, 2: 0, 3: 0, 4: 0 }
            for timestamp in timeline.values():
                if timestamp['skill_level'] == None:
                    timestamp['skill_level'] = copy.deepcopy(prev_level)
                else:
                    prev_level = copy.deepcopy(timestamp['skill_level'])

    ############################################
    ############### Event Scores ###############
    ############################################
    def score_events(self, timeline_json):
        for frame in timeline_json['info']['frames']:
            for event in frame['events']:
                if event['type'] in ['BUILDING_KILL', 'ELITE_MONSTER_KILL']:
                    self.weight_building_monster_kill(event)
                elif event['type'] == 'CHAMPION_KILL':
                    self.weight_champion_kill(event)
                elif event['type'] in ['ITEM_PURCHASED', 'ITEM_DESTROYED', 'ITEM_SOLD']:
                    self.score_item(event)
                elif event['type'] in ['WARD_PLACED', 'WARD_KILL']:
                    self.score_ward(event)
                elif event['type'] == 'LEVEL_UP':
                    self.score_level(event)
                elif event['type'] == 'SKILL_LEVEL_UP':
                    self.score_skill_level(event)

    def weight_building_monster_kill(self, event):
        timestamp = event['timestamp']
        killer, assists = event['killerId'], []
        if 'assistingParticipantIds' in event.keys():
            assists = event['assistingParticipantIds']

        event_weight = 1.0 / (len(assists) + 1.0)
        if killer != 0:
            self.player_timelines[killer-1][timestamp]["event"][event["type"]] = 1
            self.player_timelines[killer-1][timestamp]["event_weight"] = event_weight
            self.player_timelines[killer-1][timestamp]["is_valid"] = True
            for assist in assists:
                self.player_timelines[assist-1][timestamp]["event"][event["type"]+"_ASSIST"] = 1
                self.player_timelines[assist-1][timestamp]["event_weight"] = event_weight
                self.player_timelines[assist-1][timestamp]["is_valid"] = True

    def weight_champion_kill(self, event):
        timestamp = event['timestamp']
        killer, victim, assists = event['killerId'], event['victimId'], []
        if 'assistingParticipantIds' in event.keys():
            assists = event['assistingParticipantIds']

        victimDamageDealt = None
        if 'victimDamageDealt' in event: victimDamageDealt = event['victimDamageDealt']
        damage_dict = self.damage_calculator(event["victimDamageReceived"], victimDamageDealt, killer, victim, assists)

        if killer != "0":
            killer_weight = float(damage_dict[killer-1]) / float(damage_dict["total"])
            self.player_timelines[killer-1][timestamp][event["type"]] = 1
            self.player_timelines[killer-1][timestamp]["event_weight"] = killer_weight
            self.player_timelines[killer-1][timestamp]["is_valid"] = True

        victim_weight = float(damage_dict[victim-1]) / float(damage_dict["total"])
        self.player_timelines[victim-1][timestamp][event["type"]+"_VICTIM"] = 1
        self.player_timelines[victim-1][timestamp]["event_weight"] = victim_weight
        self.player_timelines[victim-1][timestamp]["is_valid"] = True

        for assist in assists:
            assist = str(assist)
            assist_weight = float(damage_dict[assist-1]) / float(damage_dict["total"])
            self.player_timelines[assist-1][timestamp][event["type"]+"_ASSIST"] = 1
            self.player_timelines[assist-1][timestamp]["event_weight"] = assist_weight
            self.player_timelines[assist-1][timestamp]["is_valid"] = True

    # calculate damage for score_champion_kill function
    def damage_calculator(self, victim_dmg_recv, victim_dmg_dealt, killer, victim, assists):
        damage_dict = {}
        # initialize damage_dict
        damage_dict["total"] = 0
        damage_dict[victim-1] = 0
        damage_dict[killer-1] = 0
        for assist in assists:
            damage_dict[assist-1] = 0

        # calculate recv damage
        for recv_dmg in victim_dmg_recv:
            quantity = recv_dmg["magicDamage"] + recv_dmg["physicalDamage"] + recv_dmg["trueDamage"]
            player = recv_dmg["participantId"]

            damage_dict["total"] += quantity
            if player != 0:
                if player in damage_dict.keys():
                    damage_dict[player-1] += quantity
        
        # calculate dealt damage
        damage_dict[victim-1] = 0
        if victim_dmg_dealt != None:
            for dealt_dmg in victim_dmg_dealt:
                quantity = dealt_dmg["magicDamage"] + dealt_dmg["physicalDamage"] + dealt_dmg["trueDamage"]
                damage_dict[victim-1] += quantity

        return damage_dict

    def score_item(self, event):
        timestamp = event['timestamp']
        player = event['participantId'] - 1
        item_value = data_parser.item_vec.loc[event['itemId']]
        if event['type'] == 'ITEM_SOLD':
            event_weight = float(item_value['gold_sell'])
        else:
            event_weight = float(item_value['gold_purchase'])

        self.player_timelines[player][timestamp][event["type"]] = 1
        self.player_timelines[player][timestamp]["event_weight"] = event_weight
        self.player_timelines[player][timestamp]["is_valid"] = True

    def score_ward(self, event):
        timestamp = event["timestamp"]
        player = (event["creatorId"]-1) if event["type"] == "WARD_PLACED" else (event["killerId"]-1)
        if (player != "0") and (event["wardType"] != "TEEMO_MUSHROOM"):
            ward_value = data_parser.ward_dict[event["wardType"]]

            event_weight = ward_value / 30.0
            self.player_timelines[player][timestamp]["event"][event["type"]] = 1
            self.player_timelines[player][timestamp]["event_weight"] = event_weight
            self.player_timelines[player][timestamp]["is_valid"] = True

    def score_level(self, event):
        timestamp = event['timestamp']
        player = event['participantId'] - 1
        level = self.player_timelines[player][timestamp]['player_level']
        level_score = 10
        for idx, other_player in enumerate(self.player_timelines):
            if idx == player: continue
            if other_player[timestamp]['player_level'] > level:
                level_score -= 1

        event_weight = level_score / 10.0
        self.player_timelines[player][timestamp][event['type']] = 1
        self.player_timelines[player][timestamp]['event_weight'] = event_weight
        self.player_timelines[player][timestamp]['is_valid'] = True

    def score_skill_level(self, event):
        timestamp = event['timestamp']
        player = event['participantId']
        
        skill_slot = event['skillSlot']
        max_level = self.skill_infos[player-1][skill_slot]['maxrank']
        level = self.player_timelines[player][timestamp]['skill_level'][skill_slot]

        event_weight = float(level) / float(max_level)
        self.player_timelines[player][timestamp]['event'][event['type']] = 1
        self.player_timelines[player][timestamp]['event_weight'] = event_weight
        self.player_timelines[player][timestamp]['is_valid'] = True