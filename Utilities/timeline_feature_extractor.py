import math
import copy

class data_parser:
    item_dict = {}
    champion_dict = {}

    feature = {
        "time": None,
        # champion
        "champion": {
            "mage": 0,
            "fighter": 0,
            "support": 0,
            "tank": 0,
            "assassin": 0,
            "marksman": 0,
        },
        # player role
        "role": {
            "TOP": 0,
            "MIDDLE": 0,
            "BOTTOM": 0,
            "UTILITY": 0,
            "JUNGLE": 0,
        },
        # position
        "x_position": None,
        "y_position": None,
        "deviation": None,
        # event
        "event": {
            "ITEM_PURCHASED": 0,
            "ITEM_SOLD": 0,
            "ITEM_DESTROYED": 0,
            "SKILL_LEVEL_UP": 0,
            "LEVEL_UP": 0,
            "WARD_PLACED": 0,
            "WARD_KILL": 0,
            "CHAMPION_KILL": 0,
            "CHAMPION_KILL_ASSIST": 0,
            "CHAMPION_KILL_VICTIM": 0,
            "BUILDING_KILL": 0,
            "BUILDING_KILL_ASSIST": 0,
            "ELITE_MONSTER_KILL": 0,
            "ELITE_MONSTER_KILL_ASSIST": 0,
        },
        "event_weight": None,
        # other info
        "player_level": None,
        "skill_level": None,
        "tower_diff": None,
        "is_valid": False
    }

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
        self.timestamps = self.timestamps_creator(timeline_json)

        self.mat_data = self.mat_data_parser(mat_json)
        # self.max_timestamp = float(list(self.timestamps.keys())[-1])
        self.max_timestamp = 4000000
        self.player_timelines = {}
        for idx in range(0, 10):
            self.player_timelines[str(idx+1)] = copy.deepcopy(self.timestamps)
        self.player_info_to_timeline()

        self.remain_turret = self.turret_remain(timeline_json)
        self.empty_turret_filler(self.remain_turret)

        self.level_calculator(timeline_json["info"]["frames"])
        self.empty_level_filler()

        self.position_calculator(timeline_json["info"]["frames"])
        self.empty_position_filler()
        self.cal_deviation()
        self.position_normalize()

        self.skill_level_calculator(timeline_json["info"]["frames"])
        self.empty_skill_level_filler()

        self.score_events(timeline_json)
        self.time_calculator()
        
    def mat_data_parser(self, mat_json):
        mat_data = {}
        for player in mat_json["info"]["participants"]:
            mat_data[str(player["participantId"])] = {
                "champ_id": player["championId"],
                "champ_cat": data_parser.champion_dict[str(player["championId"])]["category"],
                "role": player["individualPosition"]
            }
        return mat_data

    def player_info_to_timeline(self):
        for idx in range(0, 10):
            for timestamp in self.player_timelines[str(idx+1)].values():
                champ_cat = self.mat_data[str(idx+1)]["champ_cat"]
                role = self.mat_data[str(idx+1)]["role"]
                timestamp["champion"] = champ_cat
                timestamp["role"][role] = 1

    ####################################
    ############### time ###############
    ####################################
    def time_calculator(self):
        for idx in range(0, 10):
            for key, timestamp in self.player_timelines[str(idx+1)].items():
                timestamp["time"] = float(key) / self.max_timestamp

    def timestamps_creator(self, timeline_json):
        timestamps = {}
        for frame in timeline_json["info"]["frames"]:
            for event in frame["events"]:
                if str(event['timestamp']) in timestamps.keys():
                    continue
                else:
                    timestamps[str(event['timestamp'])] = copy.deepcopy(data_parser.feature)
            timestamps[str(frame['timestamp'])] = copy.deepcopy(data_parser.feature)
        return timestamps

    ########################################
    ############### position ###############
    ########################################
    def position_calculator(self, frames):
        for frame in frames:
            for event in frame["events"]:
                timestamp = str(event["timestamp"])
                if ("_KILL" in event["type"]) and ("WARD" not in event["type"]):
                    killer = str(event["killerId"])
                    if killer == "0": continue
                    assists = [] if "assistingParticipantIds" not in event.keys() else event["assistingParticipantIds"]

                    x_position = event["position"]["x"]
                    y_position = event["position"]["y"]

                    self.player_timelines[killer][timestamp]["x_position"] = x_position
                    self.player_timelines[killer][timestamp]["y_position"] = y_position
                    for assist in assists:
                        self.player_timelines[str(assist)][timestamp]["x_position"] = x_position
                        self.player_timelines[str(assist)][timestamp]["y_position"] = y_position

                    if event["type"] == "CHAMPION_KILL":
                        victim = str(event["victimId"])
                        self.player_timelines[victim][timestamp]["x_position"] = x_position
                        self.player_timelines[victim][timestamp]["y_position"] = y_position

                elif ("ITEM_" in event["type"]) and ("_DESTROYED" not in event["type"]):
                    player = event["participantId"]
                    
                    x_position = 15000 * int(player/6)
                    y_position = 15000 * int(player/6)

                    self.player_timelines[str(player)][timestamp]["x_position"] = x_position
                    self.player_timelines[str(player)][timestamp]["y_position"] = y_position

            player_timestamp = str(frame["timestamp"])
            for player, regular_frame in frame["participantFrames"].items():
                x_position = regular_frame["position"]["x"]
                y_position = regular_frame["position"]["y"]
                self.player_timelines[player][player_timestamp]["x_position"] = x_position
                self.player_timelines[player][player_timestamp]["y_position"] = y_position

    def empty_position_filler(self):
        for player, timeline in self.player_timelines.items():
            for key, timestamp in timeline.items():
                if timestamp["x_position"] != None:
                    prev_x = timestamp["x_position"]
                    prev_y = timestamp["y_position"]
                    x_chunk, y_chunk = self.cal_end_position_chunk(player, key)
                    prev_timestamp = key
                else:
                    time_diff = int(key) - int(prev_timestamp)
                    prev_timestamp = key
                    if x_chunk != None:
                        x_diff = x_chunk * time_diff
                        y_diff = y_chunk * time_diff
                        prev_x += x_diff
                        prev_y += y_diff
                    timestamp["x_position"] = prev_x
                    timestamp["y_position"] = prev_y

    def cal_end_position_chunk(self, player, start_timestamp):
        timestamp_list = list(self.player_timelines[player].keys())
        start_idx = timestamp_list.index(start_timestamp)

        found_end = False
        for timestamp in timestamp_list[start_idx+1:-1]:
            if self.player_timelines[player][timestamp]["x_position"] != None:
                end_timestamp = int(timestamp)
                end_x_position = self.player_timelines[player][timestamp]["x_position"]
                end_y_position = self.player_timelines[player][timestamp]["y_position"]
                found_end = True
                break

        if not found_end:
            x_chunk, y_chunk = None, None
        else:
            time_diff = end_timestamp - int(start_timestamp)
            x_diff = int(end_x_position) - int(self.player_timelines[player][start_timestamp]["x_position"])
            y_diff = int(end_y_position) - int(self.player_timelines[player][start_timestamp]["y_position"])

            x_chunk = float(x_diff) / float(time_diff)
            y_chunk = float(y_diff) / float(time_diff)

        return x_chunk, y_chunk

    def cal_deviation(self):
        for timestamp in self.timestamps.keys():
            x_positions = []
            y_positions = []
            for timeline in self.player_timelines.values():
                x_position = timeline[timestamp]["x_position"]
                y_position = timeline[timestamp]["y_position"]
                x_positions.append(x_position)
                y_positions.append(y_position)

            # calculate center
            blue_x_center = sum(x_positions[:5]) / 5.0
            blue_y_center = sum(y_positions[:5]) / 5.0
            red_x_center = sum(x_positions[5:]) / 5.0
            red_y_center = sum(y_positions[5:]) / 5.0

            # distance
            distances = []
            for idx in range(0, 10):
                x_distance = x_positions[idx] - blue_x_center if idx < 5 else x_positions[idx] - red_x_center
                y_distance = y_positions[idx] - blue_y_center if idx < 5 else y_positions[idx] - red_y_center
                distance = math.sqrt(x_distance**2 + y_distance**2)
                distances.append(distance)

            blue_distance_mean = sum(distances[:5]) / 5.0
            red_distance_mean = sum(distances[5:]) / 5.0

            # deviation
            deviations = []
            for idx in range(0, 10):
                distance_mean = blue_distance_mean if idx < 5 else red_distance_mean
                deviation = distances[idx] / (distance_mean * 2.5) if distance_mean != 0 else 0
                deviations.append(deviation)
            
            for idx, timeline in enumerate(self.player_timelines.values()):
                timeline[timestamp]["deviation"] = deviations[idx]

    def position_normalize(self):
        for timeline in self.player_timelines.values():
            for timestamp in timeline.values():
                timestamp["x_position"] = timestamp["x_position"] / 15000.0
                timestamp["y_position"] = timestamp["y_position"] / 15000.0

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
            timeline["0"]["player_level"] = 1
        for frame in frames:
            for event in frame["events"]:
                if event["type"] == "LEVEL_UP":
                    player = str(event["participantId"])
                    timestamp = str(event["timestamp"])
                    level = event["level"]
                    self.player_timelines[player][timestamp]["player_level"] = level

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
        for key, timeline in self.player_timelines.items():
            timeline["0"]["skill_level"] = {
                "1": 0, "2": 0, "3": 0, "4": 0
            }
            recent_level[key] = copy.deepcopy(timeline["0"]["skill_level"])

        for frame in frames:
            for event in frame["events"]:
                if event["type"] == "SKILL_LEVEL_UP":
                    player = str(event["participantId"])
                    timestamp = str(event["timestamp"])
                    slot = str(event["skillSlot"])
                    recent_level[player][slot] += 1
                    self.player_timelines[player][timestamp]["skill_level"] = recent_level[player].copy()

    def empty_skill_level_filler(self):
        for timeline in self.player_timelines.values():
            prev_level = {
                "1": 0, "2": 0, "3": 0, "4": 0
            }
            for timestamp in timeline.values():
                if timestamp["skill_level"] == None:
                    timestamp["skill_level"] = copy.deepcopy(prev_level)
                else:
                    prev_level = copy.deepcopy(timestamp["skill_level"])

    ############################################
    ############### Event Scores ###############
    ############################################
    def score_events(self, timeline_json):
        for frame in timeline_json["info"]["frames"]:
            for event in frame["events"]:
                if event["type"] in ["BUILDING_KILL", "ELITE_MONSTER_KILL"]:
                    self.score_building_monster_kill(event)
                elif event["type"] == "CHAMPION_KILL":
                    self.score_champion_kill(event)
                elif event["type"] in ["ITEM_PURCHASED", "ITEM_DESTROYED", "ITEM_SOLD"]:
                    self.score_item(event)
                elif event["type"] in ["WARD_PLACED", "WARD_KILL"]:
                    self.score_ward(event)
                elif event["type"] == "LEVEL_UP":
                    self.score_level(event)
                elif event["type"] == "SKILL_LEVEL_UP":
                    self.score_skill_level(event)

    def score_building_monster_kill(self, event):
        timestamp = str(event["timestamp"])
        killer, assists = str(event["killerId"]), []
        if "assistingParticipantIds" in event.keys():
            assists = event["assistingParticipantIds"]

        event_weight = 1.0 / (len(assists) + 1.0)
        if killer != "0":
            self.player_timelines[killer][timestamp]["event"][event["type"]] = 1
            self.player_timelines[killer][timestamp]["event_weight"] = event_weight
            self.player_timelines[killer][timestamp]["is_valid"] = True
            for assist in assists:
                assist = str(assist)
                self.player_timelines[assist][timestamp]["event"][event["type"]+"_ASSIST"] = 1
                self.player_timelines[assist][timestamp]["event_weight"] = event_weight
                self.player_timelines[assist][timestamp]["is_valid"] = True

    def score_champion_kill(self, event):
        timestamp = str(event["timestamp"])
        killer, victim, assists = str(event["killerId"]), str(event["victimId"]), []
        if "assistingParticipantIds" in event.keys():
            assists = event["assistingParticipantIds"]

        victimDamageDealt = None
        if "victimDamageDealt" in event: victimDamageDealt = event["victimDamageDealt"]
        damage_dict = self.damage_calculator(event["victimDamageReceived"], victimDamageDealt, killer, victim, assists)

        if killer != "0":
            killer_weight = float(damage_dict[killer]) / float(damage_dict["total"])
            self.player_timelines[killer][timestamp]["event"][event["type"]] = 1
            self.player_timelines[killer][timestamp]["event_weight"] = killer_weight
            self.player_timelines[killer][timestamp]["is_valid"] = True

        victim_weight = float(damage_dict[victim]) / float(damage_dict["total"])
        self.player_timelines[victim][timestamp]["event"][event["type"]+"_VICTIM"] = 1
        self.player_timelines[victim][timestamp]["event_weight"] = victim_weight
        self.player_timelines[victim][timestamp]["is_valid"] = True

        for assist in assists:
            assist = str(assist)
            assist_weight = float(damage_dict[assist]) / float(damage_dict["total"])
            self.player_timelines[assist][timestamp]["event"][event["type"]+"_ASSIST"] = 1
            self.player_timelines[assist][timestamp]["event_weight"] = assist_weight
            self.player_timelines[assist][timestamp]["is_valid"] = True

    # calculate damage for score_champion_kill function
    def damage_calculator(self, victim_dmg_recv, victim_dmg_dealt, killer, victim, assists):
        damage_dict = {}
        # initialize damage_dict
        damage_dict["total"] = 0
        damage_dict[victim] = 0
        damage_dict[killer] = 0
        for assist in assists:
            damage_dict[str(assist)] = 0

        # calculate recv damage
        for recv_dmg in victim_dmg_recv:
            quantity = recv_dmg["magicDamage"] + recv_dmg["physicalDamage"] + recv_dmg["trueDamage"]
            player = recv_dmg["participantId"]

            damage_dict["total"] += quantity
            if player != 0:
                player = str(player)
                if player in damage_dict.keys():
                    damage_dict[player] += quantity
        
        # calculate dealt damage
        damage_dict[victim] = 0
        if victim_dmg_dealt != None:
            for dealt_dmg in victim_dmg_dealt:
                quantity = dealt_dmg["magicDamage"] + dealt_dmg["physicalDamage"] + dealt_dmg["trueDamage"]
                damage_dict[victim] += quantity

        return damage_dict

    def score_item(self, event):
        timestamp = str(event["timestamp"])
        player = str(event["participantId"])
        item_value = data_parser.item_dict[str(event["itemId"])]
        if event["type"] == "ITEM_SOLD":
            event_weight = int(item_value["sell"]) / 5241.0
        else:
            event_weight = int(item_value["buy"]) / 3000.0

        self.player_timelines[player][timestamp]["event"][event["type"]] = 1
        self.player_timelines[player][timestamp]["event_weight"] = event_weight
        self.player_timelines[player][timestamp]["is_valid"] = True

    def score_ward(self, event):
        timestamp = str(event["timestamp"])
        player = str(event["creatorId"]) if event["type"] == "WARD_PLACED" else str(event["killerId"])
        if (player != "0") and (event["wardType"] != "TEEMO_MUSHROOM"):
            ward_value = data_parser.ward_dict[event["wardType"]]

            event_weight = ward_value / 30.0
            self.player_timelines[player][timestamp]["event"][event["type"]] = 1
            self.player_timelines[player][timestamp]["event_weight"] = event_weight
            self.player_timelines[player][timestamp]["is_valid"] = True

    def score_level(self, event):
        timestamp = str(event["timestamp"])
        player = str(event["participantId"])
        level = self.player_timelines[player][timestamp]["player_level"]
        level_score = 10
        for id, other_player in self.player_timelines.items():
            if id == other_player: continue
            elif other_player[timestamp]["player_level"] > level:
                level_score -= 1

        event_weight = level_score / 10.0
        self.player_timelines[player][timestamp]["event"][event["type"]] = 1
        self.player_timelines[player][timestamp]["event_weight"] = event_weight
        self.player_timelines[player][timestamp]["is_valid"] = True

    def score_skill_level(self, event):
        timestamp = str(event["timestamp"])
        player = str(event["participantId"])
        player_champ = str(self.mat_data[player]["champ_id"])
        skill_slot = str(event["skillSlot"])
        level = self.player_timelines[player][timestamp]["skill_level"][skill_slot]
        max_level = data_parser.champion_dict[player_champ]["spells"][skill_slot]

        event_weight = float(level) / float(max_level)
        self.player_timelines[player][timestamp]["event"][event["type"]] = 1
        self.player_timelines[player][timestamp]["event_weight"] = event_weight
        self.player_timelines[player][timestamp]["is_valid"] = True