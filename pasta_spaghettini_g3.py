import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from Utilities import file_io
from os.path import exists
# import copy # for debug

team = {"red": 0, "blue": 1}

class PASTA():
    def __init__(self, input_size, hidden_size, num_layers, learning_rate):
        self.h_0 = torch.zeros([num_layers, 1, hidden_size], device="cuda")
        self.noodles = [spaghettini(input_size, hidden_size, num_layers).to("cuda") for _ in range(2)]
        self.noodle_optims = [optim.Adam(self.noodles[idx].parameters(), lr=learning_rate) for idx in range(2)]
        self.input_size = input_size

    ########## Train Function ##########
    def train(self, features, winner):
        scores = []
        for idx in range(2):
            self.noodles[idx].train()
            features[idx] = features[idx].to("cuda")
            score = self.noodles[idx](features[idx], self.h_0)
            scores.append(score)

        blue_total, red_total = scores[0].sum(), scores[1].sum()
        loss = self.garnish(blue_total, red_total, winner)
        
        predict = "blue" if blue_total > red_total else "red"

        # print("blue_total: %9.3f / red_total: %9.3f / predict: %4s / winner: %4s"%(blue_total, red_total, predict, winner), end="")
        
        for idx in range(2): self.noodle_optims[idx].zero_grad()
        loss.backward()

        for idx in range(2): self.noodle_optims[idx].step()

        self.param_average()
        return team[winner], team[predict]

    ########## Test Function ##########
    def test(self, features, winner, match_id):
        scores = []
        for idx in range(2):
            self.noodles[idx].eval()
            features[idx] = features[idx].to("cuda")
            score = self.noodles[idx](features[idx], self.h_0)
            scores.append(score)

        blue_total, red_total = scores[0].sum(), scores[1].sum()
        # loss = self.garnish(blue_total, red_total, winner)

        predict = "blue" if blue_total > red_total else "red"

        self.save_result_and_score(match_id, features, scores, predict, winner)

        return team[winner], team[predict]

    ########## The Garnish ##########
    def garnish(self, blue_total, red_total, winner=None):
        if winner == "blue": loss = nn.ReLU()(red_total - blue_total)
        else: loss = nn.ReLU()(blue_total - red_total)
        return loss

    ########## Utility Functions ##########
    def save_result_and_score(self, match_id, features, scores, prediction, winner):
        # file_io.save_prediction(match_id, prediction, f"./processed_ftr/result_predict_spa_g3.csv", winner)
        # file_io.g3_save_features(match_id, features, scores, f"./processed_ftr/scores_spa_g3.csv", winner)
        file_io.save_prediction(match_id, prediction, f'./processed_ftr/challenger_g3.csv', winner)
        file_io.g3_save_features(match_id, features, scores, f'./processed_ftr/challenger_g3_score.csv', winner)
        
    def save_parameter(self, fname='./parameters/param_spa_g3'):
        torch.save(self.noodles[0].state_dict(), fname)

    def load_parameter(self, fname='./parameters/param_spa_g3'):
        noodle_param = fname
        if exists(noodle_param):
            for idx in range(2):
                self.noodles[idx].load_state_dict(torch.load(noodle_param), strict=False)

    def param_average(self):
        param = OrderedDict()
        blue_param = self.noodles[0].state_dict()
        red_param = self.noodles[1].state_dict()
        for key in blue_param.keys():
            param[key] = (blue_param[key] + red_param[key]) / 2

        for idx in range(2):
            self.noodles[idx].load_state_dict(param)

###############################
########## Submodels ##########
###############################
class spaghettini(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(spaghettini, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, input, h_0):
        output, _ = self.gru(input.float(), h_0.float())
        scores = self.linear(output)
        scores = torch.exp(scores/100)
        return scores