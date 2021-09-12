import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import file_io
from os.path import exists
# import copy # for debug

team = {"red": 0, "blue": 1}

class PASTA():
    def __init__(self, input_size, hidden_size, num_layers, learning_rate):
        self.h_0 = torch.zeros([num_layers, 1, hidden_size], device="cuda")
        self.noodles = [spaghettini(input_size, hidden_size, num_layers).to("cuda") for _ in range(0, 10)]
        self.noodle_optims = [optim.Adam(self.noodles[idx].parameters(), lr=learning_rate) for idx in range(0, 10)]
        self.input_size = input_size

    ########## Train Function ##########
    def train(self, features, winner):
        scores = []
        for idx in range(0, 10):
            self.noodles[idx].train()
            features[idx] = features[idx].to("cuda")
            score = self.noodles[idx](features[idx], self.h_0)
            scores.append(score)

        blue_total = scores[0].sum() + scores[1].sum() + scores[2].sum() + scores[3].sum() + scores[4].sum()
        red_total = scores[5].sum() + scores[6].sum() + scores[7].sum() + scores[8].sum() + scores[9].sum()
        loss = self.garnish(blue_total, red_total, winner)
        
        predict = "blue" if blue_total > red_total else "red"

        print("blue_total: %9.3f / red_total: %9.3f / predict: %4s / winner: %4s"%(blue_total, red_total, predict, winner), end="")
        
        for idx in range(0, 10): self.noodle_optims[idx].zero_grad()
        loss.backward()

        for idx in range(0, 10): self.noodle_optims[idx].step()

        self.param_average()
        return team[winner], team[predict]

    def train_shorttime_prediction(self, features, winner, duration, predict_time):
        norm_pred_time = predict_time / float(duration)
        scores = []

        for idx in range(0, 10):
            self.noodles[idx].train()
            features[idx] = features[idx].to("cuda")
            over_target_duration = features[idx].size()[1]
            for timestamp in range(0, features[idx].size()[1]):
                time = features[idx][0][timestamp][0]
                if time <= norm_pred_time:
                    continue
                else:
                    over_target_duration = timestamp
                    break
            features[idx] = features[idx][:,:over_target_duration,:] if over_target_duration > 1 else torch.zeros([1, 1, self.input_size], device="cuda")
            score = self.noodles[idx](features[idx], self.h_0)
            scores.append(score)

        blue_total = scores[0].sum() + scores[1].sum() + scores[2].sum() + scores[3].sum() + scores[4].sum()
        red_total = scores[5].sum() + scores[6].sum() + scores[7].sum() + scores[8].sum() + scores[9].sum()
        loss = self.garnish(blue_total, red_total, winner)

        predict = "blue" if blue_total > red_total else "red"

        print("blue_total: %9.3f / red_total: %9.3f / predict: %4s / winner: %4s"%(blue_total, red_total, predict, winner), end="")

        for idx in range(0, 10): self.noodle_optims[idx].zero_grad()
        loss.backward()

        for idx in range(0, 10): self.noodle_optims[idx].step()

        self.param_average()
        return team[winner], team[predict]

    ########## Test Function ##########
    def test(self, features, winner, match_id, postfix="full"):
        scores = []
        for idx in range(0, 10):
            self.noodles[idx].eval()
            features[idx] = features[idx].to("cuda")
            score = self.noodles[idx](features[idx], self.h_0)
            scores.append(score)

        blue_total = scores[0].sum() + scores[1].sum() + scores[2].sum() + scores[3].sum() + scores[4].sum()
        red_total = scores[5].sum() + scores[6].sum() + scores[7].sum() + scores[8].sum() + scores[9].sum()

        predict = "blue" if blue_total > red_total else "red"

        self.save_result_and_score(match_id, features, scores, predict, winner, postfix=postfix, train=False)

        return team[winner], team[predict]

    ########## The Garnish ##########
    def garnish(self, blue_total, red_total, winner=None):
        if winner == "blue": loss = nn.ReLU()(red_total - blue_total)
        else: loss = nn.ReLU()(blue_total - red_total)
        return loss

    ########## Utility Functions ##########
    def save_result_and_score(self, match_id, features, scores, prediction, winner, postfix="full", train=True):
        if features[0].size()[2] < 30:
            for idx in range(0, 10):
                temp_time = file_io.open_feature(f"{match_id}.ftr", train)
                temp_time = temp_time[temp_time["player"]==idx+1]["time"].values
                temp_time = torch.Tensor([[[time] for time in temp_time]]).to("cuda") if len(temp_time) != 0 else torch.zeros(1, 1, 1, device="cuda")
                features[idx] = torch.cat((temp_time, features[idx]), 2)

        file_io.save_prediction(match_id, prediction, f"./processed_ftr/result_predict_{postfix}.csv", winner)
        file_io.save_feature(match_id, features, scores, f"./processed_ftr/scores_{postfix}.csv", winner)

    def save_parameter(self, postfix="_full"):
        torch.save(self.noodles[0].state_dict(), f"./parameters/param{postfix}")

    def load_parameter(self, postfix="_full"):
        noodle_param = f"./parameters/param{postfix}"
        if exists(noodle_param):
            for idx in range(0, 10):
                self.noodles[idx].load_state_dict(torch.load(noodle_param), strict=False)

    def param_average(self):
        param = self.noodles[0].state_dict()
        for idx in range(1, 10):
            temp_param = self.noodles[idx].state_dict()
            for key in param.keys():
                param[key] += temp_param[key]

        param_mean = OrderedDict()
        for key in param:
                param_mean[key] = param[key] / 10
        for idx in range(0, 10):
            self.noodles[idx].load_state_dict(param_mean)

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