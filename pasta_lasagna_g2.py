import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import file_io
from os.path import exists
# import copy # for debug

team = {"red": 0, "blue": 1}

class PASTA():
    def __init__(self, input_size, learning_rate):
        self.lasagna = [lasagna(input_size).to("cuda") for _ in range(0, 10)]
        self.lasagna_optims = [optim.Adam(self.lasagna[idx].parameters(), lr=learning_rate) for idx in range(0, 10)]
        self.input_size = input_size

    ########## Train Function ##########
    def train(self, features, winner):
        scores = []
        for idx in range(0, 10):
            self.lasagna[idx].train()
            features[idx] = features[idx].to("cuda")
            score = self.lasagna[idx](features[idx])
            scores.append(score)

        blue_total = scores[0].sum() + scores[1].sum() + scores[2].sum() + scores[3].sum() + scores[4].sum()
        red_total = scores[5].sum() + scores[6].sum() + scores[7].sum() + scores[8].sum() + scores[9].sum()
        
        predict = "blue" if blue_total > red_total else "red"

        print("blue_total: %9.3f / red_total: %9.3f / predict: %4s / winner: %4s"%(blue_total, red_total, predict, winner), end="")
        
        loss = self.garnish(blue_total, red_total, winner)
        for idx in range(0, 10): self.lasagna_optims[idx].zero_grad()
        loss.backward()

        for idx in range(0, 10): self.lasagna_optims[idx].step()

        self.param_average()
        return team[winner], team[predict]

    ########## Test Function ##########
    def test(self, features, winner, match_id, postfix="full"):
        scores = []
        for idx in range(0, 10):
            self.lasagna[idx].eval()
            features[idx] = features[idx].to("cuda")
            score = self.lasagna[idx](features[idx])
            scores.append(score)

        blue_total = scores[0].sum() + scores[1].sum() + scores[2].sum() + scores[3].sum() + scores[4].sum()
        red_total = scores[5].sum() + scores[6].sum() + scores[7].sum() + scores[8].sum() + scores[9].sum()

        predict = "blue" if blue_total > red_total else "red"

        self.save_result_and_score(match_id, features, scores, predict, winner, postfix=postfix, train=False)

        return team[winner], team[predict]

    ########## The Garnish ##########
    def garnish(self, blue_total, red_total, winner=None):
        chance = blue_total / (blue_total + red_total)
        target = torch.tensor(1.) if winner == 'blue' else torch.tensor(0.)
        target = target.to('cuda')
        loss = nn.BCELoss()(chance, target)
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
        torch.save(self.lasagna[0].state_dict(), f"./parameters/param{postfix}")

    def load_parameter(self, postfix="_full"):
        noodle_param = f"./parameters/param{postfix}"
        if exists(noodle_param):
            for idx in range(0, 10):
                self.lasagna[idx].load_state_dict(torch.load(noodle_param), strict=False)

    def param_average(self):
        param = self.lasagna[0].state_dict()
        for idx in range(1, 10):
            temp_param = self.lasagna[idx].state_dict()
            for key in param.keys():
                param[key] += temp_param[key]

        param_mean = OrderedDict()
        for key in param:
                param_mean[key] = param[key] / 10
        for idx in range(0, 10):
            self.lasagna[idx].load_state_dict(param_mean)

###############################
########## Submodels ##########
###############################
class lasagna(nn.Module):
    def __init__(self, input_size):
        super(lasagna, self).__init__()

        self.input = nn.Linear(input_size, 16)
        self.hidden1 = nn.Linear(16, 8)
        self.hidden2 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)

        # self.sequence = nn.Sequential(
        #     self.input, nn.Sigmoid(),
        #     self.hidden1, nn.Sigmoid(),
        #     self.hidden2, nn.Sigmoid(),
        #     self.output, nn.Sigmoid()
        # )

    def forward(self, input_data):
        # scores = self.sequence(input_data.float())
        scores = self.input(input_data.float())
        scores = self.hidden1(scores)
        scores = nn.Sigmoid()(scores)
        scores = self.hidden2(scores)
        scores = nn.Sigmoid()(scores)
        scores = self.output(scores)
        scores = nn.ReLU()(scores)
        scores = torch.exp(scores/100)
        return scores
