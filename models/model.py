from torch import nn, optim, zeros, ones, tensor, exp, cuda
from os.path import exists
from os import makedirs, remove
from pickle import load, dump, HIGHEST_PROTOCOL

class Model():
    def __init__(self, hidden_size, gru_layers, lr, zero_h0):
        self.winner_h0 = ones([gru_layers, 1, hidden_size], device='cuda', requires_grad=True).float()
        self.loser_h0 = zeros([gru_layers, 1, hidden_size], device='cuda', requires_grad=True).float()
        self.zero_h0 = zero_h0

        self.sub_models = [SubModel(hidden_size, gru_layers).to('cuda') for _ in range(10)]
        self.optims = [optim.Adam(s_model.parameters(), lr=lr) for s_model in self.sub_models]

    def train(self, features, winner):
        blue_h0, red_h0 = self.get_h0(winner)
        for player in range(10): self.sub_models[player].train()

        scores = self.proceed(features, blue_h0, red_h0)
        blue_total = sum([scores[idx].sum() for idx in range(0, 5)])
        red_total = sum([scores[idx].sum() for idx in range(5, 10)]) 
        loss = self.loss_func(blue_total, red_total, winner)

        for player in range(10): self.optims[player].zero_grad()
        loss.backward()
        for player in range(10): self.optims[player].step()

        self.get_parameter_mean()

    def validate(self, features, winner):
        blue_h0, red_h0 = self.get_h0(winner)
        for player in range(10): self.sub_models[player].eval()

        scores = self.proceed(features, blue_h0, red_h0)
        for idx in range(len(scores)): scores[idx] = scores[idx].to('cpu').detach()
        blue_total = sum([scores[idx].sum() for idx in range(0, 5)])
        red_total = sum([scores[idx].sum() for idx in range(5, 10)])
        self.test_loss += self.loss_func(blue_total, red_total, winner).item()

        if blue_total >= red_total and winner == 'blue' : self.tp += 1 
        if blue_total >= red_total and winner == 'red'  : self.fp += 1 
        if blue_total < red_total and winner == 'blue'  : self.fn += 1 
        if blue_total < red_total and winner == 'red'   : self.tn += 1 

        return scores

    def test(self, features, winner, path, mat_name):
        scores = self.validate(features, winner)
        if not exists(path): makedirs(path)
        with open(f'{path}{mat_name}.pkl', 'wb') as file:
            dump(scores, file, HIGHEST_PROTOCOL)

    def loss_func(self, blue_score, red_score, winner):
        if winner == "blue" : loss = nn.ReLU()(red_score - blue_score)
        else                : loss = nn.ReLU()(blue_score - red_score)
        return loss

    def get_h0(self, winner):
        if not self.zero_h0:
            blue_h0 = self.winner_h0 if winner == 'blue' else self.loser_h0
            red_h0 = self.winner_h0 if winner == 'red' else self.loser_h0
        else:
            blue_h0, red_h0 = self.loser_h0, self.loser_h0
        return blue_h0, red_h0
    
    def proceed(self, features, blue_h0, red_h0):
        scores = []
        cuda.empty_cache()
        for player in range(10):
            feature = features[player].to('cuda')
            if feature.shape[0] == 0: feature = zeros(1, 30, device='cuda')
            feature = feature.unsqueeze(0)
            h0 = blue_h0 if player < 5 else red_h0
            score = self.sub_models[player](feature, h0)
            scores.append(score)
        return scores

    def get_parameter_mean(self):
        params = self.sub_models
        param_keys = params[0].state_dict().keys()

        tmp_params = {}
        for key in param_keys:
            tmp_params[key] = sum([param.state_dict()[key] for param in params]) / 10

        for player in range(10):
            self.sub_models[player].load_state_dict(tmp_params)

    def reset_result(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
        self.test_loss = 0

    def get_result(self):
        acc = (self.tp+self.tn) / (self.tp+self.tn+self.fp+self.fn)
        pre = (self.tp) / (self.tp+self.fp)
        rec = (self.tp) / (self.tp+self.fn)
        f1 = 2 * (pre*rec) / (pre+rec)
        return acc, pre, rec, f1, self.test_loss

    def save_parameters(self, fname):
        path = './experiment/parameters/'
        if not exists(path): makedirs(path)
        if exists(path+fname): remove(path+fname)

        param = self.sub_models[0].state_dict()
        with open(path+fname, 'wb') as file:
            dump(param, file, HIGHEST_PROTOCOL)
    
    def load_parameters(self, fname):
        path = './experiment/parameters/'
        with open(path+fname, 'rb') as file:
            params = load(file)

        for player in range(10):
            self.sub_models[player].load_state_dict(params)

class SubModel(nn.Module):
    def __init__(self, hidden_size, gru_layers):
        super(SubModel, self).__init__()
        self.gru = nn.GRU(30, hidden_size, gru_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, feature, h0):
        output, _ = self.gru(feature, h0)
        output = self.linear(output)
        output = self.tanh(output)
        return output

class BCEModel(Model):
    def __init__(self, hidden_size, gru_layers, lr, zero_h0):
        super(BCEModel, self).__init__(hidden_size, gru_layers, lr, zero_h0)

    def loss_func(self, blue_score, red_score, winner):
        device = blue_score.device.type
        criterion = nn.BCELoss()
        p_blue = exp(blue_score-red_score) / (exp(blue_score-red_score)+1)

        if winner == "blue" : target = tensor(1., device=device)
        else                : target = tensor(0., device=device)
        return criterion(p_blue, target)