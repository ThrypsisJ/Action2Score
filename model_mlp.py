from torch import nn, optim, zeros, ones, tan
from os.path import exists
from os import makedirs, remove
from pickle import load, dump, HIGHEST_PROTOCOL

class Model():
    def __init__(self, dimensions, lr):
        layers = []
        for idx, dimension in enumerate(dimensions):
            if idx == 0:
                layers.append(nn.Linear(30, dimension))
                layers.append(nn.LeakyReLU)
                prev_dim = dimension
            elif idx+1 != len(layers):
                layers.append(nn.Linear(prev_dim, dimension))
                layers.append(nn.LeakyReLU)
                prev_dim = dimension
            else:
                layers.append(nn.Linear(prev_dim, dimension))
                layers.append(nn.Sigmoid)
        self.model = nn.Sequential(*layers)
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, features, winner):
        self.model.train()

        scores = self.proceed(features)
        blue_total = sum([scores[idx].sum() for idx in range(0, 5)])
        red_total = sum([scores[idx].sum() for idx in range(5, 10)]) 
        loss = self.loss_func(blue_total, red_total, winner)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def validate(self, features, winner):
        self.model.eval()

        scores = self.proceed(features)
        blue_total = sum([scores[idx].sum() for idx in range(0, 5)])
        red_total = sum([scores[idx].sum() for idx in range(5, 10)])
        self.test_loss += self.loss_func(blue_total, red_total, winner)

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

    def proceed(self, features):
        scores = []
        for player in range(10):
            feature = features[player].to('cuda')
            if feature.shape[0] == 0: feature = zeros(1, 30, device='cuda')
            score = self.model(feature)
            scores.append(score)
        return scores
        
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