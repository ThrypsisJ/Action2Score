from torch import nn, optim, zeros, ones, exp, tensor, cuda
from os.path import exists
from os import makedirs, remove
from pickle import load, dump, HIGHEST_PROTOCOL
from collections import OrderedDict

class Model():
    def __init__(self, dimensions, lr):
        self.model = SubModel(dimensions).to('cuda')
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
        for idx in range(len(scores)): scores[idx] = scores[idx].to('cpu').detach()
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
        cuda.empty_cache()
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

        param = self.model.state_dict()
        with open(path+fname, 'wb') as file:
            dump(param, file, HIGHEST_PROTOCOL)
    
    def load_parameters(self, fname):
        path = './experiment/parameters/'
        with open(path+fname, 'rb') as file:
            params = load(file)

        self.model.load_state_dict(params)

class SubModel(nn.Module):
    def __init__(self, linear_dims):
        super(SubModel, self).__init__()
        layers = OrderedDict()
        if len(linear_dims) > 1:
            for idx in range(len(linear_dims)-1):
                layers[f'{idx}'] = nn.Linear(linear_dims[idx], linear_dims[idx+1], bias=True)
                layers[f'act_{idx}'] = nn.LeakyReLU()
        layers['last'] = nn.Linear(linear_dims[-1], 1, bias=True)
        layers['out'] = nn.Tanh()

        self.sequence = nn.Sequential(layers)

    def forward(self, feature):
        output = self.sequence(feature)
        return output

class BCEModel(Model):
    def __init__(self, dimensions, lr):
        super(BCEModel, self).__init__(dimensions, lr)

    def loss_func(self, blue_score, red_score, winner):
        criterion = nn.BCELoss()
        p_blue = exp(blue_score-red_score) / (exp(blue_score-red_score)+1)

        if winner == "blue" : target = tensor(1., device='cuda')
        else                : target = tensor(0., device='cuda')
        return criterion(p_blue, target)