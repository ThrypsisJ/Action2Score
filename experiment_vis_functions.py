from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from tqdm import tqdm
from os.path import exists
from sklearn.metrics import roc_curve, auc
from itertools import accumulate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import random

path = './processed_ftr/'
modes = ('spa', 'g2', 'ch', 'las', 'challenger_spa')
mode_title = ['Spaghettini-ReLU', 'Spaghettini-BCE', 'Spaghettini-Chronological-ReLU', 'Lasagna-BCE', 'Spaghettini-ReLU']
mpath = (path+'scores_spaghettini.ftr', path+'scores_spaghettini_g2_lr0001_2layer.ftr', path+'scores_spaghettini_nf_lr0001_2layer.ftr', path+'scores_lasagna_g2.ftr', path+'scores_challenger.csv')

teams = ('both', 'blue', 'red')

elist = ['ITEM_PURCHASED', 'ITEM_SOLD', 'ITEM_DESTROYED', 'SKILL_LEVEL_UP', 'LEVEL_UP', 'WARD_PLACED', 'WARD_KILL', 'CHAMPION_KILL', 'CHAMPION_KILL_ASSIST', 'CHAMPION_KILL_VICTIM', 'BUILDING_KILL', 'BUILDING_KILL_ASSIST', 'ELITE_MONSTER_KILL', 'ELITE_MONSTER_KILL_ASSIST']
labels = ['ITEM_PURCHASED', 'ITEM_SOLD', 'ITEM_DESTROYED', 'SKILL_LEVEL_UP', 'LEVEL_UP', 'WARD_PLACED', 'WARD_KILL', 'CHAMPION_KILL', 'CHAMPION_KILL_ASSIST', 'CHAMPION_KILL_VICTIM', 'BUILDING_KILL', 'BUILDING_KILL_ASSIST', 'MONSTER_KILL', 'MONSTER_KILL_ASSIST']
eids = [idx for idx in range(0, len(elist))]

roles = ('TOP', 'MIDDLE', 'BOTTOM', 'UTILITY', 'JUNGLE')

def scores_per_event(save=False, model='spa'):
    print('Loading datas...')
    midx = modes.index(model)
    fpath = mpath[midx]
    tmp = pd.read_feather(fpath)

    print('Organizing datas...')
    scores, times = [], []

    for event in elist:
        pair = tmp[tmp[event]==1][['time', 'score']]
        pair = pair.sample(15000)
        scores.append(pair['score'].tolist())
        times.append(pair['time'].tolist())

    max_score, min_score = max([max(scores[i]) for i in eids]), min([min(scores[i]) for i in eids])
    norm = colors.Normalize(vmax=max_score, vmin=min_score)

    print("Creating Plot...")
    plt.figure(figsize=(9, 6))
    plt.xlabel('time')
    plt.ylabel('event')
    cmap = plt.cm.autumn_r
    
    for eid in eids:
        y = [eid for _ in range(len(times[eid]))]
        plt.scatter(times[eid], y, c=scores[eid], cmap=cmap, norm=norm, marker='|', s=300, alpha=0.2)
    
    plt.yticks(eids, labels)
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ticks=[0, 1])
    cbar.set_label(label='score', labelpad=-20)
    cbar.set_ticklabels(['%5.2f'%min_score, '%5.2f'%max_score])

    if save: plt.savefig('./visualization/score_per_event.png')
    else: plt.show()
    
def box_per_event(save=False, model='spa'):
    print('Loading datas...')
    midx = modes.index(model)
    fpath = mpath[midx]
    tmp = pd.read_feather(fpath)

    print('Creating figure...')
    fig = plt.figure(figsize=(7, 7))
    win_x_axis = [i+0.8 for i in range(len(elist))]
    lose_x_axis = [i+1.2 for i in range(len(elist))]

    winners = []
    losers = []
    for i, event in enumerate(elist):
        # Winners
        winner = tmp[(tmp['win'] == True) & (tmp[event] == 1)]['score'].tolist()
        loser = tmp[(tmp['win'] == False) & (tmp[event] == 1)]['score'].tolist()

        winners.append(winner)
        losers.append(loser)

    color = 'blue'
    plt.boxplot(winners, positions=win_x_axis, patch_artist=True, widths=0.3, vert=False, boxprops=dict(facecolor=color, color=color, alpha=0.3),
    capprops=dict(color=color, alpha=0.3), flierprops=dict(marker='|', markersize=1.3, color=color, markeredgecolor=color, alpha=0.1),
    medianprops=dict(color=(0, 1, 1)), whiskerprops=dict(color=color, alpha=0.3))

    color = 'red'
    plt.boxplot(losers, positions=lose_x_axis, patch_artist=True, widths=0.3, vert=False, boxprops=dict(facecolor=color, color=color, alpha=0.3),
    capprops=dict(color=color, alpha=0.3), flierprops=dict(marker='|', markersize=1.3, color=color, markeredgecolor=color, alpha=0.1),
    medianprops=dict(color=(0, 1, 1)), whiskerprops=dict(color=color, alpha=0.3))

    plt.yticks([i+1 for i in range(len(elist))], elist)
    plt.xlabel('Score')
    plt.ylabel('Event')

    legend_elements = [Patch(facecolor='blue', alpha=0.3, edgecolor='b', label='Winner Team'),
                        Patch(facecolor='red', alpha=0.3, edgecolor='r', label='Lost Team')]

    plt.legend(handles=legend_elements, loc='upper right')

    if save:
        plt.savefig(f'./visualization/Box-Score-Event.png')
        print('Figure saved')
    else: plt.show()

def match_score_timeline(save=False, models=['spa']):
    print('Loading datas...')
    mlist = pd.read_feather(path+'match_result_test.ftr')
    mlist = mlist['match_no'].tolist()
    mno = random.sample(mlist, 1)[0]
    
    print('Organazing datas...')
    data = {}
    for model in models:
        idx = modes.index(model)
        data[model] = {
            'path': mpath[idx],
            'title': mode_title[idx]
        }
        tmp = pd.read_feather(data[model]['path'])
        data[model]['data'] = tmp[tmp['match_id']==mno][['time', 'score', 'win']]
    
    timebin = [0.01*idx for idx in range(0, 100)]
    for model in models:
        data[model]['win'] = []
        data[model]['lose'] = []

        for tbin in timebin:
            tmp = data[model]['data']
            tmp = tmp[(tmp['time']>tbin) & (tmp['time']<=(tbin+0.01))]
            data[model]['win'].append(tmp[tmp['win']==True]['score'].sum())
            data[model]['lose'].append(tmp[tmp['win']==False]['score'].sum())

        data[model]['win'] = list(accumulate(data[model]['win']))
        data[model]['lose'] = list(accumulate(data[model]['lose']))
        
    print('Creating plots...')
    fig = plt.figure(figsize=(10*len(models), 10))

    for idx, model in enumerate(models):
        ax = fig.add_subplot(len(models), 1, idx+1, title='Team score transition - %s'%data[model]['title'])
        ax.plot(timebin, data[model]['win'], color='blue', label='Winner Team')
        ax.plot(timebin, data[model]['lose'], color='red', label='Lost Team')
        ax.legend(loc='upper left')
        ax.set(xlabel='time', ylabel='total team score')
    
    if save: plt.savefig(f'./visualization/match_score_per_time_{mno}_{models}.png')
    plt.show()

def color_mapper(score, isempty, maxscore, ctype='IR'):
    colors = []
    cmap = {
        'red': ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        'green': ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        'blue': ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    }
    for i in range(len(score)):
        if isempty[i]:
            r, g, b = 1.0, 1.0, 1.0
        elif ctype == 'IR':
            r = 2*(score[i]/maxscore)
            r = 1.0 if r > 1.0 else r
            g = 2*(score[i]/maxscore) - 0.5
            g = 0.0 if g < 0.0 else g
            g = 1.0 if g > 1.0 else g
            b = 1.0 - 2*(score[i]/maxscore)
            b = 0.0 if b < 0.0 else b
            cmap['red'] = ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0))
            cmap['green'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0))
            cmap['blue'] = ((0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
        elif ctype == 'RG':
            r = score[i]/maxscore
            g = 1 - (score[i]/maxscore)
            b = 0.0
            cmap['red'] = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            cmap['green'] = ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0))
            cmap['blue'] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        elif ctype == 'RB':
            r = score[i]/maxscore
            g = 0.0
            b = 1 - (score[i]/maxscore)
            cmap['red'] = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            cmap['green'] = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
            cmap['blue'] = ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0))
        else:
            r, g, b = 1.0, 1.0, 1.0
        colors.append((r, g, b))
    return colors, cmap

def score_by_position(save=False, event='CHAMPION_KILL', isWin=True, model='spa'):
    print('Loading datas...')
    mlist = pd.read_feather(path+'match_result_test.ftr')
    mlist = mlist['match_no'].tolist()
    mlist = random.sample(mlist, 2000)

    data = {}
    idx = modes.index(model)
    fpath = mpath[idx]
    tmp = pd.read_feather(fpath)
    # tmp = tmp[(tmp['match_id'].isin(mlist)) & (tmp['win']==isWin) & (tmp[event]==1) & (tmp['player']<5)]
    tmp = tmp[(tmp['win']==isWin) & (tmp[event]==1) & (tmp['player']<5)] # No sampling

    maxscore = tmp['score'].max()
    minscore = tmp['score'].min()
    data['max'] = maxscore
    data['min'] = minscore
    # data['norm'] = colors.Normalize(vmin=minscore, vmax=maxscore)
    data['data'] = tmp[['x_position', 'y_position', 'score']]
    # data['data'] = tmp[tmp[role]==1][['x_position', 'y_position', 'score']]

    binsize = 0.005
    bincount = int(1/binsize)
    slices = []
    # slicex, slicey, slicescore = [], [], []
    # max_count = 0
    for xi in range(bincount):
        sliced_x = data['data'].loc[(xi*binsize <= tmp['x_position']) & (tmp['x_position'] <= (xi+1)*binsize)]
        for yi in range(bincount):
            # sliced_data = tmp.loc[(xi*binsize <= tmp['x_position']) & (tmp['x_position'] <= (xi+1)*binsize) & (yi*binsize <= tmp['y_position']) & (tmp['y_position'] <= (yi+1)*binsize)]
            sliced_y = sliced_x.loc[(yi*binsize <= tmp['y_position']) & (tmp['y_position'] <= (yi+1)*binsize)]
            score = sliced_y['score'].mean()
            isempty = len(sliced_y['score']) == 0

            slices.append({
                    'x': binsize*(xi+0.5),
                    'y': binsize*(yi+0.5),
                    's': score,
                    'isempty': isempty
            })
            endstring = '\n' if (xi==(bincount-1) and yi==(bincount-1)) else '\r'
            print('Slicing datas: %04s/%s x, %04s/%s y'%(xi, bincount, yi, bincount), end=endstring)

    plt.figure(figsize=(9, 7))
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('x position')
    plt.ylabel('y position')

    slices = pd.DataFrame(slices)
    x, y, score, isempty = slices['x'].tolist(), slices['y'].tolist(), slices['s'].tolist(), slices['isempty'].tolist()

    colors, cmap = color_mapper(score, isempty, maxscore, ctype='IR')
    plt.scatter(x, y, s=4, c=colors)

    cmap = LinearSegmentedColormap('IR', cmap)
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ticks=[0, 1])
    cbar.set_label(label='score', labelpad=-20)
    cbar.set_ticklabels(['%5.2f'%min(score), '%5.2f'%max(score)])

    filename = './visualization/Score_map_per_%s'%(event)
    if save: plt.savefig(filename)
    else: plt.show()

def score_per_player(save=False, model='spa', match='None', events=[]):
    print('Loading datas...')
    if match=='None':
        mlist = pd.read_feather(path+'match_result_test.ftr')
        mlist = mlist['match_no'].tolist()
        match = random.sample(mlist, 1)[0]
    print(match)

    idx = modes.index(model)
    data = {
        'idx': idx,
        'path': mpath[idx],
        'title': mode_title[idx]
    }
    tmp = pd.read_feather(data['path']) if data['idx'] != 4 else pd.read_csv(data['path'])
    tmp = tmp[tmp['match_id'] == match]

    players = []
    for player in range(10):
        if model=='challenger_spa': player += 1
        isWin = tmp[tmp['player']==player]['win'].iloc[0]
        players.append({'isWin':isWin})

    if len(events) > 0:
        for idx, event in enumerate(events):
            if idx == 0: event_norm = (tmp[event]==1)
            else: event_norm = event_norm | (tmp[event]==1)
        tmp = tmp[event_norm]

    tmp.sort_values(by='time', ignore_index=True, inplace=True)
    for player in range(10):
        if model=='challenger_spa': player += 1
        player_tmp = tmp[tmp['player']==player]
        time = player_tmp['time'].tolist()
        scores = player_tmp['score'].tolist()
        scores = list(accumulate(scores))

        if model=='challenger_spa': player -= 1
        players[player]['time'] = time
        players[player]['scores'] = scores

    print('Creating plots...')
    fig = plt.figure(figsize=(6, 6))

    colors = ['red', 'green', 'blue', 'brown', 'orange']
    title = 'Score transition by time' + ('' if len(events) == 0 else f'\n{events}')
    ax = fig.add_subplot(1, 1, 1, title=title)

    for player in range(10):
        sx = players[player]['time']
        sy = players[player]['scores']
        isWin = players[player]['isWin']
        label = 'Player %i'%(player + 1) + (' (Win)' if isWin else ' (Lost)')
        ax.plot(sx, sy, c=colors[player%5], ls='-' if isWin else ':', label=label, alpha=0.5)

        ax.legend()
        ax.set_xlabel('time')
        ax.set_ylabel('scores')
    
    if save: plt.savefig('./visualization/score_transition_per_player.png')
    else: plt.show()

def event_count_per_player(save=False, model='spa', events=['CHAMPION_KILL'], match='None'):
    print('Loading datas...')
    if match=='None':
        mlist = pd.read_feather(path+'match_result_test.ftr')
        mlist = mlist['match_no'].tolist()
        match = random.sample(mlist, 1)[0]
    print(match)

    idx = modes.index(model)
    data = {
        'idx': idx,
        'path': mpath[idx],
        'title': mode_title[idx]
    }
    tmp = pd.read_feather(data['path']) if idx != 4 else pd.read_csv(data['path'])
    norm = tmp['match_id'] == match
    data['data'] = tmp[norm]

    for idx, event in enumerate(events):
        if idx == 0: norm = norm & (tmp[event]==1)
        else: norm = norm | (tmp[event]==1)

    data['players'] = []
    for player in range(10):
        isWin = data['data'][data['data']['player']==player+1]['win'].iloc[0]
        data['players'].append({'isWin':isWin})

    etmp = data['data'][norm][['player', 'time', 'win']].copy()
    etmp.sort_values(by=['time'], inplace=True)

    for player in range(10):
        count = etmp[etmp['player']==player+1]['time']
        time = count.tolist()
        
        count = [1 for _ in range(len(count))]
        count = list(accumulate(count))
        data['players'][player]['time'] = time
        data['players'][player]['counts'] = count

    print('Creating plots...')
    fig = plt.figure(figsize=(6, 6))

    colors = ['red', 'green', 'blue', 'brown', 'orange']
    ax = fig.add_subplot(1, 1, 1, title='%s'%events)

    for player in range(10):
        isWin = data['players'][player]['isWin']
        label = 'Player %i'%(player + 1) + (' (Win)' if isWin else ' (Lost)')
        ex = data['players'][player]['time']
        ey = data['players'][player]['counts']
        ax.plot(ex, ey, c=colors[player%5], ls='-' if isWin else ':', label=label, alpha=0.5)

    ax.legend()
    ax.set_xlabel('time')
    ax.set_ylabel('count')
    
    if save: plt.savefig('./visualization/score_transition_per_player.png')
    else: plt.show()

def score_of_player_per_event(save=False, model='spa', match='None', player=0):
    idx = modes.index(model)
    path = mpath[idx]

    print('Loading datas...')
    if match=='None':
        mlist = pd.read_feather(path+'match_result_test.ftr')
        mlist = mlist['match_no'].tolist()
        match = random.sample(mlist, 1)[0]
    print(match)

    tmp = pd.read_feather(path)
    tmp = tmp[(tmp['match_id'] == match) & (tmp['player']==player)]
    tmp.sort_values(by='time', inplace=True, ignore_index=True)

    events = {}
    for event in elist:
        event_tmp = tmp[tmp[event]==1]
        time = event_tmp['time'].tolist()
        scores = event_tmp['score'].tolist()
        scores = list(accumulate(scores))

        events[event] = (time, scores)

    print('Creating plots...')
    fig = plt.figure(figsize=(6, 6))

    # colors = ['red', 'green', 'blue', 'brown', 'orange', 'pink']
    title = f'Player {player+1}\'s score transition per event'
    ax = fig.add_subplot(1, 1, 1, title=title)

    for idx, event in enumerate(elist):
        sx = events[event][0]
        sy = events[event][1]
        ls = '-' if idx < 10 else ':'
        ax.plot(sx, sy, ls=ls, label=event, alpha=0.5)

        ax.legend()
        ax.set_xlabel('time')
        ax.set_ylabel('scores')
    
    if save: plt.savefig('./visualization/score_transition_per_player.png')
    else: plt.show()

def kill_score_by_position_per_role(event, save=False):
    roles = ['TOP', 'UTILITY', 'JUNGLE']
    print('Loading datas...')
    spa = pd.read_feather(path+'scores_spaghettini.ftr')
    spa_g2 = pd.read_feather(path+'scores_spaghettini_g2_lr0001_2layer.ftr')
    spa_ch = pd.read_feather(path+'scores_spaghettini_nf_lr0001_2layer.ftr')
    las = pd.read_feather(path+'scores_lasagna_g2.ftr')
    
    print('Organizing datas...')
    spa_x, spa_y, spa_score, spa_alpha = [], [], [], []
    spa_g2_x, spa_g2_y, spa_g2_score, spa_g2_alpha = [], [], [], []
    spa_ch_x, spa_ch_y, spa_ch_score, spa_ch_alpha = [], [], [], []
    las_x, las_y, las_score, las_alpha = [], [], [], []

    for role in roles:
        pairs = spa[(spa[role] == 1) & (spa[event] == 1)]
        blue = pairs[pairs['player']<5][['x_position', 'y_position', 'score']].sample(10000)
        red = pairs[pairs['player']>=5][['x_position', 'y_position', 'score']].sample(10000)
        x = blue['x_position'].tolist() + red['x_position'].tolist()
        y = blue['y_position'].tolist() + red['y_position'].tolist()
        score = blue['score'].tolist() + red['score'].tolist()
        spa_x.append(x)
        spa_y.append(y)
        spa_score.append(score)
        spa_alpha.append([s/max(score) for s in score])
        
        pairs = spa_g2[(spa_g2[role] == 1) & (spa[event] == 1)]
        blue = pairs[pairs['player']<5][['x_position', 'y_position', 'score']].sample(10000)
        red = pairs[pairs['player']>=5][['x_position', 'y_position', 'score']].sample(10000)
        x = blue['x_position'].tolist() + red['x_position'].tolist()
        y = blue['y_position'].tolist() + red['y_position'].tolist()
        score = blue['score'].tolist() + red['score'].tolist()
        spa_g2_x.append(x)
        spa_g2_y.append(y)
        spa_g2_score.append(score)
        spa_g2_alpha.append([s/max(score) for s in score])
        
        pairs = spa_ch[(spa_ch[role] == 1) & (spa[event] == 1)]
        blue = pairs[pairs['player']<5][['x_position', 'y_position', 'score']].sample(10000)
        red = pairs[pairs['player']>=5][['x_position', 'y_position', 'score']].sample(10000)
        x = blue['x_position'].tolist() + red['x_position'].tolist()
        y = blue['y_position'].tolist() + red['y_position'].tolist()
        score = blue['score'].tolist() + red['score'].tolist()
        spa_ch_x.append(x)
        spa_ch_y.append(y)
        spa_ch_score.append(score)
        spa_ch_alpha.append([s/max(score) for s in score])
        
        pairs = las[(las[role] == 1) & (spa[event] == 1)]
        blue = pairs[pairs['player']<5][['x_position', 'y_position', 'score']].sample(10000)
        red = pairs[pairs['player']>=5][['x_position', 'y_position', 'score']].sample(10000)
        x = blue['x_position'].tolist() + red['x_position'].tolist()
        y = blue['y_position'].tolist() + red['y_position'].tolist()
        score = blue['score'].tolist() + red['score'].tolist()
        las_x.append(x)
        las_y.append(y)
        las_score.append(score)
        las_alpha.append([s/max(score) for s in score])

    print('Creating Plot...')
    fig = plt.figure(figsize=(60, 10*len(roles)))
    
    color = ['blue' if i<10000 else 'red' for i in range(20000)]
    for idx, role in enumerate(roles):
        spa_ax = fig.add_subplot(len(roles), 4, (4*idx+1), title=f'{role}-Spaghettini-ReLU')
        spa_g2_ax = fig.add_subplot(len(roles), 4, (4*idx+2), title=f'{role}-Spaghettini-BCE')
        spa_ch_ax = fig.add_subplot(len(roles), 4, (4*idx+3), title=f'{role}-Spaghettini-chronological-ReLU')
        las_ax = fig.add_subplot(len(roles), 4, (4*idx+4), title=f'{role}-lasagna-BCE')
        
        spa_ax.scatter(x=spa_x[idx], y=spa_y[idx], color=color, alpha=spa_alpha[idx], marker='.')
        spa_cbar_w = fig.colorbar(cm.ScalarMappable(cmap=cm.Blues), ax=spa_ax, ticks=[0, 1], location='left')
        spa_cbar_l = fig.colorbar(cm.ScalarMappable(cmap=cm.Reds), ax=spa_ax, ticks=[0, 1], location='right')
        spa_cbar_w.ax.set_yticklabels(['low', 'high'])
        spa_cbar_l.ax.set_yticklabels(['low', 'high'])
        spa_ax.set(xlabel='x-position', ylabel='y-position')
    
        spa_g2_ax.scatter(x=spa_g2_x[idx], y=spa_g2_y[idx], color=color, alpha=spa_g2_alpha[idx], marker='.')
        spa_g2_cbar_w = fig.colorbar(cm.ScalarMappable(cmap=cm.Blues), ax=spa_g2_ax, ticks=[0, 1], location='left')
        spa_g2_cbar_l = fig.colorbar(cm.ScalarMappable(cmap=cm.Reds), ax=spa_g2_ax, ticks=[0, 1], location='right')
        spa_g2_cbar_w.ax.set_yticklabels(['low', 'high'])
        spa_g2_cbar_l.ax.set_yticklabels(['low', 'high'])
        spa_g2_ax.set(xlabel='x-position', ylabel='y-position')
        
        spa_ch_ax.scatter(x=spa_ch_x[idx], y=spa_ch_y[idx], color=color, alpha=spa_ch_alpha[idx], marker='.')
        spa_ch_cbar_w = fig.colorbar(cm.ScalarMappable(cmap=cm.Blues), ax=spa_ch_ax, ticks=[0, 1], location='left')
        spa_ch_cbar_l = fig.colorbar(cm.ScalarMappable(cmap=cm.Reds), ax=spa_ch_ax, ticks=[0, 1], location='right')
        spa_ch_cbar_w.ax.set_yticklabels(['low', 'high'])
        spa_ch_cbar_l.ax.set_yticklabels(['low', 'high'])
        spa_ch_ax.set(xlabel='x-position', ylabel='y-position')
        
        las_ax.scatter(x=las_x[idx], y=las_y[idx], color=color, alpha=las_alpha[idx], marker='.')
        las_cbar_w = fig.colorbar(cm.ScalarMappable(cmap=cm.Blues), ax=las_ax, ticks=[0, 1], location='left')
        las_cbar_l = fig.colorbar(cm.ScalarMappable(cmap=cm.Reds), ax=las_ax, ticks=[0, 1], location='right')
        las_cbar_w.ax.set_yticklabels(['low', 'high'])
        las_cbar_l.ax.set_yticklabels(['low', 'high'])
        las_ax.set(xlabel='x-position', ylabel='y-position')
        
    plt.show()

def score_time_hist(event, save=False):
    print('Loading datas...')
    spa = pd.read_feather(path+'scores_spaghettini.ftr')
    spa_g2 = pd.read_feather(path+'scores_spaghettini_g2_lr0001_2layer.ftr')
    spa_ch = pd.read_feather(path+'scores_spaghettini_nf_lr0001_2layer.ftr')
    las = pd.read_feather(path+'scores_lasagna_g2.ftr')

    print('Organazing datas...')
    pair = spa[spa[event] == 1][['time', 'score']]
    pair = pair.sample(20000)
    spa_scores = pair['score'].tolist()
    spa_times = pair['time'].tolist()
    
    pair = spa_g2[spa_g2[event] == 1][['time', 'score']]
    pair = pair.sample(20000)
    spa_g2_scores = pair['score'].tolist()
    spa_g2_times = pair['time'].tolist()
    
    pair = spa_ch[spa_ch[event] == 1][['time', 'score']]
    pair = pair.sample(20000)
    spa_ch_scores = pair['score'].tolist()
    spa_ch_times = pair['time'].tolist()
    
    pair = las[las[event] == 1][['time', 'score']]
    pair = pair.sample(20000)
    las_scores = pair['score'].tolist()
    las_times = pair['time'].tolist()

    print('Creating figure...')
    # definitions for the axes
    left = [0.05, 0.55]
    bottom = [0.05, 0.55]
    long, short = 0.3, 0.1
    space = 0.005

    fig = plt.figure(figsize=(20, 20))

    # SPA
    spa_rect_scat = [left[0], bottom[1], long, long]
    spa_rect_time = [left[0], bottom[1]+long+space, long, short]
    spa_rect_score = [left[0]+long+space, bottom[1], short, long]

    spa_ax = fig.add_axes(spa_rect_scat)
    spa_ax_time = fig.add_axes(spa_rect_time)
    spa_ax_score = fig.add_axes(spa_rect_score)
    spa_ax_time.tick_params(axis='x', labelbottom=False)
    spa_ax_score.tick_params(axis='y', labelleft=False)

    # SPA_G2
    spa_g2_rect_scat = [left[1], bottom[1], long, long]
    spa_g2_rect_time = [left[1], bottom[1]+long+space, long, short]
    spa_g2_rect_score = [left[1]+long+space, bottom[1], short, long]

    spa_g2_ax = fig.add_axes(spa_g2_rect_scat)
    spa_g2_ax_time = fig.add_axes(spa_g2_rect_time)
    spa_g2_ax_score = fig.add_axes(spa_g2_rect_score)
    spa_g2_ax_time.tick_params(axis='x', labelbottom=False)
    spa_g2_ax_score.tick_params(axis='y', labelleft=False)

    # SPA_CH
    spa_ch_rect_scat = [left[0], bottom[0], long, long]
    spa_ch_rect_time = [left[0], bottom[0]+long+space, long, short]
    spa_ch_rect_score = [left[0]+long+space, bottom[0], short, long]

    spa_ch_ax = fig.add_axes(spa_ch_rect_scat)
    spa_ch_ax_time = fig.add_axes(spa_ch_rect_time)
    spa_ch_ax_score = fig.add_axes(spa_ch_rect_score)
    spa_ch_ax_time.tick_params(axis='x', labelbottom=False)
    spa_ch_ax_score.tick_params(axis='y', labelleft=False)

    # LAS
    las_rect_scat = [left[1], bottom[0], long, long]
    las_rect_time = [left[1], bottom[0]+long+space, long, short]
    las_rect_score = [left[1]+long+space, bottom[0], short, long]

    las_ax = fig.add_axes(las_rect_scat)
    las_ax_time = fig.add_axes(las_rect_time)
    las_ax_score = fig.add_axes(las_rect_score)
    las_ax_time.tick_params(axis='x', labelbottom=False)
    las_ax_score.tick_params(axis='y', labelleft=False)

    print('Drawing axes...')
    # Draw axes
    bins, alpha, color = 100, 0.2, 'blue'
    spa_ax.scatter(spa_times, spa_scores, alpha=alpha, color=color, marker='.')
    spa_ax_time.hist(spa_times, bins=bins, color=color)
    spa_ax_score.hist(spa_scores, bins=bins, color=color, orientation='horizontal')
    spa_ax.set_xlabel('time'); spa_ax.set_ylabel('score')

    spa_g2_ax.scatter(spa_g2_times, spa_g2_scores, alpha=alpha, color=color, marker='.')
    spa_g2_ax_time.hist(spa_g2_times, bins=bins, color=color)
    spa_g2_ax_score.hist(spa_g2_scores, bins=bins, color=color, orientation='horizontal')
    spa_g2_ax.set_xlabel('time'); spa_g2_ax.set_ylabel('score')

    spa_ch_ax.scatter(spa_ch_times, spa_ch_scores, alpha=alpha, color=color, marker='.')
    spa_ch_ax_time.hist(spa_ch_times, bins=bins, color=color)
    spa_ch_ax_score.hist(spa_ch_scores, bins=bins, color=color, orientation='horizontal')
    spa_ch_ax.set_xlabel('time'); spa_ch_ax.set_ylabel('score')

    las_ax.scatter(las_times, las_scores, alpha=alpha, color=color, marker='.')
    las_ax_time.hist(las_times, bins=bins, color=color)
    las_ax_score.hist(las_scores, bins=bins, color=color, orientation='horizontal')
    las_ax.set_xlabel('time'); las_ax.set_ylabel('score')

    if save:
        plt.savefig(f'./visualization/Time-Score histogram - {event}.png')
        print('Figure saved')
    else: plt.show()

def score_hist(event, save=False, factor='time'):
    print('Loading datas...')
    
    data = {}
    scores = {}
    for fpath, mode in zip(mpath, modes):
        print(mode)
        tmp = pd.read_feather(fpath)
        tmp = tmp[tmp[event]==1]

        data[mode] = {}
        data[mode]['win'] = tmp[tmp['win']==True].sample(10000)
        data[mode]['lose'] = tmp[tmp['win']==False].sample(10000)

    print('Organazing datas...')
    
    for mode in modes:
        print('Organizing %s...'%mode)
        scores[mode] = {}
        pair = data[mode]['win']
        pair = pair[[factor, 'score']]
        time = pair[factor].tolist()
        score = pair['score'].tolist()
        scores[mode]['win'] = (time, score)

        pair = data[mode]['lose']
        pair = pair[[factor, 'score']]
        time = pair[factor].tolist()
        score = pair['score'].tolist()
        scores[mode]['lose'] = (time, score)

    print('Creating figure...')
    # definitions for the axes
    left = [0.05, 0.55]
    bottom = [0.05, 0.55]
    long, short = 0.3, 0.1
    space = 0.005

    fig = plt.figure(figsize=(20, 20))

    graph = {}
    for idx, mode in enumerate(modes):
        rect_scat = [left[idx%2], bottom[1-int(idx/2)], long, long]
        rect_time = [left[idx%2], bottom[1-int(idx/2)]+long+space, long, short]
        rect_score = [left[idx%2]+long+space, bottom[1-int(idx/2)], short, long]

        graph[mode] = []
        graph[mode].append(fig.add_axes(rect_scat))
        graph[mode].append(fig.add_axes(rect_time))
        graph[mode].append(fig.add_axes(rect_score))
        graph[mode][1].set_title(mode_title[idx])
        graph[mode][1].tick_params(axis='x', labelbottom=False)
        graph[mode][2].tick_params(axis='y', labelleft=False)

    print('Drawing axes...')
    # Draw axes
    bins, alpha, win_color, lose_color, marker = 100, 0.3, 'blue', 'red', '.'
    for mode in modes:
        for key in scores[mode].keys():
            time, score = scores[mode][key]
            color = win_color if key=='win' else lose_color
            graph[mode][0].scatter(time, score, alpha=alpha, color=color, marker=marker)
            graph[mode][1].hist(time, bins=bins, alpha=alpha, color=color)
            graph[mode][2].hist(score, bins=bins, alpha=alpha, color=color, orientation='horizontal')
        graph[mode][0].set_xlabel(factor)
        graph[mode][0].set_ylabel('score')

    if save:
        plt.savefig(f'./visualization/Score histogram-{event}-{factor}.png')
        print('Figure saved')
    else: plt.show()

def drawbox(xdatas, color, bias, fcluster):
    position = [i+bias for i in range(fcluster)]
    plt.boxplot(xdatas, positions=position, patch_artist=True, widths=0.4,
                boxprops=dict(facecolor=color, color=color, alpha=0.3),
                capprops=dict(color=color, alpha=0.3),
                whiskerprops=dict(color=color, alpha=0.3),
                flierprops=dict(color=color, markeredgecolor=color, alpha=0.1),
                medianprops=dict(color=(0,1,1)))

def score_factor_box(event='CHAMPION_KILL', save=False, factor='time', model='spa', sample=60000):
    data = {}

    specials = ['ELITE_MONSTER_KILL', 'ELITE_MONSTER_KILL_ASSIST', 'BUILDING_KILL', 'BUILDING_KILL_ASSIST', 'SKILL_LEVEL_UP']
    tslice = 0.1 if (event not in specials) or (factor != 'event_weight') else 0.2
    tcluster = int(1/tslice)
    tsample = int(sample/tcluster)
    
    print('Loading datas...')

    midx = modes.index(model)
    fpath = mpath[midx]

    tmp = pd.read_feather(fpath)
    tmp = tmp[tmp[event]==1][['win', factor, 'score']]
    data[model] = {'midx': midx, 'win': [], 'lose': []}

    for i in range(tcluster):
        tmin, tmax = tslice * i, tslice * (i+1)

        tchunk = tmp[(tmp[factor]>=tmin) & (tmp[factor]<tmax)]
        tmp_chunk = tchunk[tchunk['win']==True]['score']
        if tmp_chunk.shape[0] == 0: data[model]['win'].append([])
        elif tmp_chunk.shape[0] >= tsample: data[model]['win'].append(tmp_chunk.sample(tsample).tolist())
        else: data[model]['win'].append(tmp_chunk.sample(tmp_chunk.shape[0]).tolist())

        tmp_chunk = tchunk[tchunk['win']==False]['score']
        if tmp_chunk.shape[0] == 0: data[model]['lose'].append([])
        elif tmp_chunk.shape[0] >= tsample: data[model]['lose'].append(tmp_chunk.sample(tsample).tolist())
        else: data[model]['lose'].append(tmp_chunk.sample(tmp_chunk.shape[0]).tolist())

    print('Creating figure...')
    plt.figure(figsize=(7, 7))

    drawbox(data[model]['win'], 'blue', 0, tcluster)
    drawbox(data[model]['lose'], 'red', 0.5, tcluster)
    plt.xticks([i for i in range(tcluster)], ['%3.2f'%(tslice*i) for i in range(tcluster)])
    plt.xlabel(factor)
    plt.ylabel('score')

    legend_elements = [Patch(facecolor='blue', alpha=0.3, edgecolor='b', label='Winner Team'),
                        Patch(facecolor='red', alpha=0.3, edgecolor='r', label='Lost Team')]

    plt.legend(handles=legend_elements, loc='upper right')

    if save:
        plt.savefig(f'./visualization/Box-{event}-{factor}-{model}.png')
        print('Figure saved')
    else: plt.show()

def score_factor_box_role(event='CHAMPION_KILL', blue=True, factor='time', roles=['JUNGLE'], sample=60000, save=False):
    if len(roles)>2: print('Only 2 roles can compared!')

    data = {}

    fslice = 0.05
    fcluster = int(1/fslice)
    fsample = int(sample/fcluster)
    
    print('Loading datas...')
    fpath = mpath[0]
    title = mode_title[0]
    tmp = pd.read_feather(fpath)

    for role in roles:
        data[role] = {'scores': []}
        if blue: data[role]['data'] = tmp[(tmp[event]==1) & (tmp[role]==1) & (tmp['player']<5)][[factor, 'score']]
        else: data[role]['data'] = tmp[(tmp[event]==1) & (tmp[role]==1) & (tmp['player']>=5)][[factor, 'score']]

        for i in range(fcluster):
            fmin, fmax = fslice * i, fslice * (i+1)

            fchunk = tmp[(tmp[factor]>=fmin) & (tmp[factor]<fmax)]
            tmp_chunk = fchunk['score']

            if tmp_chunk.shape[0] == 0: data[role]['scores'].append([])
            elif tmp_chunk.shape[0] >= fsample: data[role]['scores'].append(tmp_chunk.sample(fsample).tolist())
            else: data[role]['scores'].append(tmp_chunk.sample(tmp_chunk.shape[0]).tolist())

    print('Creating figure...')
    plt.figure(figsize=(10, 10))

    colors = ['green', 'orange']
    for idx, role in enumerate(roles):
        drawbox(data[role]['scores'], colors[idx], 0.5*idx, fcluster)

    plt.title(title)
    plt.xticks([i for i in range(fcluster)], ['%3.2f'%(fslice*i) for i in range(fcluster)])
    plt.xlabel(factor)
    plt.ylabel('score')

    legend_elements = [Patch(facecolor=colors[0], alpha=0.3, edgecolor=colors[0], label=roles[0]),
                        Patch(facecolor=colors[1], alpha=0.3, edgecolor=colors[1], label=roles[1])]
    plt.legend(handles=legend_elements, loc='upper right')

    if save:
        plt.savefig(f'./visualization/Box-{event}-{factor}-{roles}.png')
        print('Figure saved')
    else: plt.show()

def score_factor_box_rnb(event='CHAMPION_KILL', location='JUNGLE', role='assassin', fix = {'event_weight': (0.45, 0.55),}, factor='time', sample=60000, save=False):
    data = {}

    fslice = 0.05
    fcluster = int(1/fslice)
    fsample = int(sample/fcluster)
    
    print('Loading datas...')
    fpath = mpath[0]
    title = mode_title[0]
    tmp = pd.read_feather(fpath)

    rangeValues = ['time', 'x_position', 'y_position', 'deviation', 'event_weight']
    rangeValues.remove(factor)

    criterion = (tmp[event] == 1)
    if location != 'all': criterion = criterion & (tmp[location] == 1)
    if role != 'all': criterion = criterion & (tmp[role] == 1)

    for value in rangeValues:
        minvalue = 0.45 if value not in fix.keys() else fix[value][0]
        maxvalue = 0.55 if value not in fix.keys() else fix[value][1]
        criterion = criterion & (minvalue <= tmp[value]) & (tmp[value] <= maxvalue)
    tmp = tmp[criterion]

    for team in ('blue', 'red'):
        data[team] = {'scores': []}
        if team == 'blue': data[team]['data'] = tmp[(tmp[event]==1) & (tmp['player']<5)][[factor, 'score']]
        else: data[team]['data'] = tmp[(tmp[event]==1) & (tmp['player']>=5)][[factor, 'score']]

        fpointer = 0
        while fpointer < 1:
            fmin, fmax = fpointer, fpointer+0.05

            if fmax < 1: fchunk = tmp[(tmp[factor]>=fmin) & (tmp[factor]<fmax)]
            else: fchunk = tmp[(tmp[factor]>=fmin) & (tmp[factor]<=fmax)]
            tmp_chunk = fchunk['score']

            if tmp_chunk.shape[0] == 0: data[team]['scores'].append([])
            elif tmp_chunk.shape[0] >= fsample: data[team]['scores'].append(tmp_chunk.sample(fsample).tolist())
            else: data[team]['scores'].append(tmp_chunk.sample(tmp_chunk.shape[0]).tolist())

            fpointer += 0.05

    print('Creating figure...')
    plt.figure(figsize=(10, 10))

    colors = ['blue', 'red']
    for idx, team in enumerate(('blue', 'red')):
        drawbox(data[team]['scores'], colors[idx], 0.5*idx, fcluster)

    plt.title(title)
    plt.xticks([i for i in range(fcluster)], ['%3.2f'%(fslice*i) for i in range(fcluster)])
    plt.xlabel(factor)
    plt.ylabel('score')

    legend_elements = [Patch(facecolor=colors[0], alpha=0.3, edgecolor=colors[0], label='Blue Team'),
                        Patch(facecolor=colors[1], alpha=0.3, edgecolor=colors[1], label='Red Team')]
    plt.legend(handles=legend_elements, loc='upper right')

    if save:
        plt.savefig(f'./visualization/TeamBox-{event}-{factor}.png')
        print('Figure saved')
    else: plt.show()

def score_by_multiple(event, save=False, factor=['deviation', 'event_weight']):
    print('Loading datas...')
    
    data = {}
    scores = {}
    for fpath, mode in zip(mpath, modes):
        print(mode)
        tmp = pd.read_feather(fpath)
        data[mode] = tmp.sample(20000 if len(factor)==2 else 10000)

    print('Organazing datas...')
    
    cols = factor + ['score']
    for mode in modes:
        print('Organizing %s...'%mode)
        scores[mode] = {}
        pair = data[mode]
        pair = pair[pair[event]==1][cols]

        tmp = []
        for key in pair.keys():
            tmp.append(pair[key].tolist())
        scores[mode] = tmp

    print('Creating figure...')

    fig = plt.figure(figsize=(25, 20))

    if len(factor) == 3:
        graph = {}
        for mid, mode in enumerate(modes):
            graph[mode] = fig.add_subplot(2, 2, (mid+1), projection='3d')
            x = scores[mode][0]
            y = scores[mode][1]
            z = scores[mode][2]
            c = scores[mode][3]

            graph[mode].scatter(x, y, z, cmap=cm.autumn_r, c=c, marker='o')
            graph[mode].bar3d(x, y, 0, 0.001, 0.001, z, color='grey', alpha=0.1)
            graph[mode].set(xlabel=factor[0], ylabel=factor[1], zlabel=factor[2])
            cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.autumn_r), ax=graph[mode], ticks=[0, 1], location='left')
            cbar.ax.set_yticklabels(['%6.3f'%min(c), '%6.3f'%max(c)])
    else:
        graph = {}
        for mid, mode in enumerate(modes):
            graph[mode] = fig.add_subplot(2, 2, (mid+1))
            x = scores[mode][0]
            y = scores[mode][1]
            c = scores[mode][2]

            graph[mode].scatter(x, y, cmap=cm.autumn_r, c=c, marker='.')
            graph[mode].set(xlabel=factor[0], ylabel=factor[1])
            cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.autumn_r), ax=graph[mode], ticks=[0, 1], location='left')
            cbar.ax.set_yticklabels(['%6.3f'%min(c), '%6.3f'%max(c)])

    if save:
        plt.savefig(f'./visualization/score_by_two-{event}-{factor}.png')
        print('Figure saved')
    else: plt.show()

def build_roc_datas(*models):
    for model in models:
        print(f'roc data for model {model} is being built...')
        file_name = path+f'for_roc_{model}.csv'

        results = pd.read_feather(path+f'match_result_test.ftr')
        winners = results['win'].tolist()
        mlist = results['match_no'].tolist()

        data = pd.read_feather(path+f'scores_{model}.ftr')
        data = data[['match_id', 'player', 'score']]

        matches = {}
        for _, row in tqdm(data.iterrows(), ncols=50, total=data.shape[0]):
            if not row['match_id'] in matches:
                matches[row['match_id']] = {
                    'blue': 0, 'red':0
                }

            if row['player'] < 5: matches[row['match_id']]['blue'] += row['score']
            else: matches[row['match_id']]['red'] += row['score']

        roc_file = {
            'match_id': [], 'blue': [], 'red': [], 'ratio': [], 'winner': [], 'winner_digit': []
        }

        for match, winner in zip(mlist, winners):
            blue = matches[match]['blue']
            red = matches[match]['red']
            ratio = blue / (blue+red)
            winner_digit = 1 if winner == 'blue' else 0
            
            roc_file['match_id'].append(match)
            roc_file['blue'].append(blue)
            roc_file['red'].append(red)
            roc_file['ratio'].append(ratio)
            roc_file['winner'].append(winner)
            roc_file['winner_digit'].append(winner_digit)

        roc_file = pd.DataFrame(roc_file)
        if not exists(file_name):
                roc_file.to_csv(file_name, mode="w", encoding="utf-8", index=False)
        else:
                roc_file.to_csv(file_name, mode="a", encoding="utf-8", index=False, header=False)

def draw_roc_curve():
    models = ['spaghettini', 'spaghettini_g2_lr0001_2layer', 'spaghettini_nf_lr0001_2layer', 'spaghettini_nf_g2_lr0001_2layer', 'lasagna', 'lasagna_g2']
    mnames = ['spaghettini-ReLU', 'spaghettini-BCE', 'spaghettini-ch-ReLU', 'spaghettini-ch-BCE', 'lasagna-ReLU', 'lasagna-BCE']
    colors = ['red', 'blue', 'yellow', 'purple', 'green', 'orange', 'black']
    
    roc_datas = {}
    plt.figure(figsize=(6, 6))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    curve = {}
    for idx, values in enumerate(zip(models, mnames)):
        model, name = values[0], values[1]
        file = pd.read_csv('./processed_ftr/for_roc_'+model+'.csv')

        roc_datas[model] = {
            'match_id': file['match_id'].tolist(),
            'blue': file['blue'].tolist(),
            'red': file['red'].tolist(),
            'ratio': file['ratio'].tolist(),
            'winner': file['winner'].tolist(),
            'winner_digit': file['winner_digit'].tolist()
        }
        fpr, tpr, _ = roc_curve(roc_datas[model]['winner_digit'], roc_datas[model]['ratio'], pos_label=1)
        area = auc(fpr, tpr)
        curve[model] = plt.plot(fpr, tpr, color=colors[idx], label='%s, AUC=%.4f'%(name, area))
    
    plt.plot([0, 1], [0, 1], colors[6], label='random guess', linestyle='dashed')
    plt.legend()
    plt.show()