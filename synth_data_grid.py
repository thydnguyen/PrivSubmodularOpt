import numpy as np
from scipy.spatial.distance import cdist as distance
import math
import random
import pandas as pd
import csv

import argparse
import time
parser = argparse.ArgumentParser()

parser.add_argument("--numIter", type = int, help = "Number of iterations for taking average", default = 20)
parser.add_argument("--gridsize", type = int, help = "Size of each grid dimension", default = 30)
parser.add_argument("--theta", type = float, help = "Theta", default = 0.2)
parser.add_argument("--epsilon", type = float, help = "Epsilon privacy")
parser.add_argument("--delta", type = float, help = "Theta privacy", default = 8.9e-8)
parser.add_argument("--K", type = float, help = "Number of centers as a fraction of the dataset")
parser.add_argument("--seed", type = int, help = "Seed", default = 2022)

parser.add_argument('--random', action='store_true')
parser.add_argument('--laplace', action='store_true')
parser.add_argument('--gumbel', action='store_true')
parser.add_argument('--nonprivate', action='store_true')

parser.add_argument('--batch', action='store_true')
parser.add_argument('--batchpriv', action='store_true')

parser.add_argument('--overwrite', action='store_true', help = "Whether to overwrite the output file")


args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

PATH = "synth.npz"
resultFileFormatted = 'result_synth_formatted.csv'
data = np.load(PATH)
F = data['x']



F_size = len(F)

theta, epsilon, delta =  args.theta, args.epsilon, args.delta

M = len(F[0]) * 40


low_x, low_y = -20 , -20
high_x, high_y = 20, 20
grid_size = args.gridsize

 
x_grid = np.linspace(low_x, high_x, num = grid_size)
y_grid = np.linspace(low_y, high_y, num = grid_size)
numRounds = grid_size ** 2
k = int(numRounds * args.K)



def exponential( scores, sensitivity, epsilon):

    # Calculate the probability for each element, based on its score
    probabilities = [np.exp(epsilon * score / (2 * sensitivity)) for score in scores]
    
    # Normalize the probabilties so they sum to 1
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

    # Choose an element from R based on the probabilities
    return np.random.choice(scores, 1, p=probabilities)[0]

def report_noisy_max( scores, sensitivity, epsilon):
    # Calculate the score for each element of R

    # Add noise to each score
    noisy_scores = [score + np.random.laplace(sensitivity/epsilon) for score in scores]

    # Find the index of the maximum score
    max_idx = np.argmax(noisy_scores)
    
    # Return the element corresponding to that index
    return scores[max_idx]

def report_noisy_max_batch( scores, sensitivity, epsilon):
    # Calculate the score for each element of R

    # Add noise to each score
    
    noisy_scores = [score[0] + np.random.laplace(sensitivity/epsilon) for score in scores]

    # Find the index of the maximum score
    max_idx = np.argmax(noisy_scores)
    
    # Return the element corresponding to that index
    return scores[max_idx]


def exponential_batch( scores, sensitivity, epsilon):
    scale = max(score[0] for score in scores)

    # Calculate the probability for each element, based on its score
    probabilities = [np.exp(epsilon * (score[0] - scale) / (2 * sensitivity)) for score in scores]

    # Normalize the probabilties so they sum to 1
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

    # Choose an element from R based on the probabilities
    return scores[np.random.choice(range(len(scores)), 1, p=probabilities)[0]]



def eval(F, S):
    distMatrix = distance(F,S, 'cityblock')
    return len(F) - sum(np.min(distMatrix, dim = 1))



best = 0
E = min(math.log(grid_size) * k /epsilon, F_size / 2)
numThresholds = math.ceil(math.log(F_size / E, 1+theta))


delta_p = delta / (2 * numThresholds)
epsilon_p = epsilon / (2* numThresholds)
gumbel_scale = (8/epsilon_p) * np.log(2 / (epsilon_p * delta_p)) / np.log(2)
laplace_scale = math.sqrt(32  * k * np.log(1/delta_p)) / epsilon_p

print("Num rounds:", numRounds)
print("F size:", F_size)
print("k:", k)
print("Num copies:", numThresholds )

for r in range(numRounds):
    stream_r = [[x_grid[r//grid_size], y_grid[r % grid_size]]]        
    dist_  = F_size - sum(distance(stream_r,F,  'cityblock')[0])/M
    if dist_ > best:
        best = dist_

best_ = min(E,best)

numThresholds_nonpriv = math.ceil( math.log(F_size / best_, 1+theta))
print("Num copies non-private:", numThresholds_nonpriv )
iters = list(range(numRounds))
random.shuffle(iters)
def master(method = 'gumbel'):
    random.shuffle(iters)
    if method == 'random':
        idx = np.array(random.sample(range(numRounds), k = k))
        centers = []
        for r in idx:
            center = [x_grid[r//grid_size], y_grid[r % grid_size]]
            centers.append(center)
        distMat = sum(np.min(distance(centers, F, 'cityblock'), axis = 0))
        return distMat, F_size - distMat/M

    result = []
    if method == 'gumbel':
        def noise_gain():
            return np.random.gumbel(scale = gumbel_scale)
        def noise_thres():
            return np.random.gumbel(scale = gumbel_scale)
    elif method == 'laplace':
        def noise_gain():
            return np.random.laplace(scale = 2*laplace_scale)
        def noise_thres():
            return np.random.laplace(scale = laplace_scale)
    elif method == 'nonprivate':
        def noise_gain():
            return 0
        def noise_thres():
            return 0
    if method == 'nonprivate':    
        for c in range(numThresholds_nonpriv+1):
            result.append(instance(best_  * (1+theta)**c / (2 * k) , noise_gain, noise_thres, True))
    else:
        for c in range(numThresholds+1):
            result.append(instance(F_size / ((1+theta)**c * (2 * k)) , noise_gain, noise_thres))

    if method == 'nonprivate':
        submod = max(result)
    else:
        submod = report_noisy_max(result, 1, epsilon / 2)
    return (F_size -submod) * M, submod
    

def instance(threshold, noise_gain, noise_thres, nonPriv = False):


    dist  = [math.inf] * len(F)
    dist_value = len(F) 
    count = 0
    noiseThreshold = noise_thres()
    selected = []

    for r_index, r in enumerate(iters):
        stream_r = [[x_grid[r//grid_size], y_grid[r % grid_size]]] 
        point_dist = distance(stream_r, F, 'cityblock')[0]
        new_dist = np.min([dist, point_dist], axis = 0)
        new_dist_value  = sum(new_dist) / M
        gain = dist_value - new_dist_value
        gain_noise = noise_gain()

        if  gain + gain_noise > threshold + noiseThreshold or (nonPriv and numRounds-r_index <= k-count) :
            noiseThreshold = noise_thres()
            dist = np.copy(new_dist)
            dist_value = new_dist_value
            count+=1
            selected.append(r)
            if count == k:
                return (F_size - dist_value) 
            
    return (F_size - dist_value)

def batch(private):
    selected = set()
    selected.add((0,0))
    dist  = distance([[x_grid[0], y_grid[0]]] ,F,  'cityblock')[0] 
    best_new_dist = np.copy(dist)
    
    for r in range(1,k):
        selected_list = []
        best_idx = (-1,-1)
        cur_gain = sum(dist) / M
        for i in range(grid_size):
            for j in range(grid_size):
                if (i,j) not in selected:
                    stream_r = [[x_grid[i], y_grid[j]]] 
                    new_dist  = np.min([distance(stream_r,F,  'cityblock')[0], dist], axis = 0)
                    new_gain = cur_gain - sum(new_dist) / M 
                    selected_list.append([new_gain, i,j])
        if private:
            _, i,j = exponential_batch(selected_list, 1, epsilon)
            best_idx = (i,j)
        else:
            best_gain, i, j = selected_list[0]
            for s in selected_list:
                if s[0] >= best_gain:
                    best_idx = (s[1], s[2])
                    best_gain = s[0]
            
        i,j = best_idx     
        stream_r = [[x_grid[i], y_grid[j]]] 
        best_new_dist = np.min([distance(stream_r,F,  'cityblock')[0], dist], axis = 0)
        dist = np.copy(best_new_dist)
        selected.add(best_idx)
    return sum(best_new_dist)
            
            
                        


result = []
result_full = []

if args.nonprivate:
    dist_nonpriv, submod_nonprive = master('nonprivate')
    
if args.batch:
    dist_nonpriv_batch = batch(False)
    

for i in range(args.numIter):
    result_ = []
    result_full_ = []
    if args.random:
        dist, submod = master('random')
        result_.append(dist)
        result_full_.append(submod)
    else:
        result_.append(np.nan)
        result_full_.append(np.nan)
        
    if args.laplace:
        dist, submod = master('laplace')
        result_.append(dist)
        result_full_.append(submod)
    else:
        result_.append(np.nan)
        result_full_.append(np.nan)
        
    if args.gumbel:
        dist, submod = master('gumbel')
        result_.append(dist)
        result_full_.append(submod)
    else:
        result_.append(np.nan)
        result_full_.append(np.nan)
        
    if args.nonprivate:
        result_.append(dist_nonpriv)
        result_full_.append(submod_nonprive)
    else:
        result_.append(np.nan)
        result_full_.append(np.nan)
    
    if args.batch:
        result_.extend([dist_nonpriv_batch])
    else:
        result_.extend([np.nan])
        
    if args.batchpriv:
        result_.extend([batch(True)])
    else:
        result_.extend([np.nan])

        
    result.append(result_)
    result_full.append(result_full_)
        
    
mean = np.mean(result, axis = 0)
std = np.std(result, axis = 0)
result = np.expand_dims(np.append(mean, std), 0)

mean_full = np.mean(result_full, axis = 0)
std_full = np.std(result_full, axis = 0)
result_full = np.expand_dims(np.append(mean_full, std_full), 0)

print("Mean:", mean)
print("Std:", std)
header = ["Params", "Random", "Laplace", "Ours", "Non-private", "BatchNon-private", "BatchPriv" , "RandomEB", "LaplaceEB", "OursEB", "Non-privateEB", "BatchNon-privateEB", "BatchPrivEB" ]
if args.overwrite:
    w = 'w'
else:
    w = 'a'

params = ["K-{} Espilon-{} Theta-{}".format(args.K, epsilon, theta)]
params_result = list(map(str, result[0].tolist()))
params.extend(params_result)

with open(resultFileFormatted,w) as fd:
    writer = csv.writer(fd, delimiter=',')
    if args.overwrite:
        writer.writerows([header, params])
    else:
        writer.writerows([params])
