from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import gym
import numpy as np
import sys
import os
from numpy.random import choice
import pandas as pd
import seaborn as sns
from hiive.mdptoolbox.example import forest
import gym
np.random.seed(65)


prob, reward = forest(S=600, r1=150, r2= 15, p=0.01)

def adds(x, num):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[num:] - cumsum[:-num]) / float(num)

def tQ(prob, rew, disc=0.9, alpha_dec=[0.99], alpha_min=[0.001], 
            epsilon=[1.0], epsilon_decay=[0.99], niter=[1000000]):
    qdf = pd.DataFrame(columns=["Iters", "Alpha Decay", "Alpha Min", 
                                 "Epsilon", "Epsilon Decay", "Time","Reward",
                                  "Policy", "Value Function",
                                 "Training Rewards"])
    
    count = 0
    for i in niter:
        for eps in epsilon:
            for eps_dec in epsilon_decay:
                for a_dec in alpha_dec:
                    for a_min in alpha_min:
                        q = QLearning(prob, rew, disc, alpha_decay=a_dec, 
                                      alpha_min=a_min, epsilon=eps, 
                                      epsilon_decay=eps_dec, n_iter=i)
                        q.run()
                        reward = tp(prob, rew, q.policy)
                        count += 1
                        print("{}: {}".format(count, reward))
                        st = q.run_stats
                        rews = [s['Reward'] for s in st]
                        info = [i, a_dec, a_min, eps, eps_dec, q.time, reward, 
                                q.policy, q.V, rews]
                        
                        df_length = len(qdf)
                        qdf.loc[df_length] = info
    return qdf

def tVI(prob, rew, disc=0.9, epsilon=[1e-9]):
    vidf = pd.DataFrame(columns=["Policy", "Epsilon", "Iter", 
                                  "Time", "Reward", "Val Function"])
    for eps in epsilon:
        vi = ValueIteration(prob, rew, gamma=disc, epsilon=eps, max_iter=int(1e15))
        vi.run()
        reward = tp(prob, rew, vi.policy)
        info = [ vi.policy, float(eps), vi.iter, vi.time, reward, vi.V]
        df_len = len(vidf)
        vidf.loc[df_len] = info
    return vidf

def tp(prob, rew, pol, tc=100, gamma=0.9):
    ns = prob.shape[-1]
    te = ns * tc
    tr = 0
    for sta in range(ns):
        sr = 0
        for sta_episode in range(tc):
            episode_reward = 0
            disc_rate = 1
            while True:
                action = pol[sta]
                probs = prob[action][sta]
                candidates = list(range(len(prob[action][sta])))
                next_sta =  choice(candidates, 1, p=probs)[0]
                reward = rew[sta][action] * disc_rate
                episode_reward += reward
                disc_rate *= gamma
                if next_sta == 0:
                    break
            sr += episode_reward
        tr += sr
    return tr / te

def runvi():
    eps_list = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12]
    value_iter_df = tVI(prob, reward, epsilon=eps_list)
    value_iter_df.to_html('files/VI_forest.html')

def runpi():
    pol_i = PolicyIteration(prob, reward, gamma=0.9, max_iter=1e6)
    pol_i.run()
    pol_i_pol = pol_i.policy
    pol_i_reward = tp(prob, reward, pol_i_pol)
    pol_i_iter = pol_i.iter
    pol_i_time = pol_i.time
    print("Policy Iteration results forest")
    print("Iters","Time","Reward")
    print(pol_i_iter, pol_i_time, pol_i_reward)

def runq():
    a_decs = [0.99, 0.999]
    #a_decs = [0.99]
    a_mins =[0.001, 0.0001]
    
    #a_mins =[0.001]
    eps = [10.0, 1.0]
    
    #eps = [1.0]
    e_dec = [0.99, 0.999]
    #e_dec = [0.99]
    iters = [1000000, 10000000]
    #iters = [10000]
    qdf = tQ(prob, reward, disc=0.9, alpha_dec=a_decs, alpha_min=a_mins, 
                epsilon=eps, epsilon_decay=e_dec, niter=iters)

    
    qdf.to_html('files/Qlearning_forest.html')

    qdf.groupby("Epsilon Decay").mean().to_html('files/Qlearning_forest_gb_episolon_decay.html')
    qdf.groupby("Iters").mean().to_html('files/Qlearning_forest_gb_iters.html')
    qdf.groupby("Alpha Decay").mean().to_html('files/Qlearning_forestgb_alpha_decay.html')
    qdf.groupby("Alpha Min").mean().to_html('files/Qlearning_forest_gb_alpha_min.html')
    qdf.groupby("Epsilon").mean().to_html('files/Qlearning_forest_gb_epsilon.html')

def main():
    #runvi()
    #runpi()
    runq()




if __name__ == "__main__":
    main()






