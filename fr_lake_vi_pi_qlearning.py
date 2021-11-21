import gym
import numpy as np
import random
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pylab as plt
import pandas as pd
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import seaborn
np.random.seed(65)

sixteen_lake = generate_random_map(16)
ten_lake = generate_random_map(10)

map_l_ql = {"sixteen":sixteen_lake}
map_l = {"ten":ten_lake}

def adds(x, num):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[num:] - cumsum[:-num]) / float(num)

def map_d(map):
    size = len(map)
    dis_map = np.zeros((size,size))
    for i, row in enumerate(map):
        for j, loc in enumerate(row):
            if loc == "S":
                dis_map[i, j] = 0
            elif loc == "F":
                dis_map[i, j] = 0
            elif loc == "H":
                dis_map[i, j] = -1
            elif loc == "G":
                dis_map[i, j] = 1
    return dis_map


def p_conv(policy):
    size = int(np.sqrt(len(policy)))
    pol = np.asarray(policy)
    pol = pol.reshape((size, size))
    return pol


def look_p(m_size, policy, m_name, alg_name):
    plt.clf()
    data = map_d(map_l[m_name])
    np_pol = p_conv(policy)
    plt.imshow(data, interpolation="nearest")

    for i in range(np_pol[0].size):
        for j in range(np_pol[0].size):
            arrow = '\u2190'
            if np_pol[i, j] == 1:
                arrow = '\u2193'
            elif np_pol[i, j] == 2:
                arrow = '\u2192'
            elif np_pol[i, j] == 3:
                arrow = '\u2191'
            text = plt.text(j, i, arrow,
                           ha="center", va="center", color="w")
    plt.savefig("files/fr_lake_" +str(m_size)+"_"+ alg_name + "_grid.png")

def run_q_learning(env, disc=[0.9], t_eps=[1e5], alphas=[0.1], d_rates=[0.01]):
    print("Qlearning output")
    m_epsi = 0.01
    
    qd = {}
    for dis in disc:
        qd[dis] = {}
        for eps in t_eps:
            qd[dis][eps] = {}
            for alpha in alphas:
                qd[dis][eps][alpha] = {}
                for dr in d_rates:
                    qd[dis][eps][alpha][dr] = {}
                    
                    q_policy, q_solve_iter, q_solve_time, q_table, rewards = ql(env, dis, eps, alpha, dr, m_epsi)
                    q_mrews, q_meps, _, __ = tp(env, q_policy, nepoch=1000,ep_max=10000)
                    qd[dis][eps][alpha][dr]["mean_reward"] = q_mrews
                    qd[dis][eps][alpha][dr]["mean_eps"] = q_meps
                    qd[dis][eps][alpha][dr]["q-table"] = q_table
                    qd[dis][eps][alpha][dr]["rewards"] = rewards 
                    qd[dis][eps][alpha][dr]["iteration"] = q_solve_iter
                    qd[dis][eps][alpha][dr]["time_spent"] = q_solve_time
                    qd[dis][eps][alpha][dr]["policy"] = q_policy

                    print("gamma: {} total_eps: {} lr: {}, dr: {}".format(dis, eps, alpha, dr))
                    print("Iteration: {} time: {}".format(q_solve_iter, q_solve_time.total_seconds()))
                    print("Mean reward: {} - mean eps: {}".format(q_mrews, q_meps))
                    print()
    return qd

def run_pi_vi(env, disc=[0.9], epsilon=[1e-9]):
    
    vid = {}
    dicts = [{},{}]
    funcs = [vi,pi]
    names = ["Value Iter","Policy Iter"]

    for zz in [0,1]:
        vid = dicts[zz]
        ff = funcs[zz]
        for dis in disc:
            vid[dis] = {}
            for eps in epsilon:
                vid[dis][eps] = {}
                
                vi_policy, vi_solve_iter, vi_solve_time = ff(env, dis, eps)
                vi_mrews, vi_meps, _, __ = tp(env, vi_policy)    
                vid[dis][eps]["mean_reward"] = vi_mrews
                vid[dis][eps]["mean_eps"] = vi_meps
                vid[dis][eps]["iteration"] = vi_solve_iter
                vid[dis][eps]["time_spent"] = vi_solve_time
                vid[dis][eps]["policy"] = vi_policy
                print(names[zz] + " for {} disc and {} eps done".format(dis, eps))
                print("Iter: {} time: {}".format(vi_solve_iter, vi_solve_time.total_seconds()))
                print("Mean reward: {} - mean eps: {}".format(vi_mrews, vi_meps))
                print()

    
    return dicts[0],dicts[1]

def ql(env, disc=0.9, t_eps=1e5, alpha=0.1, d_rate=None,
               m_epsi=0.01):
    
    st = timer()
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    qlt = np.zeros((num_states, num_actions))
    learning_rate = alpha
    gamma = disc


    epsilon = 1.0
    max_epsilon = 1.0
    
    if not d_rate:
        d_rate = 1./t_eps
    
    rewards = []
    for episode in range(int(t_eps)):

        state = env.reset()
        step = 0
        done = False
        total_reward = 0
        while True:


            exp_exp_tradeoff = random.uniform(0,1)


            if exp_exp_tradeoff > epsilon:
                b = qlt[state, :]
                action = np.random.choice(np.where(b == b.max())[0])

            else:
                action = env.action_space.sample()


            new_state, reward, done, info = env.step(action)
            total_reward += reward
            if not done:
                qlt[state, action] = qlt[state, action] + learning_rate*(reward + gamma*np.max(qlt[new_state, :]) - qlt[state, action])
            else:
                qlt[state, action] = qlt[state,action] + learning_rate*(reward - qlt[state,action])


            state = new_state

            if done:
                break
                
        rewards.append(total_reward)
        epsilon = max(max_epsilon -  d_rate * episode, m_epsi) 

    
    end = timer()
    time_spent = timedelta(seconds=end-st)
    print("Solved in: {} episodes and {} seconds".format(t_eps, time_spent.total_seconds()))
    return np.argmax(qlt, axis=1), t_eps, time_spent, qlt, rewards

def pi(env, disc=0.9, epsilon=1e-3):
    
    st = timer()
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    pol = np.random.randint(num_actions, size=(1,num_states))

    value_list = np.zeros((1, num_states))
    episode = 0
    sigma = disc
    

    pol_stable = False
    while not pol_stable:
        episode += 1
        eval_acc = True
        while eval_acc:
            eps = 0
            for s in range(num_states):

                v = value_list[0][s]


                a = pol[0][s]
                t_val_n_state = 0
                for prob, new_state, reward, done in env.P[s][a]:
                    value_new_state = value_list[0][new_state]

                    cand_value = 0
                    if done:
                        cand_value = reward                     
                    else:
                        cand_value = reward + sigma*value_new_state
                    t_val_n_state += cand_value*prob 
                value_list[0][s] = t_val_n_state
                    

                eps = max(eps, np.abs(v-value_list[0][s]))
            if eps < epsilon:
                eval_acc = False



        pol_stable = True
        for s in range(num_states):


            old_action = pol[0][s]

            max_value = -np.inf
            for a in range(num_actions):

                total_cand_value = 0
                for prob, new_state, reward, done in env.P[s][a]:
                    value_new_state = value_list[0][new_state]
                    cand_value = 0
                    if done:
                        cand_value = reward
                    else:
                        cand_value = reward + sigma*value_new_state
                    total_cand_value += prob*cand_value
                if total_cand_value > max_value:
                    max_value = total_cand_value
                    pol[0][s] = a


            if old_action != pol[0][s]:
                pol_stable = False
    
    
    en = timer()
    ts = timedelta(seconds=en-st)
    print("Solved in: {} episodes and {} seconds".format(episode, ts.total_seconds()))
    return pol[0], episode, ts

def vi(env, disc=0.9, epsilon=1e-12):
    
    st = timer()
    
    num_states = env.observation_space.n
    pol = np.zeros((1, num_states))
    val_list = np.zeros((1, num_states))
    num_actions = env.action_space.n
    old_val_list = val_list.copy()
    episode = 0
    m_change = 1
    sigma = disc
    while m_change > epsilon:
        episode += 1
        for s in range(num_states):
            assigned_value = -np.inf
            for a in range(num_actions): 
                total_cand_value = 0
                for prob, new_state, reward, done in env.P[s][a]:
                    value_new_state = old_val_list[0][new_state]
                    cand_value = 0
                    if done:
                        cand_value = reward 
                    else:
                        cand_value = reward + sigma*value_new_state
                    total_cand_value += cand_value*prob 
                        
                if total_cand_value > assigned_value:
                    assigned_value = total_cand_value
                    pol[0][s] = a
                    val_list[0][s] = assigned_value
        changes = np.abs(val_list - old_val_list)
        m_change = np.max(changes)
        old_val_list = val_list.copy()
        
    en = timer()
    ts = timedelta(seconds=en-st)
    print("VI done in : {} epis and {} seconds".format(episode, ts.total_seconds()))
    return pol[0], episode, ts

def tp(env, pol, nepoch=1000, ep_max = 1000):
    rews = []
    eps_counts = []
    for i in range(nepoch):
        current_state = env.reset()
        ep = 0
        done = False
        epi_reward = 0
        while not done and ep < ep_max:
            ep += 1
            act = int(pol[current_state])
            new_state, reward, done, _ = env.step(act)
            epi_reward += reward
            current_state = new_state
            
        rews.append(epi_reward)
        eps_counts.append(ep)
    
    m_rew = sum(rews)/len(rews)
    mean_eps = sum(eps_counts)/len(eps_counts)
    return m_rew, mean_eps, rews, eps_counts 

def p_dict(dictionary, value="Score", size=4, variable="Disc Rate", alg_name=None, log=False):
    plt.clf()

    plt.figure(figsize=(12, 8))
    title = "avg and Max {} on {}x{} Fr Lake".format(value, size, size)
    the_val = value
    value = "avg {}".format(the_val)
    val_type = "Type of {}".format(the_val)
    the_df = pd.DataFrame(columns=[variable, value, val_type])
    for k, v in dictionary.items():
        for val in v:
            if not log:
                dic = {variable: k, value: float(val), val_type: "avg with std"}
            else:
                dic = {variable: np.log10(k), value: float(val), val_type: "avg with std"}                
            the_df = the_df.append(dic, ignore_index=True)
        if not log:
            dic = {variable: k, value: float(max(v)), val_type: "Max"}
        else:
            dic = {variable: np.log10(k), value: float(max(v)), val_type: "Max"}
        the_df = the_df.append(dic, ignore_index=True)
    seaborn.lineplot(x=variable, y=value, hue=val_type, style=val_type, markers=True, data=the_df).set(title=title)
    plt.savefig("files/fr_lake_" +str(size)+"_"+ alg_name + "_" + value+"_graph.png")

def reconvert_dict(td):
    
    disc_rewards = {}
    disc_iterations = {}
    disc_times = {}


    for disc in td:
        disc_rewards[disc] = []    
        disc_iterations[disc] = []    
        disc_times[disc] = []

        for eps in td[disc]:
            disc_rewards[disc].append(td[disc][eps]['mean_reward'])
            disc_iterations[disc].append(td[disc][eps]['iteration'])        
            disc_times[disc].append(td[disc][eps]['time_spent'].total_seconds())  

            
    epsi_rewards = {}
    epsi_iterations = {}
    epsi_times = {}
    for eps in td[0.5]:
        epsi_rewards[eps] = []    
        epsi_iterations[eps] = []    
        epsi_times[eps] = []
    
        for disc in td:
            epsi_rewards[eps].append(td[disc][eps]['mean_reward'])
            epsi_iterations[eps].append(td[disc][eps]['iteration'])        
            epsi_times[eps].append(td[disc][eps]['time_spent'].total_seconds()) 
            
    return disc_rewards, disc_iterations, disc_times, epsi_rewards, epsi_iterations, epsi_times

def d_to_df(td):
    the_df = pd.DataFrame(columns=["Discount Rate", "Training Episodes", "Learning Rate", 
                                   "Decay Rate", "Reward", "T Spent"])
    for dis in td:
        for eps in td[dis]:
            for lr in td[dis][eps]:
                for dr in td[dis][eps][lr]:
                    rew = td[dis][eps][lr][dr]["mean_reward"]
                    time_spent = td[dis][eps][lr][dr]["time_spent"].total_seconds()
                    dic = {"Discount Rate": dis, "Training Episodes": eps, "Learning Rate":lr, 
                           "Decay Rate":dr, "Reward": rew, "T Spent": time_spent}
                    the_df = the_df.append(dic, ignore_index=True)
    return the_df

def main():
    #VI 
    envv = FrozenLakeEnv(desc=map_l["ten"])

    vi_d, pi_d = run_pi_vi(envv, disc=[0.5, 0.75, 0.9, 0.95, 0.99, 0.9999], 
                                          epsilon=[1e-3, 1e-6, 1e-9, 1e-12, 1e-15])
    vi = reconvert_dict(vi_d)
    pi = reconvert_dict(pi_d)
    p = vi_d[0.95][1e-09]['policy']
    look_p(10, p, "ten","VI")
    p = pi_d[0.95][0.001]['policy']
    look_p(10, p, "ten","PI")
    p_dict(vi[0], value="Score", size=10, alg_name="VI")
    p_dict(vi[1], value="Iteration", size=10, alg_name="VI")
    p_dict(vi[2], value="Time", size=10, alg_name="VI")
    p_dict(pi[0], value="Score", size=10, alg_name="PI")
    p_dict(pi[1], value="Iteration", size=10, alg_name="PI")
    p_dict(pi[2], value="Time", size=10, alg_name="PI")
    
    #Q learning

    #envv = FrozenLakeEnv(desc=map_l_ql["sixteen"])

    eps = [1e4, 1e5, 1e6]
    #eps = [1e4]
    decs = [1e-3, 1e-5]
    #decs = [1e-3]
    alphas_inp = [0.1,0.01]
    #alphas_inp = [0.01]
    q_d = run_q_learning(envv, disc=[0.9999], t_eps=eps, alphas=alphas_inp, d_rates=decs)
    p = q_d[0.9999][int(1e5)][0.1][1e-03]['policy']
    look_p(10, p, "ten","Qlearning")
    print((q_d[0.9999][int(1e5)][0.1][1e-03]['q-table'] > 0).any())
    q = d_to_df(q_d)
    q.to_html("files/fr_lake_10_qlearning.html")





if __name__ == "__main__":
    main()





