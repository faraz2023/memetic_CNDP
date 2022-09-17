
import numpy as np
import networkx as nx
import networkx as nx
import os 
from datetime import datetime
import pandas as pd
import json 
import random 

def sol_to_txt(sol_arr, export_path):
    sol_np = np.array(sol_arr, dtype=int).reshape(-1)
    np.savetxt(export_path, sol_np, fmt="%i", delimiter=',') 


def dict_to_json(d, p):
    with open(p, "w") as write_file:
        json.dump(d, write_file, indent=4)

def calc_weighted_avg(arr):
    w_arr = np.zeros(arr.shape)
    unique, counts = np.unique(arr, return_counts=True)
    freq_dict = dict(zip(unique,counts))
    for i in range(len(arr)):
        w_arr[i] = freq_dict[arr[i]]
    return np.average(arr, weights=w_arr)

def calc_graph_connectivity(G, experiment_type, T=1):
    if(G.number_of_nodes() in [0, 1]): return 0
    N = G.number_of_nodes()
    _CN_denom = N * (N - 1)/2
    if(experiment_type=="CN"):
        pairwise_connectivity = 0
        for i in list(nx.connected_components(G)): pairwise_connectivity += (len(i) * (len(i) -1)) / 2
        pc = pairwise_connectivity / _CN_denom
        return pc
    elif(experiment_type=="R0"):
        #get gcc
        gcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc_nodes)
        
        degree_array = np.array(list(dict(nx.degree(G)).values()))
        square_degree_array = degree_array**2

        denom = calc_weighted_avg(degree_array)
        if(denom == 0):
            R_0 = 0
        else:
            R_0 = round(T * ((calc_weighted_avg(square_degree_array)/denom) - 1), 2)
        return R_0
    elif(experiment_type=="GCC"):
        maxCC = len(max(nx.connected_components(G), key=len))
        #print(maxCC , _g_num_nodes)
        #return maxCC / _g_num_nodes
        return maxCC
    print("experiment type not impelemented")
    return -1

def AdaptiveBaselines(G, k=1,approach='HDA', experiment_types=['CN'], early_stopping=None, node_limit=0, T=1, ret_sol=False):
    scores_dict = {}
    for e_t in experiment_types: scores_dict[e_t] = [calc_graph_connectivity(G, experiment_type=e_t)]


    #print("{} for k: {}".format(approach, k))
    if(node_limit > 0):
        node_limit = min(node_limit, G.number_of_nodes())
    elif(early_stopping):
        node_limit = int(G.number_of_nodes()*early_stopping)
    else:
        node_limit = G.number_of_nodes()

    node_cnt = 0
    sol = []
    while(True):
        if(approach=='HDA'):
            node_scores = dict(G.degree)
        elif(approach=='KatzA'):
            lam = max(nx.adjacency_spectrum(G)) 
            alpha = 1 / lam - 0.1
            node_scores = nx.katz_centrality(G, alpha=alpha)

        node_scores_sorted = list(sorted(node_scores.items(), key=lambda item: item[1], reverse=True))
        isTerminal = k > len(node_scores_sorted)

        num_removals = int(min(k, len(node_scores_sorted)))
        node_score_pairs = node_scores_sorted[0:num_removals]
        node_removals = [i[0] for i in node_score_pairs]
        sol += node_removals
        G.remove_nodes_from(node_removals)
        node_cnt += len(node_removals)

        for e_t in experiment_types: scores_dict[e_t].append(calc_graph_connectivity(G, experiment_type=e_t))


        if(isTerminal or node_cnt>= node_limit): break



    if(ret_sol):
        return G, scores_dict, sol
    
    return G, scores_dict

def make_dir(path):
    try: os.makedirs(path)
    except: return -1

def getDTString():
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M")
    return dt_string

# Generate a graph with the give generator and return graph plus some useful statistics
def GenerateGraph(rg, **kwds):
    if(rg=="barabasi_albert_graph"):
        G = nx.barabasi_albert_graph(**kwds)
    elif(rg=="newman_watts_strogatz"):
        G = nx.newman_watts_strogatz_graph(**kwds)
    elif(rg=='path_graph'):
        G = nx.path_graph(**kwds)

    return G


def get_Exp_Label(exp_dict):
    output_str = ""
    for key in exp_dict.keys(): 
        #if key not in ['seed']:
        output_str += "{}{}-".format(key, str(exp_dict[key]).replace(".", ""))
    #output_str += "-" + getDTString()
    return output_str

def get_gurobi_sol(G, V_vars):
    solution_arr = []
    G_nodes = list(G.nodes)
    for i in G_nodes:
        if(V_vars[i].X == 1):
            solution_arr.append(i)
            G.remove_node(i)
            
    new_pairwise_connectivity = calc_graph_connectivity(G, experiment_type='CN')
    #print("\tNew Pariwise connectivity: {}".format(new_pairwise_connectivity))

    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    #nx.draw(G, node_size=7, ax=ax1)
    #nx.draw(new_G, node_size=7, ax=ax2)
    #ax1.set_title("Original PC: {}".format(pairwise_connectivity))
    #ax2.set_title("Solution PC: {}".format(new_pairwise_connectivity))
    #fig.suptitle(exp_export_dir.split("\\")[-1])
    
    #fig.savefig(os.path.join(exp_export_dir, 'before-after.pdf'))
    #nx.write_edgelist(G, os.path.join(exp_export_dir, "G.el"))
    #nx.write_edgelist(new_G, os.path.join(exp_export_dir, "sol_G.el"))
    #np.savetxt(os.path.join(exp_export_dir, "sol.txt"), solution_arr, fmt='%i', delimiter=',') 

    return solution_arr, new_pairwise_connectivity  

def get_exp_list(seeds, rgs, avg_ns, avg_ks=None, avg_ms=None):
    exp_list = []
    for rg in rgs:
        if rg == 'newman_watts_strogatz':
            for seed in seeds:
                for avg_n, avg_k in zip(avg_ns, avg_ks):
                    range_oscil = min(50, int(avg_n/6))
                    n = avg_n - random.randint(-range_oscil, range_oscil)
                    k = avg_k
                    exp_list.append({'rg': rg, 'n': n, 'k': k, "p": 0.05, 'seed':seed})

        elif rg == 'barabasi_albert_graph':
            for seed in seeds:
                for avg_n, avg_m in zip(avg_ns, avg_ms):
                    range_oscil = min(50, int(avg_n/6))
                    n = avg_n - random.randint(-range_oscil, range_oscil)
                    m = avg_m
                    exp_list.append({'rg': rg, 'n': n, 'm': m, 'seed':seed})
        elif rg == 'path_graph':
            for seed in seeds:
                for avg_n in avg_ns:
                    range_oscil = min(50, int(avg_n/6))
                    n = avg_n - random.randint(-range_oscil, range_oscil)
                    exp_list.append({'rg': rg, 'n': n, 'seed':seed})

    return exp_list