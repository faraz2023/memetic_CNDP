import networkx as nx
import os
import shutil
import numpy as np
import pandas as pd
from graph_utils import calc_graph_connectivity

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sol_to_txt(sol_arr, export_path):
    sol_np = np.array(sol_arr, dtype=int).reshape(-1)
    np.savetxt(export_path, sol_np, fmt="%i", delimiter=',') 

def nx_to_macnp(G, export_path, export_name):
    N = G.number_of_nodes()

    make_dir(export_path)
    #write text file
    with open(os.path.join(export_path, export_name), 'w') as f:
        f.write(f"{int(N)}\n")
        for n in G.nodes():
            f.write(f"{int(n)}: ")
            for nbr in G.neighbors(n):
                f.write(f"{int(nbr)} ")
            f.write("\n")

def run_MACNP(InstanceFile, filename, K,ExecuteFile=os.path.join('MACNP.exe')
    , Dataset='model', RunTime=120, NumberRepeats=1):
    #copy instance file to MACNP folder
    true_instance_file = os.path.join('instances', Dataset, filename)
    print("File exists: ", os.path.exists(InstanceFile))
    
    shutil.copyfile(InstanceFile, true_instance_file)
    print("Trying to run MACNP")
    #command = 
    #print(command)
    os.system(f"./{str(ExecuteFile)} {filename} {Dataset} {K} {RunTime} {NumberRepeats}")

def solve_MACNP_pipeline(G, remove_ratio, export_path, g_export_name = "G_MACNP.txt", sol_export_name = "MACNP_sol.txt", time_limit=40, rewrite=False):

    N = G.number_of_nodes()
    K = int(remove_ratio * N)
    nx_to_macnp(G, export_path, g_export_name)

    Dataset="model"
    temp_export_file_name = g_export_name + str(K) + ".res1"
    print(temp_export_file_name)
    temp_export_path = os.path.join('results', Dataset, temp_export_file_name)
    
    if(rewrite):
        run_MACNP(os.path.join(export_path, g_export_name), g_export_name, K=K, RunTime=time_limit)
    else:
        if (os.path.exists(temp_export_path)):
            print("Not running MACNP. Sol already exists") 
        else:
            run_MACNP(os.path.join(export_path, g_export_name), g_export_name, K=K, RunTime=time_limit)

    final_export_path = os.path.join(export_path, sol_export_name)
    with open(temp_export_path, 'r') as f:
        lines = f.readlines()
        sol = lines[-1]
        sol = [int(s) for s in sol.split(" ") if s != '\n']
        sol_to_txt(sol, final_export_path)
        print("Sol created")


if "__main__" == __name__:

    sol_path = os.path.join("small-world_2022-09-10-20-26")
    hybrid_exp_report_df = pd.read_csv(os.path.join(sol_path, "report_fp.csv"))

    curr_exp_df = pd.DataFrame()
    for exp_label in hybrid_exp_report_df["exp_label"].unique():
        print("============================={exp_label}=============================".format(exp_label=exp_label))
        exp_path = os.path.join(sol_path, exp_label)
        G_path = os.path.join(exp_path, "G.el")

        G = nx.read_edgelist(G_path, nodetype=int)
        solve_MACNP_pipeline(G, remove_ratio=0.3, export_path=exp_path, g_export_name = "G_MACNP.txt", sol_export_name = "MACNP_sol.txt", time_limit=240, rewrite=True)

        curr_df_dict = {"exp_label": exp_label}

        gurobi_G = G.copy()
        gurobi_sol_path = os.path.join(exp_path, "hr0-", 'overall_sol.txt')
        gurobi_sol = list(np.loadtxt(gurobi_sol_path, dtype=int))
        gurobi_G.remove_nodes_from(gurobi_sol)
        gurobi_connectivity = calc_graph_connectivity(gurobi_G, experiment_type="CN")

        hybrid_08_G = G.copy()
        hybrid_08_sol_path = os.path.join(exp_path, "hr08-", 'overall_sol.txt')
        hybrid_08_sol = list(np.loadtxt(hybrid_08_sol_path, dtype=int))
        hybrid_08_G.remove_nodes_from(hybrid_08_sol)
        hybrid_08_connectivity = calc_graph_connectivity(hybrid_08_G, experiment_type="CN")

        MACNP_G = G.copy()
        MACNP_sol_path = os.path.join(exp_path, "MACNP_sol.txt")
        MACNP_sol = list(np.loadtxt(MACNP_sol_path, dtype=int))
        MACNP_G.remove_nodes_from(MACNP_sol)
        MACNP_connectivity = calc_graph_connectivity(MACNP_G, experiment_type="CN")

        curr_df_dict["gurobi_connectivity"] = gurobi_connectivity
        curr_df_dict["hybrid_08_connectivity"] = hybrid_08_connectivity
        curr_df_dict["MACNP_connectivity"] = MACNP_connectivity

        curr_exp_df = curr_exp_df.append(curr_df_dict, ignore_index=True)

    curr_exp_df.to_csv(os.path.join(sol_path, "MACNP_compare_report.csv"))

    print("Mean gurobi connectivity: ", curr_exp_df["gurobi_connectivity"].mean())
    print("Mean hybrid_08 connectivity: ", curr_exp_df["hybrid_08_connectivity"].mean())
    print("Mean MACNP connectivity: ", curr_exp_df["MACNP_connectivity"].mean())

    #https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh