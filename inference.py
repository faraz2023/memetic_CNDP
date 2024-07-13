import networkx as nx
import os
import shutil
import numpy as np
import pandas as pd
import time
from graph_utils import calc_graph_connectivity

BUDGET = [ 0.1, 0.2, 0.3, 0.4]  # Example budget values 0.1, 0.2, 0.3, 
TIME_LIMIT = 5  # in seconds



def make_dir(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def sol_to_txt(sol_arr, export_path):
    """Save solution array to a text file."""
    sol_np = np.array(sol_arr, dtype=int).reshape(-1)
    np.savetxt(export_path, sol_np, fmt="%i", delimiter=',')

def nx_to_macnp(G, export_path, export_name):
    """Convert a NetworkX graph to MACNP format and save it to a file."""
    N = G.number_of_nodes()
    make_dir(export_path)
    with open(os.path.join(export_path, export_name), 'w') as f:
        f.write(f"{int(N)}\n")
        for n in G.nodes():
            f.write(f"{int(n)}: ")
            for nbr in G.neighbors(n):
                f.write(f"{int(nbr)} ")
            f.write("\n")

def run_MACNP(InstanceFile, filename, K, ExecuteFile=os.path.join('MACNP.exe'), Dataset='model', RunTime=120, NumberRepeats=1):
    """Run the MACNP executable with the specified parameters and return the execution time."""
    true_instance_file = os.path.join('instances', Dataset, filename)
    shutil.copyfile(InstanceFile, true_instance_file)
    start_time = time.time()
    os.system(f"./{str(ExecuteFile)} {filename} {Dataset} {K} {RunTime} {NumberRepeats}")
    end_time = time.time()
    return end_time - start_time

def solve_MACNP_pipeline(G, budget, export_path, g_export_name="G_MACNP.txt", sol_export_name="MACNP_sol.txt", time_limit=40, rewrite=False):
    """Solve the MACNP problem for a given graph and budget, and return the solution and execution time."""
    N = G.number_of_nodes()
    K = int(budget * N)
    nx_to_macnp(G, export_path, g_export_name)

    Dataset = "model"
    temp_export_file_name = g_export_name + str(K) + ".res1"
    temp_export_path = os.path.join('results', Dataset, temp_export_file_name)
    
    # Run MACNP if the solution doesn't already exist or rewrite is True
    if rewrite or not os.path.exists(temp_export_path):
        macnp_time = run_MACNP(os.path.join(export_path, g_export_name), g_export_name, K=K, RunTime=time_limit)
    else:
        macnp_time = None

    final_export_path = os.path.join(export_path, sol_export_name)
    if os.path.exists(temp_export_path):
        with open(temp_export_path, 'r') as f:
            lines = f.readlines()
            sol = lines[-1]
            sol = [int(s) for s in sol.split(" ") if s != '\n']
            sol_to_txt(sol, final_export_path)
    else:
        sol = []

    return macnp_time, sol

if __name__ == "__main__":
    
    sol_path = os.path.join("EXP_LIST")
    hybrid_exp_report_df = pd.read_csv(os.path.join(sol_path, "report_fp.csv"))
    
    report_file_path = os.path.join("results", "experiment_report.csv")
    
    # Create report file if it does not exist
    if not os.path.exists(report_file_path) or os.path.getsize(report_file_path) == 0:
        report_df = pd.DataFrame(columns=["exp_label", "budget", "number_of_nodes", "nodes_removed", "macnp_time", "macnp_connectivity", "gurobi_connectivity", "hybrid_08_connectivity"])
        report_df.to_csv(report_file_path, index=False)
    
    all_results = []

    for budget in BUDGET:
        for exp_label in hybrid_exp_report_df["exp_label"].unique():
            print(f"============================={exp_label}=============================")
            exp_path = os.path.join(sol_path, exp_label)
            G_path = os.path.join(exp_path, "G.el")

            # Read the graph from the edge list file
            G = nx.read_edgelist(G_path, nodetype=int)
            number_of_nodes = G.number_of_nodes()
            
            # Solve MACNP for the given budget
            macnp_time, macnp_sol = solve_MACNP_pipeline(G, budget, export_path=exp_path, g_export_name="G_MACNP.txt", sol_export_name="MACNP_sol.txt", time_limit=TIME_LIMIT, rewrite=True)
            # Collect result data for the current experiment
            result_data = {
                "exp_label": exp_label,
                "budget": budget,
                "number_of_nodes": number_of_nodes,
                "nodes_removed": len(macnp_sol),
                "removed_nodes": macnp_sol,
                "macnp_time": macnp_time,
                "pairwise_connectivity": np.nan,
                "gurobi_connectivity": np.nan,
                "hybrid_08_connectivity": np.nan
            }

            # Calculate MACNP connectivity
            MACNP_G = G.copy()
            if macnp_sol:
                MACNP_G.remove_nodes_from(macnp_sol)
                MACNP_connectivity = calc_graph_connectivity(MACNP_G, experiment_type="CN")
                result_data["macnp_connectivity"] = MACNP_connectivity

            # Calculate Gurobi connectivity
            gurobi_G = G.copy()
            gurobi_sol_path = os.path.join(exp_path, "hr0-", 'overall_sol.txt')
            if os.path.exists(gurobi_sol_path) and os.path.getsize(gurobi_sol_path) > 0:
                gurobi_sol = list(np.loadtxt(gurobi_sol_path, dtype=int))
                gurobi_G.remove_nodes_from(gurobi_sol)
                gurobi_connectivity = calc_graph_connectivity(gurobi_G, experiment_type="CN")
                result_data["gurobi_connectivity"] = gurobi_connectivity

            # Calculate hybrid_08 connectivity
            hybrid_08_G = G.copy()
            hybrid_08_sol_path = os.path.join(exp_path, "hr08-", 'overall_sol.txt')
            if os.path.exists(hybrid_08_sol_path) and os.path.getsize(hybrid_08_sol_path) > 0:
                hybrid_08_sol = list(np.loadtxt(hybrid_08_sol_path, dtype=int))
                hybrid_08_G.remove_nodes_from(hybrid_08_sol)
                hybrid_08_connectivity = calc_graph_connectivity(hybrid_08_G, experiment_type="CN")
                result_data["hybrid_08_connectivity"] = hybrid_08_connectivity

            # Store the result data
            all_results.append(result_data)
            print(f"Collected results for {exp_label} with budget {budget}")

            # Print mean connectivity results
            print("Mean gurobi connectivity: ", result_data["gurobi_connectivity"])
            print("Mean hybrid_08 connectivity: ", result_data["hybrid_08_connectivity"])
            print("Mean MACNP connectivity: ", result_data["macnp_connectivity"])

    # Append all results to the report file and sort by exp_label and budget
    if os.path.exists(report_file_path) and os.path.getsize(report_file_path) > 0:
        report_df = pd.read_csv(report_file_path)
    else:
        report_df = pd.DataFrame(columns=["exp_label", "budget", "number_of_nodes", "nodes_removed", "macnp_time", "macnp_connectivity", "gurobi_connectivity", "hybrid_08_connectivity"])
        
    new_results_df = pd.DataFrame(all_results)
    report_df = pd.concat([report_df, new_results_df])
    report_df = report_df.sort_values(by=["exp_label", "budget"])
    report_df.to_csv(report_file_path, index=False)
    print(f"Appended and grouped results in {report_file_path}")

"""

PPI_2
PPI_3
PPI_4
PPI_5
PPI_6
PPI_7
PPI_8
PPI_9
PPI_10
PPI_11
PPI_12
PPI_13
PPI_14
PPI_15
PPI_16
PPI_17
PPI_18
PPI_19

"""
