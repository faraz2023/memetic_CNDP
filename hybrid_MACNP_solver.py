from graph_utils import *
from MACNP_interface import *
import networkx as nx
import os
import time
from gurobi_solvers import Optimize_CNDP4_solver, Optimize_CNDP41_solver
import multiprocessing

TIMELIMIT = 3 * 60 * 60
TIMELIMIT = 60
#DATA_PATH = os.path.join(".", "Data_Synth_full_picture", 'small-world_2022-08-19-18-27', 'test')

num_threads = multiprocessing.cpu_count()

#export_tag = 'barabasi_albert'       
export_tag = 'small-world'
seeds = [ 404, 442, 467, 485, 495, 501]
h_rs = [1, 0.8]
if export_tag == 'barabasi_albert':
    exp_params = {'export_tag': 'barabasi_albert', 'rgs': ['barabasi_albert_graph'],
    'seeds': seeds, 'remove_ratios' :[0.3],
    'avg_ns': [200, 300, 400], 'avg_ks' : [8,8,8], #for newman_watts_strogatz
        'avg_ms': [4,4,4], #for barabasi_albert_graph ,
        'hybrid_rates': h_rs
        }
elif export_tag == 'small-world':
    exp_params = {'export_tag': 'small-world', 'rgs': ['newman_watts_strogatz'],
    'seeds': seeds, 'remove_ratios' :[0.3],
    'avg_ns': [100,200,300], 'avg_ks' : [7,7,7], #for newman_watts_strogatz
        'avg_ms': [4,4,3], #for barabasi_albert_graph ,
        'hybrid_rates': h_rs
    }

root_export_dir = os.path.join(".", "synth_hybrid_full_picture", exp_params['export_tag']+'_'+getDTString())
#root_export_dir = os.path.join(".", "CNDP_exact_synth", 'small-world_2022-08-17-10-46')
#instance_export_dir = os.path.join(root_export_dir, 'instances')
make_dir(root_export_dir)
dict_to_json(exp_params, os.path.join(root_export_dir, 'exp_params.json'))



exps_list = get_exp_list(exp_params['seeds'], exp_params['rgs'], exp_params['avg_ns'], avg_ms=exp_params['avg_ms'], avg_ks=exp_params['avg_ks'])
print("Running experiments total number: ", len(exps_list))


report_path = os.path.join(root_export_dir, 'report_fp.csv')
if not os.path.exists(report_path):
    df = pd.DataFrame(columns=['exp_label', 'exp_path', 'hybrid_rate'])
else:
    df = pd.read_csv(report_path)

for hybrid_rate in exp_params['hybrid_rates']:
    print(f"====================={hybrid_rate}=======================")
for r_r in exp_params['remove_ratios']:
    for exp_dict in exps_list:
        for hybrid_rate in exp_params['hybrid_rates']:
            print(f"====================={hybrid_rate}=======================")

            row_dict = {}
            if(exp_dict['rg'] == 'path_graph'): exp_dict.pop('seed', None)
            G = GenerateGraph(**exp_dict)
            N = G.number_of_nodes()
            #if(N < 80): continue #not enough nodes
            K = int(r_r * N)

            copy_exp_dict = exp_dict.copy()
            copy_exp_dict['rr'] = r_r
            exp_label = get_Exp_Label(copy_exp_dict)
            print(f"====================={exp_label}=======================")


            exp_export_topdir = os.path.join(root_export_dir, exp_label)
            make_dir(exp_export_topdir)
            nx.write_edgelist(G, os.path.join(exp_export_topdir, "G.el"))
            row_dict['exp_label'] = exp_label
            row_dict['exp_path'] = exp_export_topdir
            row_dict['hybrid_rate'] = hybrid_rate

            hr_str = str(hybrid_rate)
            h_r_path = os.path.join(exp_export_topdir, get_Exp_Label({'hr':hybrid_rate}))
            h_r_callback_export_path = os.path.join(h_r_path, "MIP_callbacks")
            make_dir(h_r_path)
            temp_df = df.loc[df['hybrid_rate'] == hybrid_rate]
            if(exp_label in temp_df['exp_label'].unique()):
                print("{} solution already exists".format(h_r_path))
                continue
            
            heuristic_K = int(hybrid_rate * K)
            gurobi_K = K - heuristic_K
            overall_sol = []
            overall_time = 0
            original_pc = calc_graph_connectivity(G, experiment_type='CN')

            heuristic_timelimit = TIMELIMIT * hybrid_rate
            gurobi_timelimit = TIMELIMIT * (1-hybrid_rate)

            print("\t\tGurobi budget: ", gurobi_K)
            print("\t\tHeuristic budget: ", heuristic_K)

            if(heuristic_K > 0):
                    
                heuristic_start_time = time.time()
                solve_MACNP_pipeline(G, K=heuristic_K, export_path=h_r_path,\
                        g_export_name = "G_MACNP.txt", sol_export_name = "MACNP_sol.txt",\
                            time_limit=heuristic_timelimit, rewrite=True)
                heuristic_end_time = time.time()
                heuristic_time = heuristic_end_time - heuristic_start_time
                overall_time += heuristic_time

                heuristic_sol = list(np.loadtxt(os.path.join(h_r_path, "MACNP_sol.txt"), dtype=int))
                overall_sol += heuristic_sol
                G.remove_nodes_from(heuristic_sol)
                
                heuristic_pc = calc_graph_connectivity(G, experiment_type='CN')
                print("\t\tHeuristic sol pairwise connectivity: ", heuristic_pc)
                row_dict['heuristic_time'] = heuristic_time
                row_dict['heuristic_pc'] = heuristic_pc
                overall_pc = heuristic_pc
            
            if(gurobi_K > 0):
                print("SIZE of the connectec compoents in the residual graph: \n\t",
                        [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
                gurobi_start_time = time.time()
                node_scores = dict(G.degree)
                warm_start = list(sorted(node_scores.items(), key=lambda item: item[1], reverse=True))[0:gurobi_K]
                warm_start = [i[0] for i in warm_start]
                
                m, U_vars, V_vars = Optimize_CNDP41_solver(G, gurobi_K, warm_start=warm_start, timelimit = TIMELIMIT, MIPgap=None,  k_constr = "leq",
                        threads = num_threads, callback_export_dir=h_r_callback_export_path)
                gurobi_sol_nodes, gurobi_pc = get_gurobi_sol(G, V_vars)
                print("\t\tGurobi sol pairwise connectivity: ", gurobi_pc)
                gurobi_sol_path = os.path.join(h_r_path, "gurobi_sol.txt")
                sol_to_txt(gurobi_sol_nodes, gurobi_sol_path)

                overall_sol += gurobi_sol_nodes
                gurobi_time = time.time() - gurobi_start_time
                overall_time += gurobi_time
                row_dict["gurobi_time"] = gurobi_time
                row_dict["gurobi_ObjVale"] = m.ObjVal
                row_dict["gurobi_gap"] = m.MIPGap
                row_dict["gurobi_status"] = m.status

                row_dict["gurobi_pc"] = gurobi_pc
                overall_pc = gurobi_pc
                m.dispose()

            
            row_dict["overall_time"] = overall_time
            row_dict["overall_pc"] = overall_pc
            row_dict["original_pc"] = original_pc

            sol_to_txt(overall_sol, os.path.join(h_r_path, "overall_sol.txt"))


            df = df.append(row_dict, ignore_index=True)
            df.to_csv(report_path, index=False)