from gurobipy import GRB
import gurobipy as gp
import networkx as nx
import os
import numpy as np

def sol_to_txt(sol_arr, export_path):
    sol_np = np.array(sol_arr, dtype=int).reshape(-1)
    np.savetxt(export_path, sol_np, fmt="%i", delimiter=',') 
def make_dir(path):
    try: os.makedirs(path)
    except: return -1
def Optimize_CNDP5_solver(G, k, gnn_nodes, k_constr = "leq", timelimit = 5*60, MIPgap=None, threads=10, 
    warm_start=None, env=None):
    try:
        print("Optimizing CNDP5 solver... for {} nodes".format(G.number_of_nodes()))
        #get G complimnetary graph
        N = G.number_of_nodes()
        G_compl = nx.complement(G)
        #A_compl_dense = nx.adjacency_matrix(G_compl).todense()
        G_nodes = G.nodes
        exclude_nodes = G_nodes - gnn_nodes
        # Create a new model
        m = gp.Model("cndp5", env=env)
        m.Params.Threads = threads

        if(timelimit):m.setParam('TimeLimit', timelimit)
        # gap = | ObjBound - ObjVal | / | ObjVal |
        if(MIPgap): m.setParam('MIPGap', MIPgap)

        #if(N>60): m.Params.MIPFocus = 1


        # Create variables
        U_vars = m.addVars(G_nodes, G_nodes, vtype=GRB.BINARY, name="u") # Binary for constraint 4
        V_vars = m.addVars(G_nodes, vtype=GRB.BINARY, name='v') # Binary for constraint 5
        
        if(warm_start):
            for v in warm_start:
                V_vars[v].Start = 1

        # fixing half the variables to zero {i = 1, ..., (n − 1), j = (i + 1), ..., n}
        for i in G_nodes:
            for j in G_nodes:
                if(i >= j): U_vars[i,j].ub = 0
                
        for i in exclude_nodes:
            V_vars[i].ub = 0

        # Add constraints
        #u_i_j +v_i +v_j ≥ 1, ∀ (i, j) ∈ E (1)
        m.addConstrs((U_vars[min(i,j),max(i,j)] + V_vars[i] + V_vars[j] >= 1 for (i,j) in G.edges)
            , name="constraint_1")

        if(k_constr == "leq"):
        #∑_(i∈V) vi ≤ k (2)
            m.addConstr(gp.quicksum(V_vars) <= k)
        elif(k_constr == "eq"):
            m.addConstr(gp.quicksum(V_vars) == k)

        # get graph degrees
        G_degrees = G.degree()

        for edge in G_compl.edges:
            i, j = edge[0], edge[1]

            selected = i if G_degrees[i] < G_degrees[j] else j
            for k in G.neighbors(selected):
                m.addConstr(U_vars[min(i,j), max(i,j)] - U_vars[min(i,k),max(i,k)] \
                    - U_vars[min(j,k),max(k, j)] - V_vars[k] + 1 >= 0, name="constraint_8")
                
        
        #Set objective
        m.setObjective(gp.quicksum(U_vars), GRB.MINIMIZE)

        #Optimize model
        m.optimize()

        print('\tObj: %g' % m.ObjVal)

        return m, U_vars, V_vars

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        return -1

    except AttributeError:
        print('Encountered an attribute error')
        print(AttributeError.__traceback__())
        return -1
        
#### Model version 4 with callback start ####
def mycallback(model, where):
    export_path = model._export_path
    if where == GRB.callback.MIP:
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        solcnt = model.cbGet(GRB.Callback.MIP_SOLCNT)
        gap = abs(objbst - objbnd) / (abs(objbst))
        runtime = model.cbGet(GRB.Callback.RUNTIME)

        with open(os.path.join(export_path, "MIP_report.csv"), "a") as f:
            f.write(f"{nodecnt},{objbst},{objbnd},{solcnt},{gap},{runtime}\n")

    if where == GRB.Callback.MIPSOL:
        #nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        gap = abs(objbst - objbnd) / (abs(objbst))
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        #Caused error on cluster commenting for now
        sol_phase = "" #model.cbGet(GRB.Callback.MIPSOL_PHASE)
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        
        export_path = model._export_path
        crr_sol_path = os.path.join(export_path, str(solcnt))
        make_dir(crr_sol_path)

        v_vars = model.cbGetSolution([var for var in model.getVars() if "v" in var.VarName])
        sol = []
        
        for i in range(len(v_vars)):
            if(v_vars[i] > 0.5):
                sol.append(i)
        sol_to_txt(sol, os.path.join(crr_sol_path, "MIP_sol.txt"))

        with open(os.path.join(export_path, "MIP_sols_report.csv"), "a") as f:
            f.write(f"{nodecnt},{obj},{solcnt},{sol_phase},{gap},{runtime}\n")
            



def Optimize_CNDP41_solver(G, k, k_constr = "leq", timelimit = 5*60, MIPgap=None, threads=10, 
    warm_start=None, env=None, callback_export_dir=None):
    try:
        print("Optimizing CNDP41 solver... for {} nodes".format(G.number_of_nodes()))
        #get G complimnetary graph
        N = G.number_of_nodes()
        G_compl = nx.complement(G)
        #A_compl_dense = nx.adjacency_matrix(G_compl).todense()
        G_nodes = G.nodes

        # Create a new model

        m = gp.Model("cndp4", env=env)
        m.Params.Threads = threads

        if(timelimit):m.setParam('TimeLimit', timelimit)
        # gap = | ObjBound - ObjVal | / | ObjVal |
        if(MIPgap): m.setParam('MIPGap', MIPgap)

        #if(N>60): m.Params.MIPFocus = 1


        # Create variables
        U_vars = m.addVars(G_nodes, G_nodes, vtype=GRB.BINARY, name="u") # Binary for constraint 4
        V_vars = m.addVars(G_nodes, vtype=GRB.BINARY, name='v') # Binary for constraint 5
        
        if(warm_start):
            for v in warm_start:
                V_vars[v].Start = 1

        # fixing half the variables to zero {i = 1, ..., (n − 1), j = (i + 1), ..., n}
        for i in G_nodes:
            for j in G_nodes:
                if(i >= j): U_vars[i,j].ub = 0

        # Add constraints
        #u_i_j +v_i +v_j ≥ 1, ∀ (i, j) ∈ E (1)
        m.addConstrs((U_vars[min(i,j),max(i,j)] + V_vars[i] + V_vars[j] >= 1 for (i,j) in G.edges)
            , name="constraint_1")

        if(k_constr == "leq"):
        #∑_(i∈V) vi ≤ k (2)
            m.addConstr(gp.quicksum(V_vars) <= k)
        elif(k_constr == "eq"):
            m.addConstr(gp.quicksum(V_vars) == k)

        # get graph degrees
        G_degrees = G.degree()

        for edge in G_compl.edges:
            i, j = edge[0], edge[1]

            selected = i if G_degrees[i] < G_degrees[j] else j
            for k in G.neighbors(selected):
                m.addConstr(U_vars[min(i,j), max(i,j)] - U_vars[min(i,k),max(i,k)] \
                    - U_vars[min(j,k),max(k, j)] - V_vars[k] + 1 >= 0, name="constraint_8")
                
        
        #Set objective
        m.setObjective(gp.quicksum(U_vars), GRB.MINIMIZE)

        #Optimize model
        if(callback_export_dir):
            make_dir(callback_export_dir)
            #write csv file with just headers
            with open(os.path.join(callback_export_dir, "MIP_sols_report.csv"), "w") as f:
                f.write("nodecnt,obj,solcnt,sol_phase,gap,runtime\n")
            with open(os.path.join(callback_export_dir, "MIP_report.csv"), "w") as f:
                f.write(f"nodecnt,objbst,objbnd,solcnt,gap,runtime\n")
            m._export_path = callback_export_dir
            m.optimize(mycallback)
        else:
            m.optimize()

        print('\tObj: %g' % m.ObjVal)

        return m, U_vars, V_vars

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        return -1

    except AttributeError:
        print('Encountered an attribute error')
        print(AttributeError.__traceback__())
        return -1
#### Model version 4 with callback end ####

# # https://www.sciencedirect.com/science/article/pii/S1571065318300349

def Optimize_CNDP4_solver(G, k, k_constr = "leq", timelimit = 5*60, MIPgap=None, threads=10, 
    warm_start=None, env=None):
    try:
        print("Optimizing CNDP4 solver... for {} nodes".format(G.number_of_nodes()))
        #get G complimnetary graph
        N = G.number_of_nodes()
        G_compl = nx.complement(G)
        #A_compl_dense = nx.adjacency_matrix(G_compl).todense()
        G_nodes = G.nodes

        # Create a new model
        m = gp.Model("cndp4", env=env)
        m.Params.Threads = threads

        if(timelimit):m.setParam('TimeLimit', timelimit)
        # gap = | ObjBound - ObjVal | / | ObjVal |
        if(MIPgap): m.setParam('MIPGap', MIPgap)

        #if(N>60): m.Params.MIPFocus = 1


        # Create variables
        U_vars = m.addVars(G_nodes, G_nodes, vtype=GRB.BINARY, name="u") # Binary for constraint 4
        V_vars = m.addVars(G_nodes, vtype=GRB.BINARY, name='v') # Binary for constraint 5
        
        if(warm_start):
            for v in warm_start:
                V_vars[v].Start = 1

        # fixing half the variables to zero {i = 1, ..., (n − 1), j = (i + 1), ..., n}
        for i in G_nodes:
            for j in G_nodes:
                if(i >= j): U_vars[i,j].ub = 0

        # Add constraints
        #u_i_j +v_i +v_j ≥ 1, ∀ (i, j) ∈ E (1)
        m.addConstrs((U_vars[min(i,j),max(i,j)] + V_vars[i] + V_vars[j] >= 1 for (i,j) in G.edges)
            , name="constraint_1")

        if(k_constr == "leq"):
        #∑_(i∈V) vi ≤ k (2)
            m.addConstr(gp.quicksum(V_vars) <= k)
        elif(k_constr == "eq"):
            m.addConstr(gp.quicksum(V_vars) == k)

        # get graph degrees
        G_degrees = G.degree()

        for edge in G_compl.edges:
            i, j = edge[0], edge[1]

            selected = i if G_degrees[i] < G_degrees[j] else j
            for k in G.neighbors(selected):
                m.addConstr(U_vars[min(i,j), max(i,j)] - U_vars[min(i,k),max(i,k)] \
                    - U_vars[min(j,k),max(k, j)] - V_vars[k] + 1 >= 0, name="constraint_8")
                
        
        #Set objective
        m.setObjective(gp.quicksum(U_vars), GRB.MINIMIZE)

        #Optimize model
        m.optimize()

        print('\tObj: %g' % m.ObjVal)

        return m, U_vars, V_vars

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        return -1

    except AttributeError:
        print('Encountered an attribute error')
        print(AttributeError.__traceback__())
        return -1


def solve_DCNP2(G, k, timelimit = 5*60, MIPgap=None, threads=10, 
    warm_start=None, env=None):
    
    print("Solving DCNP ...")
    
    # create power graph
    PG = nx.power(G, 2)
    
    m = gp.Model(env=env)
    m.Params.Threads = threads

    if(timelimit):m.setParam('TimeLimit', timelimit)
    # gap = | ObjBound - ObjVal | / | ObjVal |
    if(MIPgap): m.setParam('MIPGap', MIPgap)
    
    

    
    # create y variables
    Y = m.addVars(PG.nodes, vtype=GRB.BINARY)

    if(warm_start):
        for v in warm_start:
            Y[v].Start = 1
    
    # create x variables
    new_edges = [edge for edge in PG.edges if edge not in G.edges]
    
    print("# of short pairs: ", len(new_edges))
    
    X = m.addVars(new_edges, vtype=GRB.BINARY)
    
    # minimize the number of short connections
    m.setObjective( gp.quicksum(X), GRB.MINIMIZE )
    
    # add covering constraints
    for edge in new_edges:
        common_neighbors = list(nx.common_neighbors(G, edge[0], edge[1]))
        for vertex in common_neighbors:
            m.addConstr( X[edge] + Y[vertex] >= 1)
            
    # add budget constraints
    m.addConstr( gp.quicksum(Y) <= k)    
            
    # optimize!
    m.optimize()
    
    # retrieve solutions
    critical_nodes = [ vertex for vertex in PG.nodes if Y[vertex].x > 0.5 ]
    
    print("# of critical nodes: ", len(critical_nodes))
    
    short_pairs = [ edge for edge in new_edges if X[edge].x > 0.5 ]
    
    print("# of remaining short pairs: ", len(short_pairs))

    return m, critical_nodes, short_pairs