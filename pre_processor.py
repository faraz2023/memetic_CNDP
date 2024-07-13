# this file reads the data, and sets up everything in advance, so that you can have your experiments 
# and run inference quickly.

import os
import pandas as pd
import networkx as nx
# for the graphs, make sure they are names as graph_(#the same number as the experiment)
# if experiment is PPI_0, graph should be graph_0.
# if the experiment is helloman_42
# then teh graph needs to be graph_42

def create_directories(base_dir, labels):
    """Create directories for each label and their subdirectories."""
    for label in labels:
        try:
            label_dir = os.path.join(base_dir, label)
            os.makedirs(os.path.join(label_dir, "hr0-"), exist_ok=True)
            os.makedirs(os.path.join(label_dir, "hr08-"), exist_ok=True)
        except Exception as e:
            print(f"Error creating directories for label {label}: {e}")

def process_graphs(graphs_dir, base_dir, labels):
    """Process graphs and save them in the corresponding directories."""
    for label in labels:
        try:
            label_number = label.split('_')[-1]
            graph_file = os.path.join(graphs_dir, f"graph_{label_number}.txt")
            
            if not os.path.exists(graph_file):
                print(f"Graph file {graph_file} does not exist.")
                continue
            
            G = nx.read_edgelist(graph_file)
            output_file = os.path.join(base_dir, label, "G.el")
            nx.write_edgelist(G, output_file)
        except Exception as e:
            print(f"Error processing graph for label {label}: {e}")

def main():
    base_dir = 'EXP_LIST'
    graphs_dir = 'Graphs'
    report_file = os.path.join(base_dir, 'report_fp.csv')
    
    try:
        # Read the CSV file
        df = pd.read_csv(report_file)
        
        # Extract labels
        labels = df['exp_label'].tolist()
        
        # Create directories
        create_directories(base_dir, labels)
        
        # Process graphs
        process_graphs(graphs_dir, base_dir, labels)
        
        print("Processing completed.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


''' 
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
'''