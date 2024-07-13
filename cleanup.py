
'''1: to delete all the graphs in the Graphs dir, and put them in past_graphs, then the # of time they were cleaned up.
And then it can also delete the experiments in the EXP list and put them in past_exp, with their cleanup #.

2: it can restore the LATEST cleanup, so if you have:
cleanup #1, and cleanup #2, when restoring, it will get all the graphs in cleanup #2 and put them back in Graphs.
'''
import os
import shutil

def get_next_cleanup_number(base_dir, prefix):
    """Get the next cleanup number for naming the cleanup directories."""
    if not os.path.exists(base_dir):
        return 1
    
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    if not existing_dirs:
        return 1
    
    existing_numbers = [int(d.split("#")[1].strip()) for d in existing_dirs]
    return max(existing_numbers) + 1

def move_files_to_cleanup_dir(source_dir, cleanup_base_dir, prefix):
    """Move all files from the source directory to a new cleanup directory within the cleanup base directory."""
    cleanup_number = get_next_cleanup_number(cleanup_base_dir, prefix)
    new_cleanup_dir = os.path.join(cleanup_base_dir, f'{prefix}#{cleanup_number}')
    
    os.makedirs(new_cleanup_dir, exist_ok=True)
    
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        # Ensure that past_graphs or past_exp directory itself is not moved
        if os.path.basename(item_path) in ['past_graphs', 'past_exp']:
            continue
        if os.path.isfile(item_path) or os.path.isdir(item_path):
            shutil.move(item_path, new_cleanup_dir)
            print(f"Moved {item} to {new_cleanup_dir}")

def restore_files_from_latest_cleanup(cleanup_base_dir, target_dir, prefix):
    """Restore files from the most recent cleanup directory to the target directory and delete the cleanup directory."""
    if not os.path.exists(cleanup_base_dir):
        print(f"The directory {cleanup_base_dir} does not exist.")
        return
    
    existing_dirs = [d for d in os.listdir(cleanup_base_dir) if d.startswith(prefix)]
    if not existing_dirs:
        print(f"No past cleanups found in {cleanup_base_dir}.")
        return
    
    existing_numbers = [int(d.split("#")[1].strip()) for d in existing_dirs]
    latest_cleanup_dir = os.path.join(cleanup_base_dir, f'{prefix}#{max(existing_numbers)}')
    
    for item in os.listdir(latest_cleanup_dir):
        item_path = os.path.join(latest_cleanup_dir, item)
        shutil.move(item_path, target_dir)
        print(f"Restored {item} from {latest_cleanup_dir} to {target_dir}")
    
    shutil.rmtree(latest_cleanup_dir)
    print(f"Deleted cleanup directory: {latest_cleanup_dir}")

def main():
    base_dir = 'EXP_LIST'
    graphs_dir = 'Graphs'
    past_graphs_dir = os.path.join(graphs_dir, 'past_graphs')
    past_exp_dir = os.path.join(base_dir, 'past_exp')
    
    if not os.path.exists(base_dir):
        print(f"The directory {base_dir} does not exist.")
        return
    
    if not os.path.exists(graphs_dir):
        print(f"The directory {graphs_dir} does not exist.")
        return
    
    while True: 
        action = input("Enter 'cleanup' to clean up, 'restore' to restore the latest graphs and experiments, or 0 to exit: ").strip().lower()
        
        if action == 'cleanup':
            move_files_to_cleanup_dir(graphs_dir, past_graphs_dir, 'cleanup')
            move_files_to_cleanup_dir(base_dir, past_exp_dir, 'cleanup')
            print("Cleanup completed.")
        elif action == 'restore':
            restore_files_from_latest_cleanup(past_graphs_dir, graphs_dir, 'cleanup')
            restore_files_from_latest_cleanup(past_exp_dir, base_dir, 'cleanup')
            print("Restore completed.")
        elif action == "0":
            print("Exiting.")
            break
        else:
            print("Invalid action. Please enter 'cleanup' or 'restore'.")
            print("")

if __name__ == "__main__":
    main()
