import numpy as np
import os
import yaml
import random

origin = [70.0, -40.0]
upper_left = [60.0, -10.0]
upper_center = [70.0, -10.0]
upper_right = [80.0, -10.0]

lower_left = [60.0, -45.0]
lower_center = [70.0, -45.0]
lower_right = [80.0, -45.0]

# in order of (yaw, pitch)
pathlist_0 = [origin,
              upper_left,
              lower_left,
              upper_right,
              lower_right]

pathlist_1 = [origin,
              upper_center,
              lower_left,
              upper_right,
              lower_center]

pathlist_2 = [origin,
              upper_right,
              lower_left,
              upper_right,
              lower_right]

pathlist_3 = [origin,
              lower_center,
              upper_right,
              lower_right,
              upper_left]

pathlist_4 = [origin,
              upper_right,
              lower_right,
              upper_left,
              lower_center]

small_trial_pathlist = [pathlist_0, pathlist_1, pathlist_2, pathlist_3, pathlist_4]


experiment_session_cases = [{"label" : "Default", "ptp_time": 3.0, "wait_time": 2.0, "trial_number_per_block": 5},
                {"label" : "Default", "ptp_time": 3.0, "wait_time": 2.0, "trial_number_per_block": 5},
                {"label" : "Default", "ptp_time": 3.0, "wait_time": 2.0, "trial_number_per_block": 5},
                {"label" : "Default", "ptp_time": 3.0, "wait_time": 2.0, "trial_number_per_block": 5}]

experiment_block_cases = [{'label': 'No Gravity Compensation', 'running': True, "use_gravity_compensation": False, "use_dynamic_compensation": False, "prefix_label" : ""},
                    {'label': 'Static Gravity Compensation', 'running': True, "use_gravity_compensation": True, "use_dynamic_compensation": False, "prefix_label" : ""},
                    {'label': 'Dynamic Gravity Compensation', 'running': True, "use_gravity_compensation": True, "use_dynamic_compensation": True, "prefix_label" : ""}]

from collections import deque
def generate_session_group(session_configs, case_configs, shuffle_blocks = False, shuffle_trials = False):
    session_group = dict()
    #small_trial_pathlist = get_targets_pathlist()

    session_number = len(session_configs)
    case_num = len(case_configs)

    small_trial_num = len(small_trial_pathlist)

    global_trial_index_deque_dict = dict()
    
    trial_pathlist = []

    trial_num_per_block = session_configs[0]['trial_number_per_block'] # fix this!

    for i in range(trial_num_per_block * session_number):
        trial_pathlist.append(small_trial_pathlist[i % small_trial_num])
    print(f"total trials per case in experiment: {len(trial_pathlist)}")

    for b_idx in range(case_num):
            global_trial_index_deque_dict[b_idx] = deque([i for i in range(len(trial_pathlist))])

    for s_idx in range(session_number):
        block_group = dict()

        block_order = [i for i in range(case_num)] # could just shuffle the case_configs...
        if shuffle_blocks: random.shuffle(block_order)
        print(f"Session {s_idx}: Block Order: {block_order}")
        for b_idx in range(case_num): # one block per case
            
            trial_order = [i for i in range(trial_num_per_block)]
            if shuffle_trials: random.shuffle(trial_order)
            #print(f"Block {b_idx}: Trial Order: {trial_order}")
            block_pathlist = []
            block_trial_index = []
            for t_idx in range(trial_num_per_block):
                block_pathlist.append(trial_pathlist[trial_order[t_idx]])

                #block_trial_index.append(global_trial_index_deque_dict[b_idx].popleft())
                block_trial_index.append(t_idx)

            block_case_config = case_configs[block_order[b_idx]]

            sesssion_config = session_configs[s_idx]
            #print(block_trial_index)
            block_group[b_idx] = {"session": sesssion_config, "case": block_case_config, "trials": block_pathlist, "trial_index": block_trial_index, "session_label" : session_configs[s_idx]["label"]}
        #print(block_group)
        session_group[s_idx] = block_group
    #print(session_group)
    return session_group


if __name__ == '__main__':
    # https://python.land/data-processing/python-yaml
    print("Generating experiment session group...")
    experiment_session_group = generate_session_group(experiment_session_cases, experiment_block_cases, shuffle_blocks=True, shuffle_trials=True)
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    resources_directory = os.path.join(current_directory, "res")

    # Create directories if they don't exist
    for directory in [resources_directory]:
        print(f"Checking directory {directory}...")
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created {directory}!")
            # os.chmod(directory, 0o777)  # Set full permissions

    os.chmod(resources_directory, 0o777)  # Set full permissions

    # Save each session as a separate YAML file
    for session_idx, session_data in experiment_session_group.items():
        # Add session number as the top-level key
        session_with_key = {session_idx: session_data}

        session_file_path = os.path.join(resources_directory, f"session_{session_idx}.yaml")
        with open(session_file_path, 'w') as outfile:
            yaml.dump(session_with_key, outfile)
        print(f"Saved session {session_idx} to {session_file_path}")

    print("All sessions have been saved.")
