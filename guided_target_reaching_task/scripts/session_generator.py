import numpy as np
from utils import *
import os
import yaml
import random

# define config dictionary for each experimental case
use_testing_settings = False #True
if use_testing_settings:
    calibration_session_cases = [{"label" : "Slow", "trial_time_length": 5.0, "trial_number_per_block": 3},
                    {"label" : "Fast", "trial_time_length": 3.0, "trial_number_per_block": 3}]

    calibration_block_cases = [{'label': 'Calibration', 'ac_type': 1, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": False, "prefix_label" : ""}]

    experiment_session_cases = [{"label" : "Slow", "trial_time_length": 5.0, "trial_number_per_block": 3},
                    {"label" : "Slow", "trial_time_length": 5.0, "trial_number_per_block": 3},
                    {"label" : "Fast", "trial_time_length": 3.0, "trial_number_per_block": 3},
                    {"label" : "Fast", "trial_time_length": 3.0, "trial_number_per_block": 3}]

    # try with fixed windows just to see what happens
    experiment_block_cases = [{'label': 'SAC No Assistance', 'ac_type': 1, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'UI-VAC Zero Fixed Window', 'ac_type': 2, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'UI-VAC Short Fixed Window', 'ac_type': 2, 'running': True, "fixed_time_horizon": 0.150, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'UI-VAC Long Fixed Window', 'ac_type': 2, 'running': True, "fixed_time_horizon": 0.300, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'UI-VAC Variable Window', 'ac_type': 2, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": True, "prefix_label" : ""},
                        {'label': 'C-VAC Zero Fixed Window', 'ac_type': 3, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'C-VAC Short Fixed Window', 'ac_type': 3, 'running': True, "fixed_time_horizon": 0.150, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'C-VAC Long Fixed Window', 'ac_type': 3, 'running': True, "fixed_time_horizon": 0.300, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'C-VAC Variable Window', 'ac_type': 3, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": True, "prefix_label" : ""}]
else:
    calibration_session_cases = [{"label" : "Slow", "trial_time_length": 5.0, "trial_number_per_block": 10},
                    {"label" : "Fast", "trial_time_length": 3.0, "trial_number_per_block": 10}]

    calibration_block_cases = [{'label': 'Calibration', 'ac_type': 1, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": False, "prefix_label" : ""}]

    experiment_session_cases = [{"label" : "Slow", "trial_time_length": 5.0, "trial_number_per_block": 10},
                    {"label" : "Slow", "trial_time_length": 5.0, "trial_number_per_block": 10},
                    {"label" : "Fast", "trial_time_length": 3.0, "trial_number_per_block": 10},
                    {"label" : "Fast", "trial_time_length": 3.0, "trial_number_per_block": 10}]

    # try with fixed windows just to see what happens
    experiment_block_cases = [{'label': 'SAC No Assistance', 'ac_type': 1, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'UI-VAC Zero Fixed Window', 'ac_type': 2, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'UI-VAC Short Fixed Window', 'ac_type': 2, 'running': True, "fixed_time_horizon": 0.150, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'UI-VAC Long Fixed Window', 'ac_type': 2, 'running': True, "fixed_time_horizon": 0.300, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'UI-VAC Variable Window', 'ac_type': 2, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": True, "prefix_label" : ""},
                        {'label': 'C-VAC Zero Fixed Window', 'ac_type': 3, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'C-VAC Short Fixed Window', 'ac_type': 3, 'running': True, "fixed_time_horizon": 0.150, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'C-VAC Long Fixed Window', 'ac_type': 3, 'running': True, "fixed_time_horizon": 0.300, "use_var_time": False, "prefix_label" : ""},
                        {'label': 'C-VAC Variable Window', 'ac_type': 3, 'running': True, "fixed_time_horizon": 0.0, "use_var_time": True, "prefix_label" : ""}]

from collections import deque
def generate_session_group(session_configs, case_configs, shuffle_blocks = False, shuffle_trials = False, override_cases_data_by_session_info = True, force_first_case_first = True):
    session_group = dict()
    small_trial_pathlist = get_targets_pathlist()

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
        if force_first_case_first:
            block_order.remove(0)
            block_order.insert(0,0)
        print(f"Session {s_idx}: Block Order: {block_order}")
        for b_idx in range(case_num): # one block per case
            
            trial_order = [i for i in range(trial_num_per_block)]
            if shuffle_trials: random.shuffle(trial_order)
            block_pathlist = []
            block_trial_index = []
            for t_idx in range(trial_num_per_block):
                block_pathlist.append(trial_pathlist[trial_order[t_idx]])

                block_trial_index.append(t_idx)

            block_case_config = case_configs[block_order[b_idx]]
            if override_cases_data_by_session_info:
                block_case_config["trial_time_length"] = session_configs[s_idx]["trial_time_length"]
            

            block_group[b_idx] = {"case": block_case_config, "trials": block_pathlist, "trial_index": block_trial_index, "session_label" : session_configs[s_idx]["label"], "session_trial_time" : session_configs[s_idx]["trial_time_length"]}

        session_group[s_idx] = block_group
    return session_group

if __name__ == '__main__':
    # https://python.land/data-processing/python-yaml
    print("Generating calibration session group...")
    calibration_session_group = generate_session_group(calibration_session_cases, calibration_block_cases)
    print("Generating experiment session group...")
    experiment_session_group = generate_session_group(experiment_session_cases, experiment_block_cases, shuffle_blocks=True, shuffle_trials=True)
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    resources_directory = os.path.join(current_directory, "res")
    calibration_session_path = os.path.join(resources_directory, "calibration_session_group.yaml")
    experiment_session_path = os.path.join(resources_directory, "experiment_session_group.yaml")

    # Create directories if they don't exist
    for directory in [resources_directory]:
        print(f"checking directory {directory}...")
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created {directory}!")

    os.chmod(resources_directory, 0o777)  # Set full permissions
    

    with open(calibration_session_path, 'w') as outfile:
        yaml.dump(calibration_session_group, outfile)

    with open(experiment_session_path, 'w') as outfile:
        yaml.dump(experiment_session_group, outfile)
