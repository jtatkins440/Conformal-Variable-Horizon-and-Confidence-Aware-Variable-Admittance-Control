#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import os
import numpy as np
from sensor_msgs.msg import JointState
import time
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float64MultiArray
from enum import Enum
from conf_exps.srv import *
from conf_exps.msg import *
from geometry_msgs.msg import WrenchStamped, Quaternion, PoseStamped

# helper functions
def get_valid_int_input(request_string):
    b_invalid_input = True
    while b_invalid_input:
        try:
            raw_num_input = input(f"{request_string}: \n-- ")
            int_num = int(raw_num_input)
            b_invalid_input = False
        except ValueError as e:
            print(f"Invalid number! Expected an integer, got {raw_num_input}. Please try again.")
    return int_num

def get_valid_bool_input(request_string):
    b_invalid_input = True
    bool_dict = {0: True, 1: False}
    while b_invalid_input:
        try:
            raw_num_input = input(f"{request_string}: {bool_dict}\n-- ")
            int_num = int(raw_num_input)
            valid_num = bool_dict[int_num]
            b_invalid_input = False
        except ValueError as e:
            print(f"Invalid number! Expected an integer, got {raw_num_input}. Please try again.")
    return valid_num

def get_valid_int_range_input(request_string, min_val = 0, max_val = 10):
    valid_range_list = [i for i in range(min_val, max_val)]
    b_invalid_input = True
    while b_invalid_input:
        try:
            raw_num_input = input(f"{request_string} in range [{min_val}, {max_val}]: \n-- ")
            int_num = int(raw_num_input)
            valid_num = valid_range_list[int_num]
            b_invalid_input = False
        except ValueError as e:
            print(f"Invalid number! Expected an integer, got {raw_num_input}. Please try again.")
        except IndexError as e:
            print(f"Invalid range! Expected an integer within [{min_val}, {max_val}], got {raw_num_input}. Please try again.")
    return valid_num

def get_valid_int_key_input(request_string, in_dict):
    #valid_list = keys_list #[i for i in range(min_val, max_val)]
    b_invalid_input = True
    while b_invalid_input:
        try:
            raw_key = input(f"{request_string} in  {in_dict.keys()}: \n-- ")
            valid_key = int(raw_key)#valid_range_list[int_num]
            temp = in_dict[valid_key]
            b_invalid_input = False
        except ValueError as e:
            print(f"Invalid number! Expected a string, got {raw_key}. Please try again.")
        except KeyError as e:
            print(f"Invalid key! Expected a string within {in_dict.keys()}, got {raw_key}. Please try again.")
    return valid_key

def get_valid_str_key_input(request_string, in_dict):
    #valid_list = keys_list #[i for i in range(min_val, max_val)]
    b_invalid_input = True
    while b_invalid_input:
        try:
            raw_key = input(f"{request_string} in  {in_dict.keys()}: \n-- ")
            valid_key = str(raw_key)#valid_range_list[int_num]
            temp = in_dict[valid_key]
            b_invalid_input = False
        except ValueError as e:
            print(f"Invalid number! Expected a string, got {raw_key}. Please try again.")
        except KeyError as e:
            print(f"Invalid key! Expected a string within {in_dict.keys()}, got {raw_key}. Please try again.")
    return valid_key

# Define the individual states
class Initial_State(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['initalized_subject', 'failed'], output_keys=['subject_num'])

    def execute(self, userdata):
        # Initialize relevant variables
        int_subject_num = get_valid_int_input("Enter subject number")

        rospy.set_param('current_subject_id', int_subject_num)
        userdata.subject_num = int_subject_num
        return 'initalized_subject'

class Home_State(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['begin_calibration', 'begin_session', 'move_to_origin', 'finished', 'failed'], input_keys=['subject_num'],
                             output_keys=['subject_num', 'session_index', 'block_index', 'trial_index'])

    def execute(self, userdata):
        # Initialize relevant variables
        command = int(input("Enter desired command: \n - 0: Exit\n - 1: Move robot to origin\n - 2: Begin calibration trials\n - 3: Begin trial session\n"))
        if command == 0:
            return 'finished'
        elif command == 1:
            return 'move_to_origin'
        elif command == 2:
            userdata.session_index = 0
            userdata.block_index = 0
            return 'begin_calibration'
        elif command == 3:
            return 'begin_session'
        return 'failed'

class Move_To_Origin(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['centered', 'failed'])
        self.origin_joint_angles = [-1.5708, 1.5708, 0, 1.5708, 0, -1.5708, -0.958709]
        self.current_joint_angles = []

        # publishers
        self.desired_joint_pub = rospy.Publisher('/iiwa/PositionController/command', Float64MultiArray, queue_size=1)

        # subscribers
        self.current_joint_sub = rospy.Subscriber('/iiwa/joint_states', JointState, self.current_joint_callback)

        # Setting controller behaviour
        self.controller_toggle_srv = rospy.ServiceProxy('admit/set_admittance_controller_behavior', SetInt) # CHANGE THIS

        # Toggling IK node
        self.ik_toggle_srv = rospy.ServiceProxy('/ik/toggle_publishing', SetBool)



    def lerp(self, A, B, t):
        return [a + (b - a)*t for a, b in zip(A, B)]

    def calculate_max_velocity(self, start_angles, end_angles, total_duration):
        maxvel = max([abs(end - start) for start, end in zip(start_angles, end_angles)]) / total_duration
        return maxvel

    def current_joint_callback(self, msg):
        self.current_joint_angles = msg.position

    def toggle_ik(self, enable):
        try:
            # resp_controller = self.controller_toggle_srv(enable)
            resp_ik = self.ik_toggle_srv(enable)

            if resp_ik.success:
                rospy.loginfo("Node Successfully Toggled: {}".format("ON" if enable else "OFF"))
            else:
                rospy.loginfo("Toggle Failed: {}".format(resp_ik.message))
        except rospy.ServiceException as e:
            rospy.logerr("Service Call Failed: {}".format(e))

    def set_controller_behaviour(self, value):
        try:
            resp_controller = self.controller_toggle_srv(value)
            if resp_controller.success:
                rospy.loginfo("Controller Behaviour set with value: {}".format(value))
            else:
                rospy.loginfo("Controller Behaviour couldn't be set (It's going to through a phase): {}"
                              .format(resp_controller.message))
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

    def move_to_origin(self):
        if not self.current_joint_angles:
            rospy.logwarn("Current Joint Angles Unavailable: Cannot move to Origin")
            return 'failed'

        # Using a timed interval instead of sleep()
        rospy.loginfo("Current Joint Angles: {}".format(self.current_joint_angles))
        total_duration = 5.0
        max_velocity = self.calculate_max_velocity(self.current_joint_angles, self.origin_joint_angles, total_duration)
        velocity_threshold = 0.25
        while max_velocity > velocity_threshold:
            total_duration += 1
            max_velocity = self.calculate_max_velocity(self.current_joint_angles, self.origin_joint_angles, total_duration)

        freq = 20
        num_points = int(total_duration*freq)
        time_points = np.linspace(0.0, 1.0, num_points)
        dt = total_duration / num_points

        trajectory = [self.lerp(self.current_joint_angles, self.origin_joint_angles, t) for t in time_points]

        start_time = time.time()
        # May have to make changes to make the message compatible with Desired Joint Space Topic
        for idx, joint_angles in enumerate(trajectory):
            joint_state_msg = Float64MultiArray()
            joint_state_msg.data = joint_angles
            while time.time() - start_time < idx * dt:
                pass
            self.desired_joint_pub.publish(joint_state_msg)
            #rospy.loginfo("Moving to Origin:- Current Joint Angle: {}".format(joint_angles))
            # time.sleep(dt)

        rospy.loginfo("Robot Centered")



    def execute(self, userdata):
        # Bring the robot to the center
        print(f"Executing Move_to_Origin state...")
        self.toggle_ik(False)
        self.set_controller_behaviour(0)
        self.move_to_origin()

        self.toggle_ik(True)

        return 'centered'


class Session_State(smach.State):
    def __init__(self, session_config, break_between_session = False, get_opt_sizes_on_startup = False, hold_after_blocks = False, block_holding_time = 6.0, sub_experiment_name = "Experiment"):
        smach.State.__init__(self, outcomes=['finished_session', 'failed'], input_keys=['subject_num', 'session_index', 'block_index', 'trial_index'],
                             output_keys=['subject_num', 'session_index', 'block_index', 'trial_index'])
        self.session_config = session_config
        self.break_between_session = break_between_session
        self.get_opt_sizes_on_startup = get_opt_sizes_on_startup
        self.hold_after_blocks = hold_after_blocks
        self.block_holding_time = block_holding_time

        self.initial_pose = np.array([0.0,-0.4231, 0.7589])
        self.current_state_index = 0
        self.guide_target = np.array([0.0, 0.0])

        self.pretrial_close_dist = rospy.get_param("pretrial_close_threshold")
        self.endtrial_close_dist = rospy.get_param("endtrial_close_threshold")

        #self.experiment_name = str(rospy.get_param("trial_data_logger/exp_name"))

        self.experiment_name = rospy.get_param("exp_name")
        self.sub_experiment_name = sub_experiment_name

        self.endEffector = np.array([0.0, 0.0])
        self.sub = rospy.Subscriber('/iiwa/admit_state', PoseStamped, self.end_effector_callback)
        #self.sub_guide = rospy.Subscriber('/GuideTargets', Float64MultiArray, self.guide_target_callback)
        self.sub_guide = rospy.Subscriber('/GuideTargets', GuideState, self.guide_target_callback)
        #self.sub = rospy.Subscriber('/iiwa/admit_state', PoseStamped, self.end_effector_callback)

        self.target_pub = rospy.Publisher('CurrentTargets', Float64MultiArray, queue_size=10)
        self.prev_target_pub = rospy.Publisher('PreviousTargets', Float64MultiArray, queue_size=10)
        self.trial_targets_pub = rospy.Publisher('TrialTargets', Float64MultiArray, queue_size=10)

        # for the logging node
        self.start_logger_service = rospy.ServiceProxy('start_logging', StartLogging)
        self.stop_logger_service = rospy.ServiceProxy('stop_logging', Trigger)

        # for the visualizer node
        self.start_guiding = rospy.ServiceProxy('start_guiding', SetFloat)
        self.set_guide_time = rospy.ServiceProxy('set_guide_time', SetFloat)
        self.new_trial = rospy.ServiceProxy('new_trial', SetInt)
        self.update_targets = rospy.ServiceProxy('update_targets', UpdateTargets)

        # for the guide node
        self.guide_start_guiding = rospy.ServiceProxy('guide/start_guiding', SetFloat)
        self.guide_set_guide_time = rospy.ServiceProxy('guide/set_guide_time', SetFloat)
        self.guide_update_targets = rospy.ServiceProxy('guide/update_targets', UpdateTargets)


        # for the ik node
        self.ik_toggle_orientation_srv = rospy.ServiceProxy('/ik/toggle_ignore_orientation', SetBool)
        self.ik_set_behavior_srv = rospy.ServiceProxy('/ik/set_ik_behavior', SetInt)

        # for the confidnece admittance controller
        self.set_running_behavior_srv = rospy.ServiceProxy("admit/set_running_behavior", SetBool)
        self.set_admittance_controller_behavior_srv = rospy.ServiceProxy("admit/set_admittance_controller_behavior", SetInt)
        self.set_holding_behavior_srv = rospy.ServiceProxy("admit/set_holding_behavior", SetBool)
        self.trigger_user_intent_bounds_srv = rospy.ServiceProxy("admit/trigger_user_intent_bounds", Trigger)

        self.set_fixed_setpoint_srv = rospy.ServiceProxy("mint/set_fixed_setpoint", SetFloat)
        self.set_use_variable_setpoint_bool_srv = rospy.ServiceProxy("mint/set_use_variable_setpoint", SetBool)
        self.trigger_opt_step_sizes_srv = rospy.ServiceProxy("mint/trigger_get_opt_step_sizes", Trigger)
        
        self.exp_dir = rospy.get_param("/trial_data_logger/exp_name")

    def getInitialState(self, userdata):
        continue_from_last = get_valid_bool_input(f"Continue from session {userdata.session_index}, block {userdata.block_index}?")
        if continue_from_last:
            return userdata.session_index, userdata.block_index, 0
        else:
            return self.getStartingIndexSet()
        

    def getStartingIndexSet(self):
        starting_session_key = get_valid_int_key_input("Enter starting session index", self.session_config) #str(get_valid_int_range_input("Enter starting session index", min_val=0, max_val=len(self.session_config.keys())))
        starting_block_key = get_valid_int_key_input("Enter starting block index", self.session_config[starting_session_key]) #str(get_valid_int_range_input("Enter starting block index", min_val=0, max_val=len(self.session_config[starting_session_key].keys())))
        starting_trial_key = get_valid_int_range_input("Enter starting trial index (UNUSED!)", min_val=self.session_config[starting_session_key][starting_block_key]["trial_index"][0], 
                                                       max_val=self.session_config[starting_session_key][starting_block_key]["trial_index"][-1])
        return starting_session_key, starting_block_key, starting_trial_key
    
    def end_effector_callback(self, msg):
        current_pose_msg = msg.pose.position # RELATIVE TO BASE
        abs_pose = np.array([current_pose_msg.x, current_pose_msg.y, current_pose_msg.z]) - self.initial_pose
        
        self.endEffector[0] = abs_pose[2]
        self.endEffector[1] = abs_pose[0]

    '''
    def guide_target_callback(self, msg):
        self.guide_target = np.array([[msg.data[0]], [msg.data[1]]])
        #print(self.guide_target)
        return 
    '''

    def guide_target_callback(self, msg):
        self.guide_target = np.array([msg.position.x, msg.position.y])
        return 
    
    # {'label': 'SAC Fixed Window 500ms', 'ac_type': 'Static', 'running': True, "fixed_time_index": 10, "use_var_time": False}
    def setExperimentState(self, case_dict):
        running_val = case_dict['running']
        admit_val = case_dict['ac_type']
        fixed_index_val = case_dict['fixed_time_horizon']
        var_time_val = case_dict['use_var_time']

        running_resp = self.set_running_behavior_srv(running_val)
        admit_resp = self.set_admittance_controller_behavior_srv(admit_val)
        try:
            fixed_resp = self.set_fixed_setpoint_srv(fixed_index_val)
            var_resp = self.set_use_variable_setpoint_bool_srv(var_time_val)
        except:
            print("Couldn't update setpoint!")
        

        print(f"Changed to {case_dict['label']}.")
        #print(f"Running Flag Changed to {running_val} with sucess: {running_resp.success} and message {running_resp.message}.")
        #print(f"Admit Index Changed to {admit_val} with sucess: {admit_resp.success} and message {admit_resp.message}.")
        print(f"Fixed Time Index Changed to {fixed_index_val} with sucess: {fixed_resp.success} and message {fixed_resp.message}.")
        print(f"Variable Time Flag Changed to {var_time_val} with sucess: {var_resp.success} and message {var_resp.message}.")

        return

    def is_close_enough(self, coord1, coord2, close_radius = 0.025):
        distance = np.linalg.norm(coord1 - coord2)
        #print(f"||{coord1} - {coord2}|| = {distance}) < {close_radius} -> {distance <= close_radius}")
        return distance <= close_radius
    
    def send_target_update(self, x, y, timeout = None):
        try:
            resp = self.update_targets(x, y)
            resp_2 = self.guide_update_targets(x, y)
            return resp.success
        except Exception as e:
            print("Service call failed: %s" % e)
            return False

    def new_trial_service(self, val):
        try:
            resp = self.new_trial(val)
            if not resp.success:
                rospy.logerr("Failed to communicate new trial")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def set_visualizer_flag(self, flag):
        try:
            set_vis = rospy.ServiceProxy('start_visualizing', SetBool)
            resp = set_vis(flag)
            #print(f"Set visualizer flag response: {resp.success}")
            return resp.success
        except Exception as e:
            print("Service call failed: %s" % e)
            return False
        
    def start_guiding_service(self, val):
        try:
            #resp = self.start_guiding(val)
            resp = self.guide_start_guiding(val)
            #print("Guiding service succeed!")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    '''
    def start_logging(self, subject_num, trial, method=None, trial_type=None):
        try:
            resp = self.start_logger_service(subject_num, trial, method, trial_type)
            if not resp.success:
                rospy.logerr("Failed to start logging.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
    '''

    def start_logging(self, subject_num, trial_type, subfolder_name, trial):
        try:
            resp = self.start_logger_service(subject_num, trial_type, subfolder_name, trial)
            if not resp.success:
                rospy.logerr("Failed to start logging.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def stop_logging(self):
        try:
            resp = self.stop_logger_service()
            if not resp.success:
                rospy.logerr("Failed to stop logging.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def publish_previous_target(self, target):
        target_msg = Float64MultiArray()
        try:
            target_msg.data = [target[0][0], target[1][0]]
        except:
            target_msg.data = [target[0], target[1]]
        self.prev_target_pub.publish(target_msg)
        return

    def publish_current_target(self, target):
        target_msg = Float64MultiArray()
        try:
            target_msg.data = [target[0][0], target[1][0]]
        except:
            target_msg.data = [target[0], target[1]]
        self.target_pub.publish(target_msg)
        return
    
    def block_until_close(self, close_target, use_guide_target = False, sleep_rate = 20.0, close_radius = 0.025, publish = True):
        rate = rospy.Rate(sleep_rate)
        if use_guide_target:
            while (not self.is_close_enough(close_target, self.guide_target, close_radius=close_radius)) and (not rospy.is_shutdown()):
                if publish: self.publish_current_target(close_target)
                rate.sleep()
        else:
            while (not self.is_close_enough(close_target, self.endEffector, close_radius=close_radius)) and (not rospy.is_shutdown()):
                if publish: self.publish_current_target(close_target)
                rate.sleep()
        return
    
    def buildSubjectDirectories(self, session_config, exp_name, exp_type, subject_idx):
        print(f"SM: Building directories for {exp_name}, subject {subject_idx}, {exp_type}...")
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory, _ = os.path.split(current_directory)
        data_directory = os.path.join(parent_directory, 'DATA')
        experiment_directory = os.path.join(data_directory, exp_name) # save this one and on
        subject_directory = os.path.join(experiment_directory, f'subject_{str(subject_idx)}')
        sub_experiment_directory = os.path.join(subject_directory, exp_type) # save this one and on

        for s_idx in range(0, len(session_config.keys())):
            #print(f"On session {s_idx}...")
            current_session_config = session_config[s_idx]
            for b_idx in range(0, len(current_session_config.keys())):
                #print(f"On block {b_idx}...")

                session_directory = os.path.join(sub_experiment_directory, f'session_{str(s_idx)}')
                block_directory = os.path.join(session_directory, f'block_{str(b_idx)}')

                # Create directories if they don't exist
                for directory in [data_directory, experiment_directory, sub_experiment_directory, subject_directory, session_directory, block_directory]:
                    print(f"checking directory {directory}...")
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        print(f"created {directory}!")
                        os.chmod(directory, 0o777)  # Set full permissions
                
                # save the block configs
                current_block_config = current_session_config[b_idx]
                block_config_path = os.path.join(block_directory, "block_config.yaml")
                with open(block_config_path, 'w') as outfile:
                    yaml.dump(current_block_config, outfile)

            # save the session_configs
            session_config_path = os.path.join(session_directory, "session_config.yaml")
            with open(session_config_path, 'w') as outfile:
                yaml.dump(current_session_config, outfile)
        print(f"Finished!")
        return subject_directory

    def setHoldingUntilTime(self, release_at_end = False, holding_time = 3.0, sleep_rate = 10.0, origin = np.zeros(shape=(2,1))):
        rate = rospy.Rate(sleep_rate)
        print("Triggering holding to origin!")
        holding_resp = self.set_holding_behavior_srv(True)
        self.block_until_close(origin, sleep_rate=sleep_rate, publish=False)
        hold_time = time.time()
        while time.time() - hold_time < holding_time:
            rate.sleep()
        if release_at_end:
            print("Releasing hold!")
            holding_resp = self.set_holding_behavior_srv(False)
        return

    def runBlockGroup(self, userdata, block_dict, start_trial, s_idx, b_idx):
        #block_group[b_idx] = {"case": case_configs[block_order[b_idx]], "trials": block_pathlist, "trial_index": block_trial_index}

        #print(f"block_dict.keys(): {block_dict.keys()}")
                
        self.setExperimentState(block_dict["case"])
        holding_resp = self.set_holding_behavior_srv(False)

        # pretrial work
        # Load target data from csv file
        self.ik_toggle_orientation_srv(True)
        # Set IK to Open Loop
        self.ik_set_behavior_srv(1)
        #Trigger Force offset Service 
        #self.force_offset_service()

        trial_type = block_dict["case"]['label'] # this needs to be a string!

        rospy.set_param("/experiment_name", self.experiment_name)
        rospy.set_param("/subject_num", userdata.subject_num)
        rospy.set_param("/trial_type", trial_type)

        

        pathlist = block_dict["trials"]
        #print(f"pathlist: {pathlist}")
        patharray = np.array(pathlist)
        #print(f"patharray.shape: {patharray.shape}")

        
        targetXY = np.array([0, 0])
        targetXYold = np.array([0, 0])
        origin_target = np.array([0.0, 0.0])
        sanity_target = np.array([0.1, 0.0])

        service_wait_timeout = 0.5
        
        total_trials = len(block_dict['trial_index'])
        for trial in range(start_trial, total_trials):
            print(f"On trial {trial}...")

            trial_targets = patharray[trial, :, :]
            target_num = trial_targets.shape[0]
            target_x_gui = trial_targets[:, 0]
            target_y_gui = trial_targets[:, 1]
            targets = [trial_targets[tar, :] for tar in range(target_num)]
            #start_gui_service_time = time.time()
            #timeout = 5
            #print("Before self.send_target_update()...")
            
            #rsp = self.set_guide_time(block_dict['session_trial_time'])
            rsp = self.guide_set_guide_time(block_dict['session_trial_time'])
            #print(rsp.message)

            resp_gui_result = self.send_target_update(target_x_gui, target_y_gui, timeout = service_wait_timeout)
            #if resp_gui_result:
            #    print("Trial Number sent to Visualizer")
            #else:
            #    print("!!! ERROR: Trial number not sent to visualizer!!!")

            ## pretrial setup
            # turn on visualizer
            self.set_visualizer_flag(True)

            # publish the initial values for the topics
            self.publish_previous_target(origin_target)
            self.publish_current_target(sanity_target)

            # wait until they get to the origin
            #print("block until origin target")
            self.block_until_close(origin_target)

            # then let them move to the sanity target
            #print("block until sanity target")
            self.block_until_close(sanity_target)

            # then let them move back to the origin
            #print("block until origin target again")
            self.block_until_close(origin_target, close_radius = self.pretrial_close_dist)

            # then start the data logging
            #print("start everything") # session_string
            #subject_num, trial_type, subfolder_name, trial

            '''
            string experiment_name
            string sub_experiment_name
            int32 subject_num
            int32 session_idx
            int32 block_idx
            int32 trial_num
            '''
            resp = self.start_logger_service(self.experiment_name, self.sub_experiment_name, userdata.subject_num, s_idx, b_idx, trial)
            #self.start_logging(userdata.subject_num, trial_type, block_dict['trial_index'][trial]) # fitting_method index unused!
            # and start the guiding 
            self.start_guiding_service(0.0) # setfloat unused here!
            # Communicate start of new trials to visualizer
            self.new_trial_service(1)

            ## starts the actual trials
            for target_count, target in enumerate(targets):
                #print(f"in trial {target_count}")
                #print(f"Target No. {target_count + 1} out of {target_num}")
                targetXYold = targetXY
                targetXY = target#.reshape(2, 1)

                self.publish_previous_target(targetXYold)
                self.publish_current_target(targetXY)

                #self.block_until_close(targetXY)

                if target_count < (target_num - 1):
                    self.block_until_close(targetXY, use_guide_target=True, close_radius=0.025)
                else:
                    self.block_until_close(targetXY, close_radius=self.endtrial_close_dist)

            # Wait for 5 seconds before the next trial
            targetXY = np.array([[0], [0]])
            # Trigger Trial Data Logger to close and save the file
            self.stop_logging()
            #self.set_visualizer_flag(False)
            #time.sleep(1)
            if rospy.is_shutdown():
                break
    
    def execute(self, userdata):
        
        starting_session_key, starting_block_key, starting_trial_key = self.getInitialState(userdata) #self.getStartingIndexSet()
        
        # make sure it's holding on startup
        holding_resp = self.set_holding_behavior_srv(True)

        

        subject_dir = self.buildSubjectDirectories(self.session_config, self.experiment_name, self.sub_experiment_name, userdata.subject_num)

        cal_dir = os.path.join(subject_dir, "Calibration")
        print(cal_dir)
        rospy.set_param("/calibration_directory", cal_dir) # used by quantile server to compute stuff
        time.sleep(0.5)

        if self.get_opt_sizes_on_startup:
            self.trigger_opt_step_sizes_srv()
            self.trigger_user_intent_bounds_srv()

        for s_idx in range(int(starting_session_key), len(self.session_config.keys())):
            print(f"On session {s_idx}...")
            current_session_config = self.session_config[s_idx]
            for b_idx in range(int(starting_block_key), len(self.session_config[starting_session_key].keys())):
                print(f"On block {b_idx}...")
                current_block_config = current_session_config[b_idx]
                self.runBlockGroup(userdata, current_block_config, starting_trial_key, s_idx, b_idx)

                if self.hold_after_blocks: self.setHoldingUntilTime(holding_time=self.block_holding_time)
                starting_trial_key = 0
                userdata.block_index = b_idx
            userdata.session_index = s_idx + 1
            userdata.block_index = 0
            
            if self.break_between_session: 
                break
        if userdata.session_index == len(self.session_config.keys()):
            userdata.session_index = 0
        #running_resp = self.set_running_behavior_srv(False) # deactive main loop when the sessions aren't running! self.runBlockGroup will turn it back on.
        #holding_resp = self.set_holding_behavior_srv(True)
            
        self.setHoldingUntilTime(holding_time=self.block_holding_time)
        running_resp = self.set_running_behavior_srv(False)
        return 'finished_session'

import yaml
import os
def main():
    rospy.init_node('exp_protocol_controller')

    res_folder = "res/"
    calib_session_yaml_name = res_folder + "calibration_session_group.yaml"
    exp_session_yaml_name = res_folder + "experiment_session_group.yaml"

    current_directory = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(current_directory, calib_session_yaml_name)) as file:
        calib_session_config = yaml.safe_load(file)
        #calib_session_config = json.load(json_file)

    with open(os.path.join(current_directory, exp_session_yaml_name)) as file:
        exp_session_config = yaml.safe_load(file)

    print(f"Calibration session configuration: {calib_session_config}")

    while not rospy.is_shutdown():
        # Create a top-level state machine
        sm_top = smach.StateMachine(outcomes=['success', 'failure'], input_keys=[], output_keys=[])

        sm_top.userdata.experiment_method = -1
        sm_top.userdata.subject_num = -1
        sm_top.userdata.session_index = 0
        sm_top.userdata.block_index = 0
        sm_top.userdata.trial_index = 0
        sm_top.userdata.trial_num = -1

        with sm_top:
            smach.StateMachine.add('INITIAL_STATE', Initial_State(),
                                transitions={'initalized_subject': 'HOME_STATE', 'failed': 'failure'})

            smach.StateMachine.add('HOME_STATE', Home_State(),
                                transitions={'begin_session' : 'SESSION_STATE', 'begin_calibration': 'CALIB_STATE', 'move_to_origin': 'MOVE_TO_ORIGIN', 'failed': 'failure', 'finished': 'success'})

            smach.StateMachine.add('MOVE_TO_ORIGIN', Move_To_Origin(),
                                transitions={'centered': 'HOME_STATE', 'failed': 'failure'})

            smach.StateMachine.add('CALIB_STATE', Session_State(calib_session_config, sub_experiment_name="Calibration"),
                                transitions={'finished_session': 'HOME_STATE', 'failed': 'failure'})
            
            smach.StateMachine.add('SESSION_STATE', Session_State(exp_session_config, break_between_session=True, get_opt_sizes_on_startup = True, hold_after_blocks = True, sub_experiment_name="Experiment"),
                                transitions={'finished_session': 'HOME_STATE', 'failed': 'failure'})
            
            
        # Create and start the introspection server
        sis = smach_ros.IntrospectionServer('server_name', sm_top, '/SM-ROOT')
        sis.start()

        # Execute the state machine
        try:
            outcome = sm_top.execute()
        except rospy.ROSInterruptException:
            pass
        finally:
            sis.stop()
            rospy.signal_shutdown("State Machine execution completed or terminated")

        # Wait for ctrl-c to stop the application
        


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

