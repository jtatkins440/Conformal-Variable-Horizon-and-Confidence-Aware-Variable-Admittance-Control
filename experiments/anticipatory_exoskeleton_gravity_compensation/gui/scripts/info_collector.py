import math, os, h5py, time
import rospy
from geometry_msgs.msg import WrenchStamped, Quaternion, PoseStamped, Pose, TwistStamped, Point

from rehab_msgs.msg import RPYState, ConformalSetRadial, ConformalSetTrajRadial, ClassificationOutput, GUIInfo, ClassificationInfo, ProtocolState, EMGActivations
from rehab_msgs.srv import StartLogging, StartLoggingResponse
from std_srvs.srv import TriggerResponse, Trigger

import numpy as np


class InfoCollectorNode:
    def __init__(self, nh):
        self.nh = nh

        ### init from configs
        self.dt = rospy.get_param("dt")

        self.dt_rate = int(1.0 / self.dt)

        ### initialize subscribers and publishers
        self.global_queue_size = 25 # 50 # was 1
        self.sub_state = rospy.Subscriber("current_state", RPYState, self.callbackCurrentState, queue_size=self.global_queue_size, tcp_nodelay=True)
        self.sub_pred_state = rospy.Subscriber("predicted_state_sets", ConformalSetTrajRadial, self.callbackPredictState, queue_size=self.global_queue_size, tcp_nodelay=True)
        self.sub_class = rospy.Subscriber("classifier_output", ClassificationOutput, self.callbackClass, queue_size=self.global_queue_size, tcp_nodelay=True)
        self.sub_wrench_measured = rospy.Subscriber("torque_force_sensor_state", WrenchStamped, self.callbackTorqueMeasured, queue_size=self.global_queue_size, tcp_nodelay=True)
        self.sub_wrench_robot = rospy.Subscriber("torque_force_robot_state", WrenchStamped, self.callbackTorqueRobot, queue_size=self.global_queue_size, tcp_nodelay=True)

        self.sub_protocol_state = rospy.Subscriber("protocol_state", ProtocolState, self.callbackProtocolState, queue_size=self.global_queue_size, tcp_nodelay=True)
        #self.pub_emg = rospy.Publisher('/EMG/activation', EMGActivations, queue_size=self.global_queue_size, tcp_nodelay=True)
        self.sub_emg = rospy.Subscriber("/EMG/activation", EMGActivations, self.callbackEMG, queue_size=self.global_queue_size)

        self.pub_gui_info = rospy.Publisher("gui_info", GUIInfo, queue_size=self.global_queue_size)

        self.record_dict = dict()
        self.record_dict["torque_measured"] = [0.0, 0.0, 0.0]
        self.record_dict["torque_robot"] = [0.0, 0.0, 0.0]
        self.record_dict["position"] = [0.0, 0.0, 0.0]
        self.record_dict["velocity"] = [0.0, 0.0, 0.0]
        
        self.state_dict = dict()
        self.state_dict["starting_position"] = {}
        self.state_dict["starting_position"]["x"] = 0.0
        self.state_dict["starting_position"]["y"] = 0.0
        self.state_dict["starting_position"]["z"] = 0.0
        self.state_dict["goal_position"] = {}
        self.state_dict["goal_position"]["x"] = 0.0
        self.state_dict["goal_position"]["y"] = 0.0
        self.state_dict["goal_position"]["z"] = 0.0
        self.state_dict["behavior_label"] = "None"
        self.state_dict["countdown_time"] = 0.0
        

        prediction_number = 10
        self.pred_list = []
        for i in range(prediction_number):
            pred_dict = {"center": [0.0, 0.0, 0.0], "radius": 0.0, "time": 0.0, "is_safe": False}
            self.pred_list.append(pred_dict)

        self.prob_dict = {"Dynamic": {"probability": 0.0, "interp_index": -1}, "Static": {"probability": 1.0, "interp_index": 0}}
        self.protocol_dict = {"starting_position": {"x": -80.0, "y": 0.0, "z": 0.0},
                                 "goal_position": {"x": -80.0, "y": 0.0, "z": 0.0},
                                 "behavior_label": "Static",
                                 "countdown_time": 0.0}
        
        rospy.Service('start_logging', StartLogging, self.handleStartLogging)
        rospy.Service('stop_logging', Trigger, self.handleStopLogging)
        self.data_directory = rospy.get_param("data_dir")

        ### logging related
        self.file_handle = None
        self.data_group = None
        self.start_time = None
        self.record_state_callback_time = None

        self.is_logging_flag = False
        self.close_flag = False

        self.logged_items = {
            'elapsed_time' : None,
            'position' : None,
            'velocity' : None,
            'robot_torque' : None,
            'measured_torque' : None,
            'prediction_centers' : None,
            'prediction_radius' : None,
            'prediction_is_safe' : None,
            'class_prob' : None,
            'class_index' : None,
            'behavior_index' : None,
            'emgs' : None
            }

    ### subscriber callbacks
    def callbackEMG(self, msg):
        self.logged_items["emgs"] = np.array([msg.ad_value, msg.md_value, msg.pd_value])
        return 
    
    def callbackCurrentState(self, msg): # sub_state, RPYState
        self.record_dict["position"] = [msg.RPY.x, msg.RPY.y, msg.RPY.z]
        self.record_dict["velocity"] = [msg.angular.x, msg.angular.y, msg.angular.z]

        self.logged_items["position"] = np.array([msg.RPY.x, msg.RPY.y, msg.RPY.z])
        self.logged_items["velocity"] = np.array([msg.angular.x, msg.angular.y, msg.angular.z])
        return
    
    def callbackProtocolState(self, msg): # sub_state, RPYState
        #state_dict = dict()
        self.state_dict["starting_position"]["x"] = msg.starting_position.x
        self.state_dict["starting_position"]["y"] = msg.starting_position.y
        self.state_dict["starting_position"]["z"] = msg.starting_position.z
        self.state_dict["goal_position"]["x"] = msg.goal_position.x
        self.state_dict["goal_position"]["y"] = msg.goal_position.y
        self.state_dict["goal_position"]["z"] = msg.goal_position.z
        self.state_dict["behavior_label"] = msg.behavior_label
        self.state_dict["countdown_time"] = msg.countdown_time
        self.protocol_dict = self.state_dict
        behavior_index = 0
        if msg.behavior_label == "Static":
            behavior_index = 0
        else:
            behavior_index = 1
        

        ## Comment of Motion direction
        # motion_direction = 1 : moving up
        # motion_direction = 0 : moving down
        # motion_direction = 2 : dummy
        if behavior_index == 1 and msg.goal_position.y == -10:
            motion_direction = 1
        elif behavior_index == 1 and msg.goal_position.y == -40:
            motion_direction = 0
        else:
            motion_direction = 2
            #print("Fail to capture direction of motion")
        self.logged_items["behavior_index"] = np.array([behavior_index])
        self.logged_items["start_yaxis"] = np.array([msg.starting_position.y])
        self.logged_items["goal_yaxis"] = np.array([msg.goal_position.y])
        self.logged_items["direction_motion"] = np.array([motion_direction])
        return
    
    def callbackPredictState(self, msg): # sub_pred_state, ConformalSetTrajRadial
        center_a = np.zeros(shape=[len(msg.prediction_sets), 3])
        radius_a = np.zeros(shape=[len(msg.prediction_sets)])
        is_safe_a = np.zeros(shape=[len(msg.prediction_sets)])
        for i in range(len(msg.prediction_sets)):
            pset = msg.prediction_sets[i]
            self.pred_list[i]["center"] = [pset.center.x, pset.center.y, pset.center.z]
            self.pred_list[i]["radius"] = pset.radius
            self.pred_list[i]["time"] = pset.ahead_time
            self.pred_list[i]["is_safe"] = pset.is_safe
            center_a[i, :] = np.array([[pset.center.x, pset.center.y, pset.center.z]])
            radius_a[i] = np.array(pset.radius)
            is_safe_a[i] = np.array(pset.is_safe)
        self.logged_items["prediction_centers"] = center_a
        self.logged_items["prediction_radius"] = radius_a
        self.logged_items["prediction_is_safe"] = is_safe_a
        return
    
    def callbackClass(self, msg): # sub_class, ClassProbability
        prob_a = np.zeros(shape=[len(msg.classes)])
        class_index_a = np.zeros(shape=[len(msg.classes)])
        label_list = []
        for i in range(len(msg.classes)):
            self.prob_dict[msg.classes[i].label]["probability"] = msg.classes[i].probability
            prob_a[i] = np.array([msg.classes[i].probability])
            label_list.append(msg.classes[i].label)
            class_index_a[i] = i
        self.logged_items["class_prob"] = prob_a
        self.logged_items["class_index"] = class_index_a #np.array(label_list)
        return
    
    def callbackTorqueMeasured(self, msg):
        #self.torque_sensor = [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]

        self.logged_items["measured_torque"] = np.array([msg.wrench.torque.x, -msg.wrench.torque.z, -msg.wrench.torque.y])
        return
    
    def callbackTorqueRobot(self, msg):

        self.logged_items["robot_torque"] = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        return
    
    ### publisher methods
    def publishGuiInfo(self):
        msg = GUIInfo()
        msg.header.stamp = rospy.Time.now()
        msg.state.RPY.x = self.record_dict["position"][0]
        msg.state.RPY.y = self.record_dict["position"][1]
        msg.state.RPY.z = self.record_dict["position"][2]
        msg.state.angular.x = self.record_dict["velocity"][0]
        msg.state.angular.y = self.record_dict["velocity"][1]
        msg.state.angular.z = self.record_dict["velocity"][2]

        cmsg_list = []
        max_prob_index = 0
        max_prob = 0.0
        for i, key_label in enumerate(self.prob_dict.keys()):
            cmsg = ClassificationInfo()
            cmsg.label = key_label #self.prob_dict[key_label]["label"]
            cmsg.probability = self.prob_dict[key_label]["probability"]
            if max_prob < self.prob_dict[key_label]["probability"]:
                max_prob = self.prob_dict[key_label]["probability"]
                max_prob_index = i
            cmsg_list.append(cmsg)
        
        cout_msg = ClassificationOutput()
        cout_msg.classes = cmsg_list
        cout_msg.header.stamp = rospy.Time.now()
        cout_msg.expected_class_index = max_prob_index
        msg.classification = cout_msg

        pmsg_list = []
        for i in range(len(self.pred_list)):
            pmsg = ConformalSetRadial()
            pmsg.center.x = self.pred_list[i]["center"][0]
            pmsg.center.y = self.pred_list[i]["center"][1]
            pmsg.center.z = self.pred_list[i]["center"][2]
            pmsg.radius = self.pred_list[i]["radius"]
            pmsg.is_safe = self.pred_list[i]["is_safe"]
            pmsg.ahead_time = self.pred_list[i]["time"]
            pmsg_list.append(pmsg)

        ptraj_msg = ConformalSetTrajRadial()
        ptraj_msg.prediction_sets = pmsg_list
        msg.prediction_sets = ptraj_msg

        protocol_msg = ProtocolState()
        protocol_msg.starting_position.x = self.protocol_dict["starting_position"]["x"]
        protocol_msg.starting_position.y = self.protocol_dict["starting_position"]["y"]
        protocol_msg.starting_position.z = self.protocol_dict["starting_position"]["z"]
        protocol_msg.goal_position.x = self.protocol_dict["goal_position"]["x"]
        protocol_msg.goal_position.y = self.protocol_dict["goal_position"]["y"]
        protocol_msg.goal_position.z = self.protocol_dict["goal_position"]["z"]
        protocol_msg.behavior_label = self.protocol_dict["behavior_label"]
        protocol_msg.countdown_time = self.protocol_dict["countdown_time"]
        msg.protocol_state = protocol_msg
        self.pub_gui_info.publish(msg)
        return

    def step(self):
        ### save data
        self.syncLog()

        ### publish most recent saved info for the GUI
        self.publishGuiInfo()

        return 


    def handleStartLogging(self, req):
        res = StartLoggingResponse()

        '''
        string experiment_name
        string sub_experiment_name
        int32 subject_num
        int32 session_idx
        int32 block_idx
        int32 trial_num
        '''
        # Determine the directory based on trial type and method
        try:
            ## 1. create the folder for logging if it doesn't exist
            print("session_idx",req.session_idx)
            print("block_idx",req.block_idx)
            print("trial_num",req.trial_num)
            experiment_directory = os.path.join(self.data_directory, req.experiment_name) # save this one and on
            subject_directory = os.path.join(experiment_directory, f'subject_{str(req.subject_num)}')
            sub_experiment_directory = os.path.join(subject_directory, req.sub_experiment_name) # save this one and on
            session_directory = os.path.join(sub_experiment_directory, f'session_{str(req.session_idx)}')
            block_directory = os.path.join(session_directory, f'block_{str(req.block_idx)}')

            # Create directories if they don't exist
            # 2. 필요한 디렉토리 생성
            for directory in [experiment_directory, subject_directory, sub_experiment_directory, session_directory, block_directory]:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    os.chmod(directory, 0o777)  # 권한 설정
            
            ## 2. create the file handle to the trial
            trial_file_path = os.path.join(block_directory, f'trial_{str(req.trial_num)}.h5')
            self.file_handle = h5py.File(trial_file_path, 'w')
            self.data_group = self.file_handle.create_group("TrialData")
            print(f"File handle created as: {self.file_handle}")

            res.success = True
            res.message = "Started Logging Successfully"
            print("Try 블록 끝: success =", res.success)  # 1번
        except Exception as e:
            res.success = False
            res.message = f"Failed to start logging{e}"
            rospy.logerr(f"handle_start_logging: {e}")
            print("Except 블록: success =", res.success)  # 2번
        finally:
            before_time = self.start_time  # 3번
            self.start_time = time.time() if res.success else None
            self.close_flag = False
            self.is_logging_flag = True

            print(f"""Finally 블록 디버깅:
            - res.success: {res.success}
            - before start_time: {before_time}
            - after start_time: {self.start_time}
            - is_logging_flag: {self.is_logging_flag}
            """)

        return res
    
    def handleStopLogging(self, req):
        res = TriggerResponse()
        try:
            self.is_logging_flag = False
            self.close_flag = True
            res.message = "Set close file flag to True!"
            res.success = True

        except Exception as e:
            res.success = False
            res.message = f"Could close current file: {e}"

        return res
    
    def collectLoggedInfo(self):
        return


    def closeCurrentFile(self):

        self.file_handle.close()
        self.file_handle = None
        self.data_group = None
        self.start_time = None
        self.close_flag = False
        return 



    def syncLog(self):
        
        if self.is_logging_flag:

            if self.start_time is None:
                self.start_time = time.time()
            
            elapsed_time = time.time() - self.start_time
            self.logged_items['elapsed_time'] = [elapsed_time]
            #print("self.data_group",self.data_group)

            for dataset_name, data in self.logged_items.items():
                #print("dataset_name",dataset_name)
                if data is not None:
                    rospy.logdebug(f"Dataset: {dataset_name}, Data shape: {np.shape(data)}")
                else:
                    rospy.logwarn(f"Dataset: {dataset_name} has no data to log.")

            for dataset_name, data_to_append in self.logged_items.items():
                if data_to_append is None:
                    continue
                try:
                    # Check if dataset exists
                    if dataset_name in self.data_group:
                        dataset = self.data_group[dataset_name]
                        if isinstance(data_to_append, np.ndarray):
                            # Resize and append for ndarray
                            dataset.resize([dataset.shape[0] + 1] + list(data_to_append.shape))
                        else:
                            # Resize and append for list/other types
                            dataset.resize((dataset.shape[0] + 1, len(data_to_append)))
                        dataset[-1] = data_to_append
                    else:
                        # Create a new dataset
                        if isinstance(data_to_append, np.ndarray):
                            self.data_group.create_dataset(
                                dataset_name, data=[data_to_append],
                                maxshape=([None] + list(data_to_append.shape)), chunks=True
                            )
                        else:
                            self.data_group.create_dataset(
                                dataset_name, data=[data_to_append],
                                maxshape=(None, len(data_to_append)), chunks=True
                            )
                except ValueError as ve:
                    rospy.logerr(f"Failed to create/resize dataset {dataset_name}: {ve}")
                except KeyError as ke:
                    rospy.logerr(f"Failed to access dataset {dataset_name}: {ke}")
                except Exception as e:
                    rospy.logerr(f"Unexpected error in syncLog for {dataset_name}: {e}")
        else:
            pass #print("Not logging.")
        
        if self.close_flag:
            self.closeCurrentFile()
        return
            #rospy.loginfo("Data group is not initialized. Logging skipped.")

    
    
if __name__ == '__main__':

    nh = rospy.init_node('info_collector', anonymous=True)

    node = InfoCollectorNode(nh)

    rate = rospy.Rate(node.dt_rate)

    while not rospy.is_shutdown():
        node.step()
        rate.sleep()
    node.closeCurrentFile()
