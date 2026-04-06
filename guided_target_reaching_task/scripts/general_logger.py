#!/usr/bin/env python3

import rospy
import os
from pathlib import Path
import h5py
import time
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Float64MultiArray
from conf_exps.srv import *
from geometry_msgs.msg import PoseStamped, WrenchStamped, TwistStamped
from conf_exps.msg import * #AdmitStateStamped
from utils import *

class SyncDataLogger:
    def __init__(self):
        self.nh = rospy.init_node('trial_data_logger')
        self.file_handle = None
        self.data_group = None
        self.start_time = None
        self.hz = 200.0
        self.cycle_time = 1.0 / (1.0 * self.hz)
        self.close_flag = False
        self.record_state_callback_time = None

        rospy.Service('start_logging', StartLogging, self.handle_start_logging)
        rospy.Service('stop_logging', Trigger, self.handle_stop_logging)

        self.logged_items = {
            'elapsed_time': None,
            'ee_pose' : None,
            'ee_vel' : None,
            'ee_acc' : None,
            'ee_wrench' : None,
            'ee_robot_wrench' : None,
            'ee_pose_eq' : None,
            'ee_vel_eq' : None,
            't_eq' : None,
            'stiffness' : None,
            'vel_stiffness' : None,
            'damping' : None,
            'user_intent' : None,
            'mint_output' : None,
            'icp_quantiles' : None,
            'icp_alphas' : None,
            'CurrentTargets' : None,
            'guide_pos' : None,
            'guide_vel' : None,
            'guide_active' : None,
            'scores' : None,
            'aci_quantiles': None,
            'aci_alphas': None,
            'aci_gammas': None,
            'confidence': None,
            'q_ratio': None,
            'cycle_time': None}

        self.subscribers = {
            'admit_record_state': rospy.Subscriber('admit_record_state', AdmitRecordStateStamped, self.admitRecordCallback, queue_size=1, tcp_nodelay=True),
            'mint_conf_record_state': rospy.Subscriber('mint_conf_record_state', MIntConfRecordStateStamped, self.mintRecordCallback, queue_size=1, tcp_nodelay=True),
            '/CurrentTargets': rospy.Subscriber('/CurrentTargets', Float64MultiArray, self.currentTargetsCallback, queue_size=1, tcp_nodelay=True),
            '/GuideTargets': rospy.Subscriber('/GuideTargets', GuideState, self.currentGuideCallback, queue_size=1, tcp_nodelay=True)
        }
        
        # '/GuideTargets': rospy.Subscriber('/GuideTargets', Float64MultiArray, self.current_guide_callback, queue_size=1, tcp_nodelay=True)
    def admitRecordCallback(self, msg):
        self.logged_items['ee_pose'] = [msg.pose.position.x, msg.pose.position.z]
        self.logged_items['ee_vel'] = [msg.twist.linear.x, msg.twist.linear.z]
        self.logged_items['ee_acc'] = [msg.accel.linear.x, msg.accel.linear.z]
        self.logged_items['ee_wrench'] = [msg.wrench.force.x, msg.wrench.force.z]
        self.logged_items['ee_robot_wrench'] = [msg.robot_wrench.force.x, msg.robot_wrench.force.z]
        
        self.logged_items['ee_pose_eq'] = [msg.eq_state.position.x, msg.eq_state.position.z]
        self.logged_items['ee_vel_eq'] = [msg.eq_state.position.x, msg.eq_state.position.z]

        self.logged_items['stiffness'] = [msg.stiffness]
        self.logged_items['vel_stiffness'] = [msg.vel_stiffness]
        self.logged_items['damping'] = [msg.damping]
        self.logged_items['user_intent'] = [msg.user_intent]
        self.logged_items['admit_cycle_time'] = [msg.cycle_time]
        self.logged_items['confidence'] = [msg.confidence]
        return

    def mintRecordCallback(self, msg):
        
        self.logged_items['t_eq'] = [msg.t_eq]
        self.logged_items['pred_traj'] = unpack_multiarray_msg(msg.pred_traj)
        self.logged_items['scores'] = list(msg.scores) # already a list!
        self.logged_items['q_ratio'] = [msg.q_ratio]

        aci_sample_list = msg.aci_state.aci_batch
        aci_quantiles_list = []
        aci_alphas_list = []
        aci_gammas_list = []
        for idx in range(len(aci_sample_list)):
            aci_quantiles_list.append(unpack_multiarray_msg(aci_sample_list[idx].quantiles))
            aci_alphas_list.append(unpack_multiarray_msg(aci_sample_list[idx].alphas))
            aci_gammas_list.append(aci_sample_list[idx].gammas)
        #print(f"logger: recieved aci_quantiles list as:\n{aci_quantiles_list}")
        self.logged_items['aci_quantiles'] = np.stack(aci_quantiles_list, axis=-1)
        self.logged_items['aci_alphas'] = np.stack(aci_alphas_list, axis=-1)
        self.logged_items['aci_gammas'] = np.stack(aci_gammas_list, axis=-1)
        self.logged_items['mint_cycle_time'] = [msg.cycle_time]
        return

    # Callbacks for each topic

    def currentTargetsCallback(self, msg):
        self.logged_items['CurrentTargets'] = list(msg.data)

    def currentGuideCallback(self, msg):
        self.logged_items['guide_pos'] = [msg.position.x, msg.position.y]
        self.logged_items['guide_vel'] = [msg.velocity.x, msg.velocity.y]
        self.logged_items['guide_active'] = [msg.active_coord]
        self.logged_items['guide_pos_seq'] = np.array([msg.pos_x, msg.pos_y])
        self.logged_items['guide_vel_seq'] = np.array([msg.vel_x, msg.vel_y])


    def handle_start_logging(self, req):
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

            current_directory = os.path.dirname(os.path.abspath(__file__))
            parent_directory, _ = os.path.split(current_directory)
            data_directory = os.path.join(parent_directory, 'DATA')
            experiment_directory = os.path.join(data_directory, req.experiment_name) # save this one and on
            subject_directory = os.path.join(experiment_directory, f'subject_{str(req.subject_num)}')
            sub_experiment_directory = os.path.join(subject_directory, req.sub_experiment_name) # save this one and on
            session_directory = os.path.join(sub_experiment_directory, f'session_{str(req.session_idx)}')
            block_directory = os.path.join(session_directory, f'block_{str(req.block_idx)}')

            # Create directories if they don't exist
            ## 2. create the file handle to the trial
            trial_file_path = os.path.join(block_directory, f'trial_{str(req.trial_num)}.h5')
            self.file_handle = h5py.File(trial_file_path, 'w')
            self.data_group = self.file_handle.create_group("TrialData")

            res.success = True
            res.message = "Started Logging Successfully"
        except Exception as e:
            res.success = False
            res.message = f"Failed to start logging{e}"
            rospy.logerr(f"handle_start_logging: {e}")
        finally:
            self.start_time = time.time() if res.success else None

        return res
    
    def handle_stop_logging(self, req):
        res = TriggerResponse()
        try:
            self.close_flag = True
            res.message = "Stopped Logging Successfully"
            res.success = True

        except Exception as e:
            res.success = False
            res.message = f"Could close current file: {e}"

        return res
    
    def close_current_file(self):
        if self.close_flag:
            self.file_handle.close()
            self.file_handle = None
            self.data_group = None
            self.start_time = None
            self.close_flag = False
        return 
    
    def wait_for_time(self, start_time_point):
        end_before_rest = time.time()
        elapsed_time = end_before_rest - start_time_point

        while elapsed_time < self.cycle_time:
            elapsed_time = time.time() - start_time_point

    def sync_log(self):
        if self.close_flag:
            self.close_current_file()
            return
        
        elif self.data_group is None or self.start_time is None:
            # rospy.logwarn("Data Group is not initialized. Skipping message processing")
            return

        #elif self.start_time is not None:
        elapsed_time = time.time() - self.start_time
        self.logged_items['elapsed_time'] = [elapsed_time]
        if self.data_group:
            for index, dataset_name in enumerate(self.logged_items.keys()):
                #dataset_name = topic_name.replace('/', '_').lstrip('_')
                
                if self.logged_items[dataset_name] is not None:
                    data_to_append = self.logged_items[dataset_name]

                    try:
                    # Check if the dataset exists, if so, append, otherwise create
                        if dataset_name in self.data_group:
                            # Append to the dataset
                            dataset = self.data_group[dataset_name]
                            if isinstance(data_to_append, np.ndarray):
                                try:
                                    dataset.resize([dataset.shape[0] + 1] + list(data_to_append.shape))
                                except:
                                    print(f"logger broke at resizing {dataset_name} with array {data_to_append}")
                                    raise Exception
                                #dataset.resize((dataset.shape[0] + 1, data_to_append.shape[0], data_to_append.shape[1]))
                            else:
                                dataset.resize((dataset.shape[0] + 1, len(data_to_append)))
                            dataset[-1] = data_to_append
                        else:
                            # Create a new dataset
                            try:
                                if isinstance(data_to_append, np.ndarray):
                                    self.data_group.create_dataset(dataset_name, data=[data_to_append],
                                                                maxshape=([None] + list(data_to_append.shape)), chunks=True)
                                    #self.data_group.create_dataset(dataset_name, data=[data_to_append],
                                    #                            maxshape=(None, data_to_append.shape[0], data_to_append.shape[1]), chunks=True)
                                else:
                                    self.data_group.create_dataset(dataset_name, data=[data_to_append],
                                                                maxshape=(None, len(data_to_append)), chunks=True)
                            except ValueError as ve:
                                print(f"data_to_append: {data_to_append}")
                                rospy.logerr(f"Failed to create dataset {dataset_name}: {ve}")
                    except KeyError as e:
                        rospy.logerr("Failed to access or create dataset: {}. Error: {}".format(dataset_name, e))
        else:
            rospy.loginfo("Logging has not been initiated")
        
        return

    def main(self):
        logger_rate = 200.0
        logger_dt = (1.0 / logger_rate)
        
        while not rospy.is_shutdown():
            start_time = time.time()
            self.sync_log()
            wait_for_time(start_time, logger_dt)
            #rate.sleep()

if __name__ == '__main__':
    logger = SyncDataLogger()
    logger.main()
