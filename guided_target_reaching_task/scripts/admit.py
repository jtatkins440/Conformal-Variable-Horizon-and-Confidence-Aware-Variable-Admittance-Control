#!/usr/bin/python3

import math
import rospy
from std_msgs.msg import String
from std_msgs.msg import MultiArrayDimension
from geometry_msgs.msg import WrenchStamped, Quaternion, PoseStamped, Pose, TwistStamped, Point
from conf_exps.msg import * 
from conf_exps.srv import * 
from std_srvs.srv import * 
import numpy as np
from collections import deque
import time
import json
import scipy as sp

import os

from admit_lib import *
from utils import *

### 
class Admit:
    def __init__(self, nh):
        self.nh = nh

        pos_dim = rospy.get_param("admit/pos_dim")
        self.pos_dim = pos_dim
        self.dt = rospy.get_param("admit/dt")
        self.dt_rate = int(1.0 / self.dt)

        self.I_base = rospy.get_param("admit/static/I")
        self.B_base = rospy.get_param("admit/static/B")

        self.vel_max = rospy.get_param("safety/vel_max") # make sure these are enforced in next state equations
        self.acc_max = rospy.get_param("safety/acc_max")
        self.force_max = rospy.get_param("safety/force_max")

        self.interaction_dynamics = FastAdmittanceController(I_diag=self.I_base, B_diag=self.B_base, K_p_diag=0.0, K_d_diag = 0.0, pos_dim=pos_dim, dt = self.dt, 
                                                             max_vel=self.vel_max, max_acc=self.acc_max, max_F=self.force_max)
        self.co_interaction_dynamics = FastAdmittanceController(I_diag=self.I_base, B_diag=self.B_base, K_p_diag=0.0, K_d_diag = 0.0, pos_dim=pos_dim, dt = self.dt, 
                                                                max_vel=self.vel_max, max_acc=self.acc_max, max_F=self.force_max)
        
        self.impedance_update_rule_zero = ImpedanceUpdateRule(B_diag=self.B_base, K_diag=0.0)
        self.impedance_update_rule_static = ImpedanceUpdateRule(B_diag=self.B_base, K_diag=0.0)

        k_holding = rospy.get_param("admit/holding/K")
        b_crit = 2.0 * self.I_base * math.sqrt(k_holding / self.I_base)
        

        self.impedance_update_rule = self.impedance_update_rule_zero

        self.user_intent_min = rospy.get_param("admit/user_intent/min_user_intent_range")
        self.user_intent_max = rospy.get_param("admit/user_intent/max_user_intent_range")
        self.user_intent_bound_mult = rospy.get_param("admit/user_intent/user_intent_bound_mult")
        self.impedance_update_rule_user_intent = UserIntentImpedanceUpdateRule(b_lb = rospy.get_param("admit/user_intent/B_low"), 
                                                    b_ub = rospy.get_param("admit/user_intent/B_high"), 
                                                    k_p_lb = rospy.get_param("admit/user_intent/Kp_low"), 
                                                    k_p_ub = rospy.get_param("admit/user_intent/Kp_high"), 
                                                    k_d_lb = rospy.get_param("admit/user_intent/Kd_low"), 
                                                    k_d_ub = rospy.get_param("admit/user_intent/Kd_high"), 
                                                    s_sensitivity = rospy.get_param("admit/user_intent/s_sen"), 
                                                    r_slope =  rospy.get_param("admit/user_intent/r_slope"), 
                                                    delta_offset = rospy.get_param("admit/user_intent/delta_offset"),
                                                    min_user_intent = rospy.get_param("admit/user_intent/min_user_intent_range"), 
                                                    max_user_intent = rospy.get_param("admit/user_intent/max_user_intent_range"))
        
        self.confidence_scalar = rospy.get_param("admit/conf/confidence_gain_speed") * self.dt
        self.impedance_update_rule_confidence = ConfidenceImpedanceUpdateRule(b_lb = rospy.get_param("admit/conf/B_low"), 
                                                    b_ub = rospy.get_param("admit/conf/B_high"), 
                                                    k_p_lb = rospy.get_param("admit/conf/Kp_low"), 
                                                    k_p_ub = rospy.get_param("admit/conf/Kp_high"), 
                                                    k_d_lb = rospy.get_param("admit/conf/Kd_low"), 
                                                    k_d_ub = rospy.get_param("admit/conf/Kd_high"))
        
        self.impedance_update_rule_confidence_user_intent = ConfidenceUserIntentImpedanceUpdateRule(b_lb = rospy.get_param("admit/conf_user_intent/B_low"), 
                                                    b_ub = rospy.get_param("admit/conf_user_intent/B_high"), 
                                                    k_p_lb = rospy.get_param("admit/conf_user_intent/Kp_low"), 
                                                    k_p_ub = rospy.get_param("admit/conf_user_intent/Kp_high"), 
                                                    k_d_lb = rospy.get_param("admit/conf_user_intent/Kd_low"), 
                                                    k_d_ub = rospy.get_param("admit/conf_user_intent/Kd_high"), 
                                                    s_sensitivity = rospy.get_param("admit/conf_user_intent/s_sen"), 
                                                    min_user_intent = rospy.get_param("admit/conf_user_intent/min_user_intent_range"), 
                                                    max_user_intent = rospy.get_param("admit/conf_user_intent/max_user_intent_range"))

        # high level state behavior flags, not members of class as these are mostly for debugging
        self.b_check_if_interacting = True
        self.b_use_meas_acc_for_intent = False 
        self.b_use_meas_acc_for_prediction = False 
        self.b_limit_robot_force_by_measured_force = False 
        self.b_realign_cosystem = False 
        self.measured_force_mult = 10.0 

        # initial state dict
        self.initial_state_dict = {'pos' : np.zeros(shape=(pos_dim,)),
                'vel' : np.zeros(shape=(pos_dim,)),
                'acc' : np.zeros(shape=(pos_dim,)),
                'acc_ui' : np.zeros(shape=(pos_dim,)),
                'full_state' : np.zeros(shape=(3*pos_dim,)),
                'measured_force' : np.zeros(shape=(pos_dim,)),
                'robot_force' : np.zeros(shape=(pos_dim,)),
                'desired_pos': np.zeros(shape=(pos_dim,)),
                'desired_vel' : np.zeros(shape=(pos_dim,)),
                'pos_eq' : np.zeros(shape=(pos_dim,)),
                'vel_eq' : np.zeros(shape=(pos_dim,)),
                'user_intent': 0.0,
                'confidence': 0.0,
                'I': self.I_base,
                'B': self.B_base,
                'K_p': 0.0,
                'K_d': 0.0}
        self.raw_q_ratio = 1.0

        full_state_box_bound = [rospy.get_param("safety/pos_bound"), rospy.get_param("safety/pos_bound"),
                                rospy.get_param("safety/vel_max"), rospy.get_param("safety/vel_max"),
                                rospy.get_param("safety/acc_max"), rospy.get_param("safety/acc_max")]
        # safety filter dict
        self.safety_dict = {'pos': {'box_bound': rospy.get_param("safety/pos_bound"), 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'vel': {'box_bound': None, 'mag_bound': rospy.get_param("safety/vel_max"), 'diff_bound': None, 'apply_filter': False},
            'acc': {'box_bound': None, 'mag_bound': rospy.get_param("safety/acc_max"), 'diff_bound': None, 'apply_filter': False},
            'acc_ui': {'box_bound': None, 'mag_bound': rospy.get_param("safety/acc_max"), 'diff_bound': None, 'apply_filter': False},
            'full_state': {'box_bound': full_state_box_bound, 'mag_bound': None, 'diff_bound': None, 'apply_filter': True}, # gets filtered by other node now
            'measured_force': {'box_bound': None, 'mag_bound': rospy.get_param("safety/force_max"), 'diff_bound': None, 'apply_filter': True},
            'robot_force': {'box_bound': None, 'mag_bound': rospy.get_param("safety/force_max"), 'diff_bound': None, 'apply_filter': False},
            'desired_pos': {'box_bound': rospy.get_param("safety/pos_bound"), 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'desired_vel': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'pos_eq': {'box_bound': rospy.get_param("safety/pos_bound"), 'mag_bound': None, 'diff_bound': None, 'apply_filter': True},
            'vel_eq': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': True},
            'user_intent': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'confidence': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'I': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'B': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'K_p': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'K_d': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False}}
        
        self.t_eq_min = rospy.get_param("safety/t_eq_min")
        self.t_eq = 0.0
        self.raw_confidence = 0.0

        self.input_filter_keys = ['measured_force']
        self.mid_filter_keys = ['full_state']
        self.output_filter_keys = ['user_intent', 'pos_eq', 'vel_eq'] # did include pred_traj

        # these don't fit well in the generic safety filter dict's behavior
        self.eq_pos_max = rospy.get_param("safety/eq_pos_max")
        self.eq_pos_threshold = rospy.get_param("safety/eq_pos_threshold")
        self.measured_force_min = rospy.get_param("safety/measured_force_min") # forces under this value will be considered noise, suggests user is not interacting unless there's a nearly perfect directional change.

        # generates safety filters
        b, a = scipy.signal.iirfilter(rospy.get_param("filter/order"), 
            Wn=rospy.get_param("filter/critical_freq"), 
            fs=int(1.0 / self.dt), btype="low", ftype="butter")
        self.initializeSafetyFilters(b, a)
        #self.exp_filter_alpha = rospy.get_param("filter/exp_alpha")
        self.exp_filter_alpha = rospy.get_param("filter/exp_gain_speed") * self.dt # %change/second * seconds/step = %change/step!


        self.safe_damping = self.B_base # default to this damping if there's any weird behavior.

        ### ADMITTANCE CONTROL INIT
        self.impedance_policy = ImpedancePolicy(B_diag=0.0, K_diag=0.0, pos_dim=pos_dim)

        k_holding = 50.0
        b_crit = 2.0 * self.I_base * math.sqrt(k_holding / self.I_base)
        self.holding_stiffness = k_holding
        self.holding_vel_stiffness = 0.0
        self.holding_damping = b_crit
        self.holding_impedance_policy = ImpedancePolicy(B_diag=b_crit, K_diag=k_holding, pos_dim=pos_dim)
        self.holding_pos = np.zeros(shape=(pos_dim,)) # go back to origin carefully if in holding_state
        self.holding_vel = np.zeros(shape=(pos_dim,))

        self.b_use_pos_vel_eq = True
        self.b_use_diff_pos_eq = True

        self.b_running = False
        self.b_in_holding_state = True 

        self.admit_type = 0

        self.valid_impedance_update_dict = {0: "Zero",
                                    1: "Static", 
                                    2: "User Intent",
                                    3: "Confidence",
                                    4: "Confidence_User_Intent"}

        self.force_sensor_frame = np.zeros(shape=(3, )) # formerly measured_force
        self.force_base_frame = np.zeros(shape=(3, ))
        self.used_force = np.zeros(shape=(self.pos_dim, ))
        self.initial_pose = np.array([0.0, -0.4231, 0.7589]) # pos vector from base to ee at initial config
        self.raw_pos_eq = np.zeros(shape=(pos_dim,)) # go back to origin carefully if in holding_state
        self.raw_vel_eq = np.zeros(shape=(pos_dim,))
        self.prediction_times = None

        # various bools
        self.b_verbose = False 
        self.b_print_times = False 
        self.b_use_prediction = True

        # these change within the loop! they don't control high level state behaviors, just need to be initialized
        self.user_is_interacting = False
        self.user_intent_case = False
        self.confidence_case = False
        self.confidence_user_intent_case = False
        self.b_reset_confidence = False

        # subscribers and publishers
        self.global_queue_size = 25 # 50 # was 1
        self.pub_desired = rospy.Publisher("desired_state", PoseStamped, queue_size=self.global_queue_size) # for the ik
        self.pub_record = rospy.Publisher("admit_record_state", AdmitRecordStateStamped, queue_size=self.global_queue_size) # for the logger
        self.pub_full = rospy.Publisher("full_admit_state", FullAdmitState, queue_size=self.global_queue_size)

        self.sub_wrench = rospy.Subscriber("torque_force_sensor_state", WrenchStamped, self.callbackForce, queue_size=self.global_queue_size, tcp_nodelay=True)
        self.sub_eq = rospy.Subscriber("equilibrium_state", EquilibriumState, self.callbackEqState, queue_size=self.global_queue_size, tcp_nodelay=True)


        self.set_running_behavior_service = rospy.Service("set_running_behavior", SetBool, self.setRunningBehavior)
        self.set_admittance_controller_behavior_service = rospy.Service("set_admittance_controller_behavior", SetInt, self.setAdmittanceControllerBehavior)
        self.set_holding_behavior_service = rospy.Service("set_holding_behavior", SetBool, self.setHoldingBehavior)

        self.trigger_user_intent_bounds = rospy.Service("trigger_user_intent_bounds", Trigger, self.triggerUserIntentBounds)
        self.get_user_intent_bounds_srv = rospy.ServiceProxy("/quantile_server/get_user_intent_bounds", GetUserIntentBounds)

        print(f"admit: Ready!")
        return

    def initializeSafetyFilters(self, filter_b, filter_a):
        state_keys = self.initial_state_dict.keys()
        safety_filter_dict = dict()
        for key in state_keys:
            new_filter_obj = LiveLFilter(filter_b, filter_a, self.initial_state_dict[key])
            safety_filter_dict[key] = BoundAndFilter(new_filter_obj, 
                                    box_bound = self.safety_dict[key]['box_bound'],
                                    mag_bound = self.safety_dict[key]['mag_bound'],
                                    diff_bound = self.safety_dict[key]['diff_bound'],
                                    apply_filter = self.safety_dict[key]['apply_filter'])
        self.safety_filter_dict = safety_filter_dict
        return

    def applySafetyFilters(self, array_dict, filter_keys):
        for key in filter_keys:
            try:
                array_dict[key] = self.safety_filter_dict[key].process(array_dict[key])
            except:
                print(f"error in filtering component {key}...")
                print(array_dict[key])
                raise NotImplementedError("haven't fixed this yet")
        return array_dict

    ### ROS functions
    def callbackForce(self, msg):
        self.force_sensor_frame[0] = msg.wrench.force.x
        self.force_sensor_frame[1] = msg.wrench.force.y
        self.force_sensor_frame[2] = msg.wrench.force.z
        return

    def callbackEqState(self, msg):
        if self.b_use_diff_pos_eq:
            self.raw_pos_eq[0] = msg.diff_position.x
            self.raw_pos_eq[1] = msg.diff_position.z
            self.raw_vel_eq[0] = msg.diff_velocity.x
            self.raw_vel_eq[1] = msg.diff_velocity.z
        else:
            self.raw_pos_eq[0] = msg.position.x
            self.raw_pos_eq[1] = msg.position.z
            self.raw_vel_eq[0] = msg.velocity.x
            self.raw_vel_eq[1] = msg.velocity.z
        self.t_eq = msg.t_eq
        self.raw_q_ratio = msg.q_ratio
        return

    def publishDesiredState(self, desired_pos, desired_vel = None):
        desired_pose_msg = PoseStamped()
        desired_pose_msg.header.frame_id = "ee"
        desired_pose_msg.header.stamp = rospy.Time.now() 

        # ee_pose
        desired_pose_msg.pose.position.x = desired_pos[0] + self.initial_pose[0]
        desired_pose_msg.pose.position.y = 0.0 + self.initial_pose[1]
        desired_pose_msg.pose.position.z = desired_pos[1] + self.initial_pose[2]
        desired_pose_msg.pose.orientation.w = 1.0

        self.pub_desired.publish(desired_pose_msg)
        return 

    def publishFullState(self, full_state, ref_pos, rollout_pos = None):
        msg = FullAdmitState()
        
        msg.position.x = full_state[0]
        msg.position.y = 0.0
        msg.position.z = full_state[1]

        msg.velocity.x = full_state[2]
        msg.velocity.y = 0.0
        msg.velocity.z = full_state[3]

        msg.acceleration.x = full_state[4]
        msg.acceleration.y = 0.0
        msg.acceleration.z = full_state[5]

        msg.ref_position.x = ref_pos[0]
        msg.ref_position.y = 0.0
        msg.ref_position.z = ref_pos[1]

        if rollout_pos is not None:
            pnt_list = []
            for i in range(rollout_pos.shape[1]):
                pnt = Point()
                pnt.x = rollout_pos[0, i]
                pnt.y = rollout_pos[1, i]
                pnt_list.append(pnt)
            msg.rollout_positions = pnt_list

        self.pub_full.publish(msg)
        return 

    def publishAdmitRecordState(self, state_dict, dt):
        msg = AdmitRecordStateStamped()

        msg.header.frame_id = "ee"
        
        # ee_pose
        msg.pose.position.x = state_dict["pos"][0]
        msg.pose.position.z = state_dict["pos"][1]
        msg.pose.orientation.w = 1.0

        # ee_vel
        msg.twist.linear.x = state_dict["vel"][0]
        msg.twist.linear.z = state_dict["vel"][1]

        # ee_acc
        msg.accel.linear.x = state_dict["acc"][0]
        msg.accel.linear.z = state_dict["acc"][1]

        # ee force
        msg.wrench.force.x = state_dict["measured_force"][0]
        msg.wrench.force.y = 0.0
        msg.wrench.force.z = state_dict["measured_force"][1]

        # robot force
        msg.robot_wrench.force.x = state_dict["robot_force"][0]
        msg.robot_wrench.force.z = state_dict["robot_force"][1]

        msg.eq_state.position.x = state_dict["pos_eq"][0]
        msg.eq_state.position.z = state_dict["pos_eq"][1]

        msg.eq_state.velocity.x = state_dict["vel_eq"][0]
        msg.eq_state.velocity.z = state_dict["vel_eq"][1]

        msg.stiffness = state_dict["K_p"]
        msg.vel_stiffness = state_dict["K_d"]
        msg.damping = state_dict["B"]
        msg.user_intent = state_dict["user_intent"]
        msg.confidence = state_dict["confidence"]

        msg.cycle_time = dt

        self.pub_record.publish(msg)
        return

    ### ROS services
    def setRunningBehavior(self, req): # setBool
        res = SetBoolResponse()
        if (isinstance(req.data, bool)):
            self.b_running = req.data
            res.success = True
            res.message = "Running behavior set to " + str(self.b_running) + "!!!"
        else:
            res.success = False
            res.message = "!!!Error!!! Invalid request data type, got " + str(req.data) + " but needed a bool."
        return res
    
    def setHoldingBehavior(self, req): # setBool
        res = SetBoolResponse()
        if (isinstance(req.data, bool)):
            self.b_in_holding_state = req.data
            res.success = True
            res.message = "Running holding set to " + str(self.b_in_holding_state) + "!!!"
            print(res.message)
        else:
            res.success = False
            res.message = "!!!Error!!! Invalid request data type, got " + str(req.data) + " but needed a bool."
        return res

    def setAdmittanceControllerBehavior(self, req): # setInt service
        res = SetIntResponse()
        # res.success is a bool, res.message is a string
        if (req.data not in self.valid_impedance_update_dict.keys()):
            res.success = False
            res.message = "Invalid type! Got " + str(req.data) + ", expected one of " + str(self.valid_impedance_update_dict.keys()) + " for corresponding states " + str(self.valid_impedance_update_dict.values())
        else:
            b_behavior_changed = self.setAdmitType(req.data)
            res.success = b_behavior_changed
            if (b_behavior_changed):
                res.message = "Set admittance controller behavior to type: " + str(self.valid_impedance_update_dict[self.admit_type]) + "!!!"
            else:
                res.message = "!!!Error!!! Internal state switching function returned false!!!"
        return res

    ### service-oriented functions
    def setAdmitType(self, desired_admit_type):
        if (self.valid_impedance_update_dict[desired_admit_type] == "Zero"):
            #self.impedance_update_rule = self.impedance_update_rule_zero
            self.b_use_pos_vel_eq = False
            self.user_intent_case = False
            self.confidence_case = False
            self.confidence_user_intent_case = False
        elif (self.valid_impedance_update_dict[desired_admit_type] == "Static"):
            #self.impedance_update_rule = self.impedance_update_rule_static
            self.b_use_pos_vel_eq = False
            self.user_intent_case = False
            self.confidence_case = False
            self.confidence_user_intent_case = False
        elif (self.valid_impedance_update_dict[desired_admit_type] == "User Intent"):
            #self.impedance_update_rule = self.impedance_update_rule_user_intent
            self.b_use_pos_vel_eq = False
            self.user_intent_case = True
            self.confidence_case = False
            self.confidence_user_intent_case = False
        elif (self.valid_impedance_update_dict[desired_admit_type] == "Confidence"):
            #self.impedance_update_rule = self.impedance_update_rule_confidence
            self.b_use_pos_vel_eq = True
            self.user_intent_case = False
            self.confidence_case = True
            self.confidence_user_intent_case = False
        elif (self.valid_impedance_update_dict[desired_admit_type] == "Confidence_User_Intent"):
            #self.impedance_update_rule = self.impedance_update_rule_confidence
            self.b_use_pos_vel_eq = True
            self.user_intent_case = False
            self.confidence_case = False
            self.confidence_user_intent_case = True
        else:
            return False
        self.b_reset_confidence = True
        self.admit_type = desired_admit_type
        return True
    
    def triggerUserIntentBounds(self, req): # setBool
        res = TriggerResponse()
        print(f"admit: Getting user intent bounds...")
        b_suc = self.requestUserIntentBounds()
        res.success = b_suc
        res.message = "admit: Finished service."
        return res
    
    def requestUserIntentBounds(self):
        ready = True

        resp = self.get_user_intent_bounds_srv()
        '''
        bool success
        string message
        float64 user_intent_min
        float64 uesr_intent_max
        '''
        if resp.success:
            
            self.user_intent_min = resp.user_intent_min * self.user_intent_bound_mult
            self.user_intent_max = resp.user_intent_max * self.user_intent_bound_mult
            print(f"admit: Got user intent bounds as [{self.user_intent_min}, {self.user_intent_max}]!")
            self.impedance_update_rule_user_intent.setUserIntentBounds(self.user_intent_min, self.user_intent_max)
            self.impedance_update_rule_confidence_user_intent.setUserIntentBounds(self.user_intent_min, self.user_intent_max)
        return resp.success

    ### helper functions
    def enforcePosEqStateLimits(self, pos_eq, pos):
        pos_eq, _ = bound_vector_mag(pos_eq, self.eq_pos_max, vec_mean = pos)
        pos_eq, _ = notch_vector_mag(pos_eq, self.eq_pos_threshold, vec_mean = pos) # gets rid of small predictions
        return pos_eq

    def enforceUserIntentLimits(self, user_intent):
        if user_intent < self.user_intent_min:
            user_intent = self.user_intent_min
        if self.user_intent_max < user_intent:
            user_intent = self.user_intent_max
        return user_intent

    ### getter functions
    def getMeasuredForce(self):
        self.force_base_frame[0] = -self.force_sensor_frame[0] # x goes to -x
        self.force_base_frame[1] = self.force_sensor_frame[2]
        self.force_base_frame[2] = self.force_sensor_frame[1]

        self.used_force[0] = self.force_base_frame[0]
        self.used_force[1] = self.force_base_frame[2]
        return self.used_force.copy()

    def getCurrentState(self, des_pos, des_vel):
        return des_pos, des_vel # "should" do something like a Kalman filter here.

    def getEqState(self, pos, vel):
        if self.b_use_diff_pos_eq:
            pos_eq = self.raw_pos_eq + pos
            vel_eq = self.raw_vel_eq + vel
            pos_eq = self.enforcePosEqStateLimits(pos_eq, pos)
        else:
            pos_eq = self.enforcePosEqStateLimits(self.raw_pos_eq, pos)
            vel_eq = self.raw_vel_eq

        if self.user_intent_case:
            vel_eq = vel_eq * 0.0 # make sure the vel_eq is always zero when we're doing VAC
        return pos_eq, vel_eq

    def computeRolloutTrajectory(self, ptime_sequence, initial_pos, initial_vel, pos_eq, vel_eq, measured_force, force_decay_rate = 0.5):
        pos_list = [initial_pos]
        vel_list = [initial_vel]
        pos_eq_list = [pos_eq]
        vel_eq_list = [vel_eq]
        f_meas_list = [measured_force]
        acc_list = []
        dt = ptime_sequence[1] - ptime_sequence[0] # need initial value
        # compute rollout trajectory
        for t_idx in range(0, len(ptime_sequence)):
            if t_idx != 0:
                dt = ptime_sequence[t_idx] - ptime_sequence[t_idx-1]
                pos_list.append(new_pos)
                vel_list.append(new_vel)
                pos_eq_list.append(pos_eq_list[t_idx-1])
                vel_eq_list.append(vel_eq_list[t_idx-1])
                f_meas_list.append(f_meas_list[t_idx-1] * force_decay_rate)
            new_pos, new_vel, (pos, vel, acc), (F_sys, F_r, F_meas) = self.interaction_dynamics.step_safe(pos_list[t_idx],
                                                vel_list[t_idx],
                                                pos_eq_list[t_idx],
                                                vel_eq_list[t_idx], 
                                                f_meas_list[t_idx],
                                                dt=dt)
            acc_list.append(acc)
        rollout_traj = dict()
        rollout_traj['pos'] = np.stack(pos_list, axis=1)
        rollout_traj['vel'] = np.stack(vel_list, axis=1)
        rollout_traj['acc'] = np.stack(acc_list, axis=1)
        rollout_traj['pos_eq'] = np.stack(pos_eq_list, axis=1)
        rollout_traj['vel_eq'] = np.stack(vel_eq_list, axis=1)
        rollout_traj['f_meas'] = np.stack(f_meas_list, axis=1)
        return rollout_traj

    def limitVelAcc(self, vel, acc, dt):
        acc, _ = bound_vector_mag(acc, self.acc_max)
        vel, _ = bound_vector_mag(vel + 0.5 * dt * acc, self.vel_max) # limits vel including effect of acceleration. should keep the total in desired vel range, more stable.
        return vel, acc
    
    ### core admittance controller loop functions
    def pipelineGetInputs(self, state_dict):
        ### GET INPUTS
        pos, vel = self.getCurrentState(state_dict['desired_pos'], state_dict['desired_vel'])
        pos_eq, vel_eq = self.getEqState(pos, vel)
        measured_force = self.getMeasuredForce()
        return pos, vel, pos_eq, vel_eq, measured_force

    def pipelineGetRobotImpedanceForce(self, pos, vel, pos_eq, vel_eq, impedance_policy):
        robot_force, (stiffness_force, damping_force) = impedance_policy.getForce(pos,
                                                vel,
                                                pos_eq,
                                                vel_eq)
        return robot_force, (stiffness_force, damping_force)

    def pipelineCheckIfInteracting(self, measured_force):
        return self.measured_force_min < np.linalg.norm(measured_force)

    def pipelineComputeAdmittanceControl(self, pos, vel, pos_eq, vel_eq, F_meas, dt, admit_cont):
        new_pos, new_vel, (pos, vel, acc), (F_sys, F_r, F_meas) = admit_cont.step_safe(pos, vel, pos_eq, vel_eq, F_meas, dt=dt)
        return new_pos, new_vel, (pos, vel, acc), (F_sys, F_r, F_meas)
    
    def applyStateBehaviors(self, state_dict, b_remove_measured_force):
        if b_remove_measured_force:
            state_dict['measured_force'] = state_dict['measured_force'] * 0.0
        state_dict = self.applySafetyFilters(state_dict, ['measured_force'])
        return state_dict

    def pipelineHandleStateBehaviorFlags(self, state_dict):
        b_user_is_interacting = self.pipelineCheckIfInteracting(state_dict['measured_force'])
        b_remove_robot_stiffness_force = False
        if self.b_in_holding_state:
            b_remove_robot_force = False
            b_remove_measured_force = True
            state_dict['pos_eq'] = self.holding_pos
            state_dict['vel_eq'] = self.holding_vel
        else:
            if b_user_is_interacting:
                b_remove_robot_force = False
                b_remove_measured_force = False
            else:
                b_remove_robot_force = False #True
                b_remove_measured_force = True

            if not self.b_use_pos_vel_eq:
                state_dict['vel_eq'] = self.holding_vel

        return state_dict, b_remove_robot_force, b_remove_measured_force

    def pipelineGetAdmittanceParameters(self, state_dict, b_remove_robot_force, min_damping_offset = 5.0):
        if self.b_in_holding_state:
            B = self.holding_damping
            K_p = self.holding_stiffness
            K_d = self.holding_vel_stiffness
            B_co = self.holding_damping
            K_p_co = self.holding_stiffness
            K_d_co = self.holding_vel_stiffness
            return B, K_p, K_d, B_co, K_p_co, K_d_co
        else:
            if self.user_intent_case:
                B, K_p, K_d = self.impedance_update_rule_user_intent.updateImpedanceParams(state_dict['user_intent'])
            elif self.confidence_case:
                B, K_p, K_d = self.impedance_update_rule_confidence.updateImpedanceParams(state_dict['confidence'])
            elif self.confidence_user_intent_case:
                B, K_p, K_d = self.impedance_update_rule_confidence_user_intent.updateImpedanceParams(state_dict['confidence'], state_dict['user_intent'])
            else:
                B, K_p, K_d = self.impedance_update_rule_static.updateImpedanceParams(0.0)
            B_co, K_p_co, K_d_co = self.impedance_update_rule_static.updateImpedanceParams(0.0)

        if b_remove_robot_force:
            K_p = 0.0
            K_d = 0.0
            K_p_co = 0.0
            K_d_co = 0.0

        B = B + min_damping_offset
        
        B = exp_filt(B, state_dict['B'], self.exp_filter_alpha)
        K_p = exp_filt(K_p, state_dict['K_p'], self.exp_filter_alpha)
        K_d = exp_filt(K_d, state_dict['K_d'], self.exp_filter_alpha)

        return B, K_p, K_d, B, K_p_co, K_d_co
    
    def pipelineUpdateAdmittanceParameters(self, B, K_p, K_d, B_co, K_p_co, K_d_co):
        self.interaction_dynamics.updateDynamics(B, K_p, K_d)
        self.co_interaction_dynamics.updateDynamics(B_co, K_p_co, K_d_co)
        return
    
    def mainLoopPipeline(self, state_dict):
        
        state_dict['pos'], state_dict['vel'], state_dict['pos_eq'], state_dict['vel_eq'], state_dict['measured_force'] = self.pipelineGetInputs(state_dict)

        if self.b_reset_confidence: # should be triggered 
            state_dict['confidence'] = 0.0
            self.b_reset_confidence = False
        else:
            state_dict['confidence'] = np.clip(state_dict['confidence'] + self.confidence_scalar * (1.0 - self.raw_q_ratio), a_min=0.0, a_max=1.0)

        state_dict = self.applySafetyFilters(state_dict, ['pos', 'vel', 'pos_eq', 'vel_eq', 'confidence'])

        state_dict, b_remove_robot_force, b_remove_measured_force = self.pipelineHandleStateBehaviorFlags(state_dict)
        state_dict = self.applyStateBehaviors(state_dict, b_remove_measured_force)

        state_dict['B'], state_dict['K_p'], state_dict['K_d'], B_co, K_p_co, K_d_co = self.pipelineGetAdmittanceParameters(state_dict, b_remove_robot_force)
        state_dict = self.applySafetyFilters(state_dict, ['B', 'K_p', 'K_d'])
        self.pipelineUpdateAdmittanceParameters(state_dict['B'], state_dict['K_p'], state_dict['K_d'], B_co, K_p_co, K_d_co)

        state_dict['desired_pos'], state_dict['desired_vel'], (pos, vel, acc), (F_sys, F_r, F_meas) = self.pipelineComputeAdmittanceControl(state_dict['pos'], state_dict['vel'], 
                                        state_dict['pos_eq'], state_dict['vel_eq'], state_dict['measured_force'], self.dt, self.interaction_dynamics)
        state_dict['robot_force'] = F_r
        state_dict['acc'] = acc

        state_dict["user_intent"] = self.enforceUserIntentLimits(np.dot(vel, acc))
        
        state_dict['full_state'][0:2] = pos
        state_dict['full_state'][2:4] = vel 
        state_dict['full_state'][4:6] = acc 

        state_dict = self.applySafetyFilters(state_dict, ['desired_pos', 'desired_vel', 'full_state', 'user_intent'])

        return state_dict

    def publish(self, state_dict, dt):
        self.publishDesiredState(state_dict['desired_pos'], state_dict['desired_vel'])
        self.publishAdmitRecordState(state_dict, dt)

        if self.prediction_times is None:
            try:
                self.prediction_times = rospy.get_param("/prediction_times")
            except:
                pass
            self.publishFullState(state_dict['full_state'], state_dict['desired_pos'])
        else:
            rollout_traj = self.computeRolloutTrajectory(self.prediction_times, state_dict['pos'], state_dict['vel'], state_dict['pos_eq'], state_dict['vel_eq'], state_dict['measured_force'])

            self.publishFullState(state_dict['full_state'], state_dict['desired_pos'], rollout_traj['pos'])

        


if __name__ == '__main__':

    nh = rospy.init_node('admit', anonymous=True)

    admit = Admit(nh)
    state_dict = admit.initial_state_dict

    rate = rospy.Rate(admit.dt_rate)
    main_timer = time.time()
    while not rospy.is_shutdown():

        state_dict = admit.mainLoopPipeline(state_dict)
        _ = wait_for_time(main_timer, admit.dt)

        real_dt = time.time() - main_timer
        main_timer = time.time()
        admit.publish(state_dict, real_dt)
