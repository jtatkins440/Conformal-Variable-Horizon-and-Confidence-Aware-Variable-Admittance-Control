#!/usr/bin/python3

import math
import rospy

from conf_exps.msg import * #AdmitStateStamped, RecordStateStamped
from conf_exps.srv import * # SetInt and others
from std_srvs.srv import * # SetBool
import numpy as np
from collections import deque
import time

import os

from admit_lib import *
from utils import *

### 
class MIntConf:
    def __init__(self, nh):
        self.nh = nh

        ## get all ros params

        ### MODEL INFERENCE INIT
        model_name = rospy.get_param("mint/model_name")

        self.inf_model, params = get_prediction_model(model_name)

        self.output_eq_sample_time = 0.01

        ### ROS PARAMS
        pos_channels = rospy.get_param("pos_channels")
        pos_dim = len(pos_channels)
        self.full_state_dim = pos_dim * 3
        self.raw_full_state = np.zeros(shape=(self.full_state_dim))
        self.raw_ref_pos = np.zeros(shape=(pos_dim))

        alphas = rospy.get_param("confidence/alphas")
        self.tau_const = rospy.get_param("confidence/tau_const")
        score_buffer_size = rospy.get_param("confidence/score_buffer_size")
        self.safe_eq_error_threshold = rospy.get_param("confidence/safe_eq_error_threshold")
        self.safe_var_time_eq_scalar = rospy.get_param("confidence/safe_var_time_eq_scalar")
        aci_step_sizes = rospy.get_param("confidence/aci/step_sizes")
        aci_window_length = rospy.get_param("confidence/aci/window_length")
        self.max_gamma = rospy.get_param("confidence/aci/max_step_size")
        self.vel_min = rospy.get_param("vel_min")
        self.t_limit_min = rospy.get_param("mint/t_eq_min")

        self.pos_guide_eq_mag_max = rospy.get_param("mint/pos_eq_guide_mag") #0.025
        self.vel_guide_eq_mag_max = rospy.get_param("mint/vel_eq_guide_mag") #0.05
        self.pos_assit_tol = 0.001

    
        self.safe_eq_time_threshold = self.safe_eq_error_threshold * self.safe_var_time_eq_scalar

        # model inputs
        try:
            base_dt = params["base_delta_time"]
        except:
            base_dt = 0.005

        #self.desired_loop_time = base_dt
        self.dt = base_dt
        self.dt_rate = int(1.0 / base_dt)

        model_params = params["model_params"]

        self.mint_output_time_seq = np.array([i * self.dt for i in model_params["output_idx_seq"]])
        rospy.set_param("/prediction_times", self.mint_output_time_seq.tolist())


        mint_input_history_size = abs(min(model_params["input_idx_seq"])) + 1 # expected -124 to 0
        mint_prediction_size = abs(max(model_params["output_idx_seq"]))
        print(f"mint_input_history_size {mint_input_history_size}")
        print(f"mint_prediction_size {mint_prediction_size}")
        full_mint_history_size = mint_input_history_size + mint_prediction_size
        self.score_calc_start_index = mint_input_history_size - 1 #mint_prediction_size #full_mint_history_size - 1

        scalar_params = params["scalar_params"]

        # set up the key objects for core loop
        self.pos_dim = pos_dim

        self.mint_input_indices = model_params["input_idx_seq"]
        self.mint_output_indices = model_params["output_idx_seq"]
        self.mint_output_rec = None
        
        self.prediction_length = len(self.mint_output_indices)

        self.raw_rollout_pos = np.zeros(shape=(pos_dim, len(model_params["output_idx_seq"])))

        # initial state dict
        self.init_state_dict = {'full_state' : np.zeros(shape=(3*pos_dim,)),
                'pred_traj' : np.zeros(shape=(pos_dim, len(model_params["output_idx_seq"]))),
                'pos_eq' : np.zeros(shape=(pos_dim,)),
                'vel_eq' : np.zeros(shape=(pos_dim,)),
                't_eq': 0.0,
                'scores' : np.zeros(shape = (len(model_params["output_idx_seq"]))),
                'quantiles' : np.zeros(shape = (len(aci_step_sizes), len(model_params["output_idx_seq"]))),
                'alphas' : np.zeros(shape = (len(aci_step_sizes), len(model_params["output_idx_seq"]))),
                'confidence': 0.0,
                'q_ratio': 1.0}
        
        self.raw_guide_state = np.zeros(shape=(2*pos_dim + 1))

        # safety filter dict
        
        self.safety_dict = {'full_state': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': True},
            'pred_traj': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': True},
            'pos_eq': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'vel_eq': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            't_eq': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False}, # WANT to filter the equilibrium time to avoid the very jumpy behavior, should worry about ringing though!
            'scores': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'quantiles': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'alphas': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'confidence': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False},
            'q_ratio': {'box_bound': None, 'mag_bound': None, 'diff_bound': None, 'apply_filter': False}}
        
        self.guide_points_pos = np.zeros(shape=(pos_dim, len(model_params["output_idx_seq"])))
        self.guide_points_vel = np.zeros(shape=(pos_dim, len(model_params["output_idx_seq"])))


        self.mid_filter_keys = ['full_state']
        self.late_mid_filter_keys = ['pred_traj']
        self.output_filter_keys = ['quantiles', 'pos_eq', 'vel_eq', 't_eq'] # did include pred_traj

        # generates safety filters
        b, a = scipy.signal.iirfilter(rospy.get_param("filter/order"), 
            Wn=rospy.get_param("filter/critical_freq"), 
            fs=int(1.0 / self.dt), btype="low", ftype="butter")
        self.initializeSafetyFilters(b, a)


        # params for high level services
        

        # various bools
        self.b_running = False
        self.b_verbose = False # True #True #False #True
        self.b_print_times = False # True #True #True

        # high level state behavior flags, not members of class as these are mostly for debugging
        self.b_use_variable_eq_horizon = False
        self.used_safe_quantile_index = -1 # of the full quantile array, use the most 

        # these change within the loop! they don't control high level state behaviors, just need to be initialized
        self.user_is_interacting = False
        self.is_mint_input_ready = False
        self.is_state_buffer_full = False
        self.is_score_buffer_full = False
        self.has_opt_step_sizes = False
        self.adjuts_pred_with_rollout = True #True #False
        self.swap_guide_dim = False

        # smoothing term to make input and predicted trajectories smoother. Equivalent to an exponetial filter applied over trajectories.
        self.input_traj_weight = 0.99 #
        self.pred_traj_weight = 0.90 #0.0 #0.6

        self.last_input_traj = None
        self.input_timer = time.time()

        # subscribers, publishers, and services
        self.global_queue_size = 25 # 50 # was 1

        self.sub_full_admit_state = rospy.Subscriber('full_admit_state', FullAdmitState, self.updateFullAdmitState, queue_size=1, tcp_nodelay=True)
        self.sub_guide_state = rospy.Subscriber('/GuideTargets', GuideState, self.guideCallback, queue_size=1, tcp_nodelay=True)

        self.pub_mint_conf_record_state = rospy.Publisher("mint_conf_record_state", MIntConfRecordStateStamped, queue_size=self.global_queue_size)
        self.pub_eq_state = rospy.Publisher("equilibrium_state", EquilibriumState, queue_size=self.global_queue_size)

        self.set_running_behavior_service = rospy.Service("set_running_behavior", SetBool, self.setRunningBehavior)
        self.set_fixed_setpoint_index_service = rospy.Service("set_fixed_setpoint", SetFloat, self.setFixedSetpointHorizon)
        self.set_use_variable_setpoint_bool_service = rospy.Service("set_use_variable_setpoint", SetBool, self.setUseVariableSetpoint)
        self.trigger_opt_step_sizes = rospy.Service("trigger_get_opt_step_sizes", Trigger, self.triggerOptStepSizes)

        self.get_opt_step_sizes_srv = rospy.ServiceProxy("/quantile_server/get_optimal_step_sizes", GetStepSizes)

        ### CONF AND MEMORY INIT
        self.target_alpha = alphas
        self.mem_buffer = MemoryHelper(state_chn_num = int(pos_dim * 3), min_state_buffer_size = mint_input_history_size, 
            state_buffer_size = full_mint_history_size, score_buffer_size = score_buffer_size, score_len = self.prediction_length, prediction_buffer_size = mint_prediction_size)

        self.aci_helper = AdaptiveConformalPredictionHelper(target_alpha = alphas, prediction_length = self.prediction_length, 
            step_sizes=aci_step_sizes, window_length=aci_window_length, init_quantiles = None, use_jit=True, use_parallel=False)

        print(f"mint_conf: Ready!")
        return

    def guideCallback(self, msg):
        if self.swap_guide_dim:
            self.raw_guide_state[0] = msg.position.y
            self.raw_guide_state[1] = msg.position.x
            self.raw_guide_state[2] = msg.velocity.y
            self.raw_guide_state[3] = msg.velocity.x
        else:
            self.raw_guide_state[0] = msg.position.x
            self.raw_guide_state[1] = msg.position.y
            self.raw_guide_state[2] = msg.velocity.x
            self.raw_guide_state[3] = msg.velocity.y
        self.raw_guide_state[4] = msg.active_coord

        self.guide_points_pos = np.array([msg.pos_x, msg.pos_y])
        self.guide_points_vel = np.array([msg.vel_x, msg.vel_y])
        return
    
    ### helper functions
    def initializeSafetyFilters(self, filter_b, filter_a):
        #raise NotImplementedError("Still working on it.")
        state_keys = self.init_state_dict.keys()
        safety_filter_dict = dict()
        for key in state_keys:
            new_filter_obj = LiveLFilter(filter_b, filter_a, self.init_state_dict[key])
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
    
    def getSafeEqTimeFromQuantileSeq(self, quantile_seq, safe_eq_error_threshold = None, interpolate = False):
        #print(f"quantile_seq_array: {quantile_seq}, \nshape: {quantile_seq.shape}\n\n")
        if interpolate:
            raise NotImplementedError
        else:
            if safe_eq_error_threshold is None:
                safe_eq_error_threshold = self.safe_eq_time_threshold #self.safe_eq_error_threshold
            diff_quantile_seq = quantile_seq - safe_eq_error_threshold
            if np.all(diff_quantile_seq < 0.0): # if all are very below the size, keep it at max
                quantile_index = len(diff_quantile_seq) - 1
            else: #
                quantile_index = np.argmax(0.0 <= diff_quantile_seq)

            safe_eq_time = self.mint_output_time_seq[quantile_index]
            quantile = quantile_seq[quantile_index]
            '''
            condition = ((quantile_seq - safe_eq_error_threshold) < 0.0)
            
            if len(condition) == 0:
                safe_eq_time = 0.0
                quantile_index = 0
            else:
                quantile_index = np.max(self.mint_output_time_seq[condition])
                safe_eq_time = np.max(self.mint_output_time_seq[condition]) # gets the largest time that meets the condition
            '''
        #print(f"safe_eq_time: {safe_eq_time}")
                
        return safe_eq_time, quantile, quantile_index

    ### publishers
    
    def _publishMIntConfRecordState(self, pos_eq, vel_eq, t_eq, pred_traj, scores, quantiles, alphas, target_alphas, step_sizes, q_ratio, dt):
        msg = MIntConfRecordStateStamped()
        msg.eq_state.position.x = pos_eq[0]
        msg.eq_state.position.z = pos_eq[1]
        msg.eq_state.velocity.x = vel_eq[0]
        msg.eq_state.velocity.z = vel_eq[1]
        msg.t_eq = t_eq

        msg.pred_traj = pack_array_to_multiarray_msg(pred_traj)
        msg.scores = scores.tolist()

        msg.q_ratio = q_ratio
        #msg.confidence = confidence

        pred_len = quantiles.shape[-1] # last dim is length, should have three
        aci_sample_list = []

        for i in range(pred_len):
            aci_sample = ACISample()
            aci_sample.quantiles = pack_array_to_multiarray_msg(quantiles[0:1,i:i+1])
            aci_sample.alphas = pack_array_to_multiarray_msg(alphas[0:1,i:i+1])
            aci_sample.target_alphas = target_alphas
            aci_sample.gammas = step_sizes
            aci_sample_list.append(aci_sample)
        msg.aci_state.sample_number = len(aci_sample_list)
        msg.aci_state.aci_batch = aci_sample_list

        msg.cycle_time = dt

        self.pub_mint_conf_record_state.publish(msg)
        return

    def publishMIntConfRecordState(self, state_dict, dt):
        self._publishMIntConfRecordState(state_dict["pos_eq"], 
                    state_dict["vel_eq"], 
                    state_dict["t_eq"], 
                    state_dict["pred_traj"], 
                    state_dict["scores"], 
                    state_dict["quantiles"], 
                    state_dict["alphas"], 
                    self.aci_helper.target_alphas_list, 
                    self.aci_helper.step_sizes_list,
                    state_dict["q_ratio"], 
                    dt)
        return

    def publishEquilibriumState(self, pos_eq, vel_eq, t_eq = 0.0, full_state = None, q_ratio = 0.0):
        msg = EquilibriumState()
        msg.position.x = pos_eq[0]
        msg.position.z = pos_eq[1]
        msg.velocity.x = vel_eq[0]
        msg.velocity.z = vel_eq[1]
        msg.t_eq = t_eq

        msg.diff_position.x = pos_eq[0] - self.raw_ref_pos[0]
        msg.diff_position.z = pos_eq[1] - self.raw_ref_pos[1]

        if full_state is not None:
            msg.reference_position.x = full_state[0]
            msg.reference_position.z = full_state[1]
            msg.reference_velocity.x = full_state[2]
            msg.reference_velocity.z = full_state[3]
            msg.diff_velocity.x = vel_eq[0] - full_state[2]
            msg.diff_velocity.z = vel_eq[1] - full_state[3]
        msg.q_ratio = q_ratio
        self.pub_eq_state.publish(msg)
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
    
    def triggerOptStepSizes(self, req): # setBool
        res = TriggerResponse()
        print(f"admit_conf: Getting optimial step sizes for ACI...")
        start_time = time.time()
        b_opt_step_sizes_found = self.requestOptimalStepSizes()
        req_time = time.time() - start_time
        if b_opt_step_sizes_found:
            print(f"admit_conf: Recieved! Took {req_time}s to get optimal step sizes!")
            res.success = False
            res.message = f"Recieved! Took {req_time}s to get optimal step sizes!"
        else:
            res.success = False
            res.message = "!!!Error!!! Couldn't get optimal step sizes!"
        print(res.message)
        return res
    
    def setFixedSetpointHorizon(self, req): # setInt service
        res = SetFloatResponse()
        # res.success is a bool, res.message is a string
        if (isinstance(req.data, float)):
            self.output_eq_sample_time = float(req.data)
            diff_time_seq = self.mint_output_time_seq - self.output_eq_sample_time
            if np.all(diff_time_seq < 0.0): # if all are very below the size, keep it at max
                quantile_index = len(diff_time_seq) - 1
            else:
                quantile_index = np.argmax(0.0 <= diff_time_seq)

            if self.output_eq_sample_time < self.t_limit_min:
                self.output_eq_sample_time = 0.0
                quantile_index = 0
            self.fixed_quantile_index = quantile_index

            res.success = True
            res.message = f"Equilibrium time set to {self.output_eq_sample_time}"
        else:
            res.success = False
            res.message = "!!!Error!!! Invalid request data type, got " + str(req.data) + " but needed a float."

        return res

    def setUseVariableSetpoint(self, req): # setBool
        res = SetBoolResponse()
        if (isinstance(req.data, bool)):
            self.b_use_variable_eq_horizon = req.data
            if (self.b_use_variable_eq_horizon and not self.has_opt_step_sizes):
                print(f"admit_conf: Getting optimial step sizes for ACI...")
                self.b_running = False # disable main loop while the solver is running
                start_time = time.time()
                b_opt_step_sizes_found = self.requestOptimalStepSizes()
                req_time = time.time() - start_time
                self.b_running = True
                if b_opt_step_sizes_found:
                    print(f"admit_conf: Recieved! Took {req_time}s to get optimal step sizes!")
            res.success = True
            res.message = "b_use_variable_eq_horizon set to " + str(self.b_use_variable_eq_horizon) + "!!!"
        else:
            res.success = False
            res.message = "!!!Error!!! Invalid request data type, got " + str(req.data) + " but needed a bool."
        return res
    
    def requestOptimalStepSizes(self):

        resp = self.get_opt_step_sizes_srv(self.target_alpha, 
                                            self.mem_buffer.score_buffer_size, #input_window_length
                                            self.aci_helper.window_length, #avg_window_size
                                            int(self.aci_helper.window_length), #half_coverage_window
                                            self.max_gamma)
        print(f"requestOptimalStepSizes: resp.message: {resp.message}")
        opt_step_sizes = resp.step_sizes # list of step sizes that are pred_len long
        full_step_sizes = np.stack([np.zeros_like(opt_step_sizes), np.array(opt_step_sizes)])
        self.aci_helper.updateStepSizes(full_step_sizes)
        self.has_opt_step_sizes = resp.success
        return resp.success
        
    ### PIPELINE FUNCTIONS
    def getFullState(self, state_dict, vel_channels = [2, 3]):
        state_dict["full_state"] = self.raw_full_state
        state_dict = self.applySafetyFilters(state_dict, ['full_state'])
        b_user_is_interacting = self.vel_min < np.linalg.norm(state_dict["full_state"][vel_channels])

        self.input_timer = time.time()
        return state_dict, b_user_is_interacting

    def pipelineUpdateMemoryBuffer(self, full_state):
        b_is_mint_input_ready, b_is_state_buffer_full = self.mem_buffer.updateStateBuffer(full_state)
        return b_is_mint_input_ready, b_is_state_buffer_full

    # if self.is_mint_input_ready:
    def pipelineUpdatePrediction(self, b_is_mint_input_ready, state_dict):
        if b_is_mint_input_ready:
            mint_input = get_mint_input_from_state_buffer(self.mem_buffer.state_buffer.shape[1] - 1, self.mint_input_indices, self.mem_buffer)

            # basic exp filter for input_traj, try to smooth it out a little more for better consistency
            if self.last_input_traj is not None:
                input_traj = self.input_traj_weight * mint_input + (1.0 - self.input_traj_weight) * self.last_input_traj
            else:
                input_traj = mint_input
            self.last_input_traj = input_traj

            pred_traj = np.squeeze(self.inf_model.predict(input_traj)) # 2xL

            # offset traj
            pred_init = pred_traj[:,0]
            pred_offset = self.raw_ref_pos - pred_init
            pred_traj = pred_traj + pred_offset[:,np.newaxis]
            

            if self.adjuts_pred_with_rollout:
                pred_first = pred_traj[:,0]
                rollout_pos_offset = pred_first - self.raw_rollout_pos[:,0]
                output_pred_traj = self.pred_traj_weight * pred_traj + (1.0 - self.pred_traj_weight) * (self.raw_rollout_pos)
            else:
                output_pred_traj = pred_traj
            
        else:
            output_pred_traj = np.squeeze(self.inf_model.zero_output)
        return output_pred_traj

    # b_is_state_buffer_full and b_update_quantiles:
    def pipelineUpdateScores(self, b_update_score_buffer,
                            b_adjust_output_from_filtering = True):
        if b_update_score_buffer:
            (old_mint_input, old_mint_target) = get_mint_input_target_from_state_buffer(self.score_calc_start_index, 
                                                                            self.mint_input_indices, self.mint_output_indices, self.mem_buffer)
            old_mint_output = self.mem_buffer.prediction_buffer[0] # oldest prediction, should line up with outputs from above

            if b_adjust_output_from_filtering:
                filt_offset = old_mint_output[:,0] - old_mint_target[:,0]
                old_mint_output = old_mint_output - filt_offset[:,np.newaxis]
                scores = vec_loss(old_mint_target, old_mint_output)
            else:
                scores = vec_loss(old_mint_target, old_mint_output)

            b_is_score_buffer_full = self.mem_buffer.updateScoreBuffer(scores)

        return self.mem_buffer.getLastScore(), self.mem_buffer.isScoreBufferFull()

    # once we have enough scores, start computing quantiles and confidences
    # if b_is_score_buffer_full
    def pipelineUpdateACIQuantiles(self, b_is_score_buffer_full):
        b_quantiles_ready = False
        if b_is_score_buffer_full:
            #print(self.mem_buffer.score_buffer)
            _, _ = self.aci_helper.updateQuantiles(self.mem_buffer.score_buffer) #  updates internal values
            b_quantiles_ready = True
        aci_quantiles, aci_alphas = self.aci_helper.getQuantilesAndAlphas() # gives the initial values for the quantiles even if there's not enough values to compute anything correctly.
        return aci_quantiles, aci_alphas, b_quantiles_ready

    def pipelineUpdateEquilibriumTime(self, quantiles, time_tol = 0.001):
        if self.b_use_variable_eq_horizon:
            eq_sample_time, quantile, quantile_index = self.getSafeEqTimeFromQuantileSeq(quantiles[self.used_safe_quantile_index,:])

        else:
            eq_sample_time = self.output_eq_sample_time
            quantile_index = self.fixed_quantile_index
            
            quantile = quantiles[self.used_safe_quantile_index,quantile_index]

        return eq_sample_time, quantile, quantile_index

    def pipelineUpdateEquilibrium(self, pred_traj, t_eq,
                                    b_is_mint_input_ready):
        if b_is_mint_input_ready:
            self.mem_buffer.updatePredictionBuffer(pred_traj)
            self.inf_model.interpolateOutput(output_seq=pred_traj)

            pred_pos_at_sample_time, pred_vel_at_sample_time = self.inf_model.sampleOutputPoly(t_eq)

            pos_eq = pred_pos_at_sample_time
            vel_eq = pred_vel_at_sample_time

        else:
            pos_eq = self.init_state_dict['pos_eq']
            vel_eq = self.init_state_dict['vel_eq']
        return pos_eq, vel_eq

    def updateFullAdmitState(self, msg):
        self.raw_full_state[0] = msg.position.x
        self.raw_full_state[1] = msg.position.z
        self.raw_full_state[2] = msg.velocity.x
        self.raw_full_state[3] = msg.velocity.z
        self.raw_full_state[4] = msg.acceleration.x
        self.raw_full_state[5] = msg.acceleration.z

        self.raw_ref_pos[0] = msg.ref_position.x
        self.raw_ref_pos[1] = msg.ref_position.z

        for i in range(len(msg.rollout_positions)):
            self.raw_rollout_pos[0, i] = msg.rollout_positions[i].x
            self.raw_rollout_pos[1, i] = msg.rollout_positions[i].y
        return

    def mainPipelineStep(self, state_dict):
        
        # filters state_dict with full state in order to check for interaction after filtering and returns the filtered state dict
        state_dict, b_user_is_interacting = self.getFullState(state_dict)

        b_is_mint_input_ready, b_is_state_buffer_full = self.pipelineUpdateMemoryBuffer(state_dict["full_state"])

        state_dict["pred_traj"] = self.pipelineUpdatePrediction(b_is_mint_input_ready, state_dict)
        state_dict = self.applySafetyFilters(state_dict, ['pred_traj'])

        b_update_score_buffer = (b_user_is_interacting and b_is_state_buffer_full)
        state_dict["scores"], b_is_score_buffer_full = self.pipelineUpdateScores(b_update_score_buffer)

        state_dict["quantiles"], state_dict["alphas"], b_quantiles_ready = self.pipelineUpdateACIQuantiles(b_is_score_buffer_full)

        if b_quantiles_ready:
            #print("quantiles ready!")
            state_dict['t_eq'], quantile, quantile_index = self.pipelineUpdateEquilibriumTime(state_dict["quantiles"])
            if self.b_use_variable_eq_horizon:
                state_dict['q_ratio'] = 0.0
            else:
                state_dict['q_ratio'] = quantile / self.safe_eq_error_threshold
            state_dict = self.applySafetyFilters(state_dict, ['t_eq'])
        else:
            quantile_index = 0
            quantile = 0.0
        
        pos_eq, vel_eq = self.pipelineUpdateEquilibrium(state_dict["pred_traj"], state_dict['t_eq'],
                                    b_is_mint_input_ready)
        
        guide_pos_t_eq = self.guide_points_pos[:, quantile_index]
        guide_vel_t_eq = self.guide_points_vel[:, quantile_index]

        guide_active = self.raw_guide_state[4]
        pos_assist = guide_active * (guide_pos_t_eq - pos_eq)
        vel_assist = guide_active * (guide_vel_t_eq - vel_eq)

        
        pos_assist_mag = np.linalg.norm(pos_assist)
        if pos_assist_mag < quantile:
            state_dict["pos_eq"] = pos_eq
            state_dict["vel_eq"] = vel_eq
        else:
            state_dict["pos_eq"] = pos_eq + fast_bound_vector_mag(pos_assist, self.pos_guide_eq_mag_max)
            state_dict["vel_eq"] = vel_eq + fast_bound_vector_mag(vel_assist, self.vel_guide_eq_mag_max)
    

        return state_dict

    def finalPublish(self, state_dict, dt):
        state_dict = self.applySafetyFilters(state_dict, self.output_filter_keys)
        self.publishEquilibriumState(state_dict['pos_eq'], state_dict['vel_eq'], state_dict['t_eq'], state_dict['full_state'], state_dict['q_ratio'])
        self.publishMIntConfRecordState(state_dict, dt)


if __name__ == '__main__':

    print(os.getcwd())
    print("Starting main...")
    nh = rospy.init_node('conf_ad_mint', anonymous=True)

    conf_node = MIntConf(nh)
    rate = rospy.Rate(conf_node.dt_rate)

    state_dict = conf_node.init_state_dict

    main_timer = time.time()
    while not rospy.is_shutdown():
        state_dict = conf_node.mainPipelineStep(state_dict)
        _ = wait_for_time(main_timer, conf_node.dt)

        real_dt = time.time() - main_timer
        main_timer = time.time()

        conf_node.finalPublish(state_dict, real_dt)
