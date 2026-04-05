import warnings
import time
from collections import deque

import numpy as np

from scipy.interpolate import PchipInterpolator

import json

import torch
from torch import nn
from utils import *
from numba.typed import List # experimental typed list object
import numba
cache_option = True

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

class AdditiveMaskModule(nn.Module):
    def __init__(self, mask_size, mask_initial_weight = -0.1):
        super(AdditiveMaskModule, self).__init__()
        # takes whole input and passes it to unique layers and appends the outputs, no sharing
        self.mask = nn.parameter.Parameter(mask_initial_weight * torch.ones(mask_size))
        self.act = nn.Hardtanh() #nn.Softmax(dim=1)

    def forward(self, input):
        masked_input = self.act(self.mask) * input
        output = 0.5 * (masked_input + input)
        return output

class LinearMaskModule(nn.Module):
    def __init__(self, features):
        super(LinearMaskModule, self).__init__()
        # takes whole input and passes it to unique layers and appends the outputs, no sharing
        self.mask = nn.Linear(features, features)
        self.act = nn.SiLU()

    def forward(self, input):
        mask = self.mask(input)
        output = torch.mul(self.act(mask), input) # self.act(mask)+ input
        return output

class LinearResModule(nn.Module):
    def __init__(self, features):
        super(LinearResModule, self).__init__()
        # takes whole input and passes it to unique layers and appends the outputs, no sharing
        self.mask = nn.Linear(features, features)
        self.act = nn.SiLU()

    def forward(self, input):
        mask = self.mask(input)
        output = self.act(mask) + input # self.act(mask)+ input
        return output
 
class SequentialLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(SequentialLinear, self).__init__()
        # takes whole input and passes it to unique layers and appends the outputs, no sharing
        self.linear = torch.nn.Linear(input_features, output_features)

    def forward_debug(self, input):
        input_swap = torch.swapaxes(input, -2, -1)
        output_swap = self.linear(input_swap)
        output = torch.swapaxes(output_swap, -2, -1)
        print(f"input: {input.size()}")
        print(f"input_swap: {input_swap.size()}")
        print(f"output_swap: {output_swap.size()}")
        print(f"output: {output.size()}")
        return output

    def forward(self, input): # assumes input is at least size()==2 like (N, Chn, L) or (Chn, L)
        return torch.swapaxes(self.linear(torch.swapaxes(input, -2, -1)), -2, -1)

class MIntNet_Double_Conditioner_Acc(nn.Module):
    def __init__(self, input_channels = [0, 1, 2, 3, 4, 5], input_idx_seq = [-40, -20, -10, 0], input_stride = 20, input_dialation = 5, input_kernels = 3,
                 hidden_size = 64, latent_features = 8, proj_features = 6, output_channels = [0, 1], output_idx_seq = [0, 10, 20, 40], output_dt = 0.05,
                 rel_predict = True, b_use_dropout = True, dropout_rate = 0.2):
        super(MIntNet_Double_Conditioner_Acc, self).__init__() 
        pos_dim = 2
        vel_dim = 2
        acc_dim = 2

        output_dim = acc_dim

        pos_proj_dim = proj_features
        vel_proj_dim = proj_features
        acc_proj_dim = proj_features

        state_proj_dim = pos_proj_dim + vel_proj_dim

        hidden_dim = hidden_size
        lat_features = latent_features

        dropout_alpha = dropout_rate

        act = nn.CELU # function handle
        self.act = act()
        self.pos_proj = nn.Sequential(nn.Linear(pos_dim, hidden_dim), act(), nn.Linear(hidden_dim, pos_proj_dim))
        self.vel_proj = nn.Sequential(SequentialLinear(vel_dim, hidden_dim), act(), SequentialLinear(hidden_dim, vel_proj_dim))
        self.acc_proj = nn.Sequential(SequentialLinear(acc_dim, hidden_dim), act(), SequentialLinear(hidden_dim, acc_proj_dim))

        seq_conv_feat_size = pos_proj_dim + vel_proj_dim + acc_proj_dim

        self.seq_encoder_0 = nn.Conv1d(seq_conv_feat_size, hidden_size, kernel_size=input_kernels, stride=input_stride, dilation=input_dialation)
        self.seq_encoder_1 = nn.Conv1d(hidden_size, latent_features, kernel_size=input_kernels)

        input_length = len(input_idx_seq)
        lat_out_0 = int((input_length - input_dialation * (input_kernels - 1) - 1)/input_stride) + 1
        latent_size = lat_out_0 - input_kernels + 1

        
        self.drop = nn.Dropout(dropout_alpha)
        self.b_apply_drop = True

        output_length = len(output_idx_seq)
        self.output_len = output_length
        self.pred_length = output_length - 1
        decoder_num = self.output_len #self.pred_length
        

        seq_flat_feature_size = latent_features * latent_size

        self.acc_decoders = nn.ModuleList()
        for i in range(0, decoder_num):
            decoder_block = nn.ModuleList()
            decoder_block.append(LinearMaskModule(seq_flat_feature_size)) # basically attention layer
            decoder_block.append(nn.Linear(seq_flat_feature_size, hidden_dim))
            decoder_block.append(nn.Linear(hidden_dim + state_proj_dim, hidden_dim))
            decoder_block.append(nn.Linear(hidden_dim, output_dim))
            self.acc_decoders.append(decoder_block)

        self.rel_predict = rel_predict
        self.output_dt = output_dt
        self.flatten = nn.Flatten()

    def train_encoder(self, pos, vel_seq, acc_seq):
        pos_proj = self.pos_proj(pos) # N x 2 - > N x p_lat x 1
        pos_proj_seq_ = pos_proj.unsqueeze(-1)
        #pos_proj_seq = torch.swapaxes(pos_proj_seq.expand(-1, pos_proj.size()[1], vel_seq.size()[-1]), -2, -1)
        pos_proj_seq = pos_proj_seq_.expand(-1, pos_proj.size()[1], vel_seq.size()[-1])
        #print(pos_proj_seq.size())
        vel_proj_seq = self.vel_proj(vel_seq)
        acc_proj_seq = self.acc_proj(acc_seq)

        stack_proj_seq = torch.cat([pos_proj_seq, vel_proj_seq, acc_proj_seq], dim=1) # concat on the channel dim
        if self.b_apply_drop: stack_proj_seq = self.drop(stack_proj_seq)
        stack_seq_0 = self.act(self.seq_encoder_0(stack_proj_seq))
        if self.b_apply_drop: stack_seq_0 = self.drop(stack_seq_0)
        lat_seq = self.seq_encoder_1(stack_seq_0)

        state_proj = torch.cat([pos_proj, vel_proj_seq[:,:,-1]], dim=-1)
        return lat_seq, state_proj
        
    def train_decode_acc_seq(self, lat_seq, state_proj):
        lat_vec = self.flatten(lat_seq)
        out_list = []
        for pred_index in range(self.output_len):
            decoder_block = self.acc_decoders[pred_index]
            out = decoder_block[1](decoder_block[0](lat_vec))
            if self.b_apply_drop: out = self.drop(out)
            out = decoder_block[2](torch.cat([out, state_proj], dim=1))
            if self.b_apply_drop: out = self.drop(out)
            out = decoder_block[3](torch.cat([out], dim=1))
            if self.rel_predict and (0 < pred_index):
                out_list.append(out + out_list[pred_index-1])
            else:
                out_list.append(out)
        return torch.stack(out_list, dim=-1)

    def computePosVelFromAccSeq(self, initial_full_state, acc_output):
        traj_list = [torch.cat([initial_full_state[:,0:4], acc_output[:,:,0]], dim=1)]
        #traj_list[0][:,4:6] = acc_output[:,:,0] # overwrite current acc with initial predicted acc
        #for t_idx in range(1, self.pred_length):
        for t_idx in range(1, self.output_len):
            traj_acc = traj_list[t_idx-1][:, 4:6]
            traj_vel = traj_list[t_idx-1][:, 2:4] + traj_acc * self.output_dt
            traj_pos = traj_list[t_idx-1][:, 0:2] + traj_vel * self.output_dt + traj_acc * self.output_dt ** (2.0) * 0.5
            traj_list.append(torch.cat([traj_pos, traj_vel, acc_output[:, :, t_idx]], dim=1))
        return torch.stack(traj_list, dim=-1)

    def forward(self, input):
        pos = input[:, 0:2,-1]
        vel_seq = input[:, 2:4,:]
        acc_seq = input[:, 4:6,:]

        lat_seq, state_proj = self.train_encoder(pos, vel_seq, acc_seq)
        acc_output_seq = self.train_decode_acc_seq(lat_seq, state_proj)
        #print(f"vec_output_seq.size: {vec_output_seq.size()}")
        output = self.computePosVelFromAccSeq(input[:, :, -1], acc_output_seq)
        #print(f"output.size: {output.size()}")
        return output



### INTERACTION DYNAMICS CLASSES
class SimpleAdmittanceController(object):
    # Admittance controller class
    def __init__(self, I_diag=10.0, B_diag=30.0, K_diag=150.0, pos_dim=2):
        self.pos_dim = pos_dim
        self.updateDynamicsMatrices(I_diag, B_diag, K_diag)


    def getIBK(self):
        return self.I[0,0], self.B[0,0], self.K[0,0]
        
    def updateDynamicsMatrices(self, I, B, K):
        self.I = I * np.eye(self.pos_dim)
        self.B = B * np.eye(self.pos_dim)
        self.K = K * np.eye(self.pos_dim)
        
        self.I_inv = np.linalg.inv(self.I)
        return

    def updateStiffnessDiag(self, K_diag):
        self.K = K_diag * np.eye(self.pos_dim)
        return
        
    def updateDampingDiag(self, B_diag):
        self.B = B_diag * np.eye(self.pos_dim)
        return
        
    # functional static method for computing next state [pos, vel] and differential of next state [dpos, dvel] = [vel, acc]
    def step_old(self, pos, vel, F_meas, F_robot, pos_eq = None, verbose = False):

        if (pos_eq is not None):
            ddx_system = np.dot(self.I_inv, np.dot(self.B, -vel) + np.dot(self.K, pos_eq - pos))
        else:
            ddx_system = np.dot(self.I_inv, np.dot(self.B, -vel))
        ddx_meas = np.dot(self.I_inv, F_meas)
        ddx_robot = np.dot(self.I_inv, F_robot)
        ddx = ddx_system + ddx_meas + ddx_robot

        return ddx, (ddx_system, ddx_meas, ddx_robot)

    def step(self, pos, vel, F_meas, F_robot, pos_eq = None, verbose = False):
        ddx_system = np.dot(self.I_inv, np.dot(self.B, -vel))
        ddx_meas = np.dot(self.I_inv, F_meas)
        ddx_robot = np.dot(self.I_inv, F_robot)
        ddx = ddx_system + ddx_meas + ddx_robot

        return ddx, (ddx_system, ddx_meas, ddx_robot)
    
    def computeSystemForce(self, vel, acc):
        f_sys = np.dot(self.I, acc) + np.dot(self.B, vel)
        return f_sys

### VARIABLE ADMITTANCE CONTROL
# make it functional to keep loop as fast as possible. 
@numba.njit(cache = cache_option)
def compute_robot_force(pos, vel, pos_eq, vel_eq, K_p, K_d):
    F_r = np.dot(K_p, pos_eq - pos) + np.dot(K_d, vel_eq - vel)
    return F_r

@numba.njit(cache = cache_option)
def compute_system_force(vel, B):
    F_sys = np.dot(-B, vel)
    return F_sys

@numba.njit(cache = cache_option)
def compute_system_dynamics(pos, vel, pos_eq, vel_eq, F_meas, I, B, K_p, K_d):
    F_r = compute_robot_force(pos, vel, pos_eq, vel_eq, K_p, K_d)
    F_sys = compute_system_force(vel, B)
    I_inv = np.linalg.inv(I)
    acc = np.dot(I_inv, F_sys + F_r + F_meas)
    return acc, (F_sys, F_r, F_meas)

@numba.njit(cache = cache_option)
def compute_system_dynamics_safe(pos, vel, pos_eq, vel_eq, F_meas, I, B, K_p, K_d, max_F):
    F_r = fast_bound_vector_mag(compute_robot_force(pos, vel, pos_eq, vel_eq, K_p, K_d), max_F)
    F_meas = fast_bound_vector_mag(F_meas, max_F)
    F_sys = compute_system_force(vel, B)
    I_inv = np.linalg.inv(I)
    acc = np.dot(I_inv, F_sys + F_r + F_meas)
    return acc, (F_sys, F_r, F_meas)

@numba.njit(cache = cache_option)
def compute_diag_dynamics_matrices(I_diag, B_diag, K_p_diag, K_d_diag, eye):
    I = I_diag * eye
    B = B_diag * eye
    K_p = K_p_diag * eye
    K_d = K_d_diag * eye
    return I, B, K_p, K_d

I, B, K_p, K_d = compute_diag_dynamics_matrices(10.0, 30.0, 1.0, 2.0, np.eye(2,2))

@numba.njit(cache = cache_option)
def compute_euler_step(pos, vel, acc, dt):
    new_pos = pos + dt * vel + 0.5 * (dt ** 2.0) * acc
    new_vel = vel + dt * acc
    return new_pos, new_vel

@numba.njit(cache = cache_option)
def full_dynamics_step(pos, vel, pos_eq, vel_eq, F_meas, I, B, K_p, K_d, dt):
    acc, (F_sys, F_r, F_meas) = compute_system_dynamics(pos, vel, pos_eq, vel_eq, F_meas, I, B, K_p, K_d)
    new_pos, new_vel = compute_euler_step(pos, vel, acc, dt)
    return new_pos, new_vel, (pos, vel, acc), (F_sys, F_r, F_meas)

@numba.njit(cache = cache_option)
def compute_safe_vel_acc(vel, acc, max_vel, max_acc):
    acc_safe = fast_bound_vector_mag(acc, max_acc)
    vel_safe = fast_bound_vector_mag(vel, max_vel)
    return vel_safe, acc_safe

@numba.njit(cache = cache_option)
def full_dynamics_step_safe(pos, vel, pos_eq, vel_eq, F_meas, I, B, K_p, K_d, dt, max_vel, max_acc, max_F):
    acc, (F_sys, F_r, F_meas) = compute_system_dynamics_safe(pos, vel, pos_eq, vel_eq, F_meas, I, B, K_p, K_d, max_F)
    vel_safe, acc_safe = compute_safe_vel_acc(vel, acc, max_vel, max_acc)
    new_pos, new_vel = compute_euler_step(pos, vel_safe, acc_safe, dt)
    return new_pos, new_vel, (pos, vel_safe, acc_safe), (F_sys, F_r, F_meas)

# keep it very simple, all heavy lifting done by njit functions
class FastAdmittanceController(object):
    def __init__(self, I_diag=10.0, B_diag=30.0, K_p_diag=150.0, K_d_diag = 30.0, pos_dim=2, dt = 0.005, max_vel = 0.1, max_acc = 1.0, max_F = 30.0):
        self.pos_dim = pos_dim
        self.eye = np.eye(self.pos_dim, self.pos_dim)
        self.I_diag = I_diag
        self.B_diag = B_diag
        self.K_p_diag = K_p_diag
        self.K_d_diag = K_d_diag
        self.dt = dt
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_F = max_F

    def updateDynamics(self, B_diag=30.0, K_p_diag=150.0, K_d_diag = 30.0):
        self.B_diag = B_diag
        self.K_p_diag = K_p_diag
        self.K_d_diag = K_d_diag
        return
    
    def step(self, pos, vel, pos_eq, vel_eq, F_meas, dt=None):
        if dt is None:
            dt = self.dt
        
        I, B, K_p, K_d = compute_diag_dynamics_matrices(self.I_diag, self.B_diag, self.K_p_diag, self.K_d_diag, self.eye)
        new_pos, new_vel, (pos, vel, acc), (F_sys, F_r, F_meas) = full_dynamics_step(pos, vel, pos_eq, vel_eq, F_meas, I, B, K_p, K_d, dt)
        return new_pos, new_vel, (pos, vel, acc), (F_sys, F_r, F_meas)
    
    def step_safe(self, pos, vel, pos_eq, vel_eq, F_meas, dt=None, max_vel = None, max_acc = None, max_F = None):
        if dt is None:
            dt = self.dt
        if max_vel is None:
            max_vel = self.max_vel
        if max_acc is None:
            max_acc = self.max_acc
        if max_F is None:
            max_F = self.max_F
        
        I, B, K_p, K_d = compute_diag_dynamics_matrices(self.I_diag, self.B_diag, self.K_p_diag, self.K_d_diag, self.eye)
        new_pos, new_vel, (pos, vel, acc), (F_sys, F_r, F_meas) = full_dynamics_step_safe(pos, vel, pos_eq, vel_eq, F_meas, I, B, K_p, K_d, dt, max_vel, max_acc, max_F)
        return new_pos, new_vel, (pos, vel, acc), (F_sys, F_r, F_meas)


### FORCE CONTROL CLASSES
# generates forces based on a simple impedance control law
class ImpedancePolicy(object):
    def __init__(self, B_diag=30.0, K_diag=0.0, pos_dim=2):
        self.pos_dim = pos_dim
        self.updateStiffnessAndDamping(B_diag, K_diag)

    def updateStiffnessAndDamping(self, K_diag, B_diag):
        self.K = K_diag * np.eye(self.pos_dim)
        self.B = B_diag * np.eye(self.pos_dim)
        return

    def getForce(self, pos, vel, pos_eq, vel_eq):
        stiffness_force = np.dot(self.K, (pos_eq - pos))
        damping_force = np.dot(self.B, (vel_eq - vel))
        total_force = stiffness_force + damping_force
        return total_force, (stiffness_force, damping_force)

# base class for varying the ibk parameters, base works for a static controller
class ImpedanceUpdateRule(object):
    def __init__(self, B_diag=30.0, K_diag=0.0):
        self.B_nominal = B_diag
        self.K_p_nominal = K_diag
        self.K_d_nominal = K_diag * 0.0

    def updateImpedanceParams(self, unused = None):
        return self.B_nominal, self.K_p_nominal, self.K_d_nominal

class UserIntentImpedanceUpdateRule(object):
    def __init__(self,
                b_lb = 5.0, b_ub = 30.0, 
                k_p_lb = 0.0, k_p_ub = 300.0, 
                k_d_lb = 0.0, k_d_ub = 60.0, 
                s_sensitivity = 0.95, r_slope = 75.0, delta_offset = 3.0,
                max_user_intent = 2.0, min_user_intent = -2.0):
        super().__init__()
        self.s_sensitivity = s_sensitivity
        self.setUserIntentBounds(min_user_intent, max_user_intent)
        
        self.vac_b_lb = b_lb
        self.vac_b_ub = b_ub
        self.vac_k_p_ub = k_p_ub
        self.vac_k_d_ub = k_d_ub 
        self.vac_r = 0.0 * r_slope # unused
        self.vac_delta = 0.0 * delta_offset # unused

    def setUserIntentBounds(self, user_intent_min, user_intent_max):
        self.max_user_intent = user_intent_max
        self.min_user_intent = user_intent_min
        self.user_intent_range = self.max_user_intent - self.min_user_intent
        
        self.k_p = - np.log(((1.0 - self.s_sensitivity) / (1.0 + self.s_sensitivity))) / self.max_user_intent
        self.k_n = - np.log(((1.0 + self.s_sensitivity) / (1.0 - self.s_sensitivity))) / self.min_user_intent

        log_sen = np.log(self.s_sensitivity / (1.0 - self.s_sensitivity))
        self.vac_r = (2.0 / (self.max_user_intent - self.min_user_intent)) * log_sen
        self.vac_delta = (self.max_user_intent + self.min_user_intent / (self.max_user_intent - self.min_user_intent)) * log_sen

    def getNormUserIntent(self, user_intent):
        if user_intent <= self.min_user_intent:
            n_ui = 0.0
        elif self.max_user_intent <= user_intent:
            n_ui = 1.0
        else:
            n_ui = (user_intent - self.min_user_intent) / self.user_intent_range
        return n_ui
    
    def adjustVariableDamping(self, user_intent):
        #s_ui = self.getNormUserIntent(user_intent)
        #new_damping = interp(s_ui, self.vac_b_ub, self.vac_b_lb)
        
        #new_damping = (self.vac_b_ub - self.vac_b_lb) / (1.0 + np.exp(self.k_p * user_intent)) + self.vac_b_lb
        
        if (0.0 <= user_intent):
            new_damping = (2.0 * self.vac_b_lb) / (1.0 + np.exp(-self.k_p * user_intent)) - self.vac_b_lb #
        else:
            new_damping = -(2.0 * self.vac_b_ub) / (1.0 + np.exp(-self.k_n * user_intent)) + self.vac_b_ub
        
        return new_damping

    def adjustVariableStiffness(self, user_intent):
        if (0.0 <= user_intent):
            #new_stiffness = (self.vac_k_ub) / (1.0 + np.exp(- self.vac_r * user_intent + self.vac_delta))
            new_stiffness = (self.vac_k_p_ub) / (1.0 + np.exp(- self.vac_r * user_intent + self.vac_delta))
            #new_stiffness = (self.vac_k_p_ub) / (1.0 + np.exp(-self.k_p * user_intent))
        else:
            new_stiffness = 0.0 # if there's negative intent, don't apply asssistance force.
        return new_stiffness

    def adjustVariableVelStiffness(self, user_intent):
        if (0.0 <= user_intent):
            new_stiffness = (self.vac_k_d_ub) / (1.0 + np.exp(- self.vac_r * user_intent + self.vac_delta))
            #new_stiffness = (self.vac_k_d_ub) / (1.0 + np.exp(-self.k_p * user_intent))
        else:
            new_stiffness = 0.0 # if there's negative intent, don't apply asssistance force.
        return new_stiffness

    def updateImpedanceParams(self, user_intent):
        if user_intent < self.min_user_intent:
            user_intent = self.min_user_intent
        elif self.max_user_intent < user_intent:
            user_intent = self.max_user_intent
        B = self.adjustVariableDamping(user_intent)
        K_p = self.adjustVariableStiffness(user_intent)
        K_d = self.adjustVariableVelStiffness(user_intent)

        return B, K_p, K_d

class ConfidenceImpedanceUpdateRule(object):
    def __init__(self,
                b_lb = 30.0, b_ub = 5.0, 
                k_p_lb = 0.0, k_p_ub = 300.0, 
                k_d_lb = 0.0, k_d_ub = 60.0):
        self.b_lb = b_lb
        self.b_ub = b_ub
        self.k_p_lb = k_p_lb
        self.k_p_ub = k_p_ub
        self.k_d_lb = k_d_lb
        self.k_d_ub = k_d_ub

    def updateImpedanceParams(self, x):
        B = interp(x, self.b_lb, self.b_ub)
        K_p = interp(x, self.k_p_lb, self.k_p_ub)
        K_d = interp(x, self.k_d_lb, self.k_d_ub)
        return B, K_p, K_d

class ConfidenceUserIntentImpedanceUpdateRule(object):
    def __init__(self,
                b_lb = 30.0, b_ub = 5.0, 
                k_p_lb = 0.0, k_p_ub = 300.0, 
                k_d_lb = 0.0, k_d_ub = 60.0,
                s_sensitivity = 0.95,
                max_user_intent = 2.0, min_user_intent = -2.0):
        self.b_lb = b_lb
        self.b_ub = b_ub
        self.k_p_lb = k_p_lb
        self.k_p_ub = k_p_ub
        self.k_d_lb = k_d_lb
        self.k_d_ub = k_d_ub
        self.s_sensitivity = s_sensitivity
        self.min_user_intent = min_user_intent
        self.max_user_intent = max_user_intent

        self.s_sensitivity = s_sensitivity
        self.setUserIntentBounds(min_user_intent, max_user_intent)

    def setUserIntentBounds(self, user_intent_min, user_intent_max):
        self.max_user_intent = user_intent_max
        self.min_user_intent = user_intent_min
        self.user_intent_range = self.max_user_intent - self.min_user_intent
        
        self.k_p = - np.log(((1.0 - self.s_sensitivity) / (1.0 + self.s_sensitivity))) / self.max_user_intent
        self.k_n = - np.log(((1.0 + self.s_sensitivity) / (1.0 - self.s_sensitivity))) / self.min_user_intent

    def getNormUserIntent(self, user_intent):
        if user_intent <= self.min_user_intent:
            n_ui = 0.0
        elif self.max_user_intent <= user_intent:
            n_ui = 1.0
        else:
            n_ui = (user_intent - self.min_user_intent) / self.user_intent_range
        return n_ui
    
    def adjustVariableDamping(self, user_intent):
        #s_ui = self.getNormUserIntent(user_intent)
        #new_damping = interp(s_ui, self.b_lb, self.b_ub)
        
        if (0.0 <= user_intent):
            new_damping = (2.0 * self.b_ub) / (1.0 + np.exp(-self.k_p * user_intent)) - self.b_ub # UB AND LB ARE SWITCHED ON PURPOSE HERE
        else:
            new_damping = -(2.0 * self.b_lb) / (1.0 + np.exp(-self.k_n * user_intent)) + self.b_lb 
        
        return new_damping

    def adjustVariableStiffness(self, user_intent):
        if (0.0 <= user_intent):
            #new_stiffness = (self.vac_k_ub) / (1.0 + np.exp(- self.vac_r * user_intent + self.vac_delta))
            new_stiffness = (self.k_p_ub) / (1.0 + np.exp(-self.k_p * user_intent))
        else:
            new_stiffness = 0.0 # if there's negative intent, don't apply asssistance force.
        return new_stiffness

    def adjustVariableVelStiffness(self, user_intent):
        if (0.0 <= user_intent):
            #new_stiffness = (self.vac_k_ub) / (1.0 + np.exp(- self.vac_r * user_intent + self.vac_delta))
            new_stiffness = (self.k_d_ub) / (1.0 + np.exp(-self.k_p * user_intent))
        else:
            new_stiffness = 0.0 # if there's negative intent, don't apply asssistance force.
        return new_stiffness
    
    def updateImpedanceParams(self, x, user_intent):
        if user_intent < self.min_user_intent:
            user_intent = self.min_user_intent
        elif self.max_user_intent < user_intent:
            user_intent = self.max_user_intent
    
        #B = interp(x, self.b_lb, self.adjustVariableDamping(user_intent)) # use this if we want to couple confidence and basic damping. I think it makes sense to leave them unconnected.
        B = interp(x, self.b_lb, self.adjustVariableDamping(user_intent)) # 
        K_p = interp(x, self.k_p_lb, self.adjustVariableStiffness(user_intent))
        K_d = interp(x, self.k_d_lb, self.adjustVariableVelStiffness(user_intent))
        #K_p = interp(x, self.k_p_lb, self.k_p_ub)
        #K_d = interp(x, self.k_d_lb, self.k_p_ub)
        return B, K_p, K_d
    
class AdaptiveConformalPredictionHelper:
    def __init__(self, target_alpha = 0.1, prediction_length = 6, step_sizes=[0.0, 0.001, 0.005], window_length=1, init_quantiles = None, use_jit = True, use_parallel = True):
        
        self.prediction_length = prediction_length # maybe don't need this as class member
        self.step_sizes_list = step_sizes
        self.target_alphas_list = [target_alpha for g in range(len(step_sizes))]
        self.window_length = window_length
        self.use_jit = use_jit
        self.use_parallel = use_parallel

        self.target_alphas_array = np.array(self.target_alphas_list)
        self.current_alphas_array = np.zeros(shape=(len(step_sizes), self.prediction_length))
        self.step_sizes_array = np.zeros(shape=(len(step_sizes), self.prediction_length))
        self.quantiles_array = np.zeros(shape=(len(step_sizes), self.prediction_length))

        for t_idx in range(0, self.prediction_length):
            self.current_alphas_array[:, t_idx] = self.target_alphas_array.copy()
            self.step_sizes_array[:, t_idx] = np.array(self.step_sizes_list)
            #self.quantiles_array[:, t_idx] = self.target_alphas_array.copy() # leave as zeros

        if self.use_jit: # need to initalize it
            print(f"AdaptiveConformalPredictionHelper: Using JIT, parallel={self.use_parallel}! Initalizing JIT function...")
            fake_score_array = np.zeros(shape=(window_length * 2, prediction_length))
            out = self.updateQuantiles(fake_score_array)

        if init_quantiles is None:
            self.quantiles_ready = False
        else:
            self.quantiles_array = init_quantiles
            self.quantiles_ready = True

        print("AdaptiveConformalPredictionHelper: Ready!")
        return

    def updateStepSizes(self, new_step_sizes):
        #raise NotImplementedError
        
        for t_idx in range(0, self.prediction_length):
            self.step_sizes_array[:, t_idx] = np.array(new_step_sizes[:, t_idx])
        print(f"ACI Helper: Step sizes updated to: \n{self.step_sizes_array}")
        return

    def _updateQuantilesJIT(self, scores_typed_list):
        if self.use_parallel:
            alpha_quantile_list = update_alphas_and_quantiles_list_njit_par(scores_typed_list, self.current_alphas_array,
                                                                        self.quantiles_array, self.target_alphas_array, 
                                                                        self.step_sizes_array, self.window_length)
        else:
            alpha_quantile_list = update_alphas_and_quantiles_list_njit(scores_typed_list, self.current_alphas_array,
                                                                    self.quantiles_array, self.target_alphas_array, 
                                                                    self.step_sizes_array, self.window_length)
        return alpha_quantile_list
    
    def updateQuantiles(self, score_array):
        scores_typed_list = List()
        for t_idx in range(0, score_array.shape[1]):
            scores_typed_list.append(score_array[:, t_idx].copy()) # hopefully this doesn't kill speed gains
        
        if self.use_jit:
            alpha_quantile_list = self._updateQuantilesJIT(scores_typed_list)
        else:
            alpha_quantile_list = update_alphas_and_quantiles_list(scores_typed_list, self.current_alphas_array,
                                                                        self.quantiles_array, self.target_alphas_array, 
                                                                        self.step_sizes_array, self.window_length)
        self.current_alphas_array = alpha_quantile_list[0]
        self.quantiles_array = alpha_quantile_list[1]
        return self.quantiles_array, self.current_alphas_array

    def getQuantilesAndAlphas(self):
        return self.quantiles_array, self.current_alphas_array

class MemoryHelper:
    def __init__(self, state_chn_num = 6, min_state_buffer_size = 125, state_buffer_size = 200, score_buffer_size = 200, score_len = 6, prediction_buffer_size = 125):

        # state buffer stuff, keep it as a numpy array for easier math
        self.state_chn_num = state_chn_num
        self.state_buffer_size = state_buffer_size
        self.state_buffer_min_size = min_state_buffer_size
        self.clearStateBuffer()
        self.state_buffer_list = [] # list of state buffers

        # score buffer stuff
        self.score_buffer_size = score_buffer_size
        self.score_len = score_len
        self.score_buffer = np.zeros(shape=(self.score_buffer_size, self.score_len))
        self.score_buffer_counter = 0
        self.is_score_buffer_full = False

        # prediction buffer stuff
        self.prediction_buffer = deque(maxlen = prediction_buffer_size)

    def updateStateBuffer(self, new_state):
        if (self.state_buffer_counter < self.state_buffer_size):
            self.state_buffer[:, self.state_buffer_counter] = new_state
            self.state_buffer_counter += 1
        else:
            self.state_buffer[:, 0:self.state_buffer_size-1] = self.state_buffer[:, 1:self.state_buffer_size]
            self.state_buffer[:, -1] = new_state
        return (self.state_buffer_min_size < self.state_buffer_counter, self.state_buffer_size <= self.state_buffer_counter)
        
    def clearStateBuffer(self):
        self.state_buffer = np.zeros(shape=(self.state_chn_num, self.state_buffer_size))
        self.state_buffer_counter = 0 # counts how many elements have been added to the buffer
        return

    def segmentStateBuffer(self):
        self.state_buffer_list.append(self.state_buffer.copy())
        self.clearStateBuffer()
        return

    def updateScoreBuffer(self, new_score):
        if (self.score_buffer_counter < self.score_buffer_size):
            self.score_buffer[self.score_buffer_counter, :] = new_score
            self.score_buffer_counter += 1
        else:
            self.score_buffer[0:self.score_buffer_size-1, :] = self.score_buffer[1:self.score_buffer_size, :]
            self.score_buffer[-1, :] = new_score
        return 

    def isScoreBufferFull(self):
        return self.score_buffer_size <= self.score_buffer_counter

    def getLastScore(self):
        return self.score_buffer[-1,:]

    def updatePredictionBuffer(self, new_prediction):
        self.prediction_buffer.append(new_prediction)
        return 

class SimulatedSubject:
    def __init__(self):
        self.dt = 0.005
        self.state_dim = 2
        self.error_vec = np.zeros(shape=(self.state_dim,))
        self.error_derivative_vec = np.zeros(shape=(self.state_dim,))
        self.error_integral_vec = np.zeros(shape=(self.state_dim,))
        # control gains
        self.P_gain = 100.0 #20.0
        self.D_gain = 20.0 #1.0
        self.I_gain = 0.0 #0.5
    def refreshErrorSignals(self):
        self.error_vec = np.zeros(shape=(self.state_dim,))
        self.error_derivative_vec = np.zeros(shape=(self.state_dim,))
        self.error_integral_vec = np.zeros(shape=(self.state_dim,))
    
    def updateErrorSignals(self, current_state, desired_state, used_dims = [0, 1]):
        last_error_vec = self.error_vec.copy()
        self.error_vec = desired_state[used_dims] - current_state[used_dims]
        self.error_derivative_vec = self.dt * (self.error_vec - last_error_vec)
        self.error_integral_vec = self.error_integral_vec + self.dt * self.error_vec
        return
        
    def getControlSignal(self):
        return self.P_gain * self.error_vec + self.D_gain * self.error_derivative_vec + self.I_gain * self.error_integral_vec

    def setCritDampedPDGains(self, I, B, time_constant = 0.5):
        # damping_ratio of 1
        K_D = I * B / time_constant - B
        K_P = ((1.0 - K_D / B) ** 2.0) / I
        self.P_gain = K_P
        self.D_gain = K_D
        return

class ModelInferenceWrapperAcceleration(): ## NEURAL MODEL THAT PREDICTS ACCELERATION
    def __init__(self, model, model_params, scalar_params, use_floats = True, force_cpu = True, compile_model = True, return_numpy = True, base_dt = 0.005):
        self.base_model = model
        self.model_params = model_params
        self.scalar_params = scalar_params
        self.input_scaler, self.input_center, self.output_scaler, self.output_center = self.makeScalingTensors()
        self.use_floats = use_floats
        if self.use_floats:
            self.base_model = self.base_model.float()
            self.input_scaler = self.input_scaler.float()
            self.input_center = self.input_center.float()
            self.output_scaler = self.output_scaler.float()
            self.output_center = self.output_center.float()
        self.return_numpy = return_numpy

        # put everything torch-related on the right devices and compile if we can
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if force_cpu:
            self.device = "cpu"
        self.base_model.to(self.device)
        self.input_scaler = self.input_scaler.to(self.device)
        self.input_center = self.input_center.to(self.device)
        self.output_scaler = self.output_scaler.to(self.device)
        self.output_center = self.output_center.to(self.device)
        self.last_output = None
        if compile_model:
            self.base_model = torch.compile(self.base_model).eval()

        # interpolating components for getting smooth outputs
        self.base_dt = base_dt
        self.output_time_seq = np.array(self.model_params["output_idx_seq"]) * self.base_dt
        print(f"NN Model Output Time Sequence: {self.output_time_seq}")
        self.base_big_dt = self.output_time_seq[1] - self.output_time_seq[0]
        self.dense_output_time_seq = np.arange(np.min(self.output_time_seq), np.max(self.output_time_seq) + self.base_dt, self.base_dt)
        #print(f"Output time sequence: \n{self.output_time_seq}")
        #print(f"Dense output time sequence: \n{self.dense_output_time_seq}")
        self.last_output_poly = None
        self.last_output_dpoly = None

        # initialize inputs and outputs
        temp_input = np.zeros(shape=(len(self.model_params['input_channels']), len(self.model_params['input_idx_seq'])))
        temp_output = self.predict(temp_input) # should JIT compile the model and initialize last inputs and outputs

        self.zero_output = temp_output * 0.0 # 

    def makeScalingTensors(self):
        input_scales = []
        input_centers = []
        for chn_idx, chn in enumerate(self.model_params["input_channels"]):
            row_scales = []
            row_centers = []
            for step in enumerate(self.model_params["input_idx_seq"]):
                row_scales.append(self.scalar_params["scales"][chn_idx])
                row_centers.append(self.scalar_params["centers"][chn_idx])
            input_scales.append(row_scales)
            input_centers.append(row_centers)
        output_scales = []
        output_centers = []
        for chn_idx, chn in enumerate(self.model_params["output_channels"]):
            row_scales = []
            row_centers = []
            for step in enumerate(self.model_params["output_idx_seq"]):
                row_scales.append(self.scalar_params["scales"][chn_idx])
                row_centers.append(self.scalar_params["centers"][chn_idx])
            output_scales.append(row_scales)
            output_centers.append(row_centers)
        input_scaler_t = 1.0 / torch.tensor(input_scales)
        input_center_t = -torch.tensor(input_centers)
        output_scaler_t = torch.tensor(output_scales)
        output_center_t = torch.tensor(output_centers)
        return input_scaler_t, input_center_t, output_scaler_t, output_center_t
        
    def scaleInput(self, input):
        return self.input_scaler * input + self.input_center
    
    def scaleOutput(self, output):
        return self.output_scaler * output + self.output_center
        
    def predictOutput(self, input):
        if self.use_floats:
                input_t = torch.tensor(input, dtype=torch.float)
        else:
            input_t = torch.tensor(input, dtype=torch.double)
        if len(input_t.size()) == 2:
            input_t = torch.unsqueeze(input_t, 0)
        input_t = input_t.to(self.device)

        output = self.scaleOutput(self.base_model(self.scaleInput(input_t)))
        if self.return_numpy:
            output = output.detach().cpu().numpy()
        output = output[-1,:,:] # no batch dim
        return output

    def predictTrajectory(self, input):
        acc_output = self.predictOutput(input)
        output_traj = self.computeTrajectoryFromAcceleration(acc_output, input[:,-1], self.base_big_dt)
        return output_traj

    def predict(self, input, fit_output_poly = False):
        output_traj = self.predictTrajectory(input)
        self.last_input = input[:,:].copy() # should be numpy
        self.last_output = output_traj[0:2, :].copy() #output[-1,:,:].copy()
        return output_traj[0:2, :] #output

    def computeTrajectoryFromAcceleration(self, acc_output, initial_state, output_dt = 0.050):
        output_traj = np.zeros(shape=(initial_state.shape[0], acc_output.shape[1]))
        output_traj[4:6, :] = acc_output
        output_traj[0:4, 0] = initial_state[0:4]
        for t_idx in range(1, output_traj.shape[1]):
            output_traj[2:4, t_idx] = output_traj[2:4, t_idx-1] + output_traj[4:6, t_idx-1] * output_dt
            output_traj[0:2, t_idx] = output_traj[0:2, t_idx-1] + output_traj[2:4, t_idx-1] * output_dt + output_traj[4:6, t_idx-1] * output_dt ** (2.0) * 0.5
        return output_traj

    def interpolateOutput(self, output_seq = None):
        if output_seq is None:
            output_seq = self.last_output.T
        else:
            output_seq = output_seq.T
        self.last_output_poly = PchipInterpolator(self.output_time_seq, output_seq)
        self.last_output_dpoly = self.last_output_poly.derivative()
        return 

    def sampleOutputPoly(self, sample_idx = None):
        if (self.last_output_poly is not None):
            if (sample_idx is not None):
                output_pos = self.last_output_poly(sample_idx)
                output_vel = self.last_output_dpoly(sample_idx)
            else:
                output_pos = self.last_output_poly(self.dense_output_time_seq)
                output_vel = self.last_output_dpoly(self.dense_output_time_seq)
        else:
            output_pos = None
            output_vel = None
        return output_pos, output_vel



class RegModelInferenceWrapper(): # REGRESSION MODEL
    def __init__(self, model_params, poly_degree = 3, base_dt = 0.005):
        
        self.input_channels = model_params['input_channels']
        self.input_idx_seq = model_params['input_idx_seq']
        self.input_time_seq = [idx * base_dt for idx in self.input_idx_seq] #self.input_idx_seq * base_dt
        self.input_time_array = np.array(self.input_time_seq)

        self.output_channels = model_params['output_channels']
        self.output_idx_seq = model_params['output_idx_seq']
        self.output_time_seq = [idx * base_dt for idx in self.output_idx_seq]
        self.output_time_array = np.array(self.output_time_seq)
        
        self.model_params = model_params

        self.poly_degree = poly_degree # poly features are [1, x, x^2, ...] with biggest exponent as the degree. Also does interactions if there are any.
        self.base_model = self._makeModelPipeline()

        self.time_fit = self.input_time_array[np.newaxis, :].T
        self.time_predict = self.output_time_array[np.newaxis, :].T
        
        self.last_input = np.zeros(shape=(len(model_params['input_channels']), len(model_params['input_idx_seq'])))
        self.last_output = np.zeros(shape=(len(model_params['output_channels']), len(model_params['output_idx_seq'])))
        

    def _makeModelPipeline(self):
        model = Pipeline([('poly', PolynomialFeatures(degree=self.poly_degree)),
                  ('linear', LinearRegression(fit_intercept=False))]) # intercept is generated by PolynomialFeatures
        #model = Pipeline([('linear', LinearRegression(fit_intercept=True))]) # intercept is generated by PolynomialFeatures
        return model

    def predict(self, input, b_correct_initial_offset = True):
        self.last_input = input
        pos_fit = input[self.output_channels, :].T
        self.base_model.fit(self.time_fit, pos_fit)
        
        output = self.base_model.predict(self.time_predict).T

        # fix initial shift
        if b_correct_initial_offset:
            current_pos = input[self.output_channels, -1]
            pred_current_pos = output[:, 0]
            init_shift = pred_current_pos - current_pos
            output = output - init_shift[:, np.newaxis]

        self.last_output = output
        return output
        
    def interpolateOutput(self):
        self.last_output_poly = PchipInterpolator(self.output_time_seq, self.last_output.T)
        self.last_output_dpoly = self.last_output_poly.derivative()
        return 
        
    def sampleOutputPoly(self, sample_idx = None):
        if (self.last_output_poly is not None):
            if (sample_idx is not None):
                output_pos = self.last_output_poly(sample_idx)
                output_vel = self.last_output_dpoly(sample_idx)
            else:
                output_pos = self.last_output_poly(self.dense_output_time_seq)
                output_vel = self.last_output_dpoly(self.dense_output_time_seq)
        else:
            output_pos = None
            output_vel = None
        return output_pos, output_vel


def get_mint_input_from_state_buffer(start_index, input_index_seq, memory_helper, input_chn = [0, 1, 2, 3, 4, 5]):
    input_seq = memory_helper.state_buffer[:, start_index + np.array(input_index_seq, dtype=np.int32)]
    return input_seq[input_chn, :]

def get_mint_target_from_state_buffer(start_index, output_index_seq, memory_helper, output_chn = [0, 1]):
    target_seq = memory_helper.state_buffer[:, start_index + np.array(output_index_seq, dtype=np.int32)]
    return target_seq[output_chn, :]

def get_mint_input_target_from_state_buffer(start_index, input_index_seq, output_index_seq, memory_helper):
    return (get_mint_input_from_state_buffer(start_index, input_index_seq, memory_helper), get_mint_target_from_state_buffer(start_index, output_index_seq, memory_helper))

def vec_loss(target, output):
    return np.linalg.norm(target - output, axis=0, keepdims=True)


def get_prediction_model(model_name, rel_model_dir = "res/models", hparam_extension = "_hyper_params.json", dt=0.005):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(current_directory, rel_model_dir) # should be where the model is saved

    model_path = os.path.join(model_dir, model_name + ".pt")
    print(model_name)
    hyperparam_path = os.path.join(model_dir, model_name + hparam_extension)

    f = open(hyperparam_path)
    params = json.loads(json.load(f))
    f.close()
    model_params = params["model_params"]
    scalar_params = params["scalar_params"]
    
    base_mdl = MIntNet_Double_Conditioner_Acc(input_channels = model_params["input_channels"], input_idx_seq = model_params["input_idx_seq"], 
        input_stride = model_params["input_stride"], input_dialation = model_params["input_dialation"], 
        input_kernels = model_params["input_kernels"], hidden_size = model_params["hidden_size"], 
        latent_features = model_params["latent_features"], proj_features = model_params["proj_features"], output_channels = model_params["output_channels"], 
        output_idx_seq = model_params["output_idx_seq"], output_dt = model_params['output_dt'], rel_predict = model_params['rel_predict'],
        b_use_dropout = model_params["b_use_dropout"], dropout_rate = model_params["dropout_rate"])

    try:
        base_mdl.load_state_dict(torch.load(model_path))
    except:
        print("\nERROR LOADING PYTORCH INFERENCE MODEL, trying with map_location=torch.device('cpu')...")
        base_mdl.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    inf_model = ModelInferenceWrapperAcceleration(base_mdl, model_params, scalar_params, force_cpu=True, base_dt = params["base_delta_time"])
    return inf_model, params