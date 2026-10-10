#!/usr/bin/python3

import numpy as np
import numba
cache_option = True

### from old utils
@numba.njit(cache=True)
def fast_bound_vector_mag(vec, mag_bound):
    vec_mag = np.linalg.norm(vec, ord=2)
    if (mag_bound < vec_mag):
        scaling_term = mag_bound / vec_mag
        rescaled_vec = vec * scaling_term
    else:
        rescaled_vec = vec
    return rescaled_vec

@numba.njit(cache=True)
def fast_rescale_vector_mag(vec, reference_vec, tol):
    vec_mag = np.linalg.norm(vec, ord=2)
    reference_vec_mag = np.linalg.norm(reference_vec, ord=2)
    if tol < vec_mag:
        scaling_term = reference_vec_mag / vec_mag
    else:
        scaling_term = 1.0
    rescaled_vec = vec * scaling_term
    return rescaled_vec

### basic dynamics
@numba.njit(cache = cache_option)
def compute_diag_dynamics_matrices(I_diag, B_diag, K_p_diag, K_d_diag, eye):
    I = I_diag * eye
    B = B_diag * eye
    K_p = K_p_diag * eye
    K_d = K_d_diag * eye
    return I, B, K_p, K_d

@numba.njit(cache = cache_option)
def compute_euler_step(pos, vel, acc, dt):
    new_pos = pos + dt * vel + 0.5 * (dt ** 2.0) * acc
    new_vel = vel + dt * acc
    return new_pos, new_vel

@numba.njit(cache = cache_option)
def full_dynamics_step_safe(pos, vel, pos_eq, vel_eq, F_meas, F_robot, I, B, K_p, K_d, dt, safe_vel_max, safe_acc_max, safe_F_max):
    F_sys = np.dot(-B, vel) + np.dot(K_p, pos_eq - pos) + np.dot(K_d, vel_eq - vel)
    I_inv = np.linalg.inv(I)
    F_sum = fast_bound_vector_mag(F_sys + F_robot + F_meas, safe_F_max)
    acc = np.dot(I_inv, F_sum)

    #acc, (F_sys, F_r, F_meas) = compute_system_dynamics_safe(pos, vel, pos_eq, vel_eq, F_meas, I, B, K_p, K_d, max_F)
    acc_safe = fast_bound_vector_mag(acc, safe_acc_max)
    vel_safe = fast_bound_vector_mag(vel, safe_vel_max)
    new_pos, new_vel = compute_euler_step(pos, vel_safe, acc_safe, dt)
    return new_pos, new_vel, (pos, vel_safe, acc_safe)

# keep it very simple, all heavy lifting done by njit functions
class FastAdmittanceController(object):
    def __init__(self, pos_init, vel_init, I_diag=10.0, B_diag=30.0, K_p_diag=150.0, K_d_diag = 30.0, pos_dim=2, dt = 0.005, max_vel = 0.1, max_acc = 1.0, max_F = 30.0):
        self.pos_dim = len(pos_init)
        self.eye = np.eye(self.pos_dim, self.pos_dim)
        self.dt = dt
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_F = max_F

        self.updateDynamics(I_diag, B_diag, K_p_diag, K_d_diag)
        self.initalizeState(pos_init, vel_init)

    def updateDynamics(self, I_diag=10.0, B_diag=30.0, K_p_diag=150.0, K_d_diag = 30.0):
        self.I, self.B, self.K_p, self.K_d = compute_diag_dynamics_matrices(I_diag, B_diag, K_p_diag, K_d_diag, self.eye)
        return
    
    def initalizeState(self, pos_init, vel_init):
        self.pos = np.array(pos_init)
        self.vel = np.array(vel_init)
        self.acc = np.zeros_like(self.vel) #np.array(vel_init)
        return
    
    # handles variable inputs and computes discrete dynamics update, does not update internal state!
    def step(self, pos, vel, F_meas, F_robot, pos_eq, vel_eq, dt=None, max_vel = None, max_acc = None, max_F = None):
        if dt is None:
            dt = self.dt
        if max_vel is None:
            safe_vel_max = self.max_vel
        if max_acc is None:
            safe_acc_max = self.max_acc
        if max_F is None:
            safe_F_max = self.max_F

        new_pos, new_vel, (pos, vel, acc) = full_dynamics_step_safe(pos, vel, 
                                                                    pos_eq, vel_eq, 
                                                                    F_meas, F_robot, 
                                                                    self.I, self.B, self.K_p, self.K_d, dt, 
                                                                    safe_vel_max, safe_acc_max, safe_F_max)

        return new_pos, new_vel, (pos, vel, acc)
    
    def admittanceUpdate(self, F_meas, F_robot, pos_eq, vel_eq):
        new_pos, new_vel, (pos, vel, acc) = self.step(self.pos, self.vel, F_meas, F_robot, pos_eq, vel_eq)
        # update info for state dict here
        self.pos = new_pos
        self.vel = new_vel
        self.acc = acc
        return #self.pos, self.vel
    
    def getStateDictonary(self): # add other info we want to publish or work with here!
        state_dict = dict()
        state_dict['pos'] = self.pos.copy()
        state_dict['vel'] = self.vel.copy()
        state_dict['acc'] = self.acc.copy()
        return state_dict
