import warnings
import time
from collections import deque

import scipy.signal
import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import splprep, splev, CubicHermiteSpline, PchipInterpolator

from scipy.stats import randint
from sklearn.preprocessing import RobustScaler
import json

import torch
from torch import nn
import os, fnmatch
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import numba
from numba.typed import List # experimental typed list object

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

### LOOSE FUNCTIONS used to make everything easier to import, altough it is messy. Consider organizing later.
def pack_array_to_multiarray_msg(array, row_name = "rows", col_name = "columns"):
    msg = Float64MultiArray()
    msg.data = array.copy().flatten().tolist()
    msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
    msg.layout.dim[0].label = row_name
    msg.layout.dim[0].size = array.shape[0] # rows
    msg.layout.dim[0].stride = array.shape[0] * array.shape[1] # full size
    msg.layout.dim[1].label = col_name
    msg.layout.dim[1].size = array.shape[1] # col num
    msg.layout.dim[1].stride = array.shape[1] # sizes to end
    return msg

def unpack_multiarray_msg(msg):
    #print(msg)
    row_num = msg.layout.dim[0].size
    row_stride = msg.layout.dim[0].stride
    col_num = msg.layout.dim[1].size
    col_stride = msg.layout.dim[1].stride
    flat_array = msg.data
    array = np.reshape(flat_array, (row_num, col_num))
    return array

def test_msg_packing():
    ary_shape = (9, 11)
    ary_data = [i for i in range(ary_shape[0] * ary_shape[1])]
    ary = np.reshape(np.array(ary_data), ary_shape)
    print(f"ary: {ary}")

    msg = pack_array_to_multiarray_msg(ary)
    print(f"packed msg: {msg.data}")

    up_ary = unpack_multiarray_msg(msg)
    print(f"unpacked ary: {up_ary}")

def OLD_wait_for_time(start_time_point, cycle_time):
    end_before_rest = time.time()
    elapsed_time = end_before_rest - start_time_point
    while elapsed_time < cycle_time:
        elapsed_time = time.time() - start_time_point

def wait_for_time_block(start_time_point, wait_time, warning_prefix = "!!! Loop time warning: ", verbose = False):
    end_before_rest = time.time()
    elapsed_time = end_before_rest - start_time_point

    if (wait_time < elapsed_time):
        if verbose:
            print(f"{warning_prefix}Wanted to wait until {wait_time}s but elasped_time was already {elapsed_time}!!!")
    else:
        while elapsed_time < wait_time:
            elapsed_time = time.time() - start_time_point
    return elapsed_time

def wait_for_time(start_time_point, wait_time, warning_prefix = "!!! Loop time warning: ", verbose = False):
    end_before_rest = time.time()
    elapsed_time = end_before_rest - start_time_point
    time_until_end = wait_time - elapsed_time
    if 0.0 < time_until_end:
        time.sleep(time_until_end)
    
    return time.time() - start_time_point #elapsed_time
    #return

### SAFETY FUNCTIONS
def bound_vector_box(vec, box_bound, vec_mean = None):
    b_hit_bound = False
    if vec_mean is None:
        vec_mean = np.zeros_like(vec)

    shifted_vec = vec - vec_mean
    if isinstance(box_bound, float):
        for idx in range(shifted_vec.shape[0]):
            if shifted_vec[idx] < -box_bound:
                b_hit_bound = True
                shifted_vec[idx] = -box_bound
            elif box_bound < shifted_vec[idx]:
                b_hit_bound = True
                shifted_vec[idx] = box_bound
    elif isinstance(box_bound, list):
        for idx in range(shifted_vec.shape[0]):
            if shifted_vec[idx] < -box_bound[idx]:
                b_hit_bound = True
                shifted_vec[idx] = -box_bound[idx]
            elif box_bound[idx] < shifted_vec[idx]:
                b_hit_bound = True
                shifted_vec[idx] = box_bound[idx]
    else:
        print(f"Unrecognized box_bound type! Got {box_bound} and type(box_bound) of {type(box_bound)}")

    rescaled_vec = shifted_vec + vec_mean
    return rescaled_vec, b_hit_bound

def bound_vector_mag(vec, mag_bound, vec_mean = None):
    b_hit_bound = False
    if vec_mean is None:
        vec_mean = np.zeros_like(vec)

    shifted_vec = vec - vec_mean
    vec_mag = np.linalg.norm(shifted_vec, ord=2)
    if (mag_bound < vec_mag):
        b_hit_bound = True
        scaling_term = mag_bound / vec_mag
        rescaled_shifted_vec = shifted_vec * scaling_term
        rescaled_vec = rescaled_shifted_vec + vec_mean
    else:
        rescaled_vec = vec
    return rescaled_vec, b_hit_bound

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

def notch_vector_mag(vec, mag_bound, vec_mean = None):
    b_hit_bound = False
    if vec_mean is None:
        vec_mean = np.zeros_like(vec)

    shifted_vec = vec - vec_mean
    vec_mag = np.linalg.norm(shifted_vec, ord=2)
    if (vec_mag < mag_bound):
        b_hit_bound = True
        rescaled_shifted_vec = shifted_vec * 0.0
        rescaled_vec = rescaled_shifted_vec + vec_mean
    else:
        rescaled_vec = vec
    return rescaled_vec, b_hit_bound

def bound_scalar_mag(scalar, scalar_bound, scalar_mean = None):
    b_hit_bound = False
    if scalar_mean is None:
        scalar_mean = 0.0

    shifted_scalar = scalar - scalar_mean
    scalar_mag = abs(shifted_scalar)
    if (scalar_bound < scalar_mag):
        b_hit_bound = True
        scaling_term = scalar_bound / scalar_mag
        rescaled_shifted_scalar = shifted_scalar * scaling_term
        rescaled_scalar = rescaled_shifted_scalar + scalar_mean
    else:
        rescaled_scalar = scalar
    return rescaled_scalar, b_hit_bound

# live filter class based on this:
# https://www.samproell.io/posts/yarppg/yarppg-live-digital-filter/
# edited to allow it to work on multidim arrays

class ExpLiveFilter(object):
    def __init__(self, alpha, shape):
        self.last_value = np.zeros(shape=shape)
        self.alpha = alpha
    def process(self, raw_value):
        self.last_value = (1.0 - self.alpha) * raw_value + self.alpha * self.last_value
        return self.last_value

class LiveFilter:
    """Base class for live filters.
    """
    def process(self, x):
        # do not process NaNs
        #if np.isnan(x):
        #    return x

        return self._process(x)

    def __call__(self, x):
        return self.process(x)

    def _process(self, x):
        raise NotImplementedError("Derived class must implement _process")

class LiveLFilter(LiveFilter):
    def __init__(self, b, a, initial_array):
        """Initialize live filter based on difference equation.

        Args:
            b (array-like): numerator coefficients obtained from scipy.
            a (array-like): denominator coefficients obtained from scipy.
        """
        self.b = b
        self.a = a

        self.is_float = isinstance(initial_array, float)

        self._xs_init = deque([initial_array] * len(b), maxlen=len(b))
        self._ys_init = deque([initial_array] * (len(a) - 1), maxlen=len(a)-1)

        self._xs = deque([initial_array] * len(b), maxlen=len(b))
        self._ys = deque([initial_array] * (len(a) - 1), maxlen=len(a)-1)

    def getLastFileredValue(self):
        if self.is_float:
            return self._ys[0]
        else:
            return self._ys[0].copy()

    def _process(self, x):
        """Filter incoming data with standard difference equations.
        """
        self._xs.appendleft(x)
        try:
            y = np.tensordot(self.b, self._xs, (0, 0)) - np.tensordot(self.a[1:], self._ys, (0, 0))
        except:
            print(f"self.b: {self.b}, self._xs: {self._xs}, self.a[1:]: {self.a[1:]}, self._ys: {self._ys}")
            raise NotImplementedError("fixing it")
        y = y / self.a[0]
        self._ys.appendleft(y)

        return y

class BoundAndFilter(object):
    def __init__(self, filter_obj, box_bound = None, mag_bound = None, diff_bound = None, apply_filter = False):
        self.filter = filter_obj
        self.box_bound = box_bound
        self.mag_bound = mag_bound
        self.diff_bound = diff_bound
        self.apply_filter = apply_filter

    def process(self, new_value):
        if self.apply_filter:
            new_value =  self.filter.process(new_value)
        if self.filter.is_float:
            if self.mag_bound is not None:
                new_value, _ = bound_scalar_mag(new_value, self.mag_bound)
            if self.diff_bound is not None:
                new_value, _ = bound_scalar_mag(new_value, self.diff_bound, scalar_mean = self.filter.getLastFileredValue())
            #return self.filter.process(new_value)
        else:
            if self.box_bound is not None:
                new_value, _ = bound_vector_box(new_value, self.box_bound)
            if self.mag_bound is not None:
                new_value, _ = bound_vector_mag(new_value, self.mag_bound)
            if self.diff_bound is not None:
                new_value, _ = bound_vector_mag(new_value, self.diff_bound, vec_mean = self.filter.getLastFileredValue())
        return new_value
            
def computeOsculatingCircle(vel, acc, vel_min = 0.01):
        vel_mag = np.linalg.norm(vel)
        if vel_min < np.linalg.norm(vel):
            vel_meas_stack = np.stack([vel, acc], axis=1)
            rot_intent = np.linalg.det(vel_meas_stack)
            vel_mag_cubed = vel_mag ** (3.0)
            signed_curvature = rot_intent / vel_mag_cubed
            normal_dir = np.array([-vel[1], vel[0]]) / vel_mag
            circ_center = (1.0 / signed_curvature) * normal_dir
            circ_radius = np.abs(1.0 / signed_curvature)
        else:
            circ_center = np.array([0.0, 0.0])
            circ_radius = 0.0
        return circ_center, circ_radius

### CURVATURE HELPERS
def computeOsculatingCircleSplineFromCenterRadius(pos, circ_center, circ_radius, point_res = 100, vel_min = 0.01):
    circ_x = [pos[0] + circ_center[0] + circ_radius * np.cos((i / point_res) * 2.0 * 3.14) for i in range(point_res + 1)]
    circ_y = [pos[1] + circ_center[1] + circ_radius * np.sin((i / point_res) * 2.0 * 3.14) for i in range(point_res + 1)]
    return circ_x, circ_y

def computeOsculatingCircleSpline(pos, vel, acc, point_res = 100, vel_min = 0.01):
    circ_center, circ_radius = computeOsculatingCircle(vel, acc, vel_min=vel_min)
    circ_x = [pos[0] + circ_center[0] + circ_radius * np.cos((i / point_res) * 2.0 * 3.14) for i in range(point_res + 1)]
    circ_y = [pos[1] + circ_center[1] + circ_radius * np.sin((i / point_res) * 2.0 * 3.14) for i in range(point_res + 1)]
    return circ_x, circ_y

def rotmat_2d(angle):
    rotmat = np.array([[np.cos(angle), -1.0 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotmat
        
### functional quantiles and conformal prediction stuff

### NO JIT
def compute_quantiles(score_array, alphas):
    q = np.quantile(score_array, 1.0 - alphas)
    return q

def compute_miscoverage_array(score_array, quantiles_array):
    q_flat = quantiles_array.flatten()
    miscoverage_counts = np.zeros(shape=(score_array.shape[0], q_flat.shape[0]))
    for q_id in range(len(q_flat)):
        miscoverage_counts[:, q_id] = (q_flat[q_id] < score_array[:]).astype(np.float64)
    return miscoverage_counts.reshape([miscoverage_counts.shape[0]] + list(quantiles_array.shape))

# ASSUMES quantiles is flat!
def compute_miscoverage(score_array, quantiles):
    miscoverage_counts = np.zeros(shape=(score_array.shape[0], quantiles.shape[0]))
    for q_idx in range(quantiles.shape[0]):
        miscoverage_counts[:, q_idx] = (quantiles[q_idx] < score_array[:]).astype(np.float64)
    return miscoverage_counts

def compute_average_miscoverage(score_array, quantiles_array):
    miscoverage_counts = compute_miscoverage(score_array, quantiles_array)
    return np.mean(miscoverage_counts, axis=0) # returns (len(alphas)) or (len(gammas), len(alphas))

def update_alphas_from_miscoverage(miscoverage_array, current_alphas, target_alphas, step_sizes):
    new_alphas = current_alphas + step_sizes * (target_alphas - miscoverage_array)
    #new_alphas = current_alphas + step_sizes * (target_alphas - miscoverage_array * np.ones(shape=(target_alphas.shape[0])))
    return new_alphas

# assumes scores is a single numpy array, the other terms should not have a time dim!
def update_alphas_and_quantiles(scores, current_alphas, current_quantiles, target_alphas, step_sizes, avg_window_size):

    scores_len = scores.shape[0] - 1

    miscov = compute_average_miscoverage(scores[scores_len - avg_window_size: scores_len], current_quantiles)
    current_alphas = update_alphas_from_miscoverage(miscov, current_alphas, target_alphas, step_sizes)
    current_alphas = np.clip(current_alphas, 0.00001, 0.99999)
    current_quantiles = compute_quantiles(scores, current_alphas)

    output = List()
    output.append(current_alphas)
    output.append(current_quantiles)
    return output #List(current_alphas, current_quantiles)

# assumes scores is a typed list (List)
def update_alphas_and_quantiles_list(scores, current_alphas, current_quantiles, target_alphas, step_sizes, avg_window_size):
    new_alphas = np.zeros_like(current_alphas)
    new_quantiles = np.zeros_like(current_quantiles)
    
    for t_idx in numba.prange(len(scores)):
        alphas_quantile_list = update_alphas_and_quantiles(scores[t_idx], current_alphas[:, t_idx], current_quantiles[:, t_idx], target_alphas, step_sizes[:, t_idx], avg_window_size)
        new_alphas[:, t_idx] = alphas_quantile_list[0]
        new_quantiles[:, t_idx] = alphas_quantile_list[1]
    output = List()
    output.append(new_alphas)
    output.append(new_quantiles)
    return output #List(current_alphas, current_quantiles)

cache_option = True
### NJIT
@numba.njit(cache = cache_option)
def compute_quantiles_njit(score_array, alphas):
    q = np.quantile(score_array, 1.0 - alphas)
    return q

@numba.njit(cache = cache_option)
def compute_miscoverage_array_njit(score_array, quantiles_array):
    q_flat = quantiles_array.flatten()
    miscoverage_counts = np.zeros(shape=(score_array.shape[0], q_flat.shape[0]))
    for q_id in range(len(q_flat)):
        miscoverage_counts[:, q_id] = (q_flat[q_id] < score_array[:]).astype(np.float64)
    return miscoverage_counts.reshape([miscoverage_counts.shape[0]] + list(quantiles_array.shape))

# ASSUMES quantiles is flat!
@numba.njit(cache = cache_option) #(numba.float64[:,:](numba.float64[:], numba.float64[:]))
def compute_miscoverage_njit(score_array, quantiles):
    miscoverage_counts = np.zeros(shape=(score_array.shape[0], quantiles.shape[0]))
    for q_idx in range(quantiles.shape[0]):
        miscoverage_counts[:, q_idx] = (quantiles[q_idx] < score_array[:]).astype(np.float64)
    return miscoverage_counts

@numba.njit(cache = cache_option) #(numba.float64(numba.float64[:], numba.float64[:]))
def compute_average_miscoverage_njit(score_array, quantiles_array):
    miscoverage_counts = compute_miscoverage_njit(score_array, quantiles_array)
    return float(np.mean(miscoverage_counts)) # returns (len(alphas)) or (len(gammas), len(alphas))

@numba.njit(cache = cache_option)
def update_alphas_from_miscoverage_njit(miscoverage_array, current_alphas, target_alphas, step_sizes):
    new_alphas = current_alphas + step_sizes * (target_alphas - miscoverage_array)
    #new_alphas = current_alphas + step_sizes * (target_alphas - miscoverage_array * np.ones(shape=(target_alphas.shape[0])))
    return new_alphas

# assumes scores is a single numpy array, the other terms should not have a time dim!
@numba.njit(cache = cache_option)
def update_alphas_and_quantiles_njit(scores, current_alphas, current_quantiles, target_alphas, step_sizes, avg_window_size):

    scores_len = scores.shape[0] - 1

    miscov = compute_average_miscoverage_njit(scores[scores_len - avg_window_size: scores_len], current_quantiles)
    current_alphas = update_alphas_from_miscoverage_njit(miscov, current_alphas, target_alphas, step_sizes)
    current_alphas = np.clip(current_alphas, 0.00001, 0.99999)
    current_quantiles = compute_quantiles_njit(scores, current_alphas)

    output = List()
    output.append(current_alphas)
    output.append(current_quantiles)
    return output #List(current_alphas, current_quantiles)

# assumes scores is a typed list (List)
@numba.njit(cache = cache_option)
def update_alphas_and_quantiles_list_njit(scores, current_alphas, current_quantiles, target_alphas, step_sizes, avg_window_size):
    new_alphas = np.zeros_like(current_alphas)
    new_quantiles = np.zeros_like(current_quantiles)
    
    for t_idx in numba.prange(len(scores)):
        alphas_quantile_list = update_alphas_and_quantiles_njit(scores[t_idx], current_alphas[:, t_idx], current_quantiles[:, t_idx], target_alphas, step_sizes[:, t_idx], avg_window_size)
        new_alphas[:, t_idx] = alphas_quantile_list[0]
        new_quantiles[:, t_idx] = alphas_quantile_list[1]
    output = List()
    output.append(new_alphas)
    output.append(new_quantiles)
    return output 

@numba.njit(parallel=True)
def update_alphas_and_quantiles_list_njit_par(scores, current_alphas, current_quantiles, target_alphas, step_sizes, avg_window_size):
    new_alphas = np.zeros_like(current_alphas)
    new_quantiles = np.zeros_like(current_quantiles)
    
    for t_idx in numba.prange(len(scores)):
        alphas_quantile_list = update_alphas_and_quantiles_njit(scores[t_idx], current_alphas[:, t_idx], current_quantiles[:, t_idx], target_alphas, step_sizes[:, t_idx], avg_window_size)
        new_alphas[:, t_idx] = alphas_quantile_list[0]
        new_quantiles[:, t_idx] = alphas_quantile_list[1]
    output = List()
    output.append(new_alphas)
    output.append(new_quantiles)
    return output 

# assumes a single np.array of scores, no L dim! Only use one step_size and one target_alpha for this function
@numba.njit(cache = cache_option)
def compute_local_coverage_njit(trial_scores, target_alpha, step_size, avg_window_size,
                   input_window_length = 1200, local_cov_half_window = 600):
    trial_score_length = trial_scores.shape[0] # N
    trial_last_valid_index = trial_score_length - local_cov_half_window

    local_coverage_length = trial_last_valid_index - input_window_length - 1 #trial_score_length - input_window_length - local_cov_half_window
    local_coverage = np.ones(shape=(local_coverage_length))

    target_alpha_a = np.array([target_alpha])
    current_alpha = target_alpha_a.copy()
    current_quantile = compute_quantiles_njit(trial_scores[0:input_window_length], current_alpha) #np.array([compute_quantiles_njit(trial_scores[0:input_window_length], current_alpha)])

    cov_idx = 0
    for n_idx in range(input_window_length, trial_last_valid_index-1):
        scores_window = trial_scores[n_idx-local_cov_half_window:n_idx+local_cov_half_window]
        local_coverage[n_idx - input_window_length] = 1.0 - compute_average_miscoverage_njit(scores_window, current_quantile)
        alphas_quantile_list = update_alphas_and_quantiles_njit(trial_scores[n_idx-input_window_length:n_idx], current_alpha, current_quantile, target_alpha_a, step_size, avg_window_size)
        current_alpha = alphas_quantile_list[0]
        current_quantile = alphas_quantile_list[1]
    return local_coverage

# assumes a single np.array of scores, no L dim!
@numba.njit(cache = cache_option)
def compute_mean_absolute_coverage_error_njit(trial_scores, target_alpha, step_size, avg_window_size,
                   input_window_length = 1200, local_cov_half_window = 600):
    local_coverage = compute_local_coverage_njit(trial_scores, target_alpha, step_size, avg_window_size,
                   input_window_length = input_window_length, local_cov_half_window = local_cov_half_window)
    mean_local_coverage = np.mean(local_coverage)
    target_coverage = 1.0 - target_alpha
    abs_coverage_gap = np.abs(target_coverage - mean_local_coverage)
    return abs_coverage_gap

def get_targets_pathlist(scale_mult = 0.01, targets_per_trial = 6):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(current_directory, '..', 'include', 'targets.csv')
    data = np.loadtxt(csv_path, delimiter=",")
    
    # Waypoints
    targetx_data = data[:, 0] * scale_mult
    targety_data = data[:, 1] * scale_mult

    pathlist = []

    for i in range(0, targetx_data.size, targets_per_trial):
        path_num = (i // targets_per_trial) + 1
        pathlist.append(np.column_stack((targetx_data[i:i + targets_per_trial], targety_data[i:i + targets_per_trial])).tolist())
    return pathlist

@numba.njit(cache = cache_option)
def compute_exp_confidence(x, t_const):
    return np.exp(- x * t_const)
    #return 1.0 - np.exp(- x * t_const)

@numba.njit(cache = cache_option)
def interp(x, P_min, P_max):
    P = P_min + (P_max - P_min) * x
    return P

@numba.njit(cache = cache_option)
def exp_filt(new, old, alpha):
    filt_new = alpha * new + (1.0 - alpha) * old
    return filt_new