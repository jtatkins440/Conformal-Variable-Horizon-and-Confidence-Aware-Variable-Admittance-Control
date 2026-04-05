import numpy as np
import numba
from numba.typed import List # experimental typed list object

from collections import deque
import pickle
from statsmodels.tsa.api import VAR

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

use_jit = False
cache_option = True #False #True
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
    return output #List(current_alphas, current_quantiles)

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
    return output #List(current_alphas, current_quantiles)

class AdaptiveConformalPredictionHelper:
    def __init__(self, target_alpha = 0.1, prediction_length = 6, step_sizes=[0.0, 0.001, 0.005], window_length=1, init_quantiles = None, use_jit = True, use_parallel = True):
        
        self.prediction_length = prediction_length # maybe don't need this as class member
        self.step_sizes_list = step_sizes
        self.target_alphas_list = [target_alpha for g in range(len(step_sizes))]
        self.window_length = window_length
        self.use_jit = use_jit
        self.use_parallel = use_parallel

        self.target_alphas_array = np.squeeze(np.array(self.target_alphas_list)) #np.ones(shape=(len(step_sizes, )))#np.array(self.target_alphas_list)
        self.current_alphas_array = np.zeros(shape=(len(step_sizes), self.prediction_length))
        self.step_sizes_array = np.zeros(shape=(len(step_sizes), self.prediction_length))
        self.quantiles_array = np.zeros(shape=(len(step_sizes), self.prediction_length))

        for t_idx in range(0, self.prediction_length):
            self.current_alphas_array[:, t_idx] = self.target_alphas_array #self.target_alphas_array[:,0]
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
        #self.state_buffer = np.zeros(shape=(state_chn_num, state_buffer_size))
        #self.state_buffer_counter = 0 # counts how many elements have been added to the buffer

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
        return self.isScoreBufferFull()

    def isScoreBufferFull(self):
        return self.score_buffer_size <= self.score_buffer_counter

    def getLastScore(self):
        return self.score_buffer[-1,:]

    def updatePredictionBuffer(self, new_prediction):
        self.prediction_buffer.append(new_prediction)
        return 
    
def vec_loss(target, output):
    return np.linalg.norm(target - output, axis=0, keepdims=True)

def get_mint_input_from_state_buffer(start_index, input_index_seq, memory_helper, input_chn = [0, 1, 2, 3, 4, 5]):
    input_seq = memory_helper.state_buffer[:, start_index + np.array(input_index_seq, dtype=np.int32)]
    return input_seq[input_chn, :]

def get_mint_target_from_state_buffer(start_index, output_index_seq, memory_helper, output_chn = [0, 1]):
    target_seq = memory_helper.state_buffer[:, start_index + np.array(output_index_seq, dtype=np.int32)]
    return target_seq[output_chn, :]

def get_mint_input_target_from_state_buffer(start_index, input_index_seq, output_index_seq, memory_helper):
    return (get_mint_input_from_state_buffer(start_index, input_index_seq, memory_helper), get_mint_target_from_state_buffer(start_index, output_index_seq, memory_helper))


class PredictEval:
    def __init__(self, window_steps, horizon_steps, down_sample_interval, dt, aci_alphas, aci_step_sizes, aci_window_length, safe_pos_error_level, prediction_model_path, state_history_dim = 4, predict_input_spacing = None, predict_output_spacing = None):

        # from Dr.Hwang's code, move these to inputs!
        self.window_steps = window_steps #240 # 960ms at 4ms/step
        self.horizon_steps = horizon_steps #120 # 480ms at 4ms/step
        self.down_sample_interval = down_sample_interval #12
        dt = dt #0.004 # s/step
        
        alphas = aci_alphas #[0.1]
        aci_step_sizes = aci_step_sizes #[0.0, 0.001] #[0.00001, 0.0001] #[0.001, 0.005, 0.01]
        aci_window_length = aci_window_length #100 #100
        self.safe_pos_error_level = safe_pos_error_level #10.0 # degrees, TODO: CHECK TO MAKE SURE CONTROLLER IS USING DEGREES AND NOT RADIANS
        predict_model_path = prediction_model_path #'/home/antigrav_ws/src/rehab_antigrav/predict_eval/scripts/Var_lag21.pkl'

        self.window_points = (self.window_steps // self.down_sample_interval) + 1 #20 plus 1 for inclusive!
        self.horizon_points = (self.horizon_steps // self.down_sample_interval) # 10 <-- len([t+1:t+H]), doesn't include t!
        
        self.state_history_len = max(self.window_steps, self.horizon_steps)
        self.state_history = np.zeros([self.state_history_len, state_history_dim])
        self.prediction_history = np.zeros([self.state_history_len, state_history_dim, self.horizon_points])

        self.predict_input_stencil = np.linspace(self.state_history_len - self.window_steps, self.state_history_len, self.window_points, dtype=np.int32)
        #self.predict_output_stencil = 
        self.eval_stencil = np.linspace(self.state_history_len - self.horizon_steps + self.down_sample_interval, self.state_history_len, self.horizon_points, dtype=np.int32) - 1
        self.predict_times = np.linspace(self.down_sample_interval, self.horizon_steps, self.horizon_points) * dt
        #with open('Var_lag21.pkl', 'rb') as f:
        #    self.prediction_model = pickle.load(f)

        #with open('/home/linny/antigrav_ws/src/rehab_antigrav/predict_eval/scripts/Var_lag21.pkl', 'rb') as f:
        #    self.prediction_model = pickle.load(f)

        with open(predict_model_path, 'rb') as f:
            self.prediction_model = pickle.load(f)

        
        self.mem_buffer = MemoryHelper(state_chn_num = state_history_dim, min_state_buffer_size = self.state_history_len, state_buffer_size = self.state_history_len + 50, 
                        score_buffer_size = self.state_history_len, score_len = self.horizon_points, prediction_buffer_size = self.horizon_steps)

        self.aci_helper = AdaptiveConformalPredictionHelper(target_alpha = alphas, prediction_length = self.horizon_points, 
            step_sizes=aci_step_sizes, window_length=aci_window_length, init_quantiles = None, use_jit=use_jit, use_parallel=False)
        
        self.aci_channels = [0, 1] # channels for computing aci info, don't use whole [x, dx] vector! Different units!
        
    def predict(self, input):
        output_t = self.prediction_model.forecast(input.T, steps = self.horizon_points) # assumes a statsmodels VARResults object
        #print("output_t",output_t.T)
        return output_t.T

    def evaluate(self, predicted, observed):
        pass

    def predictEvaluate(self, new_state):
        b_output_ready = False
        b_is_score_buffer_full = False
        b_quantiles_ready = False
        
        b_is_mint_input_ready, b_is_state_buffer_full = self.mem_buffer.updateStateBuffer(new_state)

        if b_is_mint_input_ready:
            mdl_input = self.mem_buffer.state_buffer[:, self.predict_input_stencil]
            output_traj = self.predict(mdl_input)
            self.mem_buffer.updatePredictionBuffer(output_traj)

        #start_index = mint_input_history_size - 1
        #old_mint_target = self.mem_buffer.state_buffer[:, start_index + np.array(output_index_seq, dtype=np.int32)]

        
        if b_is_state_buffer_full:
            old_mint_target = self.mem_buffer.state_buffer[:, self.eval_stencil]
            old_mint_output = self.mem_buffer.prediction_buffer[0]

            #scores = vec_loss(old_mint_target, old_mint_output)
            scores = vec_loss(old_mint_target[self.aci_channels,:], old_mint_output[self.aci_channels,:])

            b_is_score_buffer_full = self.mem_buffer.updateScoreBuffer(scores)

        
        if b_is_score_buffer_full:
            #print(self.mem_buffer.score_buffer)
            _, _ = self.aci_helper.updateQuantiles(self.mem_buffer.score_buffer) #  updates internal values
            b_quantiles_ready = True

        if b_quantiles_ready:
            aci_quantiles, aci_alphas = self.aci_helper.getQuantilesAndAlphas() # gives the initial values for the quantiles even if there's not enough values to compute anything correctly.
            aci_quantiles_safe = aci_quantiles < self.safe_pos_error_level # output should be boolean array
            b_output_ready = True

        #print("b_is_state_buffer_full: " + str(b_is_state_buffer_full))
        #print("b_is_score_buffer_full: " + str(b_is_score_buffer_full))
        #print("b_quantiles_ready: " + str(b_quantiles_ready))
        #print("b_output_ready: " + str(b_output_ready))

        output_dict = dict()
        output_dict["ready"] = b_output_ready
        if b_output_ready:
            output_dict["pred_traj"] = output_traj # output is (dim_alphas, horizon_steps)
            output_dict["quantiles"] = aci_quantiles[-1, :] # output is (dim_alphas, horizon_steps)
            output_dict["is_safe"] = aci_quantiles_safe[-1, :] 
            output_dict["times"] = self.predict_times.copy()
        else:
            output_dict["pred_traj"] = None
            output_dict["quantiles"] = None
            output_dict["is_safe"] = None
            output_dict["times"] = None
        return output_dict
