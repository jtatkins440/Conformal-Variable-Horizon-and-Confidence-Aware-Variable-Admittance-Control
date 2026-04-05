#!/usr/bin/env python3

import rospy
import time
import numpy as np
from std_msgs.msg import Float64MultiArray
from conf_exps.msg import AdmitStateStamped
from conf_exps.srv import * # GetQuantiles mostly
import math
import os, fnmatch
from utils import *
from admit_lib import AdaptiveConformalPredictionHelper
import h5py

class QuantileServer:
	def __init__(self):
		return
	
	def getScores(self, trial_folder_path):
		trial_files = find('*.h5', trial_folder_path)

		print(f"Quantile Server: Found {len(trial_files)} trials in directory {trial_folder_path}...")

		if len(trial_files) == 0:
			print(f"No trials found! Exiting!")
			return
		
		scores_list = []
		for trial_file in trial_files:
			f = h5py.File(trial_file, 'r')

			fs = f['TrialData']

			trial_scores = fs['scores'] # h5 dataset object
			scores_list.append(np.array(trial_scores))
		scores_array = np.concatenate(scores_list, axis=0)
		try:
			scores = scores_array[:,0,:]
		except:
			scores = scores_array
		return scores
	
	def getUserIntent(self, trial_folder_path):
		trial_files = find('*.h5', trial_folder_path)
		#print(f"trial_files: {trial_files}")
		print(f"Quantile Server: Found {len(trial_files)} trials in directory {trial_folder_path}...")

		if len(trial_files) == 0:
			print(f"No trials found! Exiting!")
			return
		
		ui_list = []
		for trial_file in trial_files:
			f = h5py.File(trial_file, 'r')
			#print(f.keys())
			fs = f['TrialData']
			#print(fs.keys())
			trial_ui = fs['user_intent'] # h5 dataset object
			ui_list.append(np.array(trial_ui))
		user_intent_array = np.concatenate(ui_list, axis=0)
		return user_intent_array

	def computeQuantiles(self, scores_array, alphas):
		#quantile_array = compute_quantiles(scores_array, alphas) # don't need to use the optimized version for speed reasons. Need it for consistent ouptut sizes.
		quantile_array = compute_quantiles_njit(scores_array, alphas)
		return quantile_array

	def stepSizeOptBisection(self, scores, target_alpha, input_window_length = 1200, avg_window_size = 300, local_cov_half_window = 600, 
					gamma_min = 0.0, gamma_max = 0.01, max_iter_num = 10, percent_gamma_change_tol = 0.001, verbose=True):
		gamma_boundary = np.array([gamma_min, gamma_max])
		gamma_boundary_performance = np.zeros_like(gamma_boundary)
		for g_idx in range(len(gamma_boundary)):
				gamma_boundary_performance[g_idx] = compute_mean_absolute_coverage_error_njit(scores, target_alpha, gamma_boundary[g_idx], avg_window_size,
							input_window_length = input_window_length, local_cov_half_window = local_cov_half_window)
		
		last_gamma_mid = gamma_boundary[0].copy()
		gamma_mid = np.mean(gamma_boundary) #(gamma_max - gamma_min) / 2.0
		last_gamma_change_mag = np.linalg.norm(gamma_mid - last_gamma_mid)
		for i in range(max_iter_num):
			
			gamma_mid_performance = compute_mean_absolute_coverage_error_njit(scores, target_alpha, gamma_mid, avg_window_size,
							input_window_length = input_window_length, local_cov_half_window = local_cov_half_window)
			if verbose: print(f"bisection iter {i}: center gamma: {gamma_mid}, gamma boundary: {gamma_boundary}, boundary_perf: {gamma_boundary_performance}, mid_perf: {gamma_mid_performance}")
			if gamma_boundary_performance[0] < gamma_mid_performance: # if lb is better than the mid, set the mid to be the upper bound
				gamma_boundary[1] = gamma_mid
				gamma_boundary_performance[1] = gamma_mid_performance
			else: # if ub is better than the mid, set the mid to be the lower bound
				gamma_boundary[0] = gamma_mid
				gamma_boundary_performance[0] = gamma_mid_performance
			last_gamma_mid = gamma_mid.copy()
			gamma_mid = np.mean(gamma_boundary)

			# early stopping, break if percent change in gammas is less than a small threshold
			gamma_change_mag = np.linalg.norm(gamma_mid - last_gamma_mid)
			percent_change = 1.0 - np.abs(gamma_change_mag - last_gamma_change_mag) / last_gamma_change_mag
			print(f"bisection iter {i}: percent change in gamma: {percent_change}")
			if percent_change < percent_gamma_change_tol:
				print(f"Early stopping criteria reached at iteration {i}!")
				break
			last_gamma_change_mag = percent_change.copy()
		return gamma_mid, gamma_mid_performance
	
	def stepSizeOptBisectionArray(self, scores_array, target_alpha, input_window_length = 1200, avg_window_size = 300, local_cov_half_window = 600, 
					gamma_min = 0.0, gamma_max = 0.01, max_iter_num = 20):
		prediction_num = scores_array.shape[1]
		opt_step_sizes = [0.0] # first prediction is identity, score is always zero
		opt_perfs = [0.0]
		for t_idx in range(1, prediction_num):
			print(f"Bisection: On index {t_idx}...")
			opt_gamma, opt_perf = self.stepSizeOptBisection(scores_array[:,t_idx], target_alpha, input_window_length = input_window_length, 
					avg_window_size = avg_window_size, local_cov_half_window = local_cov_half_window, 
					gamma_min = gamma_min, gamma_max = gamma_max, max_iter_num = max_iter_num)
			opt_step_sizes.append(opt_gamma)
			opt_perfs.append(opt_perf)
		return opt_step_sizes, opt_perfs

	def stepSizeOptPSO(self, scores, target_alpha, input_window_length = 1200, avg_window_size = 300, local_cov_half_window = 600, 
					gamma_min = 0.0, gamma_max = 0.01, particle_num = 30, max_iter_num = 100, lr = 0.1, verbose=False):
		step_sizes = np.linspace(gamma_min, gamma_max, particle_num)
		performace = np.zeros(shape=(particle_num))
		for i in range(max_iter_num):
			for g_idx in range(len(step_sizes)):
				performace[g_idx] = compute_mean_absolute_coverage_error_njit(scores, target_alpha, step_sizes[g_idx], avg_window_size,
						input_window_length = input_window_length, local_cov_half_window = local_cov_half_window)
			#print(f"performace: {performace}")
			#print(f"step_sizes: {step_sizes}")
			best_particle_idx = np.argmin(performace)
			particle_dist = (step_sizes[best_particle_idx] - step_sizes) # could impliment early stopping with this if needed

			if verbose: print(f"PSO iteration {i}: min error {performace[best_particle_idx]} for particle {best_particle_idx} with step size {step_sizes[best_particle_idx]}")
			step_sizes = step_sizes + lr * particle_dist
		return step_sizes[best_particle_idx], performace[best_particle_idx]
	
	def stepSizeOptPSOArray(self, scores_array, target_alpha, input_window_length = 1200, avg_window_size = 300, local_cov_half_window = 600, 
					gamma_min = 0.0, gamma_max = 0.01, particle_num = 30, max_iter_num = 100, lr = 0.1):
		prediction_num = scores_array.shape[1]
		opt_step_sizes = [0.0] # first prediction is identity, score is always zero
		opt_perfs = [0.0]
		for t_idx in range(1, prediction_num):
			print(f"PSO: On index {t_idx}...")
			opt_gamma, opt_perf = self.stepSizeOptPSO(scores_array[:,t_idx], target_alpha, input_window_length = input_window_length, 
					avg_window_size = avg_window_size, local_cov_half_window = local_cov_half_window, 
					gamma_min = gamma_min, gamma_max = gamma_max, particle_num = particle_num, max_iter_num = max_iter_num, lr = lr)
			opt_step_sizes.append(opt_gamma)
			opt_perfs.append(opt_perf)
		return opt_step_sizes, opt_perfs
	
	
class ROSQuantileServer:
	def __init__(self, nh):
		self.nh = nh
		self.get_score_quantiles_service = rospy.Service('/quantile_server/get_score_quantiles', GetQuantiles, self.computeScoreQuantiles)
		self.get_optimal_step_sizes_service = rospy.Service('/quantile_server/get_optimal_step_sizes', GetStepSizes, self.computeOptimalStepSizes)
		self.get_optimal_step_sizes_service = rospy.Service('/quantile_server/get_user_intent_bounds', GetUserIntentBounds, self.computeUserIntentBounds)
		self.qs = QuantileServer() # helper no-ROS class for handling the functionality
		return


	def computeScoreQuantiles(self, req):
		res = GetQuantilesResponse()

		# find path to the trials folder
		current_directory = os.path.dirname(os.path.abspath(__file__))
		data_directory = os.path.join(current_directory, 'DATA')
		experiment_directory = os.path.join(data_directory, req.experiment_name) # save this one and on
		subject_directory = os.path.join(experiment_directory, req.subject_name)
		method_directory = os.path.join(subject_directory, req.trial_type)

		scores = self.qs.getScores(method_directory)
		quantiles = self.qs.computeQuantiles(scores, req.alphas)

		res.success = True
		res.message = f"Score quantiles generated from {method_directory}!"
		res.icp_quantiles = pack_array_to_multiarray_msg(quantiles)
		return res
	
	def computeOptimalStepSizes(self, req):
		res = GetStepSizesResponse()

		print(f"Quantile Server: Recieved request with options of:\n{req}")


		calibration_dir = rospy.get_param("/calibration_directory")
		scores = self.qs.getScores(calibration_dir)

		step_sizes, cov_gap = self.qs.stepSizeOptBisectionArray(scores, req.alpha, input_window_length=req.input_window_length,
														avg_window_size = req.avg_window_size, local_cov_half_window=req.half_coverage_window, gamma_max=req.max_gamma)

		res.success = True
		res.message = f"(Sub-)Optimal step sizes of {step_sizes} with coverage gaps of {cov_gap} generated from {calibration_dir}!"
		res.step_sizes = step_sizes
		return res

	def computeUserIntentBounds(self, req):
		res = GetUserIntentBoundsResponse()

		print(f"Quantile Server: Recieved user intent bounds request with options of:\n{req}")

		calibration_dir = rospy.get_param("/calibration_directory")
		user_intent = self.qs.getUserIntent(calibration_dir)
		
		res.user_intent_min = np.min(user_intent)
		res.user_intent_max = np.max(user_intent)

		res.success = True
		res.message = f"Got user intent bounds of [{res.user_intent_min}, {res.user_intent_max}] from {calibration_dir}!"
		return res
	
	def main(self, quant_rate = 100.0):
		rate = rospy.Rate(quant_rate)
		while True: #not rospy.is_shutdown:
			rate.sleep()

import time
if __name__ == "__main__":
	nh = rospy.init_node('quantile_server', anonymous=False)

	quant_server = ROSQuantileServer(nh)
	quant_rate = 20.0 # rate that it's checking for needing to run the service
	quant_server.main(quant_rate)

	print(f"Quantile server exiting!")
