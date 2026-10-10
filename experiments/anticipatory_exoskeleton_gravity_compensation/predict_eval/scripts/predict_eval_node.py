import math
import rospy
import numpy as np

from rehab_msgs.msg import RPYState, ConformalSetRadial, ConformalSetTrajRadial, ConformalSetElementwise, ConformalSetTrajElementwise


from predict_eval.predict_eval_lib import PredictEval

class PredictEvalROS:
    def __init__(self, nh):
        self.nh = nh

        ### init from configs
        self.dt = rospy.get_param("pred_eval/dt")
        self.dt_rate = int(1.0 / self.dt)
        self.conformal_set_type = rospy.get_param("pred_eval/conformal_set_type") #"radial" # or elementwise

        window_steps = rospy.get_param("pred_eval/window_steps") 
        horizon_steps = rospy.get_param("pred_eval/horizon_steps") 
        down_sample_interval = rospy.get_param("pred_eval/down_sample_interval") #12
        
        aci_alphas = rospy.get_param("pred_eval/aci_alphas") 
        aci_step_sizes = rospy.get_param("pred_eval/aci_step_sizes") 
        aci_window_length = rospy.get_param("pred_eval/aci_window_length") 
        safe_pos_error_level = rospy.get_param("pred_eval/safe_pos_error_level") 
        predict_model_path = rospy.get_param("pred_eval/predict_model_path") #'/home/antigrav_ws/src/rehab_antigrav/predict_eval/scripts/Var_lag21.pkl'


        ### initialize subscribers and publishers
        self.global_queue_size = 25 # 50 # was 1
        self.sub_state = rospy.Subscriber("current_state", RPYState, self.callbackCurrentState, queue_size=self.global_queue_size, tcp_nodelay=True)

        if self.conformal_set_type == "radial":
            self.pub_conf_traj = rospy.Publisher("conf_set_traj", ConformalSetTrajRadial, queue_size=self.global_queue_size)
        else:
            self.pub_conf_traj = rospy.Publisher("conf_set_traj", ConformalSetTrajElementwise, queue_size=self.global_queue_size)


        ### initialize attributes
        pos_dim = 2
        self.pos_current = np.zeros([pos_dim,])
        self.vel_current = np.zeros([pos_dim,])
        self.state_current = np.zeros([int(2 * pos_dim),])

        

        self.predict_eval = PredictEval(window_steps, horizon_steps, down_sample_interval, self.dt, aci_alphas, aci_step_sizes, aci_window_length, safe_pos_error_level, predict_model_path, state_history_dim=int(2 * pos_dim))
        

    ### subscriber callbacks
    def callbackCurrentState(self, msg):
        self.state_current = np.array([msg.RPY.y, msg.RPY.z, msg.angular.y, msg.angular.z])
        return
    
    ### publisher methods
    def publishConformalSetTrajectory(self, centers, radii, is_safe, times):
        msg = ConformalSetTrajRadial()
        cset_list = []
        times_list = []
        for i in range(centers.shape[1]): # assumes large np array
            cset = ConformalSetRadial()
            cset.center.y = centers[0, i]
            cset.center.z = centers[1, i]
            cset.radius = radii[i]
            cset.is_safe = is_safe[i]
            cset.ahead_time = times[i]
            times_list.append(times[i])
            cset_list.append(cset)
        msg.prediction_sets = cset_list

        self.pub_conf_traj.publish(msg)

    def publishConformalSetTrajectoryList(self, centers, radii, times):
        msg = ConformalSetTrajRadial()
        cset_list = []
        times_list = []
        for i in range(len(centers)): # assume list of np arrays
            cset = ConformalSetRadial()
            cset.center.x = centers[i][0]
            cset.center.y = centers[i][1]
            cset.radius = radii[i]
            times_list.append(times[i])
            cset_list.append(cset)
        msg.prediction_sets = cset_list

        self.pub_conf_traj.publish(msg)

    ### getter functions
    def getCurrentState(self):
        return self.state_current.copy()

    
    def step(self):
        ### get inputs
        state_curr = self.getCurrentState()

        ### update main
        predict_dict = self.predict_eval.predictEvaluate(state_curr)

        ### package outputs in state_dict
        state_dict = dict()
        if predict_dict["ready"]:
            state_dict['centers'] = predict_dict["pred_traj"] 
            state_dict['radii'] = predict_dict["quantiles"]
            state_dict['is_safe'] = predict_dict["is_safe"]
            state_dict['times'] = predict_dict["times"]
        else:
            state_dict['centers'] = np.zeros([4, 10]) 
            state_dict['radii'] = np.zeros([10,])
            state_dict['is_safe'] = np.array([False for i in range(10)]) 
            state_dict['times'] = np.linspace(0.0, 0.480, 10) 
        
        self.publishConformalSetTrajectory(state_dict['centers'], state_dict['radii'], state_dict['is_safe'], state_dict['times'])
        return state_dict # returns the state dict for possible external logging and backwards compatibility, not strictly needed.
    
if __name__ == '__main__':

    nh = rospy.init_node('pred_eval', anonymous=True)

    pe = PredictEvalROS(nh)

    rate = rospy.Rate(pe.dt_rate)

    while not rospy.is_shutdown():
        state_dict = pe.step()
        
        rate.sleep()
