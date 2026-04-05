import math
import rospy
from geometry_msgs.msg import WrenchStamped, Quaternion, PoseStamped, Pose, TwistStamped, Point
from std_srvs.srv import SetBool, SetBoolResponse

from rehab_msgs.msg import RPYState, ConformalSetRadial, ConformalSetTrajRadial, ClassProbability, ClassificationOutput


import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import PchipInterpolator as pchip

def vec3_to_skew_sym(vec):
    skew = np.array([[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0.0]])
    return skew

def rpy_to_rot_mat(rpy):
    rot = Rotation.from_euler("xyz", rpy, degrees=True)
    rotmat = np.squeeze(rot.as_matrix())
    return rotmat #np.eye(3)

class Antigrav:
    def __init__(self, nominal_mass, center_of_mass_pos, support_ratio):
        self.com_vector = np.array(center_of_mass_pos)
        self.support_ratio = support_ratio
        self.grav_force_vector = nominal_mass * np.array([0.0, 0.0, -9.81])
        self.grav_force_skew_mat = vec3_to_skew_sym(self.grav_force_vector)

    def getArmGravityAtRPY(self, rpy):
        rot_mat = rpy_to_rot_mat(rpy)
        com_vec = np.dot(rot_mat, self.com_vector)
        grav = np.dot(self.grav_force_skew_mat, com_vec)
        return grav
    
    def getGravCompTorqueAtRPY(self, rpy):
        
        arm_grav = self.getArmGravityAtRPY(rpy) * self.support_ratio
        return arm_grav
    
class AntigravROS:
    def __init__(self, nh):
        self.nh = nh

        ### init from configs
        self.dt = rospy.get_param("antigrav/dt")
        nom_arm_mass = rospy.get_param("antigrav/nominal_mass")
        com_pos = rospy.get_param("antigrav/center_of_mass")
        support_ratio = rospy.get_param("antigrav/support_ratio")

        self.dt_rate = int(1.0 / self.dt)

        ### initialize subscribers and publishers
        self.global_queue_size = 25 # 50 # was 1
        self.sub_state = rospy.Subscriber("current_state", RPYState, self.callbackCurrentState, queue_size=self.global_queue_size, tcp_nodelay=True)
        self.sub_pred_state = rospy.Subscriber("predicted_state_sets", ConformalSetTrajRadial, self.callbackPredictState, queue_size=self.global_queue_size, tcp_nodelay=True)
        self.sub_class = rospy.Subscriber("classifier_output", ClassificationOutput, self.callbackClass, queue_size=self.global_queue_size, tcp_nodelay=True)
        self.pub_wrench = rospy.Publisher("wrench_robot", WrenchStamped, queue_size=self.global_queue_size)

        ### initialize attributes
        pos_dim = 3 # rpy
        self.pos_current = np.zeros([pos_dim,])
        self.vel_current = np.zeros([pos_dim,])

        self.safe_centers = [[0.0, 0.0, 0.0] for i in range(1)]
        self.safe_radii = [0.0 for i in range(1)]
        self.safe_times = [0.0 for i in range(1)]

        self.probs = [0.0, 1.0]
        self.prob_labels = ["Dynamic", "Static"]
        self.prob_dict = {"Dynamic": {"probability": 0.0, "interp_index": -1}, "Static": {"probability": 1.0, "interp_index": 0}}


        self.use_dynamic = False #True #False

        self.antigrav = Antigrav(nom_arm_mass, com_pos, support_ratio)

        self.set_use_dynamic_compensation_service = rospy.Service("robot_policy/set_use_dynamic_compensation", SetBool, self.setUseDynamicCompensation)

    def setUseDynamicCompensation(self, req): # setBool
        res = SetBoolResponse()
        if (isinstance(req.data, bool)):
            self.use_dynamic = req.data
            res.success = True
            res.message = "Use dynamic compensation behavior set to " + str(self.use_dynamic) + "!!!"
        else:
            res.success = False
            res.message = "!!!Error!!! Invalid request data type, got " + str(req.data) + " but needed a bool."
        return res
    
    ### subscriber callbacks
    def callbackCurrentState(self, msg): # sub_state, RPYState
        self.pos_current = np.array([msg.RPY.x, msg.RPY.y, msg.RPY.z])
        self.vel_current = np.array([msg.angular.x, msg.angular.y, msg.angular.z])
        return
    
    def callbackPredictState(self, msg): # sub_pred_state, ConformalSetTrajRadial
        centers = []
        radii = []
        times = []
        safes = []
        end_safe_index = 0
        for i in range(len(msg.prediction_sets)):
            pset = msg.prediction_sets[i]
            centers.append([pset.center.x, pset.center.y, pset.center.z])
            radii.append(pset.radius)
            times.append(pset.ahead_time)
            safes.append(pset.is_safe)
            if safes[i]:
                end_safe_index = i # will increase safe index until it's false

        if (end_safe_index == 0):
            self.safe_centers = [centers[0]]
            self.safe_radii = [radii[0]]
            self.safe_times = [times[0]]
        else:
            self.safe_centers = [centers[i] for i in range(end_safe_index)]
            self.safe_radii = [radii[i] for i in range(end_safe_index)]
            self.safe_times = [times[i] for i in range(end_safe_index)]
        return
    
    def callbackClass(self, msg): # sub_class, ClassProbability
        for i in range(len(msg.classes)):
            self.probs[i] = msg.classes[i].probability
            self.prob_labels[i] = msg.classes[i].label
            self.prob_dict[msg.classes[i].label]["probability"] = msg.classes[i].probability
        return
    
    ### publisher methods
    def publishWrench(self, torques, forces):
        msg = WrenchStamped()
        msg.wrench.force.x = forces[0]
        msg.wrench.force.y = forces[1]
        msg.wrench.force.z = forces[2]

        msg.wrench.torque.x = torques[0]
        msg.wrench.torque.y = torques[1]
        msg.wrench.torque.z = torques[2]

        self.pub_wrench.publish(msg)

    ### getter functions
    def getCurrentState(self):
        return self.pos_current.copy()

    ### anticipatory compensation
    def getInterpState(self):
        # find interpolation time
        #anti_time = self.probs[0] * self.safe_times[0] + self.probs[1] * self.safe_times[-1]
        #anti_time = self.probs[1] * self.safe_times[0] + self.probs[0] * self.safe_times[-1] # classifier output has index of 0 for dynamic and 1 for static...
        if (1 < len(self.safe_centers)):
            anti_time = 0.0
            for class_state in self.prob_dict.keys():
                anti_time += self.prob_dict[class_state]["probability"] * self.safe_times[self.prob_dict[class_state]["interp_index"]]

            # interpolate the predicted states
            safe_times_array = np.array(self.safe_times)
            safe_centers_array = np.array(self.safe_centers)
            interp_spline = pchip(safe_times_array, safe_centers_array, axis=0)

            # sample at the anticipitory time
            interp_state = interp_spline(anti_time) #interp_spline[anti_time]

            # hard coding fix the roll angle
            interp_state[0] = -80.0
            #interp_state[0] = -70.0
        else:
            interp_state = np.array(self.safe_centers)
            interp_state[0] = -80.0
            #interp_state[0] = -70.0
        return interp_state


    def step(self):
        ### update main
        if self.use_dynamic:
            # print("Use dyanmic")
            state = self.getInterpState()
            #state2 = self.getCurrentState()
        else:
            state = self.getCurrentState()
            #state2 = self.getInterpState()
        
        assistance_torque = self.antigrav.getGravCompTorqueAtRPY(state)

        ### package outputs in state_dict
        state_dict = dict()
        state_dict["assistance_torque"] = assistance_torque
        state_dict["assistance_force"] = assistance_torque * 0.0

        return state_dict
    
if __name__ == '__main__':

    nh = rospy.init_node('robot_policy', anonymous=True)

    node = AntigravROS(nh)

    rate = rospy.Rate(node.dt_rate)

    while not rospy.is_shutdown():
        state_dict = node.step()
        node.publishWrench(state_dict["assistance_torque"],state_dict["assistance_force"])
        rate.sleep()
