#!/usr/bin/python3

import math
import rospy
#from geometry_msgs.msg import WrenchStamped, Quaternion, PoseStamped, Pose, TwistStamped, Point

#from admit_lib import *
from rehab_msgs.msg import RPYState, ConformalSetRadial, ConformalSetTrajRadial, ClassProbability, ClassificationInfo, ClassificationOutput
#from rehab_msgs.scripts.utils import *

from collections import deque

import numpy as np
import numba

# Model wrapper import
from classification.TaskClassModel import *
import time
# This is the working branch for the classifier node
class ClassifierROS:
    def __init__(self, nh):
        self.nh = nh

        ### init from configs
        self.dt = rospy.get_param("class/dt")
        self.dt_rate = int(1.0 / self.dt)
        #self.conformal_set_type = rospy.get_param("pred_eval/conformal_set_type") #"radial" # or elementwise

        ### initialize subscribers and publishers
        self.global_queue_size = 25 # 50 # was 1
        self.sub_state = rospy.Subscriber("current_state", RPYState, self.callbackCurrentState, queue_size=self.global_queue_size, tcp_nodelay=True)
        #self.sub_pred_state = rospy.Subscriber("predicted_state_sets", ConformalSetTrajRadial, self.callbackPredictState, queue_size=self.global_queue_size, tcp_nodelay=True)
        #self.pub_class = rospy.Publisher("classifier_output", ClassProbability, queue_size=self.global_queue_size)
        self.pub_class = rospy.Publisher("classifier_output", ClassificationOutput, queue_size=self.global_queue_size)
    
        ### initialize attributes
        self.dim = rospy.get_param("class/input_len") # input dimension
        self.inputs = deque(maxlen=self.dim)

        ### TaskClass Model Import
        Model_name = rospy.get_param("class/model_name") #"TaskClassModel"
        self.TaskClassModel = TaskClassModelWrapper(Model_name)
        self.maxabs_weights = self.TaskClassModel.maxabs_weights

        self.states_dict = {
        "pos_pitch": 0.0,
        "vel_pitch": 0.0,
        "acc_pitch": 0.0,
        "pos_yaw": 0.0,
        "vel_yaw": 0.0,
        "acc_yaw": 0.0
        }

    ### subscriber callbacks
    def callbackCurrentState(self, msg):
        self.states_dict = {
        "pos_pitch": msg.RPY.y,
        "vel_pitch": msg.angular.y,
        "acc_pitch": msg.accel.y,
        "pos_yaw": msg.RPY.z,
        "vel_yaw": msg.angular.z,
        "acc_yaw": msg.accel.z
        }
    
    ### publisher methods
    def publishClassifierOutput(self, probs, labels, index):
        msg = ClassificationOutput()
        msg.header.stamp = rospy.Time.now()
        class_msgs = []
        for i in range(len(labels)):
            cmsg = ClassificationInfo()
            cmsg.probability = probs[i]
            cmsg.label = labels[i]
            class_msgs.append(cmsg)
        msg.classes = class_msgs
        msg.expected_class_index = int(index)
        
        self.pub_class.publish(msg)

    ### getter functions

    def get_inputs(self):
        self.input = np.array([self.states_dict["vel_yaw"] / self.maxabs_weights[0], 
                            self.states_dict["vel_pitch"] / self.maxabs_weights[1], 
                            self.states_dict["acc_yaw"] / self.maxabs_weights[2], 
                            self.states_dict["acc_pitch"] / self.maxabs_weights[3]])
        return self.input

    def taskClassfication(self):
        self.inputs.append(self.get_inputs())

        if len(self.inputs) == self.dim:
            inputs = np.array(self.inputs)
            output,probs,label_index = self.TaskClassModel.predict(inputs)
            probs = probs.flatten().tolist()
        else:
            probs = [0.0, 0.0]
            label_index = 1
        
        # if label_index == 1:
        #     print("I am doing static")
        # else:
        #     print("I am doing dynamic")

        label = ["Dynamic", "Static"]
        self.publishClassifierOutput(probs, label, label_index)
    
    def waitForTime(self, start_time_point):        
        # Sleeps if faster than 250Hz
        endBeforeRest = time.time()
        elapsedTime = endBeforeRest - start_time_point

        while (elapsedTime < self.dt):
            elapsedTime = time.time() - start_time_point

        #print("elapsedTime:", elapsedTime)
    
if __name__ == '__main__':

    nh = rospy.init_node('classifier', anonymous=True)

    node = ClassifierROS(nh)

    rate = rospy.Rate(node.dt_rate)

    while not rospy.is_shutdown():
        if node.states_dict: # task classification do not start until data is coming
            #state_dict = node.step()
            start = time.time()
            node.taskClassfication()

            node.waitForTime(start)
        else:
            print("Data is not coming yet")