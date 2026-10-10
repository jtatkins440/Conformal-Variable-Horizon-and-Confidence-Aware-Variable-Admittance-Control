
import math
import rospy
from std_msgs.msg import Int16
from std_srvs.srv import Empty
from geometry_msgs.msg import WrenchStamped, Quaternion, PoseStamped, Pose, TwistStamped, Point
from rehab_msgs.msg import ClassProbability

import numpy as np
import pandas as pd


# just start with wrench for now
class PlaybackROS:
    def __init__(self, nh):
        self.nh = nh

        ### init from configs
        self.dt = 0.004
        self.csv_path = "/home/antigrav_ws/src/rehab_antigrav/admit/scripts/res/TaskClass_data1.csv"
        
        self.playback_df = pd.read_csv(self.csv_path)

        #self.dt = rospy.get_param("playback/dt")
        #self.csv_path = rospy.get_param("playback/csv_path")
        
        self.dt_rate = int(1.0 / self.dt)

        ### initialize subscribers and publishers
        self.global_queue_size = 25 # 50 # was 1
        self.pub_wrench_measured = rospy.Publisher("ft_raw", WrenchStamped, queue_size=self.global_queue_size)
        self.pub_class = rospy.Publisher("classifier_reference", ClassProbability, queue_size=self.global_queue_size)

        ### initialize attributes
        self.force_meas = np.zeros([3,])
        self.torque_meas = np.zeros([3,])

        self.playback_active = True #False
        self.loop_playback = True
        self.playback_index = 0
        self.playback_index_max = len(self.playback_df)




    def publishMeasuredWrench(self, force, torque):
        msg = WrenchStamped()
        msg.wrench.force.x = force[0]
        msg.wrench.force.y = force[1]
        msg.wrench.force.z = force[2]
        msg.wrench.torque.x = torque[0]
        msg.wrench.torque.y = torque[1]
        msg.wrench.torque.z = torque[2]

        self.pub_wrench_measured.publish(msg)

    ### publisher methods
    def publishClassifierOutput(self, probs, labels):
        msg = ClassProbability()
        msg.probabilities = probs
        msg.labels = labels

        self.pub_class.publish(msg)

    ###
    def getForceTorqueMeasured(self):
        
        idx = self.playback_df.index[self.playback_index]
        self.torque_meas[1] = self.playback_df.loc[idx, "pitch_torque"]
        self.torque_meas[2] = self.playback_df.loc[idx, "yaw_torque"]
        #self.force_meas
        #self.torque_meas
        return self.force_meas.copy(), self.torque_meas.copy()
    
    def getClassProbabilities(self):
        class_index = int(self.playback_df.loc[self.playback_df.index[self.playback_index], "task_num"]) - 1 # starts at 1!
        class_labels = ["static", "dynamic"]
        class_probs = np.zeros([len(class_labels),])
        class_probs[class_index] = 1.0
        return class_probs, class_labels

    
    def step(self):
        if self.playback_active:
            ### get inputs
            force_measured, torque_measured = self.getForceTorqueMeasured()
            class_probs, class_labels = self.getClassProbabilities()

            ### package state_dict
            state_dict = dict()
            state_dict["force"] = force_measured
            state_dict["torque"] = torque_measured
            state_dict["class_probs"] = class_probs
            state_dict["class_labels"] = class_labels

            self.publishMeasuredWrench(state_dict['force'], state_dict['torque'])
            self.publishClassifierOutput(state_dict["class_probs"], state_dict["class_labels"])

            ### update index
            self.playback_index += 1
            if (self.playback_index == self.playback_index_max):
                self.playback_index = 0
                if not self.loop_playback: 
                    self.playback_active = False
        return 
    
if __name__ == '__main__':

    nh = rospy.init_node('admit', anonymous=True)

    playback = PlaybackROS(nh)

    rate = rospy.Rate(playback.dt_rate)

    while not rospy.is_shutdown():
        playback.step()
        #state_dict = playback.step()
        
        rate.sleep()
