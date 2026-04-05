#!/usr/bin/env python3
from mimetypes import init
import rospy
import os
import numpy as np
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from conf_exps.srv import *
from conf_exps.msg import *
from scipy import interpolate
from geometry_msgs.msg import PoseStamped
import time
from utils import *

class GuidePointGenerator():
    def __init__(self, nh):
        super().__init__()

        self.nh = nh

        ## ROS 
        # subscribers
        self.srv_set_guide_time = rospy.Service('set_guide_time', SetFloat, self.handle_set_guide_time)
        self.srv_start_guiding = rospy.Service('start_guiding', SetFloat, self.handle_start_guiding)
        self.srv_update_targets = rospy.Service('update_targets', UpdateTargets, self.update_targets)
        self.dt = rospy.get_param("dt")
        self.rate = int(1.0 / self.dt)

        '''
        self.subscribers = {
            'admit_record_state': rospy.Subscriber('admit_record_state', AdmitRecordStateStamped, self.admitRecordCallback, queue_size=1, tcp_nodelay=True),
            'mint_conf_record_state': rospy.Subscriber('mint_conf_record_state', MIntConfRecordStateStamped, self.mintRecordCallback, queue_size=1, tcp_nodelay=True),
            '/CurrentTargets': rospy.Subscriber('/CurrentTargets', Float64MultiArray, self.targetXY_callback, queue_size=1, tcp_nodelay=True)
        }
        '''

        self.guide_pub = rospy.Publisher('/GuideTargets', GuideState, queue_size=10)

        ##
        # initial vectors

        self.guideXY = np.array([0.0, 0.0])
        self.idx = 0
        self.target_count = 6
        self.new_trial = 0
        self.guide_time = 4.0
        self.tracking_error_mag = 0.0

        self.countdown_time = self.guide_time - 0.0

        current_directory = os.path.dirname(os.path.realpath(__file__))
        csv_path = os.path.join(current_directory, '..', 'include', 'targets.csv')
        data = np.loadtxt(csv_path, delimiter=",")

        # Waypoints
        targetx_data = data[:, 0] * 0.01
        targety_data = data[:, 1] * 0.01

        self.pathmap = {}

        for i in range(0, targetx_data.size, 6):
            path_num = (i // 6) + 1
            self.pathmap[path_num] = np.column_stack((targetx_data[i:i + 6], targety_data[i:i + 6]))

        self.targets = self.pathmap[1]
        self.tck, u = interpolate.splprep([self.targets[:,0], self.targets[:,1]], s=0)

        self.current_guide_point = np.array([0.0, 0.0])
        self.current_guide_point_vel = np.array([0.0, 0.0])

        wait_time = 10.0
        start_time = time.time()
        while (not rospy.has_param("/prediction_times")) or ((time.time() - start_time) < wait_time):
            time.sleep(0.5)

        self.pred_times = np.array(rospy.get_param("/prediction_times"))
        print(self.pred_times)
        self.guide_points_pos = np.zeros(shape=(2, len(self.pred_times)))
        self.guide_points_vel = np.zeros(shape=(2, len(self.pred_times)))

        self.in_pretrial = True
        self.start_time = 0.0
        self.dsdt = (1.0 - 0.0) / self.guide_time
        self.active_coord = 0.0

    def handle_set_guide_time(self, req):
        res = SetFloatResponse()
        '''
            Can add future functionality here
        '''
        #self.target_count = req.data
        self.guide_time = req.data
        res.success = True
        res.message = f"GUI: Guide Time changed to {self.guide_time}!"
        self.dsdt = (1.0 - 0.0) / self.guide_time
        #self.error_bar_active = False
        #self.in_pretrial = True
        return res
    
    def handle_start_guiding(self, req):
        res = SetFloatResponse()
        res.success = True
        res.message = "GUI: Guiding Started!"
        self.in_pretrial = False
        self.start_time = time.time()
        return res

    def update_targets(self, req):
        self.targets = np.column_stack((req.x, req.y))
        self.in_pretrial = True
        self.tck, u = interpolate.splprep([self.targets[:,0], self.targets[:,1]], s=0)

        return UpdateTargetsResponse(success=True)

    def getSPointArray(self, diff_time):
        s_times = np.clip((self.pred_times + diff_time) / self.guide_time, 0.0, 1.0) * np.array(self.active_coord)
        #print(type(s_times)) # self.active_coord
        #print(s_times)
        return s_times

    def getGuidePointArray(self, s_points):
        spline_points_xylist = interpolate.splev(s_points, self.tck)
        spline_point_vels_xylist = interpolate.splev(s_points, self.tck, der=1)
        spline_points = np.stack(spline_points_xylist, axis=0)
        spline_point_vels = np.stack(spline_point_vels_xylist, axis=0) * self.dsdt

        return spline_points, spline_point_vels

        #self.current_guide_point[0] = spline_point[0]
        #self.current_guide_point[1] = spline_point[1]

        #self.current_guide_point_vel[0] = spline_point_vel[0] * self.dsdt
        #self.current_guide_point_vel[1] = spline_point_vel[1] * self.dsdt # last part scales it to time domain. dx/ds * ds/dt = dx/dt


    def updateCurrentGuidePoint(self):
        if self.in_pretrial:
            #s_point = 0.000001
            self.countdown_time = self.guide_time
            self.active_coord = 0.0
            s_point_array = self.getSPointArray(0.0)
        else:
            self.active_coord = 1.0
            diff_time = time.time() - self.start_time
            self.countdown_time = self.guide_time - diff_time
            #s_point = diff_time / self.guide_time
            s_point_array = self.getSPointArray(diff_time)
            #if 1.0 < s_point:
            if (s_point_array[0] == 1.0):
                self.countdown_time = 0.0
                self.active_coord = 0.0

        self.guide_points_pos, self.guide_points_vel = self.getGuidePointArray(s_point_array)
        self.current_guide_point[0] = self.guide_points_pos[0,0]
        self.current_guide_point[1] = self.guide_points_pos[1,0]

        self.current_guide_point_vel[0] = self.guide_points_vel[0,0]
        self.current_guide_point_vel[1] = self.guide_points_vel[1,0]

        return 

    def publishCurrentGuidePoint(self):
        msg = GuideState()
        msg.position.x = self.current_guide_point[0]
        msg.position.y = self.current_guide_point[1]
        msg.velocity.x = self.current_guide_point_vel[0]
        msg.velocity.y = self.current_guide_point_vel[1]
        msg.countdown_time = self.countdown_time
        msg.active_coord = self.active_coord

        msg.pos_x = self.guide_points_pos[1,:].tolist()
        msg.pos_y = self.guide_points_pos[0,:].tolist()
        msg.vel_x = self.guide_points_vel[1,:].tolist()
        msg.vel_y = self.guide_points_vel[0,:].tolist()

        #print(msg)
        self.guide_pub.publish(msg)
        return
    
#from multiprocessing import Process
def main():
    nh = rospy.init_node('guide_node')
 
    guide_node = GuidePointGenerator(nh)

    rate = rospy.Rate(guide_node.rate)
    #vis_process.start()
    while not rospy.is_shutdown():

        guide_node.updateCurrentGuidePoint()
        guide_node.publishCurrentGuidePoint()
        rate.sleep()
    #vis_process.join()
    plt.close()

if __name__ == '__main__':
    main()
