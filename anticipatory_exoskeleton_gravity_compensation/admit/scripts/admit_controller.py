#!/usr/bin/python3

import math
import rospy
from std_msgs.msg import Int16
from geometry_msgs.msg import WrenchStamped, Quaternion, PoseStamped, Pose, TwistStamped, Point
from std_srvs.srv import SetBool, SetBoolResponse

#from admit_lib import *
#from admit_lib import FastAdmittanceController

#from admit import *
#from admit.admit_lib import *
#import admit #admit_lib
#print(f"dir(admit): {dir(admit)}")
#import admit.admit_lib

from admit.admit_lib import *

from rehab_msgs.msg import RPYCommand, RPYState
#from rehab_msgs.scripts.utils import *

### constants from dynamixel code
MOTOR_ENABLE = 10
MOTOR_DISABLE = 11


class AdmitROS:
    def __init__(self, nh):
        self.nh = nh

        ### init from configs
        self.dt = rospy.get_param("admit/dt")
        self.dt_rate = int(1.0 / self.dt)

        self.I_base = rospy.get_param("admit/static/I")
        self.B_base = rospy.get_param("admit/static/B")

        self.vel_max = rospy.get_param("safety/vel_max") # make sure these are enforced in next state equations
        self.acc_max = rospy.get_param("safety/acc_max")
        self.force_max = rospy.get_param("safety/force_max")

        pos_init = rospy.get_param("admit/init/pos")
        vel_init = rospy.get_param("admit/init/vel")
        pos_dim = len(pos_init)

        self.interaction_dynamics = FastAdmittanceController(pos_init, vel_init, I_diag=self.I_base, B_diag=self.B_base, K_p_diag=0.0, K_d_diag = 0.0, pos_dim=pos_dim, dt = self.dt, 
                                                             max_vel=self.vel_max, max_acc=self.acc_max, max_F=self.force_max)
        
        ### initialize subscribers and publishers
        self.global_queue_size = 25 # 50 # was 1
        self.sub_wrench_measured = rospy.Subscriber("torque_force_sensor_state", WrenchStamped, self.callbackTorqueMeasured, queue_size=self.global_queue_size, tcp_nodelay=True)
        self.sub_wrench_robot = rospy.Subscriber("torque_force_robot_state", WrenchStamped, self.callbackTorqueRobot, queue_size=self.global_queue_size, tcp_nodelay=True)

        self.pub_desired = rospy.Publisher("rpy_command", RPYCommand, queue_size=self.global_queue_size) # for the ik
        self.pub_req_state = rospy.Publisher('req_state', Int16, queue_size=self.global_queue_size)

        self.pub_state = rospy.Publisher("rpy_state", RPYState, queue_size=self.global_queue_size) # for the ik


        #self.pub_full = rospy.Publisher("full_admit_state", FullAdmitState, queue_size=self.global_queue_size)

        ### initialize attributes
        self.torque_sensor = np.zeros([3,])
        self.torque_meas = np.zeros([pos_dim,])
        self.torque_robot = np.zeros([pos_dim,])
        self.pos_eq = np.zeros([pos_dim,])
        self.vel_eq = np.zeros([pos_dim,])

        self.fix_roll = rospy.get_param("safety/fix_roll")

        ### managed by services
        self.b_running = False
        self.b_use_gravity_compensation = False
        self.set_running_behavior_service = rospy.Service("admit/set_running_behavior", SetBool, self.setRunningBehavior)
        self.set_grav_comp_service = rospy.Service("admit/set_use_gravity_compensation", SetBool, self.setGravComp)

    ### services
    def setRunningBehavior(self, req): # setBool
        res = SetBoolResponse()
        if (isinstance(req.data, bool)):
            self.b_running = req.data
            res.success = True
            res.message = "Running behavior set to " + str(self.b_running) + "!!!"
        else:
            res.success = False
            res.message = "!!!Error!!! Invalid request data type, got " + str(req.data) + " but needed a bool."
        return res
    
    def setGravComp(self, req): # setBool
        res = SetBoolResponse()
        if (isinstance(req.data, bool)):
            self.b_use_gravity_compensation = req.data
            res.success = True
            res.message = "Use gravity compensation behavior set to " + str(self.b_use_gravity_compensation) + "!!!"
        else:
            res.success = False
            res.message = "!!!Error!!! Invalid request data type, got " + str(req.data) + " but needed a bool."
        return res

    ### subscriber callbacks
    def callbackTorqueMeasured(self, msg):
        self.torque_sensor = [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]

        # TODO: update this to go in getter function. include correct frame transformation there.
        self.torque_meas[0] = msg.wrench.torque.x # verify this one!
        self.torque_meas[1] = -msg.wrench.torque.z # z aligns with perp. to ft sensor face. 
        self.torque_meas[2] = -msg.wrench.torque.y
        

        # was this before we included roll as first dim
        #self.torque_meas[0] = -msg.wrench.torque.z
        #self.torque_meas[1] = -msg.wrench.torque.y
        return
    
    def callbackTorqueRobot(self, msg):
        self.torque_robot[0] = msg.wrench.torque.x
        self.torque_robot[1] = msg.wrench.torque.y
        self.torque_robot[2] = msg.wrench.torque.z
        return
    
    ### publisher methods
    def publishRPYCommand(self, desired_angles, commandedThetaPerturb = 0.0, commandedPhiPerturb = 0.0, covert_to_radians = True):
        msg = RPYCommand()
        d_angles = desired_angles.copy()
        if covert_to_radians:
            for i in range(len(d_angles)):
                d_angles[i] = d_angles[i] * (3.14 / 180.0)
            commandedThetaPerturb = commandedThetaPerturb * (3.14 / 180.0)
            commandedPhiPerturb = commandedPhiPerturb * (3.14 / 180.0)
        msg.psi = d_angles[0] # roll
        msg.theta = d_angles[1] # pitch
        msg.phi = d_angles[2] # yaw

        msg.commandedThetaPerturb = commandedThetaPerturb
        msg.commandedPhiPerturb = commandedPhiPerturb

        self.pub_desired.publish(msg)

    def publishRPYState(self, desired_pos, desired_vel, desired_acc, dt = 0.005, fix_roll = True):
        msg = RPYState()
        msg.header.stamp = rospy.Time.now()
        msg.RPY.x = desired_pos[0]
        msg.RPY.y = desired_pos[1]
        msg.RPY.z = desired_pos[2]
        msg.angular.x = desired_vel[0]
        msg.angular.y = desired_vel[1]
        msg.angular.z = desired_vel[2]
        msg.accel.x = desired_acc[0]
        msg.accel.y = desired_acc[1]
        msg.accel.z = desired_acc[2]


        self.pub_state.publish(msg)

    ###
    def getTorqueMeasured(self):
        return self.torque_meas.copy()

    def getTorqueRobot(self):
        return self.torque_robot.copy()

    def getEqState(self):
        return self.pos_eq.copy(), self.vel_eq.copy()
    
    def step(self):
        ### get inputs
        torque_measured = self.getTorqueMeasured()
        torque_robot = self.getTorqueRobot()
        #print("torque_robot",torque_robot)
        pos_eq, vel_eq = self.getEqState()

        if self.fix_roll:
            torque_measured[0] = 0.0
            torque_robot[0] = 0.0

        if not self.b_use_gravity_compensation:
            torque_robot = torque_robot * 0.0
        ### update admittance controller
        #pos, vel = self.interaction_dynamics.admittance_update(F_meas, F_robot, pos_eq, vel_eq)

        if self.b_running:
            self.interaction_dynamics.admittanceUpdate(torque_measured, torque_robot, pos_eq, vel_eq)

            
            ### package outputs in state_dict
            state_dict = self.interaction_dynamics.getStateDictonary()
            
            # print(f"torque_measured: {torque_measured}")
            # print(f"torque_robot: {torque_robot}")
            # print(f"state_dict['pos']: {state_dict['pos']}")
            # print(f"state_dict['vel']: {state_dict['vel']}")
            
            self.publishRPYCommand(state_dict['pos'])
            self.publishRPYState(state_dict['pos'], state_dict['vel'], state_dict['acc'])
        return
    
if __name__ == '__main__':

    nh = rospy.init_node('admit', anonymous=True)

    admit = AdmitROS(nh)

    rate = rospy.Rate(admit.dt_rate)

    admit.pub_req_state.publish(MOTOR_ENABLE)

    while not rospy.is_shutdown():
        admit.step()
        rate.sleep()

    admit.pub_req_state.publish(MOTOR_DISABLE)