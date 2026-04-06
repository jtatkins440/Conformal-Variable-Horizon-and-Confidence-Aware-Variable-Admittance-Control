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
import pyqtgraph as pg
from PyQt5 import QtWidgets
import time
from utils import *

pg.setConfigOptions(useNumba = True)

class StandaloneVisualizerApp(QtWidgets.QMainWindow):
    def __init__(self, nh):
        super().__init__()

        self.nh = nh

        ## get ros params
        self.scatter_symbol_size = rospy.get_param("guide_plot/symbol_size")
        self.pen_width = rospy.get_param("guide_plot/line_width")
        self.traj_x_range = rospy.get_param("guide_plot/x_range")
        self.traj_y_range = rospy.get_param("guide_plot/y_range")
        self.b_disp_eq = rospy.get_param("guide_plot/display_equilibrium")
        self.b_disp_prediction = rospy.get_param("guide_plot/display_prediction")
        self.b_shift_prediction = rospy.get_param("guide_plot/shift_prediction")
        self.b_disp_quantiles = rospy.get_param("guide_plot/display_quantiles") #True

        self.b_disp_arrow = rospy.get_param("arrows/display_arrows")
        self.arrow_l = rospy.get_param("arrows/length")
        self.arrow_w = rospy.get_param("arrows/head_half_width")
        self.arrow_h = rospy.get_param("arrows/head_heigth") #True
        self.arrow_draw_width = rospy.get_param("arrows/line_width") #True

        self.timer_text_color = rospy.get_param("timer/text_color")
        self.timer_text_font_size = rospy.get_param("timer/text_font_size")
        self.timer_clock_font_size = rospy.get_param("timer/clock_font_size")

        self.max_error = rospy.get_param("bar/max_vis_error")
        self.zero_error_color = np.array(rospy.get_param("bar/zero_error_rgba_color")) # RGBA but convert these to ints after interp!
        self.max_error_color = np.array(rospy.get_param("bar/max_error_rgba_color"))
        self.bar_x_range = rospy.get_param("bar/x_range") 
        self.bar_y_range = rospy.get_param("bar/y_range")

        self.base_arrow_points = self.getBaseArrowArray(self.arrow_l, self.arrow_w, self.arrow_h)

        self.graphics_layout_widget = pg.LayoutWidget()
        self.graphWidget = pg.PlotWidget()
        #self.graphWidget = self.graphics_layout_widget.addPlot(row=0, col=0, rowspan=1, colspan=1)
        #self.barWidget = pg.GraphicsView() #pg.PlotWidget() # maybe change this for bar charts
        
        self.barWidget = pg.PlotWidget() #pg.plot() #pg.GraphicsView()
        
        #self.barWidget.addItem(self.barItem) # setCentralWidget
        # pg.BarGraphItem(x=x+0.66, height=y3, width=0.3, brush='b')

        graph_policy = QtWidgets.QSizePolicy()
        graph_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Preferred)
        graph_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Preferred)
        graph_policy.setVerticalStretch(3)
        graph_policy.setHorizontalStretch(3)
        self.graphWidget.setSizePolicy(graph_policy)

        bar_policy = QtWidgets.QSizePolicy()
        bar_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Preferred)
        bar_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Preferred)
        bar_policy.setVerticalStretch(3)
        bar_policy.setHorizontalStretch(1)
        self.barWidget.setSizePolicy(bar_policy)

        self.graphics_layout_widget.addWidget(self.graphWidget, row=0, col=0, colspan=1)
        self.graphics_layout_widget.addWidget(self.barWidget, row=0, col=1, colspan=1)
        self.setCentralWidget(self.graphics_layout_widget)

        # Set up plot properties
        self.graphWidget.setBackground('w')
        self.barWidget.setBackground('w')
        #self.graphics_layout_widget.setBackground('w')

        # Create separate PlotDataItems
        self.targetsPlot = self.graphWidget.plot([], [], symbol='o', pen=None, symbolBrush='b', symbolSize=self.scatter_symbol_size)
        print(f"self.targetsPlot: {self.targetsPlot}")
        self.currentTargetPlot = self.graphWidget.plot([], [], symbol='o', pen=None, symbolBrush='orange', symbolSize=self.scatter_symbol_size)
        self.endEffectorPlot = self.graphWidget.plot([], [], symbol='o', pen=None, symbolBrush='r', symbolSize=self.scatter_symbol_size)
        self.guideXYPlot = self.graphWidget.plot([], [], symbol='o', pen=None, symbolBrush='green', symbolSize=self.scatter_symbol_size)
        self.splinePlot = self.graphWidget.plot([], [], pen=pg.mkPen('k', width=self.pen_width))
        self.eeEqPlot = self.graphWidget.plot([], [], symbol='o',pen=None, symbolBrush='grey', symbolSize=self.scatter_symbol_size)

        self.endEffectorArrowPlot = self.graphWidget.plot([], [], pen=pg.mkPen('r', width=self.arrow_draw_width))
        self.guideXYArrowPlot = self.graphWidget.plot([], [], pen=pg.mkPen('green', width=self.arrow_draw_width))

        self.timer_text_item = pg.TextItem(html=self.getHTMLTimeString(0.0)) #pg.TextItem(text="TEXT ITEM TESTING", anchor=(0.0, 0.0), color='k')
        self.timer_text_item.setPos(0.1, -0.1)
        self.graphWidget.addItem(self.timer_text_item)

        # plotdataitems for the bar graphs
        self.barItemPos = self.barWidget.plot([], [], pen=pg.mkPen('r', width=10))
        self.barItemVel = self.barWidget.plot([], [], pen=pg.mkPen('r', width=10), symbolBrush='r')

        self.point_num = 6 # was 5

        self.mint_pred_plot_list = [self.graphWidget.plot([], [], pen=pg.mkPen('k', width=4)) for i in range(0, self.point_num)] #rgba(183, 0, 255, 0.8)
        self.mint_pred_plot = self.graphWidget.plot([], [], pen=pg.mkPen('k', width=self.pen_width))

        rgba = (100, 0, 0, 50)
        self.quantile_brush = pg.mkBrush(rgba)
        self.quantile_plot = self.graphWidget.plot([], [], symbol='o', pen=None, symbolBrush='r', symbolSize=0.1, pxMode=False)

        # Set plot limits
        self.graphWidget.setXRange(self.traj_x_range[0], self.traj_x_range[1])
        self.graphWidget.setYRange(self.traj_y_range[0], self.traj_y_range[1])

        self.barWidget.setXRange(self.bar_x_range[0], self.bar_x_range[1])
        self.barWidget.setYRange(self.bar_y_range[0], self.bar_y_range[1])

        ## ROS 
        # subscribers
        self.srv_start_visualizing = rospy.Service('start_visualizing', SetBool, self.handle_visualize)
        self.srv_new_trial = rospy.Service('new_trial', SetInt, self.handle_new_trial)
        self.srv_update_targets = rospy.Service('update_targets', UpdateTargets, self.update_targets)

        self.sub_prev_targets = rospy.Subscriber('/PreviousTargets', Float64MultiArray, self.targetXYold_callback, queue_size=1)
        self.sub_curr_targets = rospy.Subscriber('/CurrentTargets', Float64MultiArray, self.targetXY_callback, queue_size=1)

        self.subscribers = {
            'admit_record_state': rospy.Subscriber('admit_record_state', AdmitRecordStateStamped, self.admitRecordCallback, queue_size=1, tcp_nodelay=True),
            'mint_conf_record_state': rospy.Subscriber('mint_conf_record_state', MIntConfRecordStateStamped, self.mintRecordCallback, queue_size=1, tcp_nodelay=True),
            '/CurrentTargets': rospy.Subscriber('/CurrentTargets', Float64MultiArray, self.targetXY_callback, queue_size=1, tcp_nodelay=True),
            '/GuideTargets': rospy.Subscriber('/GuideTargets', GuideState, self.guideCallback, queue_size=1, tcp_nodelay=True)
        }
        
        ##
        # initial vectors
        # Global variables
        self.targetXYold = np.zeros(2)
        self.targetXY = np.zeros(2)
        self.endEffectorXY = np.array([0.0, 0.0])
        self.ee_eq_pose = np.array([0.0,0.0])
        self.pred_pos = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.point_num  = 6

        self.ee_vel = np.array([0.0,0.0])
        self.force_app = np.array([0.0,0.0])

        # targets = np.zeros((6, 2))
        self.initial_pose = np.array([0.0,-0.4231, 0.7589])
        self.guideXY = np.array([0.0, 0.0])
        self.idx = 0
        self.target_count = 6
        self.new_trial = 0
        self.guide_time = 4.0
        self.tracking_error_mag = 0.0
        self.countdown_time = 0.0

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

        self.b_visualize = False
        self.current_guide_point = np.array([0.0, 0.0])
        self.current_guide_point_vel = np.array([0.0, 0.0])

        self.quantiles = None
        self.in_pretrial = True
        self.start_time = None
        # arrow points stuff

    def getErrorColor(self, error):
        norm_error = error / self.max_error
        if 1.0 < norm_error:
            norm_error = 1.0
        set_color = (1.0 - norm_error) * self.zero_error_color + (norm_error) * self.max_error_color
        return set_color.astype(np.int64)

    def getHTMLTimeString(self, disp_time):
            countdown_string = "{:.1f}s".format(disp_time)
            html_string = f'<div style="text-align: center"><span style="color: {self.timer_text_color}; font-size: {self.timer_text_font_size}pt">Trial Time:</span><br><span style="color: {self.timer_text_color}; font-size: {self.timer_clock_font_size}pt;">{countdown_string}</span></div>'
            return html_string

    def getBaseArrowArray(self, l, w, h):
        origin = np.array([0.0, 0.0])
        tip = np.array([l, 0.0])
        upper_tip = np.array([l - h, w])
        lower_tip = np.array([l - h, -w])
        base_arrow_points = np.stack([origin, tip, upper_tip, tip, lower_tip], axis=1)
        return base_arrow_points

    def getArrowArray(self, dir, l, w, h):
        origin = np.array([0.0, 0.0])
        tip = l * dir
        dir_ortho = np.array([dir[1], -dir[0]])
        upper_tip = (l - h) * dir + w * dir_ortho
        lower_tip = (l - h) * dir - w * dir_ortho
        base_arrow_points = np.stack([origin, tip, upper_tip, tip, lower_tip], axis=1)
        return base_arrow_points

    def getDirectionalArrow(self, start_point, arrow_direction, min_dir_mag = 0.005):
        base_arrow_copy = self.base_arrow_points.copy()
        base_arrow_dir = base_arrow_copy[:,1] / np.linalg.norm(base_arrow_copy[:,1]) # tip vector, along x-axis
        dir_mag = np.linalg.norm(arrow_direction)
        if min_dir_mag < dir_mag:
            arrow_dir = arrow_direction / dir_mag
            rot_arrow = self.getArrowArray(arrow_dir, self.arrow_l,self.arrow_w, self.arrow_h)
            return rot_arrow + start_point[:, np.newaxis]
        else:
            return base_arrow_copy * 0.0 
        
    def old_getDirectionalArrow(self, start_point, arrow_direction, min_dir_mag = 0.005):
        base_arrow_copy = self.base_arrow_points.copy()
        base_arrow_dir = base_arrow_copy[:,1] / np.linalg.norm(base_arrow_copy[:,1]) # tip vector, along x-axis
        dir_mag = np.linalg.norm(arrow_direction)
        if min_dir_mag < dir_mag:
            arrow_dir = arrow_direction / dir_mag
            vec_ang = np.arcsin(np.linalg.det(np.stack([base_arrow_dir, arrow_dir], axis=1)))
            rot_mat = rotmat_2d(vec_ang)
            rot_arrow = np.matmul(rot_mat, base_arrow_copy)
            return rot_arrow + start_point[:, np.newaxis]
        else:
            return base_arrow_copy * 0.0 

    def admitRecordCallback(self, msg):
        current_pose_msg = msg.pose.position
        abs_pose = np.array([current_pose_msg.x, current_pose_msg.y, current_pose_msg.z])# - self.initial_pose
        self.endEffectorXY[0] = abs_pose[2]
        self.endEffectorXY[1] = abs_pose[0]

        #print(f"self.endEffectorXY: {self.endEffectorXY}")
        max_eq_distance = 0.05
        temp_data = np.array([msg.eq_state.position.x, msg.eq_state.position.y, msg.eq_state.position.z])# - self.initial_pose
        self.ee_eq_pose[0] = temp_data[2]
        self.ee_eq_pose[1] = temp_data[0]
        diff_pos = self.ee_eq_pose - [self.endEffectorXY[0], self.endEffectorXY[1]]

        self.ee_vel = np.array([msg.twist.linear.z, msg.twist.linear.x])
        self.force_app = np.array([msg.wrench.force.z, msg.wrench.force.x])

    def mintRecordCallback(self, msg):
        aci_sample_list = msg.aci_state.aci_batch
        aci_quantiles_list = []
        aci_alphas_list = []
        aci_gammas_list = []
        for idx in range(len(aci_sample_list)):
            aci_quantiles_list.append(unpack_multiarray_msg(aci_sample_list[idx].quantiles))
            aci_alphas_list.append(unpack_multiarray_msg(aci_sample_list[idx].alphas))
            aci_gammas_list.append(aci_sample_list[idx].gammas)
        self.quantiles = np.stack(aci_quantiles_list, axis=-1)

        try:
            self.pred_pos = unpack_multiarray_msg(msg.pred_traj)
            self.pred_pos = np.flip(self.pred_pos, axis=0)
        except:
            self.pred_pos = None
        return

    def recordStateCallback(self, msg):
        current_pose_msg = msg.pose.position
        abs_pose = np.array([current_pose_msg.x, current_pose_msg.y, current_pose_msg.z]) - self.initial_pose
        self.endEffectorXY[0] = abs_pose[2]
        self.endEffectorXY[1] = abs_pose[0]

        #self.ee_eq_pose
        max_eq_distance = 0.05
        temp_data = np.array([msg.pose_eq.position.x, msg.pose_eq.position.y, msg.pose_eq.position.z]) - self.initial_pose
        self.ee_eq_pose[0] = temp_data[2]
        self.ee_eq_pose[1] = temp_data[0]
        diff_pos = self.ee_eq_pose - [self.endEffectorXY[0], self.endEffectorXY[1]]

        self.ee_vel = np.array([msg.twist.linear.z, msg.twist.linear.x])
        self.force_app = np.array([msg.wrench.force.z, msg.wrench.force.x])
        
        try:
            self.pred_pos = unpack_multiarray_msg(msg.icp_traj.model_output)
            self.pred_pos = np.flip(self.pred_pos, axis=0)
        except:
            self.pred_pos = None

    def aciStateCallback(self, msg):
        pred_len = msg.sample_number
        aci_sample_list = msg.aci_batch
        aci_quantiles_list = []
        aci_alphas_list = []
        aci_gammas_list = []
        for idx in range(len(aci_sample_list)):
            aci_quantiles_list.append(unpack_multiarray_msg(aci_sample_list[idx].quantiles))
            aci_alphas_list.append(unpack_multiarray_msg(aci_sample_list[idx].alphas))
            aci_gammas_list.append(aci_sample_list[idx].gammas)
        self.quantiles = np.stack(aci_quantiles_list, axis=-1)
        return
    
    def guideCallback(self, msg):
        self.current_guide_point[0] = msg.position.x
        self.current_guide_point[1] = msg.position.y

        self.current_guide_point_vel[0] = msg.velocity.x
        self.current_guide_point_vel[1] = msg.velocity.y
        self.countdown_time = msg.countdown_time
        return

    def handle_visualize(self, req):
        res = SetBoolResponse()
        self.b_visualize = bool(req.data)
        res.success = True
        res.message = f"Set visualizer flag to: {self.b_visualize}!"
        return res

    def targetXYold_callback(self, msg):
        global targetXYold
        self.targetXYold = np.array(msg.data)

    def targetXY_callback(self, msg):
        global targetXY
        self.targetXY = np.array(msg.data)

    def getQuantilePoints(self):
        q_sizes = 1 #
        if self.quantiles is not None:
            q_sizes = self.quantiles[0,0,:].tolist()
        return q_sizes

    def handle_new_trial(self, msg):
        res = SetIntResponse()
        self.new_trial = msg.data
        res.success = True
        res.message = "New Trial"
        self.error_bar_active = True
        self.in_pretrial = False
        return res

    def update_targets(self, req):
        self.targets = np.column_stack((req.x, req.y))
        self.error_bar_active = False
        self.in_pretrial = True
        return UpdateTargetsResponse(success=True)

    def update_plot(self, target_points, current_target, end_effector, guideXY, ee_eq_pose, pred_pos, spline_points=None):
        if self.b_visualize:
            # Update targets
            self.targetsPlot.setData(target_points[0], target_points[1])

            # Update current target
            self.currentTargetPlot.setData([current_target[0]], [current_target[1]])

            # Update end effector
            self.endEffectorPlot.setData([end_effector[0]], [end_effector[1]])

            self.guideXYPlot.setData([guideXY[0]], [guideXY[1]])

            if self.b_disp_arrow:
                ee_arrow = self.getDirectionalArrow(self.endEffectorXY, self.ee_vel)
                self.endEffectorArrowPlot.setData(ee_arrow[0,:], ee_arrow[1,:])

                guide_arrow = self.getDirectionalArrow(self.current_guide_point, self.current_guide_point_vel)
                self.guideXYArrowPlot.setData(guide_arrow[0,:], guide_arrow[1,:])


            if self.b_disp_eq:
                self.eeEqPlot.setData([self.ee_eq_pose[0]],[self.ee_eq_pose[1]])

            if self.b_disp_prediction:
                if pred_pos is not None:
                    if self.b_shift_prediction:
                        pred_pos_offset = end_effector - pred_pos[:,0]
                        pred_pos = pred_pos + pred_pos_offset[:,np.newaxis]
                    self.mint_pred_plot.setData(pred_pos[0, :], pred_pos[1, :])

            if self.b_disp_quantiles:
                if (self.quantiles is not None) and (pred_pos is not None):
                    rgba = (100, 0, 0, 50)
                    quantile_brush = pg.mkBrush(rgba)

                    q_sizes = self.getQuantilePoints()
                    #print(f"GUI: q_sizes: {q_sizes}")
                    self.quantile_plot.setData(pred_pos[0, :], pred_pos[1, :])
                    self.quantile_plot.setSymbolSize(q_sizes)
                    self.quantile_plot.setSymbolBrush(quantile_brush)

            # Update spline if available
            if spline_points is not None:
                self.splinePlot.setData(spline_points[0], spline_points[1])

            self.barItemPos.setData([0.0, 0.0, 0.01, 0.01, 0.0], [0.0, self.tracking_error_mag, self.tracking_error_mag, 0.0, 0.0])
            err_col = self.getErrorColor(self.tracking_error_mag)
            self.barItemPos.setPen(pg.mkPen(err_col))
            self.barItemPos.setFillBrush(pg.mkBrush(err_col))
            self.barItemPos.setFillLevel(0.1)

            self.timer_text_item.setHtml(self.getHTMLTimeString(self.countdown_time))

    def visualize(self, app):   

        # Spline plotting
        spline_points = None

        if len(self.targets) > 1:
            interp_time_start = time.time()
            self.tck, u = interpolate.splprep([self.targets[:,0], self.targets[:,1]], s=0)
            
            unew = np.linspace(0, 1, 1000)
            spline_points = interpolate.splev(unew, self.tck)

            #global spline_points
            if (self.new_trial == 1) and not np.array_equal(self.targetXY, np.zeros(2)):
                self.current_guide_point = np.array([0.0, 0.0])
                self.start_time = time.time()
                self.new_trial = -1
                self.error_bar_active = True
            self.tracking_error_mag = np.linalg.norm(self.endEffectorXY - self.current_guide_point)

        # Target points
        target_points = (self.targets[:, 0], self.targets[:, 1])

        # Current target and end effector
        current_target = (self.targetXY[0], self.targetXY[1])
        end_effector = (self.endEffectorXY[0], self.endEffectorXY[1])

        # Update plot
        self.update_plot(target_points, current_target, end_effector, self.current_guide_point, self.ee_eq_pose, self.pred_pos, spline_points)
        app.processEvents() 
    
    def updateCurrentGuidePoint(self):
        if (self.new_trial == -1):
            s_point = (time.time() - self.start_time) / self.guide_time
            if 1.0 < s_point:
                s_point = 1.0

            if self.error_bar_active:
                self.tracking_error_mag = np.linalg.norm(self.endEffectorXY - self.current_guide_point)
            else:
                self.tracking_error_mag = 0.0

        else:
            s_point = 0.0
            self.tracking_error_mag = 0.0

        

        spline_point = interpolate.splev(s_point, self.tck)
        print(f"interpolation tck length: {len(self.tck)}")
        for i in range(len(self.tck)):
            print(f"interpolation tck {i}: {self.tck[i]}")

        if self.in_pretrial:
            s_point = 0.0001
        spline_point_vel = interpolate.splev(s_point, self.tck, der=1)

        self.current_guide_point[0] = spline_point[0]
        self.current_guide_point[1] = spline_point[1]

        self.current_guide_point_vel[0] = spline_point_vel[0]
        self.current_guide_point_vel[1] = spline_point_vel[1]

        return 

    def visualizerLoop(self, app, vis_rate_dt):
        vis_process = Process(target=self.visualize, args=(app,))
        vis_time_start = time.time()
        
        while not rospy.is_shutdown():
            if self.b_visualize and (vis_rate_dt < (time.time() - vis_time_start)):
                self.visualize(app)
                vis_process.start()
                vis_time_start = time.time()
            else:
                app.processEvents()
        vis_process.join()
        return

#from multiprocessing import Process
def main():
    # Initialize the service
    nh = rospy.init_node('visualizer_py')

   

    rangex_ = -0.18
    rangex = 0.18
    rangey_ = -0.18
    rangey = 0.18
    global d_r, u_r
    d_r = 0.015  # Radius of each target point
    u_r = 0.005  # Radius of end-effector position
    app = QtWidgets.QApplication(sys.argv)
    window = StandaloneVisualizerApp(nh)
    window.show()
    main_rate = 200.0
    vis_rate = 20 #20.0 # was 60
    rate = rospy.Rate(vis_rate)
    vis_rate_dt = 1.0 / vis_rate

    vis_time_start = time.time()

    #vis_process.start()
    while not rospy.is_shutdown():
        window.visualize(app)
        app.processEvents()
        rate.sleep()

    plt.close()

if __name__ == '__main__':
    main()
