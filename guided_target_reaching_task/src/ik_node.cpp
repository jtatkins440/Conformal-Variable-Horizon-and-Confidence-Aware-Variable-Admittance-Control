//// /*
// kuka_ik_node.cpp: This is a rework of old code which was then organized into a class with additional helper functions.
//
// */

#include "ros/ros.h"
#include <conf_exps/IKmsg.h>
#include <conf_exps/AdmitStateStamped.h>
#include "sensor_msgs/JointState.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <chrono>
#include <ctime>
#include <ros/callback_queue.h>
#include <boost/bind.hpp>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Pose.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_srvs/SetBool.h"
#include "std_srvs/Trigger.h"
#include <conf_exps/GetCurrentPoseVector.h>
#include <conf_exps/SetInt.h>

using namespace std;
using namespace Eigen;
# define M_PI 3.14159265358979323846 /* pi */

class KUKAStateHandler{
    public:
    KUKAStateHandler(){};

    KUKAStateHandler(std::vector<double> alpha_vec, std::vector<double> a_vec, std::vector<double> d_vec, bool ignore_orientation){
        ignore_orientation = ignore_orientation;
        njoints = alpha_vec.size();

        //std::cout << "In KUKAStateHandler constructor, before initalizing eigen mats" << std::endl;
        alpha = MatrixXd::Zero(1, njoints); // only eigen data types just because to not change lots of messy hardcoded code that 'works'
        a = MatrixXd::Zero(1, njoints);
        d = MatrixXd::Zero(1, njoints);
        theta = MatrixXd::Zero(1, njoints);
        qcurr = MatrixXd::Zero(njoints-1, 1);
        inv_Ja = MatrixXd::Zero(6, 6); // hardcoded size, bad!
        ee_transf = MatrixXd::Zero(4,4);

        std::vector<double> theta_vec;

        for (int i = 0; i < njoints; i++){
            alpha(0,i) = alpha_vec[i];
            a(0,i) = a_vec[i];
            d(0,i) = d_vec[i];
            theta_vec.push_back(0.0); // assumes initial joint state is zero
        }

        std::stringstream sstream_temp;
        sstream_temp << "KUKAStateHandler: alpha matrix: ";
        for (int i = 0; i < alpha.cols(); i++){
            sstream_temp << alpha(0,i) << " ";
        }
        sstream_temp << std::endl;

        sstream_temp << "a matrix: ";
        for (int i = 0; i < a.cols(); i++){
            sstream_temp << a(0,i) << " ";
        }
        sstream_temp << std::endl;

        sstream_temp << "d matrix: ";
        for (int i = 0; i < d.cols(); i++){
            sstream_temp << d(0,i) << " ";
        }
        sstream_temp << std::endl;
        std::string stemp = sstream_temp.str();
        ROS_INFO(stemp.c_str());

        ROS_INFO("KUKAStateHandler:\n njoints: %d\n", njoints);

        //std::cout << "In KUKAStateHandler constructor, before setRobotJointConfig" << std::endl;
        setRobotJointConfig(theta_vec); // defines all internal variables dependent on the joint state
    }
    

    void setRobotJointConfig(std::vector<double> joint_config){
        //ROS_INFO("In setRobotJointConfig, near 58\n");
        if (njoints != joint_config.size()){

            ROS_WARN("KUKAStateHandler Error, wrong number of joints in setRobtoJointConfig!:\n njoints: %d\n joint_config.size(): %ld", njoints, joint_config.size());
            return; // do nothing if joint vector isn't correct size
        }


        for (int i = 0; i < njoints; i++){
            theta(0,i) = joint_config[i];
        }

        //std::cout << "In setRobotJointConfig, near 64" << std::endl;
        // Defines relative transformations matrices
        theta(0, 0) += M_PI; // From DH. To match with ROS frames. MAKE SURE THIS WILL NOT HURT REAL KUKA
        MatrixXd A1(4, 4); A1 << cos(theta(0, 0)), -sin(theta(0, 0))*cos(alpha(0, 0)), sin(theta(0, 0))*sin(alpha(0, 0)), a(0, 0)*cos(theta(0, 0)),
        sin(theta(0, 0)), cos(theta(0, 0))*cos(alpha(0, 0)), -cos(theta(0, 0))*sin(alpha(0, 0)), a(0, 0)*sin(theta(0, 0)),
        0, sin(alpha(0, 0)), cos(alpha(0, 0)), d(0, 0),
        0, 0, 0, 1;
        MatrixXd A2(4, 4); A2 << cos(theta(0, 1)), -sin(theta(0, 1))*cos(alpha(0, 1)), sin(theta(0, 1))*sin(alpha(0, 1)), a(0, 1)*cos(theta(0, 1)),
        sin(theta(0, 1)), cos(theta(0, 1))*cos(alpha(0, 1)), -cos(theta(0, 1))*sin(alpha(0, 1)), a(0, 1)*sin(theta(0, 1)),
        0, sin(alpha(0, 1)), cos(alpha(0, 1)), d(0, 1),
        0, 0, 0, 1;
        MatrixXd A3(4, 4); A3 << cos(theta(0, 2)), -sin(theta(0, 2))*cos(alpha(0, 2)), sin(theta(0, 2))*sin(alpha(0, 2)), a(0, 2)*cos(theta(0, 2)),
        sin(theta(0, 2)), cos(theta(0, 2))*cos(alpha(0, 2)), -cos(theta(0, 2))*sin(alpha(0, 2)), a(0, 2)*sin(theta(0, 2)),
        0, sin(alpha(0, 2)), cos(alpha(0, 2)), d(0, 2),
        0, 0, 0, 1;
        MatrixXd A4(4, 4); A4 << cos(theta(0, 3)), -sin(theta(0, 3))*cos(alpha(0, 3)), sin(theta(0, 3))*sin(alpha(0, 3)), a(0, 3)*cos(theta(0, 3)),
        sin(theta(0, 3)), cos(theta(0, 3))*cos(alpha(0, 3)), -cos(theta(0, 3))*sin(alpha(0, 3)), a(0, 3)*sin(theta(0, 3)),
        0, sin(alpha(0, 3)), cos(alpha(0, 3)), d(0, 3),
        0, 0, 0, 1;
        MatrixXd A5(4, 4); A5 << cos(theta(0, 4)), -sin(theta(0, 4))*cos(alpha(0, 4)), sin(theta(0, 4))*sin(alpha(0, 4)), a(0, 4)*cos(theta(0, 4)),
        sin(theta(0, 4)), cos(theta(0, 4))*cos(alpha(0, 4)), -cos(theta(0, 4))*sin(alpha(0, 4)), a(0, 4)*sin(theta(0, 4)),
        0, sin(alpha(0, 4)), cos(alpha(0, 4)), d(0, 4),
        0, 0, 0, 1;
        MatrixXd A6(4, 4); A6 << cos(theta(0, 5)), -sin(theta(0, 5))*cos(alpha(0, 5)), sin(theta(0, 5))*sin(alpha(0, 5)), a(0, 5)*cos(theta(0, 5)),
        sin(theta(0, 5)), cos(theta(0, 5))*cos(alpha(0, 5)), -cos(theta(0, 5))*sin(alpha(0, 5)), a(0, 5)*sin(theta(0, 5)),
        0, sin(alpha(0, 5)), cos(alpha(0, 5)), d(0, 5),
        0, 0, 0, 1;
        MatrixXd A7(4, 4); A7 << cos(theta(0, 6)), -sin(theta(0, 6))*cos(alpha(0, 6)), sin(theta(0, 6))*sin(alpha(0, 6)), a(0, 6)*cos(theta(0, 6)),
        sin(theta(0, 6)), cos(theta(0, 6))*cos(alpha(0, 6)), -cos(theta(0, 6))*sin(alpha(0, 6)), a(0, 6)*sin(theta(0, 6)),
        0, sin(alpha(0, 6)), cos(alpha(0, 6)), d(0, 6),
        0, 0, 0, 1;

        //ROS_INFO("In setRobotJointConfig, near 98\n");

        // chains relative rotation matrices to get list of absolute transformations with respect to the origin frame
        MatrixXd T01(4, 4); T01 << A1;
        MatrixXd T02(4, 4); T02 << T01*A2;
        MatrixXd T03(4, 4); T03 << T02*A3;
        MatrixXd T04(4, 4); T04 << T03*A4;
        MatrixXd T05(4, 4); T05 << T04*A5;
        MatrixXd T06(4, 4); T06 << T05*A6;
        MatrixXd T07(4, 4); T07 << T06*A7;

        // computes (Z-Y-Z) = (phi, theta, psi) euler angles of link 6 with respect to the origin
        double phi_euler = atan2(T06(1, 2), T06(0, 2)); // check this euler angle type! 
        double theta_euler = atan2(sqrt(pow(T06(1, 2), 2) + pow(T06(0, 2), 2)), T06(2, 2));
        double psi_euler = atan2(T06(2, 1), -T06(2, 0));

        // seperates joint axis vectors from corresponding transformation matrices
        MatrixXd z0(3, 1); z0 << 0, 0, 1;
        MatrixXd z1(3, 1); z1 << T01(0, 2), T01(1, 2), T01(2, 2);
        MatrixXd z2(3, 1); z2 << T02(0, 2), T02(1, 2), T02(2, 2);
        MatrixXd z3(3, 1); z3 << T03(0, 2), T03(1, 2), T03(2, 2);
        MatrixXd z4(3, 1); z4 << T04(0, 2), T04(1, 2), T04(2, 2);
        MatrixXd z5(3, 1); z5 << T05(0, 2), T05(1, 2), T05(2, 2);
        MatrixXd z6(3, 1); z6 << T06(0, 2), T06(1, 2), T06(2, 2);
        MatrixXd z7(3, 1); z7 << T07(0, 2), T07(1, 2), T07(2, 2);

        // seperates joint axis positions from corresponding transformation matrices
        MatrixXd p0(3, 1); p0 << 0, 0, 0;
        MatrixXd p1(3, 1); p1 << T01(0, 3), T01(1, 3), T01(2, 3);
        MatrixXd p2(3, 1); p2 << T02(0, 3), T02(1, 3), T02(2, 3);
        MatrixXd p3(3, 1); p3 << T03(0, 3), T03(1, 3), T03(2, 3);
        MatrixXd p4(3, 1); p4 << T04(0, 3), T04(1, 3), T04(2, 3);
        MatrixXd p5(3, 1); p5 << T05(0, 3), T05(1, 3), T05(2, 3);
        MatrixXd p6(3, 1); p6 << T06(0, 3), T06(1, 3), T06(2, 3);
        MatrixXd p7(3, 1); p7 << T07(0, 3), T07(1, 3), T07(2, 3);

        //ROS_INFO("In setRobotJointConfig, near 134\n");
        // computes columns in the geometric jacobian from the joint axes and their positions
        MatrixXd J1(6, 1); J1 << z0(1, 0)*(p6(2, 0) - p0(2, 0)) - z0(2, 0)*(p6(1, 0) - p0(1, 0)),
                                -z0(0, 0)*(p6(2, 0) - p0(2, 0)) + z0(2, 0)*(p6(0, 0) - p0(0, 0)),
                                z0(0, 0)*(p6(1, 0) - p0(1, 0)) - z0(1, 0)*(p6(0, 0) - p0(0, 0)),
                                z0(0, 0), z0(1, 0), z0(2, 0);
        MatrixXd J2(6, 1); J2 << z1(1, 0)*(p6(2, 0) - p1(2, 0)) - z1(2, 0)*(p6(1, 0) - p1(1, 0)),
                                -z1(0, 0)*(p6(2, 0) - p1(2, 0)) + z1(2, 0)*(p6(0, 0) - p1(0, 0)),
                                z1(0, 0)*(p6(1, 0) - p1(1, 0)) - z1(1, 0)*(p6(0, 0) - p1(0, 0)),
                                z1(0, 0), z1(1, 0), z1(2, 0);
        MatrixXd J3(6, 1); J3 << z2(1, 0)*(p6(2, 0) - p2(2, 0)) - z2(2, 0)*(p6(1, 0) - p2(1, 0)),
                                -z2(0, 0)*(p6(2, 0) - p2(2, 0)) + z2(2, 0)*(p6(0, 0) - p2(0, 0)),
                                z2(0, 0)*(p6(1, 0) - p2(1, 0)) - z2(1, 0)*(p6(0, 0) - p2(0, 0)),
                                z2(0, 0), z2(1, 0), z2(2, 0);
        MatrixXd J4(6, 1); J4 << z3(1, 0)*(p6(2, 0) - p3(2, 0)) - z3(2, 0)*(p6(1, 0) - p3(1, 0)),
                                -z3(0, 0)*(p6(2, 0) - p3(2, 0)) + z3(2, 0)*(p6(0, 0) - p3(0, 0)),
                                z3(0, 0)*(p6(1, 0) - p3(1, 0)) - z3(1, 0)*(p6(0, 0) - p3(0, 0)),
                                z3(0, 0), z3(1, 0), z3(2, 0);
        MatrixXd J5(6, 1); J5 << z4(1, 0)*(p6(2, 0) - p4(2, 0)) - z4(2, 0)*(p6(1, 0) - p4(1, 0)),
                                -z4(0, 0)*(p6(2, 0) - p4(2, 0)) + z4(2, 0)*(p6(0, 0) - p4(0, 0)),
                                z4(0, 0)*(p6(1, 0) - p4(1, 0)) - z4(1, 0)*(p6(0, 0) - p4(0, 0)),
                                z4(0, 0), z4(1, 0), z4(2, 0);
        MatrixXd J6(6, 1); J6 << z5(1, 0)*(p6(2, 0) - p5(2, 0)) - z5(2, 0)*(p6(1, 0) - p5(1, 0)),
                                -z5(0, 0)*(p6(2, 0) - p5(2, 0)) + z5(2, 0)*(p6(0, 0) - p5(0, 0)),
                                z5(0, 0)*(p6(1, 0) - p5(1, 0)) - z5(1, 0)*(p6(0, 0) - p5(0, 0)),
                                z5(0, 0), z5(1, 0), z5(2, 0);


        MatrixXd Jg(6, 6); Jg << J1, J2, J3, J4, J5, J6; // Geometric Jacobian
        MatrixXd Tphi(6, 6); Tphi << 1, 0, 0, 0, 0, 0,
                                    0, 1, 0, 0, 0, 0,
                                    0, 0, 1, 0, 0, 0,
                                    0, 0, 0, 0, -sin(phi_euler), cos(phi_euler)*sin(theta_euler),
                                    0, 0, 0, 0, cos(phi_euler), sin(phi_euler)*sin(theta_euler),
                                    0, 0, 0, 1, 0, cos(theta_euler); // from eq 3.64 in Siciliano

        MatrixXd Ja(6, 6); Ja << Tphi.inverse()*Jg; // Analytical Jacobian

        // set internal variables here
        ee_transf = T06;

        /*
        std::stringstream sstream;
        sstream << "KUKAStateHandler: In setRobotJointConfig, ee_transf: ";
        for (int j = 0; j < ee_transf.cols(); j++){
            for (int i = 0; i < ee_transf.rows(); i++){
                sstream << ee_transf(i, j) << " ";
            }
            sstream << ";";
        }
        std::string temp_string = sstream.str();
        ROS_INFO(temp_string.c_str());
        */

        inv_Ja = Ja.inverse();

        for (int i = 0; i < njoints-1; i++){
            qcurr(i,0) = joint_config[i];
        }
        return;
    }

    std::vector<double> computeDesiredJointConfig(std::vector<double> xyz_rpy_new){
        MatrixXd xyz_delta(6, 1); xyz_delta << 0, 0, 0, 0, 0, 0;
        MatrixXd qc(6, 1); //qc << 0, 0, 0, 0, 0, 0;
        MatrixXd qnew(6, 1);

        //ROS_INFO("BEFORE getEndEffectorPoseVectorXYZRPY, near 199\n");
        //std::vector<double> xyz_rpy_current = getEndEffectorPoseVectorXYZRPY();

        std::vector<double> xyz_zyz_current = getEndEffectorPoseVectorXYZZYZ();
        //ROS_INFO("AFTER getEndEffectorPoseVectorXYZRPY, near 213\n");

        // Get the difference between current and next pose for XYZ.
        for (int i = 0; i < 3; i++)
        {
            xyz_delta(i) = xyz_rpy_new[i] - xyz_zyz_current[i];
        }
        //ROS_INFO("BEFORE ignore_orientation, near 220\n");


        //xyz_delta(3) = M_PI_2 - xyz_rpy_current[3]; // seems ok but current value is noisy when updated via sensors!
        //xyz_delta(4) = M_PI_2 - xyz_rpy_current[4];
        //xyz_delta(5) = 0.0 - xyz_rpy_current[5];

        // forcing to zero cleans up noise but orientation starts to drift over time
        //xyz_delta(3) = 0.0; //M_PI_2 - xyz_rpy_current[3];
        //xyz_delta(4) = 0.0; //M_PI_2 - xyz_rpy_current[4];
        //xyz_delta(5) = 0.0; // - xyz_rpy_current[5];
        
        // still getting non-zero angle changes even with flag set to true! Why isn't it working?
        
        if (ignore_orientation)
        {

            xyz_delta(3) = M_PI_2 - xyz_zyz_current[3];
            xyz_delta(4) = M_PI_2 - xyz_zyz_current[4];
            xyz_delta(5) = 0.0 - xyz_zyz_current[5];

            //xyz_delta(3) = M_PI_2 - xyz_rpy_current[3];
            //xyz_delta(4) = M_PI_2 - xyz_rpy_current[4];
            //xyz_delta(5) = M_PI - xyz_rpy_current[5];

            //xyz_delta(3) = 0.0; //M_PI_2 - xyz_rpy_current[3];
            //xyz_delta(4) = 0.0; //M_PI_2 - xyz_rpy_current[4];
            //xyz_delta(5) = 0.0; // - xyz_rpy_current[5];
        } 
        else {
            xyz_delta(3) = xyz_rpy_new[3] - xyz_zyz_current[3];
            xyz_delta(4) = xyz_rpy_new[4] - xyz_zyz_current[4];
            xyz_delta(5) = xyz_rpy_new[5] - xyz_zyz_current[5];
        }

        //ROS_INFO("BEFORE qc, near 234\n");
        //qc << inv_Ja * xyz_delta;
        qc = inv_Ja * xyz_delta; // = is probably the better operator here. maybe doesn't matter?
        //ROS_INFO("BEFORE qnew << qc + qcurr;, near 237\n");
        qnew << qc + qcurr;

        std::vector<double> new_joints;
        for (int i = 0; i < njoints-1; i++)
        {
            new_joints.push_back(qnew(i));
        }
        // Set last joint to the constant -55 degrees
        new_joints.push_back(-0.958709);
        //ROS_INFO("BEFORE return, near 246\n");
        return new_joints;
    }

        std::vector<double> getEndEffectorPoseVectorXYZRPY(){
        //ROS_INFO("In getEndEffectorPoseVectorXYZRPY, near 189\n");
        std::vector<double> xyz_rpy;
        for (int i = 0; i < 3; i++){
            xyz_rpy.push_back(ee_transf(i, 3));
        }
        //ROS_INFO("In getEndEffectorPoseVectorXYZRPY, near 189\n");
        double phi_euler = atan2(ee_transf(1, 0), ee_transf(0, 0));
        double theta_euler = atan2(-ee_transf(2, 0), sqrt(pow(ee_transf(2, 1), 2) + pow(ee_transf(2, 2), 2)));
        double psi_euler = atan2(ee_transf(2, 1), ee_transf(2, 2));
        //ROS_INFO("In getEndEffectorPoseVectorXYZRPY, near 193\n");
        xyz_rpy.push_back(phi_euler);
        xyz_rpy.push_back(theta_euler);
        xyz_rpy.push_back(psi_euler);
        return xyz_rpy;
    }

    std::vector<double> getEndEffectorPoseVectorXYZZYZ(){
        //ROS_INFO("In getEndEffectorPoseVectorXYZRPY, near 189\n");
        std::vector<double> xyz_rpy;
        for (int i = 0; i < 3; i++){
            xyz_rpy.push_back(ee_transf(i, 3));
        }
        //ROS_INFO("In getEndEffectorPoseVectorXYZRPY, near 189\n");
        double phi_euler = atan2(ee_transf(1, 2), ee_transf(0, 2));
        double theta_euler = atan2(sqrt(pow(ee_transf(1, 2), 2) + pow(ee_transf(0, 2), 2)), ee_transf(2, 2));
        double psi_euler = atan2(ee_transf(2, 1), -ee_transf(2, 0));
        //ROS_INFO("In getEndEffectorPoseVectorXYZRPY, near 193\n");
        xyz_rpy.push_back(phi_euler);
        xyz_rpy.push_back(theta_euler);
        xyz_rpy.push_back(psi_euler);
        return xyz_rpy;
    }

    bool setIgnoreOrientation(bool desired_flag){
        ignore_orientation = desired_flag;
        return true;
    }
    
    bool ignore_orientation;

    private:
    int njoints;
    MatrixXd alpha;
    MatrixXd a;
    MatrixXd d;
    MatrixXd theta;
    MatrixXd qcurr; 

    MatrixXd ee_transf; // absolute transformation from origin to 'main' end effector pose. Should be pose to joint 6 (T06 in old code)
    MatrixXd inv_Ja;
    
};


/* // WIP, don't worry about it, might not even use it
class AbstractStateFilter{
    public:
    AbstractStateFilter(){};

    private:
    int dim;
    double dt;
    std::vector<double> state_est;
    std::vector<double> dstate_est;

    void predict_(double dt){
        return; // does nothing in abstract class
    };
    void update_(double dt, std::vector<double> state_measured){
        return; // does nothing in abstract class
    }
}
*/

// state filtering/observer class to test if it would help the IK class's estimate of the joint state.
class AlphaBetaStateFilter {
    public:
    AlphaBetaStateFilter(){};
    AlphaBetaStateFilter(int state_dim, double delta_time, double alpha, double beta){
        dim = state_dim;
        dt = delta_time;

        alpha = alpha;
        beta = beta;
        beta_dt = beta / delta_time; // this is updated automatically in update just in case dt changes, mostly want to allocate memory for it just once

        // initalize state_est and dstate_est with zeros if no initial conditions given
        for (int i = 0; i < dim; i++){
            state_est.push_back(0.0);
            dstate_est.push_back(0.0);
            residual.push_back(0.0);
        }
    }

    AlphaBetaStateFilter(int state_dim, double delta_time, double alpha, double beta, std::vector<double> state){
        dim = state_dim;
        dt = delta_time;

        alpha = alpha;
        beta = beta;
        beta_dt = beta / delta_time;

        // initalize state_est and dstate_est with zeros if no initial conditions given
        for (int i = 0; i < dim; i++){
            state_est.push_back(state[i]);
            dstate_est.push_back(0.0);
            residual.push_back(0.0);
        }
    }

    AlphaBetaStateFilter(int state_dim, double delta_time, double alpha, double beta, std::vector<double> state, std::vector<double> dstate){
        dim = state_dim;
        dt = delta_time;

        alpha = alpha;
        beta = beta;
        beta_dt = beta / delta_time;

        // initalize state_est and dstate_est with zeros if no initial conditions given
        for (int i = 0; i < dim; i++){
            state_est.push_back(state[i]);
            dstate_est.push_back(dstate[i]);
            residual.push_back(0.0);
        }
    }

    void updateStateEstimate(std::vector<double> measured_state){
        predict_(dt);
        update_(dt, measured_state);
        return;
    }

    void updateStateEstimate(std::vector<double> measured_state, double delta_time){
        predict_(delta_time); 
        update_(delta_time, measured_state);
        return;
    }

    std::vector<double> getStateEstimate(){
        std::vector<double> current_state_est = state_est; // double check this is a deep copy
        return current_state_est;
    }

    std::vector<double> getUpdatedStateEstimate(std::vector<double> measured_state){
        updateStateEstimate(measured_state);
        return getStateEstimate();
    }

    std::vector<double> getUpdatedStateEstimate(std::vector<double> measured_state, double delta_time){
        updateStateEstimate(measured_state, delta_time);
        return getStateEstimate();
    }

    private:
    int dim;
    double dt;
    double alpha;
    double beta;
    double beta_dt;

    std::vector<double> state_est;
    std::vector<double> dstate_est;
    std::vector<double> residual;

    void predict_(double dt){
        for (int i = 0; i < dim; i ++){
            state_est[i] = state_est[i] + dt * dstate_est[i];
        }
        return; 
    };
    void update_(double dt, std::vector<double> measured_state){
        beta_dt = beta / dt;
        for (int i = 0; i < dim; i ++){
            residual[i] = measured_state[i] - state_est[i];
            state_est[i] = state_est[i] + alpha * residual[i];
            dstate_est[i] = dstate_est[i] + beta_dt * residual[i];
        }
        return;
    }
};



class ikClass
{
    /*
        Class for IK node: Subscribes to the joint_states and new_pose. Publishes next_joint_angles.
        Creating two nodehandles and two threads to run the subscribers. 
        Joint_states subscriber callback needs to trigger only if new_pose callback is triggered.
    */
    private:
        std_msgs::Float64MultiArray new_joint_angles;
        std::vector<double> measured_joint_angles;
        std::vector<double> joint_angles;
        std::vector<double> old_joint_angles;
        std::vector<double> new_pose;
        std::vector<double> desired_pose_vec;
        int freq;

        // two extremely simple helper methods for making sure we don't parse safety-related config parameters incorrectly
        std::vector<double> convertDegToRad_(std::vector<double> deg_vec){
            std::vector<double> rad_vec;
            for (int i = 0; i < deg_vec.size(); i++){
                rad_vec.push_back(deg_vec[i] * M_PI / 180.0);
            }
            return rad_vec;
        };
        double convertDegToRad_(double deg_val){
            double rad_val = deg_val * M_PI / 180.0;
            return rad_val;
        };

    public:
        ros::NodeHandle nh_j, nh_p, nh; // current joint angle subscriber and desired pose subscriber node handles. Why are they given node handles? They shouldn't need them.
        ros::Publisher pub_ja; // joint angle publisher
        ros::Subscriber joint_sub, pose_sub;

        ikClass(int hz);
        ~ikClass();
        void setupSubandPub(string topic_pub, string topic1_sub, string topic2_sub);
        void jointCallback(const sensor_msgs::JointState::ConstPtr& msg);
        void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg); 
        void main_loop(ros::CallbackQueue& queue_joint, ros::CallbackQueue& queue_pose);
        MatrixXd applyJointSafetyCheck(MatrixXd old_joints, MatrixXd new_joints);
        std::vector<double> applyJointSafetyCheck(std::vector<double> old_joints, std::vector<double> new_joints);
        bool triggerPublishing(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
        bool togglePublishing(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);
        bool triggerSyncWithMeasuredJoint(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
        std::vector<double> getFilteredJointAngles();
        bool getEndEffectorPose(conf_exps::GetCurrentPoseVector::Request &req, conf_exps::GetCurrentPoseVector::Response &res);
        bool setIKBehaviorMode(conf_exps::SetInt::Request &req, conf_exps::SetInt::Response &res);
        bool changeIKBehaviorMode(int desired_behavior_mode);
        bool triggerIgnoreOrientation(std_srvs::Trigger::Request & req, std_srvs::Trigger::Response & res);
        bool toggleIgnoreOrientation(std_srvs::SetBool::Request & req, std_srvs::SetBool::Response & res);
        void saveJointsRow();
        bool triggerSaveJoints(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
        void computeAndPublishDesiredJointAngles();

        std::vector<double> joint_bounds_lower;
        std::vector<double> joint_bounds_upper;
        double joint_bounds_vel;
        int joint_num;
        bool use_sampled_joints_for_ik_guess;
        ros::ServiceServer srv_trigger_publishing;
        ros::ServiceServer srv_toggle_publishing;
        ros::ServiceServer srv_trigger_sync;
        ros::ServiceServer srv_get_pose;
        ros::ServiceServer srv_set_ik_behavior;
        ros::ServiceServer srv_ik_ignore_orientation;
        ros::ServiceServer srv_ik_ignore_orientation_toggle;
        ros::ServiceServer srv_trigger_save_joints;

        bool bflag_publish_output;
        int ik_behavior_mode;
        //std::ofstream *joint_file;
        std::vector<std::vector<double>> saved_joints;
        bool bflag_save_joints;
        bool bflag_joint_callback_done;
        bool bflag_pose_callback_done;

        KUKAStateHandler kuka;
        AlphaBetaStateFilter state_filter;
};

ikClass::ikClass(int hz)
{ 
    nh = ros::NodeHandle("ik");
    freq = hz;

    nh.param<int>("joint_num", joint_num, 7);

    for (int i = 0; i < joint_num; i++) {
        joint_bounds_lower.push_back(-120.0);
        joint_bounds_upper.push_back(120.0);
    } // default joint bounds vector
    

    nh.param<std::vector<double>>("joint_bounds_lower_deg", joint_bounds_lower, joint_bounds_lower);
    nh.param<std::vector<double>>("joint_bounds_upper_deg", joint_bounds_upper, joint_bounds_upper);
    nh.param<double>("joint_bounds_velocity_deg", joint_bounds_vel, 45.0);

    std::vector<double> alpha_vec = {1.5708, -1.5708, -1.5708, 1.5708, 1.5708, -1.5708, 0.0};
    std::vector<double> a_vec = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //std::vector<double> d_vec = {0.36, 0.0, 0.42, 0.0, 0.4, 0.0, 0.081001}; // old had last value of 0.126
    std::vector<double> d_vec = {0.36, 0.0, 0.42, 0.0, 0.4, 0.0, 0.126}; // old had last value of 0.126

    nh.param<std::vector<double>>("dh_alpha", alpha_vec, alpha_vec);
    nh.param<std::vector<double>>("dh_a", a_vec, a_vec);
    nh.param<std::vector<double>>("dh_d", d_vec, d_vec);

    std::cout << "In ikClass constructor, before KUKAStateHandler creation" << std::endl;
    kuka = KUKAStateHandler(alpha_vec, a_vec, d_vec, true); // initalizes kuka object used to handle fk/ik updates
    kuka.setIgnoreOrientation(true);

    // copied from old c++ code. not ideal.
    old_joint_angles.push_back(-1.5708);
    old_joint_angles.push_back(1.5708);
    old_joint_angles.push_back(0.0);
    old_joint_angles.push_back(1.5708);
    old_joint_angles.push_back(0.0);
    old_joint_angles.push_back(-1.5708);
    old_joint_angles.push_back(-0.958709);
    
    nh.param<std::vector<double>>("initial_joint_angles", old_joint_angles, old_joint_angles);
    nh.param<bool>("use_sampled_joints_for_ik_guess", use_sampled_joints_for_ik_guess, true);

    std::cout << "In ikClass constructor, before setRobotJointConfig call" << std::endl;
    kuka.setRobotJointConfig(old_joint_angles); // recomputes everything to assume we're starting at the initail joint angle. might cause issue with gazebo!
    for (int i = 0; i < old_joint_angles.size(); i++){ measured_joint_angles.push_back(old_joint_angles[i]);}



    joint_bounds_lower = convertDegToRad_(joint_bounds_lower);
    joint_bounds_upper = convertDegToRad_(joint_bounds_upper);
    joint_bounds_vel = convertDegToRad_(joint_bounds_vel);

    srv_trigger_publishing = nh.advertiseService("/ik/trigger_publishing", &ikClass::triggerPublishing, this);
    srv_toggle_publishing = nh.advertiseService("/ik/toggle_publishing", &ikClass::togglePublishing, this);
    srv_trigger_sync = nh.advertiseService("/ik/trigger_sync_with_measured_joint", &ikClass::triggerSyncWithMeasuredJoint, this);
    srv_get_pose = nh.advertiseService("/ik/get_current_pose", &ikClass::getEndEffectorPose, this);
    srv_set_ik_behavior = nh.advertiseService("/ik/set_ik_behavior", &ikClass::setIKBehaviorMode, this);
    srv_ik_ignore_orientation = nh.advertiseService("/ik/trigger_ignore_orientation", &ikClass::triggerIgnoreOrientation, this);
    srv_ik_ignore_orientation_toggle = nh.advertiseService("/ik/toggle_ignore_orientation", &ikClass::toggleIgnoreOrientation, this);
    srv_trigger_save_joints = nh.advertiseService("/ik/trigger_save_joints", &ikClass::triggerSaveJoints, this);

    bflag_publish_output = false;
    ik_behavior_mode = 0;
    bflag_save_joints = false;

    // placeholder code, just for testing the filter settings quickly
    int filter_state_dim = 7;
    double filter_delta_time = 0.005;
    double filter_alpha = 0.5;
    double filter_beta = 0.5;
    std::vector<double> filter_initial_state = old_joint_angles;
    state_filter = AlphaBetaStateFilter(filter_state_dim, filter_delta_time, filter_alpha, filter_beta, filter_initial_state);

    bflag_joint_callback_done = false;
    bflag_pose_callback_done = false;
}

ikClass::~ikClass(){
}

void ikClass::setupSubandPub(string topic_pub, string topic1_sub, string topic2_sub){
    pub_ja = nh_p.advertise<std_msgs::Float64MultiArray>(topic_pub, 1);
    joint_sub = nh_j.subscribe(topic1_sub, 1, &ikClass::jointCallback, this);
    pose_sub = nh_p.subscribe(topic2_sub, 1, &ikClass::poseCallback, this);
}

bool ikClass::toggleIgnoreOrientation(std_srvs::SetBool::Request & req, std_srvs::SetBool::Response & res){
    bool flag = req.data;

    kuka.setIgnoreOrientation(flag);

    res.message = "Ignore orientation set to " + std::to_string(flag) + ".";
    res.success = true;

    return res.success;
}

bool ikClass::triggerIgnoreOrientation(std_srvs::Trigger::Request & req, std_srvs::Trigger::Response & res){
    res.success = false;
    if (kuka.ignore_orientation == true){
        kuka.setIgnoreOrientation(false);
        res.message = "Ignore orientation set to false.";
        res.success = true;
    } else if (kuka.ignore_orientation == false){
        kuka.setIgnoreOrientation(true);
        res.message = "Ignore orientation set to true.";
        res.success = true;
    } else{
        res.message = "!!!Error!!! kuka.ignore_orientation is in an unknown state!";
    }
    return res.success;
}

bool ikClass::setIKBehaviorMode(conf_exps::SetInt::Request &req, conf_exps::SetInt::Response &res){
    // swtiches ik behavior based on req.data (int) where
    //ik_behavior_mode = req.data;
    int desired_behavior_mode = (int) req.data;
    
    bool succ = changeIKBehaviorMode(desired_behavior_mode);
    std::string set_type;
    if (desired_behavior_mode == 0){
        set_type = "Sensor Only Mode";
    } 
    else if (desired_behavior_mode == 1){
        set_type = "Open Loop Mode";
    }
    else if (desired_behavior_mode == 2){
        set_type = "Filtered Mode";
    } 
    else{
        set_type = "UNKNOWN";
    }

    if (succ){
        res.success = true;
        res.message = "IK behavior mode set to " + set_type + "!";
    } else{
        res.success = false;
        res.message = "!!!Error!!! IK behavior mode NOT set to " + set_type + " correctly!";
    }

    return succ;
}

bool ikClass::changeIKBehaviorMode(int desired_behavior_mode){
    // very boring function, only here to make sure we have a place to do more complex behavior state changes if needed
    if (desired_behavior_mode == 0){
        // change to 'closed loop' behavior
        ik_behavior_mode = desired_behavior_mode;
        return true;
    } 
    else if (desired_behavior_mode == 1){
        // change to 'open loop' behavior
        ik_behavior_mode = desired_behavior_mode;
        return true;
    } 
    else if (desired_behavior_mode == 2){
        // change to 'filtered/mixed' behavior
        ik_behavior_mode = desired_behavior_mode;
        return true;
    } 
    else{
        return false;
    }
}

std::vector<double> ikClass::getFilteredJointAngles(){
    std::vector<double> est_joint_angles;
    if (ik_behavior_mode == 0){
        est_joint_angles = measured_joint_angles;
    }
    else if (ik_behavior_mode == 1){
        est_joint_angles = old_joint_angles;
    }
    else if (ik_behavior_mode == 2){
        est_joint_angles = state_filter.getUpdatedStateEstimate(measured_joint_angles);
        // PLACEHOLDER BEHAVIOR, USE CLASS
        /*
        double residual;
        for (int i = 0; i < measured_joint_angles.size(); i++){
            residual = measured_joint_angles[i] - old_joint_angles[i];
            est_joint_angles.push_back(old_joint_angles[i] + 0.5 * residual);
        }
        */
    }
    else {
        est_joint_angles = measured_joint_angles; // if there's some error just keep it with measured
    }
    return est_joint_angles;
}


void ikClass::jointCallback(const sensor_msgs::JointState::ConstPtr& msg){
    measured_joint_angles = msg->position;

    std::vector<double> set_joint_vec(joint_num, 0.0); // (size of vector, initial value for all elements)
    set_joint_vec = getFilteredJointAngles();
    
    // do some type of safety checking here if needed! mutex guard!
    joint_angles.clear();
    joint_angles = set_joint_vec;

    bflag_joint_callback_done = true; // should be faster than checking if it's false then setting it to true.

    return;
}

/*
void ikClass::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
        
    geometry_msgs::Pose new_desired_pose = msg->pose;

    // mutex guard here!
    desired_pose_vec.clear();
    desired_pose_vec = {new_desired_pose.position.x, new_desired_pose.position.y, new_desired_pose.position.z, 0.0, 0.0, 0.0};

    bflag_pose_callback_done = true;
    // cout << "IK loop took" << elapsed_seconds.count() << "s\n";
    return;
}
*/

/*
void ikClass::poseCallback(const conf_exps::AdmitStateStamped::ConstPtr& msg){
        
    geometry_msgs::Pose new_desired_pose = msg->pose;

    // mutex guard here!
    desired_pose_vec.clear();
    desired_pose_vec = {new_desired_pose.position.x, new_desired_pose.position.y, 
        new_desired_pose.position.z, 0.0, 0.0, 0.0};

    bflag_pose_callback_done = true;
    return;
}
*/

void ikClass::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
        
    geometry_msgs::Pose new_desired_pose = msg->pose;

    // mutex guard here!
    desired_pose_vec.clear();
    desired_pose_vec = {new_desired_pose.position.x, new_desired_pose.position.y, 
        new_desired_pose.position.z, 0.0, 0.0, 0.0};

    bflag_pose_callback_done = true;
    return;
}

void ikClass::computeAndPublishDesiredJointAngles()
{
    // updates joint-dependent params like end effector pose and jacobians using most up-to-date joint_angles from jointCallback
    kuka.setRobotJointConfig(joint_angles);

    // computes next joint vector based on jacobians and desired end effector pose using most up-to-date desired_pose_vec from poseCallback
    std::vector<double> desired_joint_vec = kuka.computeDesiredJointConfig(desired_pose_vec);

    // add safety limits check
    //desired_joint_vec = applyJointSafetyCheck(old_joint_angles, desired_joint_vec);

    // clears old joint values
    new_joint_angles.data.clear(); 
    old_joint_angles.clear();

    // updates 
    for (int i = 0; i < joint_num; i++)
    {
        new_joint_angles.data.push_back(desired_joint_vec[i]);
        old_joint_angles.push_back(desired_joint_vec[i]); // used in next callback
    }
    //new_joint_angles.tag = msg->tag;
    if (bflag_publish_output){
    pub_ja.publish(new_joint_angles);
    }
    return;
}

/*
void ikClass::jointCallback(const sensor_msgs::JointState::ConstPtr& msg){
    measured_joint_angles = msg->position;

    std::vector<double> set_joint_vec(joint_num, 0.0); // (size of vector, initial value for all elements)
    set_joint_vec = getFilteredJointAngles();
    
    // do some type of safety checking here if needed!
    joint_angles = set_joint_vec;

    // updates joint-dependent params like end effector pose and jacobians
    kuka.setRobotJointConfig(joint_angles);
    return;
}

void ikClass::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){

    auto pre = chrono::steady_clock::now();
        
    geometry_msgs::Pose new_desired_pose = msg->pose;
    std::vector<double> desired_pose_vec{new_desired_pose.position.x, new_desired_pose.position.y, new_desired_pose.position.z, 0.0, 0.0, 0.0};


    // computes next joint vector based on jacobians and desired end effector pose
    std::vector<double> desired_joint_vec = kuka.computeDesiredJointConfig(desired_pose_vec);

    // add safety limits check
    //desired_joint_vec = applyJointSafetyCheck(old_joint_angles, desired_joint_vec);

    // clears old joint values
    new_joint_angles.data.clear(); 
    old_joint_angles.clear();

    // updates 
    for (int i = 0; i < joint_num; i++)
    {
        new_joint_angles.data.push_back(desired_joint_vec[i]);
        old_joint_angles.push_back(desired_joint_vec[i]); // used in next callback
    }
    //new_joint_angles.tag = msg->tag;
    if (bflag_publish_output){
    pub_ja.publish(new_joint_angles);
    }

    auto now = chrono::steady_clock::now();
    chrono::duration<double> elapsed_seconds = now-pre;
    // cout << "IK loop took" << elapsed_seconds.count() << "s\n";
    return;
}
*/

MatrixXd ikClass::applyJointSafetyCheck(MatrixXd old_joints, MatrixXd new_joints){
    // joint velocity check
    double temp_joint_vel = 0.0; // probably shouldn't allocate new memory in a callback
    double dt = 1.0 / (double) freq;
    for (int i = 0; i << old_joints.rows(); i++){
        temp_joint_vel = abs(new_joints(i) - old_joints(i)) / dt;
        if (joint_bounds_vel < temp_joint_vel){
            new_joints(i) = old_joints(i);
        }
    }
    
    // joint position check
    for (int i = 0; i << new_joints.rows(); i++){
        // if lower than the lower bound, set to lower bound
        if (new_joints(i) < joint_bounds_lower[i]){
            new_joints(i) = joint_bounds_lower[i];
        }
        // if higher than upper bound, set to upper bound
        if (joint_bounds_upper[i] < new_joints(i)){
            new_joints(i) = joint_bounds_upper[i];
        }
    }
    return new_joints;
}

vector<double> ikClass::applyJointSafetyCheck(vector<double> old_joints, vector<double> new_joints){
    // joint velocity check
    double temp_joint_vel = 0.0;
    double dt = 1.0 / (double) freq;
    for (int i = 0; i << old_joints.size(); i++){
        temp_joint_vel = abs(new_joints[i] - old_joints[i]) / dt;
        // if estimated joint velocity is too high, goto last joint value
        if (joint_bounds_vel < temp_joint_vel){
            new_joints[i] = old_joints[i];
        }
    }
    
    // joint position check
    for (int i = 0; i << new_joints.size(); i++){
        // if lower than the lower bound, set to lower bound
        if (new_joints[i] < joint_bounds_lower[i]){
            new_joints[i] = joint_bounds_lower[i];
        }
        // if higher than upper bound, set to upper bound
        if (joint_bounds_upper[i] < new_joints[i]){
            new_joints[i] = joint_bounds_upper[i];
        }
    }
    return new_joints;
}

// unused
void ikClass::main_loop(ros::CallbackQueue& queue_joint, ros::CallbackQueue& queue_pose){
    /*
    ros::Rate loop_rate(freq);
    //ROS_INFO("Inverse Kinematics Node is ready.");
    // auto now = chrono::steady_clock::now(); 
    // auto prev = now - chrono::seconds(1/freq);
    while (ros::ok())
    {   
        // auto t_pre = chrono::steady_clock::now();
        // auto t_end = t_pre + chrono::seconds(1/freq);

        //ROS_INFO("Triggering possible joint_states calls");
        queue_joint.callAvailable(ros::WallDuration());     // Trigger joint_states topic callback
        queue_pose.callAvailable(ros::WallDuration());     // Trigger new_pose topic callback

        // auto now = chrono::steady_clock::now();
        // cout << "IK loop took" << now - prev << "s\n";
        // prev = now;

        // std::this_thread::sleep_until(t_end);
        loop_rate.sleep();
    }
    */
}

bool ikClass::togglePublishing(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res){
    bflag_publish_output = req.data;

    res.success = true;
    res.message = "ikClass publishing set to " + std::to_string(bflag_publish_output) + "!!!";
    return res.success;

    //out_string = "ikClass publishing set to " + std::to_string(bflag_publish_output) + "!!!";
    //res.message = out_string;
    //return res.success;
    /*
    bflag_publish_output = req.data;
    res.success = true;
    std::string out_string;
    out_string = "ikClass publishing set to " + std::to_string(bflag_publish_output) + "!!!";
    res.message = out_string;
    return res.success;
    */
}

bool ikClass::triggerPublishing(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
    //bflag_publish_output = req.data;
    if (bflag_publish_output){
        bflag_publish_output = false;
    }
    else{
        bflag_publish_output = true;
    }
    res.success = true;
    res.message = "ikClass publishing set to " + std::to_string(bflag_publish_output) + "!!!";
    return res.success;

    //out_string = "ikClass publishing set to " + std::to_string(bflag_publish_output) + "!!!";
    //res.message = out_string;
    //return res.success;
    /*
    bflag_publish_output = req.data;
    res.success = true;
    std::string out_string;
    out_string = "ikClass publishing set to " + std::to_string(bflag_publish_output) + "!!!";
    res.message = out_string;
    return res.success;
    */
}

bool ikClass::triggerSyncWithMeasuredJoint(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
    // get measured joint vector
    old_joint_angles.clear();
    joint_angles.clear();
    for (int i = 0; i < measured_joint_angles.size(); i++){
        old_joint_angles.push_back(measured_joint_angles[i]);
        joint_angles.push_back(measured_joint_angles[i]);
    }

    // pass it to kuka
    kuka.setRobotJointConfig(measured_joint_angles);
    res.success = true;

    std::stringstream sstream;
    sstream << "ikClass: Joint angles synced to measured joint of ";
    for (int i = 0; i < measured_joint_angles.size(); i++){
        sstream << measured_joint_angles[i];
        if (i < measured_joint_angles.size()-1) {sstream << ", ";}
    }
    sstream << "." << std::endl;
    std::string temp_string = sstream.str();
    ROS_INFO(temp_string.c_str());

    //std::string out_string;
    //out_string = "Joint angles synced to measured!!!";
    res.message = temp_string;
    return res.success;
}

bool ikClass::getEndEffectorPose(conf_exps::GetCurrentPoseVector::Request &req, conf_exps::GetCurrentPoseVector::Response &res){
    res.xyzrpy_pose.clear();
    std::vector<double> xyzrpy = kuka.getEndEffectorPoseVectorXYZRPY();
    for (int i = 0; i < xyzrpy.size(); i++){
        res.xyzrpy_pose.push_back(xyzrpy[i]);
    }

    std::stringstream sstream;
    sstream << "ikClass: Sending end effector pose of ";
    for (int i = 0; i < xyzrpy.size(); i++){
        sstream << xyzrpy[i];
        if (i < measured_joint_angles.size()-1) {sstream << ", ";}
    }
    sstream << "." << std::endl;
    std::string temp_string = sstream.str();
    ROS_INFO(temp_string.c_str());
    return true;
}

bool ikClass::triggerSaveJoints(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
    if (bflag_save_joints){
        // if on, turn it off and close the file.
        bflag_save_joints = false;

        res.message = "Set to close file and stop recording!";

        std::string file_name = "ik_mode_" + std::to_string(ik_behavior_mode) + "_rec.csv";

        std::ofstream joint_file(file_name);

        // define the header row
        for (int i = 0; i < measured_joint_angles.size(); i++){
            joint_file << "measured_joint_" << std::to_string(i) << ",";
        }
        for (int i = 0; i < old_joint_angles.size(); i++){
            joint_file << "old_joint" << std::to_string(i) << ",";
        }
        for (int i = 0; i < joint_angles.size(); i++){
            joint_file << "estimated_joint_" << std::to_string(i) << ",";
        }
        joint_file << "\n";
        
        for (int row = 0; row < saved_joints.size(); row++){
            std::vector<double> joints_row = saved_joints[row];
            for (int col = 0; col < joints_row.size(); col ++){
                joint_file << std::to_string(joints_row[col]) << ",";
            }
            joint_file << "\n";
        }
        joint_file.close();
    } 
    else {
        // if off, turn it on and open the file object
        bflag_save_joints = true;
        saved_joints.clear();
        res.message = "Set to open file and start recording!";
    }

    res.success = true;

    return true;
}

void ikClass::saveJointsRow(){
    if (bflag_save_joints){
        std::vector<double> joints_row;
        for (int i = 0; i < measured_joint_angles.size(); i++){
            joints_row.push_back(measured_joint_angles[i]);
        }
        for (int i = 0; i < old_joint_angles.size(); i++){
            joints_row.push_back(old_joint_angles[i]);
        }
        for (int i = 0; i < joint_angles.size(); i++){
            joints_row.push_back(joint_angles[i]);
        }
    saved_joints.push_back(joints_row);
    }
    //ROS_INFO("ikClass: Saving joint row! Should have %d rows", saved_joints.size());
    return;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "kuka_ik_node");
    
    int hz = 500;
    std::string logstring;
    if (ros::param::get("/ik/rate", hz)){
        logstring = "IK node set to rate of " + std::to_string(hz) + "Hz.";
    }
    else{
        logstring = "/ik/rate not found! Set to default value of 500Hz.";
    }
    ROS_INFO(logstring.c_str());

    // Declare multiple queues to control the order in which callbacks are called
    ros::CallbackQueue queue_joint, queue_pose;

    ikClass node_obj = ikClass(hz);

    node_obj.setupSubandPub("PositionController/command", "joint_states", "admit_state");

    ros::Rate loop_rate(hz);
    //ROS_INFO("Inverse Kinematics Node is ready.");
    while (ros::ok())
    {   
        //auto timer_start = std::chrono::steady_clock::now();
        
        // if we've recieved inputs and ready, start computing and publishing
        if ((node_obj.bflag_joint_callback_done) && (node_obj.bflag_pose_callback_done)){
            node_obj.computeAndPublishDesiredJointAngles();
            if (node_obj.bflag_save_joints){
                node_obj.saveJointsRow();
            }
        }

        ros::spinOnce();
        loop_rate.sleep();

        //std::chrono::duration<double, std::milli> timer_duration = std::chrono::steady_clock::now() - timer_start;
        //double measured_loop_time = timer_duration.count() * (1.0 / sec_to_millisec);
    }

    return 0;
}
