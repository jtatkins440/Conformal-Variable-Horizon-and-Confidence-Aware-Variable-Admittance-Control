Header header
geometry_msgs/Pose pose # end effector pose
geometry_msgs/Twist twist # end effector twist (6d velocity)
geometry_msgs/Accel accel # end effector acceleration
geometry_msgs/Wrench wrench # end effector applied forces
geometry_msgs/Wrench robot_wrench # robot's assistance force
geometry_msgs/Pose pose_eq # end effector equilibrium pose
geometry_msgs/Twist twist_eq # end effector equilibrium vel
float64 t_eq
float64 stiffness
float64 damping
float64 user_intent
float64 used_confidence
float64 icp_confidence
float64 err_confidence
ICPCompact icp_traj
float64 cycle_time
