Header header
geometry_msgs/Pose pose # end effector pose
geometry_msgs/Twist twist # end effector twist (6d velocity)
geometry_msgs/Accel accel # end effector acceleration
geometry_msgs/Wrench wrench # end effector applied forces
geometry_msgs/Wrench robot_wrench # robot's assistance force
EquilibriumState eq_state
float64 stiffness
float64 vel_stiffness
float64 damping
float64 user_intent
float64 confidence
float64 cycle_time

