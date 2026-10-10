geometry_msgs/Point position
geometry_msgs/Vector3 velocity
float64 t_eq
float64 q_ratio

geometry_msgs/Point reference_position # where the eq_state is relative to, or (x_eq - x_ref)
geometry_msgs/Vector3 reference_velocity

geometry_msgs/Point diff_position # just the x_eq - x_ref vector. 
geometry_msgs/Vector3 diff_velocity
