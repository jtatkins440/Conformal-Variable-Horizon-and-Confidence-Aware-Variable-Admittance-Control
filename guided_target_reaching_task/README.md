# Guided Target-Reaching Task

## Overview
This ROS1 Melodic package provides the code used to impliment the Guided Target-Reaching Task (Experiment 1) described in the paper Conformal Variable-Horizon and Confidence-Aware Variable Admittance Control for Physical Human-Robot Interaction.

## Core Python Modules
- **admit_lib.py**: Python module that implements core admittance control and motion prediction functionality. Includes neural network models for motion intention prediction, JIT-compiled admittance controllers, impedance policies, variable admittance update rules for UIVAC and CaVA, adaptive conformal prediction helper, and model inference wrappers for trajectory prediction.
- **utils.py**: Python module providing JIT-compiled conformal prediction fuctnions and utility functions for data processing. Includes ROS message packing/unpacking, timing and waiting functions, safety bounding for vectors and scalars, live filtering classes, curvature computation helpers, conformal prediction algorithms, and miscellaneous helper routines used throughout the project.

## Dependencies
- ROS1 Melodic
- C++: Eigen library (place header in /include folder)
- Python: numpy, scipy, numba, PyQt5, PyTorch, hdf5 for Python, and sklearn.
- ROS packages: smach

## Usage
### Launching the System
- Pre-experiment pipeline for startup: `roslaunch conf_exps exp_pipeline.launch`
- State machine that runs experiment protocol: `roslaunch conf_exps exp_sm.launch`

## Configuration
Configuration files are located in the `config/` directory:
- `admit_config.yaml`: Variable admittance control parameters
- `mint_conf_config.yaml`: Motion intention prediction (mint for short), adaptive conformal inference, and confidence calulcation parameters.
- `ik_config.yaml`: Inverse kinematics settings
- `sm_config.yaml`: State machine configuration
- `logger_config.yaml`: Data logging options

Note that all the key hyperparameters needed for replicating the experiment as described in the Methods section are stored in 'admit_config.yaml' and 'mint_conf_config.yaml'. Other config files are related to implimentation details and are not required for understanding the method.

## Nodes
- **admit.py**: Python node for variable admittance control. Impliments the passive admittance control (PAC), User-Intention Variable Admittance Control (UIVAC), and Confidence-aware Variable Admittance Control (CaVA). Uses 'admit_config.yaml' for defining settings.
- **mint_conf.py**: Node for handling the motion intention prediction, adaptive conformal inference, and confidence calulcation based on settings from 'mint_conf_config.yaml'.
- **quantile_server.py**: Conformal set processing server.
- **exp_state_machine.py**: Experiment state machine for implimenting experimental protocol
- **general_logger.py**: Data logging node to hdf5 files
- **gui_node_pyqt.py**: PyQT-based visual GUI shown to human subjects during testing
- **guide_node.py**: Guidance node for publishing the current reference point to the gui_node
- **ik_node**: C++ node for Jacobian-based constrained inverse kinematics for the KUKA iiwa-14. 

## Authors
John Atkins is the main author of this code. 
Thanks to Divya Prakash for contributions to the GUI and protocol mananger in a previous project that were edited for use in this project.

## License
Apache License 2.0