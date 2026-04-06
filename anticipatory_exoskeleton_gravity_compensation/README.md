# Anticipatory Exoskeleton Gravity Compensation

## Overview
This ROS1 Noetic workspace contains the code used for anticipatory exoskeleton gravity compensation experiment. It includes ROS packages for variable admittance control, intention classification, user motion prediction and confidence evaluation, GUI support, experiment protocol management, and custom message definitions.

## Dependencies
- ROS1 Noetic
- Python: numpy, scipy, pandas, numba, statsmodels, scikit-learn, pyqtgraph, PyQt5, h5py
- External: 
    - The serial C++ library from http://wjwwood.github.io/serial/. Clone the repo into the high level folder and the ROS nodes should be able to reference it.
    - The dynamixel_sdk C++ library for communicating with dynamixel MX-106 motors. Clone into this folder similar to serial. See (http://wiki.ros.org/dynamixel_sdk) for more details.
- Hardware Interface:
    - Also requires a hardware_interface package not included in this repo due to it containing propritary code we cannot share publically. None of this code is relevant to the Method itself, but is entirely related to interfacing with Axia80 Torque/Force sensors, EMG sensors, and dynamixel motors for controlling the exoskeleton. Contact the authors for more details if you need to fully replicate this section.

## Usage
### Launching the System
- Full pre-experiment dependices on startup: `roslaunch protocol pipeline.launch` (use `roslaunch protocol pipeline_no_hardware.launch` to run without the hardware_interface package)
- Full experiment dependices on startup: `roslaunch protocol protocol.launch`

## Nodes and Scripts
- **admit/scripts/admit_controller.py**: Admittance control node implementing anticipatory and gravity compensation behaviors
- **robot_policy/scripts/antigrav_policy.py**: Anticipitory gravity compensation logic based on CVH-CaVA policy designed for exposkeleton assistance
- **classification/scripts/classifier.py**: Motion intent classification node
- **gui/scripts/GravityCompGui.py**: PyQt-based graphical interface for subjects and experiment monitoring
- **gui/scripts/info_collector.py**: Information collection utilities for subject/session data
- **protocol/scripts/protocol_controller.py**: High-level experiment state and protocol management node
- **protocol/scripts/session_generator.py**: Trial/session generation utilities

## Configuration
Configuration files are located in each package `config/` directory.
- `admit/config/`: Admittance and controller parameters
- `classification/config/`: Classifier and model settings
- `gui/config/`: GUI display and layout settings
- `protocol/config/`: Protocol and session generation settings
- `predict_eval/config/`: Prediction and evaluation parameters

## Authors
The authors of this code are John Atkins and Seunghoon Hwang.
Thanks to Dongjune Chang for contributions to the hardare_interface package, which is not included for the reasons stated above.

## License
Apache License 2.0
