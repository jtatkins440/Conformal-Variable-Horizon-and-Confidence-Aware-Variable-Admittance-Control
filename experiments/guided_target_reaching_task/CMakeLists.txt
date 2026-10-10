cmake_minimum_required(VERSION 3.0.2)
project(conf_exps)

set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch/share/cmake/Torch)
set(iir_DIR /usr/local/lib/cmake/iir)
#set(iir_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/iir/build)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
add_compile_options(-std=c++17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  tf
  tf2
  message_generation
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Torch REQUIRED) # for loading the mintnet into c++
find_package(iir REQUIRED) # for filtering the inputs to the mintnet


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   FullFitIO.msg
   AdmitStateStamped.msg
   IKmsg.msg
   ICPCompact.msg
   RecordStateStamped.msg
   ACISample.msg
   ACICompact.msg
   AdmitRecordStateStamped.msg
   MIntConfRecordStateStamped.msg
   FullAdmitState.msg
   EquilibriumState.msg
   GuideState.msg
#   Message2.msg
 )

## Generate services in the 'srv' folder
add_service_files(
	FILES
	SetInt.srv
	inverse_kinematics.srv
	forward_kinematics.srv
	GetCurrentPoseVector.srv
	UpdateTargets.srv
	InitLogger.srv
	StartLogging.srv
	StopLogging.srv
	GetQuantiles.srv
	GetStepSizes.srv
	SetFloat.srv
	GetUserIntentBounds.srv
	#   Service2.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
 )


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES motion_intention
  CATKIN_DEPENDS roscpp rospy std_msgs tf2_ros
  DEPENDS system_lib message_runtime
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)


add_executable(ik_node src/ik_node.cpp)
add_dependencies(ik_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ik_node ${catkin_LIBRARIES})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
   scripts/gui_node_pyqt.py
   scripts/general_logger.py
   scripts/utils.py
   scripts/quantile_server.py
   scripts/exp_state_machine.py
   scripts/mint_conf.py
   scripts/admit.py
   scripts/guide_node.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )


## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)
