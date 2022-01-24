#!/bin/bash

source gazebo_plugin/devel/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
roslaunch agv_ppot turtlebot3_AGV_PPot.launch
