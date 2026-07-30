# landerpi-ros-stm32

This repository stores the LanderPi robot software source:

- `ros2/lander_pi`: ROS 2 workspace/source code for the Raspberry Pi side.
- `stm32/RosRobotControllerLite_ros_250811`: STM32 controller firmware source.

Large generated files are intentionally excluded, including ROS build/install/log
directories, bag/database recordings, compiled STM32 outputs, and precompiled
binary libraries. Rebuild those artifacts locally from the corresponding ROS 2
workspace or STM32 project when needed.

## Notes

- ROS odometry-related code includes the encoder odometry changes for
  `/encoder_odom` and `/wheel_encoder/state`.
- The STM32 directory is kept as firmware source plus project configuration,
  without compiler output files.
