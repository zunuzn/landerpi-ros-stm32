#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <imu_calib/accel_calib.h>

namespace imu_calib
{

class ApplyCalib : public rclcpp::Node
{
public:
  ApplyCalib();

private:
  AccelCalib calib_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr raw_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr corrected_pub_;

  void rawImuCallback(sensor_msgs::msg::Imu::SharedPtr raw);

  bool calibrate_gyros_;
  bool calibrate_accels_;
  int gyro_calib_samples_;
  int accel_calib_samples_;
  int gyro_sample_count_;
  int accel_sample_count_;

  double gyro_bias_x_;
  double gyro_bias_y_;
  double gyro_bias_z_;

  double accel_bias_x_;
  double accel_bias_y_;
  double accel_bias_z_;
  double accel_mean_x_;
  double accel_mean_y_;
  double accel_mean_z_;
  double accel_reference_;
  double accel_calib_max_range_;
  double accel_min_norm_;
  double accel_max_norm_;
};

} // namespace imu_calib
