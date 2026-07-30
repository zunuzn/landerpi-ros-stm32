import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription, LaunchService
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context):
    mode = LaunchConfiguration('mode', default=1)
    mode_arg = DeclareLaunchArgument('mode', default_value=mode)

    app_package_path = get_package_share_directory('app')
    large_models_package_path = get_package_share_directory('large_models')

    interruption = LaunchConfiguration('interruption', default=False)
    interruption_arg = DeclareLaunchArgument('interruption', default_value=interruption)

    camera_topic = LaunchConfiguration('camera_topic', default='/ascamera/camera_publisher/rgb0/image')
    camera_topic_arg = DeclareLaunchArgument('camera_topic', default_value=camera_topic)

    line_following_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(app_package_path, 'launch/line_following_node.launch.py')),
        launch_arguments={
            'debug': 'true',
        }.items(),
    )

    large_models_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(large_models_package_path, 'launch/start.launch.py')),
        launch_arguments={'mode': mode,'camera_topic': camera_topic}.items(),
    )

    llm_visual_patrol_node = Node(
        package='large_models_examples',
        executable='llm_visual_patrol',
        output='screen',
        parameters=[
        {
            'interruption': interruption,
        }],
    )

    return [mode_arg,
            camera_topic_arg,
            interruption_arg,
            line_following_node_launch,
            large_models_launch,
            llm_visual_patrol_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # Create a LaunchDescription object(创建一个LaunchDescription对象)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
