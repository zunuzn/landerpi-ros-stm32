#!/usr/bin/env python3
# encoding: utf-8
# @Author: Gcusms
# @Date: 2025/10/21
import os
import re
import cv2
import textwrap
import ast
import time
import json
import math
import rclpy
import threading
from speech import speech
from rclpy.node import Node

import numpy as np
import queue
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from geometry_msgs.msg import Twist

from std_msgs.msg import String, Bool, Float32
from std_srvs.srv import Trigger, SetBool, Empty
from large_models_msgs.msg import Tools
from large_models.config import *
from large_models_msgs.srv import SetModel, SetString, SetTools

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from interfaces.srv import SetPose2D,SetPoint
from interfaces.srv import SetString as SetColor
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# 机械臂
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position
from servo_controller.action_group_controller import ActionGroupController

# sdk 相关工具
from sdk import common
from sdk.pid import PID

# 物体追踪
from large_models_examples.track_anything import ObjectTracker


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_pixel_distance",
            "description": "获取一个或多个指定位置像素的深度距离(Get the depth distance of one or more specified positions)",
            "parameters": {
                "type": "object",
                "properties": {
                    "pixel_position": {
                        "type": "string",
                        "description": "包含一个或多个像素坐标列表的JSON字符串。例如：'[[100, 120], [350, 400]](The JSON string containing a list of one or more pixel coordinates. For example: '[[100, 120], [350, 400])'",
                    }
                },
                "required": ["pixel_position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_obejct_pixel",
            "description": "如果不是特定需要，默认不进行调用，识别物体，并获取他的像素位置，一般需要获取目标物体深度前调用。(If not for specific needs, it is not called by default. Recognize objects and get their pixel positions. Normally called before getting the target object.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "用户的问题(User's question)",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_locations",
            "description": "查询并列出机器人可以导航前往的所有预定义地点的位置列表。(Query and list the location of all pre-defined places that the robot can navigate to.)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_location",
            "description": "查询并获取机器人当前在地图中的精确位置坐标和朝向。(Query and get the exact position coordinates and orientation of the robot in the map.)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_current_view",
            "description": "详细描述机器人当前看到的画面内容，以回答用户提出的具体问题。(Describe the detailed contents of the scene that the robot sees currently, to answer the specific question that the user asks.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "关于当前画面的具体问题，例如'大门关了吗？'或'有什么东西？(A specific question about the current scene, for example 'Is the door closed?', or 'What is there?')'"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {   "type": "function",
        "function": {
            "name": "move_to_location",
            "description": "将机器人移动到指定位置。这是一个耗时操作，当函数成功返回时，表示机器人已经到达目的地。(Move the robot to a specified location. This is an expensive operation, and when the function returns successfully, it indicates that the robot has reached its destination.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "要去的目标地点名称。必须是预定义地点之一或者临时记录的位置(The name of the destination. Must be one of the predefined locations, or a temporary record of a location)",
                        "enum": ["垃圾回收站(Garbage Station)", "超市(Supermarket)", "花园(Supermarket)", "快递站(Express Station)", "家(Home)","起点(Starting Point)"]
                    }
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "line_following",
            "description": "命令机器人沿着指定颜色的线进行巡线。(Command the robot to follow a line of a specified color.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "要巡线的颜色。(red, green, blue)",
                        "enum": ["red", "green", "blue", "black", "yellow"]
                    }
                },
                "required": ["color"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lidar_scan_detect",
            "description": "巡线时默认必须调用的函数，辅助机器人避开障碍物(The function must be called by default when robot is following a line, it helps the robot avoid obstacles)",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_detect": {
                        "type": "string",
                        "description": "障碍物检测结果。(There is an obstacle, No obstacle)",
                    }
                },
                "required": ["scan_detect"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "color_track",
            "description": "命令机器人追踪指定颜色的物体,注意:只有在清楚指明特定的颜色的时候再进行调用(Command robot to track object of specified color)",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "要追踪的物体的颜色(Color of object to track)",
                        "enum": ["red", "green", "blue", "black", "yellow"]
                    }
                },
                "required": ["color"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "robot_move_control",
            "description": "以指定的线速度和角速度，控制机器人移动一段特定的时间。这是一个基础移动指令。(Move the robot with specified line speed and angular speed for a specific time. This is a basic movement instruction.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "linear_x": {
                        "type": "number",
                        "description": "X轴方向的线速度（前进/后退），单位m/s，范围[-1.0, 1.0]。用户没有说明，默认速度linear_x = 0.2m/s(The linear speed in X-axis direction (forward/backward), unit m/s, range [-1.0, 1.0]. User does not specify the default speed linear_x = 0.2m/s.)"
                    },
                    "linear_y": {
                        "type": "number",
                        "description": "Y轴方向的线速度（左/右平移），单位m/s，范围[-1.0, 1.0]。用户没有说明，默认速度 linear_y = 0.2m/s(The linear speed in Y-axis direction (left/right), unit m/s, range [-1.0, 1.0]. User does not specify the default speed linear_y = 0.2m/s.)"
                    },
                    "angular_z": {
                        "type": "number",
                        "description": "Z轴的角速度（原地左转/右转），单位rad/s，范围[-1.0, 1.0]。用户没有说明，默认速度 angular_z = 1.0m/s(The angular speed in Z-axis direction (turn left/right), unit rad/s, range [-1.0, 1.0]. User does not specify the default speed angular_z = 1.0m/s.)",
                    },
                    "duration": {
                        "type": "number",
                        "description": "移动的持续时间，单位为秒(s),。用户没有说明，默认时间为 duration = 2.0s(The duration of the movement, unit is second(s). User does not specify the default time duration = 2.0s.)"
                    }
                },
                "required": ["linear_x", "linear_y", "angular_z", "duration"]
            }
        }
    },


    {
        "type": "function",
        "function": {
            "name": "obj_track_box_detect",
            "description": "获取目标对应在图像上的像素位置，并且用方框进行框选和定位，一般在追踪非指定颜色的物体功能之前进行调用，用于追踪定位(Get the target position in pixels of the image, and use box to select and locate it. Usually called before tracking function, used for tracking location)",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "需要追踪非指定颜色的物体的需求(The demand for tracking objects that are not specified color)",
                    }
                },
                "required": ["user_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "object_track",
            "description": "命令机器人追踪目标物体，注意：只有是确定非指定颜色的物体追踪时才进行调用。需要获取目标在图像上的像素方框位置(Order the robot to track the target object, note: only when it is sure to track the object to call. Need to get the pixel square position of the target on the graph line.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "box": {
                        "type": "string",
                        "description": "目标物体的方框(The square of the target object)",
                    }
                },
                "required": ["box"]
            }
        }
    },

    
    {
        "type": "function",
        "function": {
            "name": "obj_pick_box_detect",
            "description": "获取目标对应在图像上的像素位置，并且用方框进行框选和定位，一般在物体夹取功能之前进行调用，用于物品的夹取（不带颜色的(Get the target pixel position in the image, and use box to select and locate it, usually called before the object grasping function, used for item grasping (without color))",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "需要追踪非指定颜色的物体的需求(The demand for tracking objects that are not specified color)",
                    }
                },
                "required": ["user_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "object_pick_function",
            "description": "专门用于拾取垃圾或者夹取垃圾木块的函数,需要获取目标在图线上的像素方框位置(A function dedicated to picking up garbage or holding the wooden block of garbage, need to get the target's pixel box location on the picture)",
            "parameters": {
                "type": "object",
                "properties": {
                    "box": {
                        "type": "string",
                        "description": "目标物体的方框(The box of the target object)",
                    }
                },
                "required": ["box"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "object_place_function",
            "description": "放置垃圾或者放置垃圾木块的函数,无特殊说明的话，一般调用和使用前会先进行夹取垃圾相关的动作(Function to put garbage or garbage block, no special mention, it will do the related action before using)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "color_pick",
            "description": "拾取不同颜色的物体，比如说蓝色的方块，红色的包裹的函数，如果有颜色的夹取需求，可以直接进行调用，无需确认物体具体位置(Pick a different colored object, such as blue cube, red package. If there is color picking requirement, you can directly call this function without specifying the position of the object.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "pick_color": {
                        "type": "string",
                        "description": "需要夹取对应颜色的物体(The color of the object that needs to be picked)",
                        "enum": ["red", "green", "blue", "black", "yellow"]
                    }
                },
                "required": ["pick_color"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "color_place",
            "description": "放置不同颜色的东西，比如说放下红色的包裹，蓝色的方块，无特殊说明的话，一般调用的时候，使用前会先进行夹取相关的动作(Place the red package, blue cube, etc.)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
]


#xy,rpy,m,deg
position_dict = {
    "垃圾回收站(Garbage Station)": [0.65, 0.23, 0.0, 0.0, -90.0],
    "超市(Supermarket)": [1.2, 0.7, 0.0, 0.0, 170.0],
    "花园(Garden)": [0.1, 0.1, 0.0, 0.0, -170.0],
    "快递站(Express Station)": [-0.2, 0.8, 0.0, 0.0, 130.0],
    "家(Home)": [0.0, 0.0, 0.0, 0.0, -95.0],
    "起点(Starting Point)": [0.0, 0.0, 0.0, 0.0, 0.0],
}

if os.environ.get('ASR_LANGUAGE') == 'English':
    content_string = textwrap.dedent("""
        # Role setting
        You are a real interactive robot, need to execute tasks according to user instructions,
        and interact with users in a friendly way, just like chatting with friends.
        ## Workflow
        1. **Task Planning:** Before starting a task, you need to break it down and plan it. 
        The steps will be presented in a numbered format, with each number representing an independent step.
        2. **Tool Usage:** Before each tool is used, you need to provide an explanation, no more than 20 words, 
        describing the feedback in a humorous and varied way to make the communication process more engaging.
        3. **Feedback Processing:** After the tool is used, you need to follow up with a commentary on the feedback results, 
        no more than 20 words, describing the feedback in a humorous and varied way to make the communication process more engaging.
        4. **Task Completion:** After all task steps have been completed, provide a summary explanation, no more than 40 words.
    """) 

    PROMPT = '''
    As an image recognition expert, your capability is to accurately locate objects in images sent by users through object detection, and output the final results according to the "Output Format".
    ## 1. Understand User Instructions
    I will give you a sentence. You need to make the best decision based on my words and extract the "object name" from the decision. **The name corresponding to the object must be in English**, **do not output objects that are not mentioned**.
    ## 2. Understand the Image
    I will give you an image. Analyze the image and identify all recognizable objects within it.
    For each identified object, calculate the center point coordinates of the object. **Do not output objects that are not mentioned**.
    【Special Note】: Deeply understand the positional relationships of objects.
    ## Output Format (Please only output the following content, do not say any extra words)
    [
    {
    "object": name_1,
    "center_xy": [center_x_1, center_y_1]
    },
    {
    "object": name_2,
    "center_xy": [center_x_2, center_y_2]
    }
    ]
    '''

    OBJ_TRACK_PROMPT = '''
    As an intelligent vehicle, skilled in image recognition, your capability is to accurately locate objects in images sent by users through object detection, output the final results according to the "Output Format", and then perform tracking.
    ## 1. Understand User Instructions
    I will give you a sentence. You need to extract the "object name" from my words. **The name corresponding to the object must be in English**, **do not output objects that are not mentioned**.
    ## 2. Understand the Image
    I will give you an image. From this image, find the pixel coordinates of the top-left and bottom-right corners of the object corresponding to the "object name". If not found, then xyxy should be []. **Do not output objects that are not mentioned**.
    【Special Note】: Deeply understand the positional relationships of objects. The response needs to combine the user's instruction and the detection results.
    ## Output Format (Please only output the following content, do not say any extra words)
    {
        "object": "name", 
        "xyxy": [xmin, ymin, xmax, ymax]
    }
    '''
else:

    content_string = textwrap.dedent("""
        # 角色设定
        你是一个风趣幽默的机器人助手，用第一人称与用户亲切交流，就像和朋友聊天一样自然。
        
        # 核心规则（必须遵守）
        1. 在执行任何工具调用前，必须先给我一段简短风趣的提示文字
        2. 工具调用完成后，必须给我一段简短风趣的结果说明
        3. 所有给我的文字回复都要保持轻松有趣的风格
        4. 如果任务没有需要执行工具，同样的也要用一小段文字返回
        
        # 工作流程
        1. **任务规划**：先简要说明你的行动计划（20-30字），风格要风趣幽默
        2. **调用工具**：每次调用工具前必须给我提示（15-25字），说明你要做什么
        3. **处理反馈**：工具执行后必须给我结果说明（15-25字），分享进展或趣事
        4. **任务总结**：完成后进行风趣总结（30-50字）
        
        # 重要提醒
        - 每次工具调用前后都必须给我文字回复
        - 保持对话的连贯性和趣味性
    """)

    PROMPT = '''
    你作为图像识别专家，你的能力是将用户发来的图片进行目标检测精准定位，并按「输出格式」进行最后结果的输出。
    ## 1. 理解用户指令
    我会给你一句话，你需要根据我的话做出最佳决策，从做出的决策中提取「物体名称」, **object对应的name要用英文表示**, **不要输出没有提及到的物体**
    ## 2. 理解图片
    我会给你一张图, 分析图片，找出其中所有可识别的物体。
    对于每一个被识别出的物体，并计算出物体的中心点坐标,**不要输出没有提及到的物体**
    【特别注意】： 要深刻理解物体的方位关系
    ## 输出格式（请仅输出以下内容，不要说任何多余的话)
    [
    {
    "object": name_1,
    "center_xy": [center_x_1, center_y_1]
    },
    {
    "object": name_2,
    "center_xy": [center_x_2, center_y_2]
    }
    ]
    '''

    OBJ_TRACK_PROMPT = '''
    你作为智能车，善于图像识别，你的能力是将用户发来的图片进行目标检测精准定位，并按「输出格式」进行最后结果的输出，然后进行跟随。
    ## 1. 理解用户指令
    我会给你一句话，你需要根据我的话中提取「物体名称」。 **object对应的name要用英文表示**, **不要输出没有提及到的物体**
    ## 2. 理解图片
    我会给你一张图, 从这张图中找到「物体名称」对应物体的左上角和右下角的像素坐标; 如果没有找到，那xyxy为[]。**不要输出没有提及到的物体**
    【特别注意】： 要深刻理解物体的方位关系, response需要结合用户指令和检测的结果进行回答
    ## 输出格式（请仅输出以下内容，不要说任何多余的话)
    {
        "object": "name", 
        "xyxy": [xmin, ymin, xmax, ymax]
    }
    '''


class LogColors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'

class LLMControlMove(Node):
    def __init__(self, name):
        super().__init__(name)
        
        self.initialize_variables() # 初始化变量(init variables)
        self.setup_ros_components() # 设置ROS节点(init node)
        self.setup_services_and_clients() # 设置ROS服务客户端(init services and clients)
        self.setup_subs_and_pubs() # 设置ROS订阅者与发布者(init subs and pubs)
        self.setup_timers() # 设置ROS定时器(init timers)

    def initialize_variables(self):
        """初始化所有类变量(init all class variables)"""
        self.language = os.environ.get('ASR_LANGUAGE')
        self.tools = []
        self.vllm_result = ''
        self.current_pose = None
        self.saved_pose = None
        self.obstacle_detected = False
        self.bridge = CvBridge()

        self.action = []
        self.llm_result = ''
        self.running = True
        self.interrupt = False
        self.action_finish = False
        self.play_audio_finish = False
        self.is_task_running = False
        self.reach_goal = False

        self.machine_type = os.environ.get('MACHINE_TYPE', '')
        self.camera_type = os.environ.get('DEPTH_CAMERA_TYPE', '')
        self.llm_model = 'qwen3-max'
        self.vllm_model = 'qwen-vl-max-latest'
        if self.language == 'English':
            self.llm_model = 'gpt-4o-mini'
            self.vllm_model = 'gpt-4o'
        self.cb_group = ReentrantCallbackGroup()

        # 巡线相关变量(line_following)
        self.line_following_count = 0
        self.line_following_start = False
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0

        # 普通物体分类相关变量(nomal_object_classification)
        self.obj_move_finish = False
        self.obj_pick_start = False
        self.obj_place_start = False
        self.obj_place_finish = False

        # 颜色拾取追踪相关变量(color_tracking)
        self.lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")
        self.image_proc_size = (320, 240)
        self.start_color_pick = False
        self.start_place = False
        self.pick_stop_x = 335
        self.pick_stop_y = 246
        self.place_stop_x = 345
        self.place_stop_y = 220

        self.yaw_pid = PID(P=0.005, I=0, D=0.000)
        self.linear_pid = PID(P=0.0003, I=0, D=0)
        self.angular_pid = PID(P=0.001, I=0, D=0)

        self.color_transport_stop = False
        self.linear_base_speed = 0.007
        self.angular_base_speed = 0.03
        self.stop_single = False
        self.pick = False
        self.place = False
        self.status = "approach"
        
        # 颜色追踪(color_track)
        self.color_track_status = False

        # 线程安全锁(lock)
        self.draw_lock = threading.Lock()
        self.color_detect_box = None
        self.color_detect_center = None
        self.draw_flag = False

        # 物体追踪相关变量(track)
        self.start_track = False
        self.track_box_p1 = None
        self.track_box_p2 = None
        self.object_detect_box = False
        self.object_mode = ''
        self.get_box_flag = False
        self.box = None
        self.box_count = 0

        # 图像队列(queue)
        self.image_pair_queue = queue.Queue(maxsize=2)

    def setup_ros_components(self):
        """设置ROS2组件(Setup ROS2 components)"""
        # 物体追踪器(ObjectTracker)
        self.track = ObjectTracker(use_mouse=False, automatic=True, log=self.get_logger())
        
        # PID参数(PID parameters)
        self.pid_params = {
            'kp1': 0.01, 'ki1': 0.0, 'kd1': 0.00,
            'kp2': 0.002, 'ki2': 0.0, 'kd2': 0.0,
        }
        
        for param_name, default_value in self.pid_params.items():
            self.declare_parameter(param_name, default_value)
            self.pid_params[param_name] = self.get_parameter(param_name).value

        self.track.update_pid(
            [self.pid_params['kp1'], self.pid_params['ki1'], self.pid_params['kd1']],
            [self.pid_params['kp2'], self.pid_params['ki2'], self.pid_params['kd2']]
        )

        # 物体夹取(Object Picking)
        self.obj_pick_center_x = 320
        self.obj_pick_center_y = 300

    def setup_services_and_clients(self):
        """设置服务和服务客户端(Set Services and Clients)"""
        # LLM相关客户端(LLM-related clients)
        self.client = speech.OpenAIAPI(api_key, base_url)
        self.set_tool_client = self.create_client(SetTools, 'agent_process/set_tool')
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        
        # 导航客户端(Navigation clients)
        self.set_pose_client = self.create_client(SetPose2D, 'navigation_controller/set_pose')
        
        # 巡线客户端(Line following clients)
        self.line_follower_enter_client = self.create_client(Trigger, 'line_following/enter')
        self.line_follower_exit_client = self.create_client(Trigger, 'line_following/exit')
        self.line_follower_start_client = self.create_client(SetBool, 'line_following/set_running')
        self.line_follower_set_target_client = self.create_client(SetColor, 'line_following/set_color')

        # 颜色追踪客户端(Color tracking clients)
        self.object_tracker_enter_client = self.create_client(Trigger, 'object_tracking/enter')
        self.object_tracker_start_client = self.create_client(SetBool, 'object_tracking/set_running')
        self.object_tracker_set_target_client = self.create_client(SetColor, 'object_tracking/set_color')
        
        # 语音唤醒客户端(Voice wakeup clients)
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')

    def setup_subs_and_pubs(self):
        """设置订阅者和发布者(Setup subscribers and publishers)"""
        # LLM相关(LLM related)
        self.create_subscription(Tools, 'agent_process/tools', self.tools_callback, 1, callback_group=self.cb_group)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.tools_result_pub = self.create_publisher(Tools, 'agent_process/tools_result', 1)

        # 导航相关(Navigation related)
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos_profile.reliability = QoSReliabilityPolicy.RELIABLE
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, qos_profile)
        self.create_subscription(Bool, 'navigation_controller/reach_goal', self.reach_goal_callback, 1, callback_group=self.cb_group)

        # 语音相关(Voice related)
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1, callback_group=self.cb_group)
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=self.cb_group)

        # 底盘控制(Chassis control)
        self.cmd_vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)

        # 巡线相关(Line following)
        self.cmd_vel_subscription = self.create_subscription(Twist,'/controller/cmd_vel',self.cmd_vel_callback,10)

        # 机械臂(Robot arm)
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        self.controller = ActionGroupController(
            self.joints_pub, 
            '/home/ubuntu/software/arm_pc/ActionGroups'
        )


        # 服务(Service)
        # self.create_service(Trigger, '~/joy_up', self.get_joy_up)
        self.create_service(Trigger, '~/garb_pick', self.garb_pick)
        self.create_service(Trigger, '~/garb_place', self.garb_place)

        # 图像发布(Image publisher)
        self.result_image_publisher = self.create_publisher(Image, '~/result_image', 10)

        # 图像同步(Image synchronizer)
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        image_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 3, 0.02)
        ts.registerCallback(self.image_sync_callback)

    def setup_timers(self):
        """设置定时器(Timer)"""
        self.timer = self.create_timer(0.0, self.init_process, callback_group=self.cb_group)

    def _wait_for_services(self, timeout_sec=5.0):
        """等待关键服务就绪(Wait for critical services to be available)"""
        services = [
            (self.set_model_client, 'set_model'),
            (self.set_prompt_client, 'set_prompt'),
            (self.set_tool_client, 'set_tool'),
            (self.line_follower_enter_client, 'line_follower_enter'),
            (self.line_follower_set_target_client, 'line_follower_set_target'),
            (self.line_follower_start_client, 'line_follower_start'),
            (self.object_tracker_enter_client, 'object_tracker_enter'),
            (self.object_tracker_start_client, 'object_tracker_start'),
            (self.object_tracker_set_target_client, 'object_tracker_set_target')
        ]
        
        for client, name in services:
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().warn(f'Service {name} not available after {timeout_sec} seconds')

    def amcl_pose_callback(self, msg):
        """处理AMCL位姿信息(Slove the pose information from AMCL)"""
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation

        # 四元数转欧拉角(Convert quaternion to euler angle)
        t3 = +2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        t4 = +1.0 - 2.0 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        yaw_z = math.atan2(t3, t4)
        yaw_deg = math.degrees(yaw_z)

        self.current_pose = {
            "x": position.x,
            "y": position.y,
            "yaw_degrees": yaw_deg
        }

    def reach_goal_callback(self, msg):
        """到达目标回调(Arrived at goal)"""
        self.get_logger().info('Reached goal')
        self.reach_goal = msg.data

    def llm_result_callback(self, msg):
        """LLM结果回调(LLM result)"""
        self.llm_result = msg.data
        self.get_logger().info(f'{LogColors.YELLOW}{LogColors.BOLD}LLM Reply: {self.llm_result}{LogColors.RESET}')
        
        # 非列表响应才进行语音播报(Speak out the response if it is not a list response)
        text_to_speak = self.llm_result
        is_list_response = re.search(r'^\s*\d+\.', text_to_speak, re.MULTILINE)
        if not is_list_response:
            tts_msg = String()
            tts_msg.data = text_to_speak
            self.tts_text_pub.publish(tts_msg)

    def image_sync_callback(self, ros_image, ros_depth_image):
        """同步图像回调(Callback of synchronized images)"""
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            depth_image = np.ndarray(
                shape=(ros_depth_image.height, ros_depth_image.width), 
                dtype=np.uint16, 
                buffer=ros_depth_image.data
            )

            if self.image_pair_queue.full():
                self.image_pair_queue.get()
            self.image_pair_queue.put((bgr_image, depth_image))
        except Exception as e:
            self.get_logger().error(f"Image sync error: {str(e)}")

    def get_pixel_distance(self, pixel_coords_list):
        """获取像素距离(get pixel distance)"""
        try:
            _, depth_image = self.image_pair_queue.get()
            distances = []
            
            for pixel_xy in pixel_coords_list:
                x, y = pixel_xy  
                pixel_depth = depth_image[y, x]
                if 0 < pixel_depth < 30000:
                    distance = round(float(pixel_depth / 1000), 3)
                else:
                    distance = 0.0
                distances.append(distance)
            return f"{distances}.cm"
        except Exception as e:
            self.get_logger().error(f"Get pixel distance error: {str(e)}")
            return "distance is not available"

    def get_obejct_pixel(self, user_query):
        """获取物体像素位置(x, y)，返回列表(get object pixel position (x, y), return list)"""
        try:
            rgb_image, _ = self.image_pair_queue.get()
            vllm_result_str = self.client.vllm(user_query, rgb_image, prompt=PROMPT, model=self.vllm_model)

            # 提取JSON部分(extract JSON part)
            if "```json" in vllm_result_str:
                json_part = vllm_result_str.split("```json")[1].split("```")[0]
            else:
                json_part = vllm_result_str

            detected_objects = json.loads(json_part.strip())
            return str(detected_objects)
        except Exception as e:
            self.get_logger().error(f"Get object pixel error: {str(e)}")
            return "[]"

    def move_to_location(self, destination):
        """移动到指定位置(move to a specific location)"""
        if destination not in position_dict:
            return f"移动失败：未知的目标地点 '{destination}'。(Failed to move: Unknown destination '{destination}'.)"

        self.reach_goal = False
        msg = SetPose2D.Request()
        p = position_dict[destination]
        msg.data.x = float(p[0])
        msg.data.y = float(p[1])
        msg.data.roll = p[2]
        msg.data.pitch = p[3]
        msg.data.yaw = p[4]
        
        self.send_request(self.set_pose_client, msg)
        self.get_logger().info(f"Navigation goal '{destination}' sent. Waiting for arrival...")

        # 等待到达目标
        while not self.reach_goal:
            time.sleep(0.1)

        if self.reach_goal:
            return f"已成功抵达{destination}(Success reached {destination})"
        else:
            return f"移动超时，未能到达{destination}(Failed to reach {destination})"

    def get_current_location(self):
        """获取当前位置"""
        timeout_sec = 10.0
        start_time = time.time()

        while self.current_pose is None and (time.time() - start_time) < timeout_sec:
            time.sleep(0.1)

        if self.current_pose:
            x = self.current_pose['x']
            y = self.current_pose['y']
            location_string = self.find_nearest_location(x, y, position_dict)
            return location_string
        else:
            return "抱歉，我现在还无法确定自己的位置信息。(Sorry I can't determine my location now.)"

    def find_nearest_location(self, current_x, current_y, position_dict):
        """查找最近的位置(Find the nearest location)"""
        min_distance = float('inf')
        nearest_location_name = None

        for location_name, coords in position_dict.items():
            target_x, target_y = coords[0], coords[1]
            distance = math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_location_name = location_name

        if nearest_location_name:
            if min_distance < 0.2:
                return f'我现在在{nearest_location_name}(I am at {nearest_location_name})'
            else:
                return f'我现在在{nearest_location_name}附近(I am near {nearest_location_name})'
        else:
            return '我现在暂时不知道在哪里(I am not sure where I am now)'

    def describe_current_view(self, question):
        """描述当前视图(Describe the current view)"""
        try:
            self.controller.run_action('llm_camera_up')
            time.sleep(2)
            rgb_image, _ = self.image_pair_queue.get(block=True)
            if self.language == 'Chinese':
                VLLM_PROMPT = textwrap.dedent(f"""
                作为我的机器人管家，请仔细观察摄像头捕捉到的画面，并根据以下问题给出一个简洁、人性化的回答。
                不要进行反问，字数在10到40字之间。
                问题是："{question}
            """)
                description = self.client.vllm(question, rgb_image, prompt=VLLM_PROMPT, model=self.vllm_model)
            else:
                VLLM_PROMPT = textwrap.dedent(f"""
                To be my robot butler, please observe the image captured by camera carefully. 
                Please give a concise and humanized answer to the following question. 
                Don't ask any questions back. The length of your answer should be between 10 to 40 words.
                "{question}
            """)
                description = self.client.vllm(question, rgb_image, prompt=VLLM_PROMPT, model=self.vllm_model)


            self.get_logger().info(f'{LogColors.YELLOW}{LogColors.BOLD}LLM Reply: {description}{LogColors.RESET}')

            # 直接发布到TTS进行语音播报(publish to tts)
            tts_msg = String()
            tts_msg.data = description
            self.tts_text_pub.publish(tts_msg)
            self.controller.run_action('init')

            return f"画面描述任务已成功执行。得到的结果是{description}(The description of the view has been successfully executed. The result is {description})"

        except Exception as e:
            self.get_logger().error(f"Describe current view error: {str(e)}")
            return "无法描述当前画面(Unable to describe the current view)"

    def line_following(self, color):
        """巡线功能(Line following function)"""
        self.line_follower_enter_client.call_async(Trigger.Request())
        self.line_following_start = True

        # 设置目标颜色(Set target color)
        color_msg = SetColor.Request()
        color_msg.data = color
        self.line_follower_set_target_client.call_async(color_msg)

        # 启动巡线(Start line following)
        start_msg = SetBool.Request()
        start_msg.data = True
        self.line_follower_start_client.call_async(start_msg)

        self.is_task_running = True
        return f"好的，马上开始沿着{color}线行驶。(Okay, I will follow the {color} line.)"

    def lidar_scan_detect(self, scan_detect):
        """激光雷达障碍物检测(LiDAR obstacle detection)"""
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}start lidar_scan_detect{LogColors.RESET}')
        time.sleep(2)

        stop_distance = 0.3
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0
        while self.line_following_start:
            if abs(self.current_linear_x) < 0.001 and abs(self.current_angular_z) < 0.001:
                if self.line_following_count < 500:
                    self.line_following_count += 1
                    if self.line_following_count % 20 == 0:
                        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}Detected zero velocity{LogColors.RESET}')
                else:
                    start_msg = SetBool.Request()
                    start_msg.data = False
                    self.line_follower_start_client.call_async(start_msg) 
                    self.line_following_count = 0
                    time.sleep(2)
                    self.line_follower_exit_client.call_async(Trigger.Request())
                    time.sleep(1)
                    return '检测到速度为0,已停止巡线(Detected zero velocity, stop line following)'
            else:
                self.line_following_count = 0

            time.sleep(0.01)

        if not self.line_following_start:
            return '已停止巡线(Stop line following)'
        else:
            return '巡线检测超时(Line following timeout)'


    def cmd_vel_callback(self, msg):
        """cmd_vel话题回调函数(cmd_vel topic callback function)"""
        if self.line_following_start:
            self.current_linear_x = msg.linear.x
            self.current_angular_z = msg.angular.z

    def color_track(self, color):
        """颜色追踪(Color tracking)"""
        self.object_tracker_enter_client.call_async(Trigger.Request())

        # 设置要追踪的颜色(Set color to track)
        color_msg = SetColor.Request()
        color_msg.data = color
        self.object_tracker_set_target_client.call_async(color_msg)

        # 启动追踪(Start tracking)
        start_msg = SetBool.Request()
        start_msg.data = True
        self.object_tracker_start_client.call_async(start_msg)
        
        self.is_task_running = True
        self.color_track_status = True
        return f"好的,马上开始追踪{color}的物体。(Okay, I'll start tracking {color} objects.)"


    def robot_move_control(self, linear_x, linear_y, angular_z, duration):
        """机器人移动控制(Robot move control)"""
        self.get_logger().info(f"Executing move: x={linear_x}, y={linear_y}, z={angular_z} for {duration}s")
        self.is_task_running = True

        twist_msg = Twist()
        twist_msg.linear.x = float(linear_x)
        twist_msg.linear.y = float(linear_y)
        twist_msg.angular.z = float(angular_z)

        self.cmd_vel_pub.publish(twist_msg)

        end_time = time.time() + float(duration)
        while time.time() < end_time and self.is_task_running:
            time.sleep(0.05)

        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info("Movement finished, stopping robot.")
        return f"move: x={linear_x}, y={linear_y}, z={angular_z}, duration {duration}s"

    def obj_track_box_detect(self, user_query):
        """获取物体边界框(Get object bounding box)"""
        self.get_logger().info(f"ObjTracking: {user_query}")
        time.sleep(3)
        box_message = self.box_find(user_query)
        return box_message

    def obj_pick_box_detect(self, user_query):
        """获取物体加速边界框(Get object accelerated bounding box)"""
        self.get_logger().info(f"ObjTracking: {user_query}")
        self.controller.run_action('navigation_pick_init')
        time.sleep(3)
        box_message = self.box_find(user_query)
        return box_message

    def box_find(self,user_query):
        try:
            rgb_image, _ = self.image_pair_queue.get(block=True)
            vllm_result_str = self.client.vllm(user_query, rgb_image, prompt=OBJ_TRACK_PROMPT, model=self.vllm_model)

            if "```json" in vllm_result_str:
                json_part = vllm_result_str.split("```json")[1].split("```")[0]
            else:
                json_part = vllm_result_str

            detected_objects = json.loads(json_part.strip())
            self.get_box_flag = True

            if 'xyxy' in detected_objects:
                self.box = detected_objects['xyxy']
                self.get_logger().info('Detected objects: %s' % str(self.box))
                self.is_task_running = True

            return f'已经找到了物体的位置(The object location has been found.)' + str(detected_objects)
        except Exception as e:
            self.get_logger().error(f"Get object box error: {str(e)}")
            return "could not get object box"

    def object_track(self, box):
        """物体追踪(Object Tracking)"""
        self.get_logger().info('Object Tracking: %s' % str(box))
        with self.draw_lock:
            self.draw_flag = True
            self.box = ast.literal_eval(box)
        self.object_detect_box = True
        self.object_mode = 'track'
        return 'Start Tracking'

    def object_pick_function(self,box):
        time.sleep(1)
        self.obj_move_finish = False
        self.obj_pick_start = False
        self.get_logger().info("Waiting for obj_pick_start")

        self.controller.run_action('navigation_pick_init')
        time.sleep(2)
        self.box = None
        self.object_detect_box = True

        self.object_mode = 'pick'
        with self.draw_lock:
            self.draw_flag = True
            self.box = ast.literal_eval(box)
        self.get_logger().info(f"obj_pick_start>>>>>>{self.box}")
        while not self.obj_move_finish:
            if self.obj_pick_start:
                self.get_logger().info("obj_pick_start")
                self.controller.run_action('navigation_pick')
                time.sleep(3)
                break
            time.sleep(0.01)
        self.obj_move_finish = True
        return 'pick finish'

    def object_place_function(self):
        time.sleep(1)
        self.obj_move_finish = False
        self.obj_place_start = False
        self.get_logger().info("Waiting for obj_place_start")
        time.sleep(1)
        
        timeout = time.time() + 1
        while not self.obj_move_finish and time.time() < timeout:
            if self.obj_place_start:
                self.get_logger().info("obj_place_start")
                self.controller.run_action('navigation_place')
                self.obj_place_finish = True
                time.sleep(3)
                self.obj_place_start = False
                break
            time.sleep(0.01)
            
        self.controller.run_action('init')
        self.obj_move_finish = True
        time.sleep(2)
        return 'place finish'

    def color_pick(self, color):
        """颜色拾取(Color Pick)"""
        self.get_logger().info(f"Starting to pick color: {color}")
        self.controller.run_action('navigation_pick_init')
        self.start_color_pick = True
        
        # 重置控制参数(Reset control parameters)
        self.linear_speed = 0
        self.angular_speed = 0
        self.yaw_angle = 90
        self.color_transport_stop = True
        self.d_y = 10
        self.d_x = 10
        self.pick = False
        self.place = False
        self.target_color = color
        self.count_stop = 0
        self.count_turn = 0
        self.linear_pid.clear()
        self.angular_pid.clear()
        self.start_pick = True

        timeout = time.time() + 60 
        stop_single = False
        
        while self.running and not stop_single and time.time() < timeout:
            try:
                image, _ = self.image_pair_queue.get(block=True, timeout=1)
                if self.start_color_pick:
                    self.color_transport_stop = True
                    result_image, stop_single = self.pick_handle(image)
                else:
                    if self.color_transport_stop:
                        self.color_transport_stop = False
                        self.cmd_vel_pub.publish(Twist())
                    result_image = image
                    
                # 绘制参考线(start_line, stop_line)
                cv2.line(result_image, (self.pick_stop_x, 0), (self.pick_stop_x, 480), (0, 255, 255), 2)
                cv2.line(result_image, (0, self.pick_stop_y), (640, self.pick_stop_y), (0, 255, 255), 2)
                
            except queue.Empty:
                if not self.running:
                    break
                continue

        self.cmd_vel_pub.publish(Twist())
        time.sleep(1)
        self.controller.run_action('navigation_pick')
        time.sleep(2)
        self.stop_single = False
        return "pick finish"


    def color_place(self):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start place")

        # 重置控制参数(reset control parameters)
        self.linear_speed = 0
        self.angular_speed = 0
        self.d_y = 30
        self.d_x = 30

        self.color_transport_stop = True
        self.pick = False
        self.place = False
        self.target_color = ''

        self.linear_pid.clear()
        self.angular_pid.clear()
        self.start_place = True

        # 做一个放置的动作(make a placing action)
        self.cmd_vel_pub.publish(Twist())
        time.sleep(1)
        self.controller.run_action('navigation_place')
        time.sleep(2)
        self.controller.run_action('init')
        # set_servo_position(self.joints_pub, 2, ((1, 500), (2, 700), (3, 15), (4, 215), (5, 500), (10, 200)))
        self.stop_single = False
        return f"pick finish"


    def pick_handle(self, image):
        """拾取处理(Pcick)"""
        twist = Twist()
        if not self.pick:
            object_center_x, object_center_y, object_angle, box = self.color_detect(image)

            if object_center_x > 0:
                # 线性PID控制(Linear PID control)
                self.linear_pid.SetPoint = self.pick_stop_y
                if abs(object_center_y - self.pick_stop_y) <= self.d_y:
                    object_center_y = self.pick_stop_y
                    
                if self.status != "align":
                    self.linear_pid.update(object_center_y)
                    tmp = self.linear_base_speed + self.linear_pid.output
                    self.linear_speed = max(min(tmp, 0.15), -0.15)
                    if abs(tmp) <= 0.0075:
                        self.linear_speed = 0

                # 角速度PID控制(angular velocity PID control)
                self.angular_pid.SetPoint = self.pick_stop_x
                if abs(object_center_x - self.pick_stop_x) <= self.d_x:
                    object_center_x = self.pick_stop_x
                    
                if self.status != "align":
                    self.angular_pid.update(object_center_x)
                    tmp = self.angular_base_speed + self.angular_pid.output
                    self.angular_speed = max(min(tmp, 1.2), -1.2)
                    if abs(tmp) <= 0.038:
                        self.angular_speed = 0

                # 停止条件判断(stop flag)
                if abs(self.linear_speed) == 0 and abs(self.angular_speed) == 0:
                    if 'Mecanum' in self.machine_type:
                        self.count_turn = min(self.count_turn + 1, 5)
                        if self.count_turn == 5:
                            self.status = "align"
                            if self.count_stop < 10:
                                if object_angle < 40:
                                    object_angle += 90
                                self.yaw_pid.SetPoint = 90
                                if abs(object_angle - 90) <= 1:
                                    object_angle = 90
                                self.yaw_pid.update(object_angle)
                                self.yaw_angle = self.yaw_pid.output
                                
                                if object_angle != 90:
                                    if abs(self.yaw_angle) <= 0.038:
                                        self.count_stop += 1
                                    else:
                                        self.count_stop = 0
                                    twist.linear.y = float(-2 * 0.3 * math.sin(self.yaw_angle / 2))
                                    twist.angular.z = float(self.yaw_angle)
                                else:
                                    self.count_stop += 1
                            elif self.count_stop <= 20:
                                self.d_x = 5
                                self.d_y = 5
                                self.count_stop += 1
                                self.status = "adjust"
                            else:
                                self.count_stop = 0
                                self.pick = True
                    else:
                        self.count_stop += 1
                        if self.count_stop > 15:
                            self.count_stop = 0
                            self.pick = True
                else:
                    self.count_turn = 0
                    if self.status != 'align':
                        twist.linear.x = float(self.linear_speed)
                        twist.angular.z = float(self.angular_speed)

        self.stop_single = self.pick
        with self.draw_lock:
            if not self.stop_single:
                self.color_detect_box = box
                self.color_detect_center = [object_center_x, object_center_y]
                self.draw_flag = True
            else:
                self.draw_flag = False

        self.cmd_vel_pub.publish(twist)
        return image, self.stop_single

    def color_detect(self, img):
        """颜色检测(color detection)"""
        img_h, img_w = img.shape[:2]
        frame_resize = cv2.resize(img, self.image_proc_size, interpolation=cv2.INTER_NEAREST)
        frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)
        frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

        frame_mask = cv2.inRange(frame_lab, 
                               tuple(self.lab_data['lab']['Stereo'][self.target_color]['min']),
                               tuple(self.lab_data['lab']['Stereo'][self.target_color]['max']))

        eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        center_x, center_y, angle, box = -1, -1, -1, None

        if contours:
            areaMaxContour, area_max = common.get_area_max_contour(contours, 10)
            if areaMaxContour is not None and area_max > 10:
                rect = cv2.minAreaRect(areaMaxContour)
                angle = rect[2]
                box = np.intp(cv2.boxPoints(rect))
                
                for j in range(4):
                    box[j, 0] = int(common.val_map(box[j, 0], 0, self.image_proc_size[0], 0, img_w))
                    box[j, 1] = int(common.val_map(box[j, 1], 0, self.image_proc_size[1], 0, img_h))

                cv2.drawContours(img, [box], -1, (0, 255, 255), 2)
                ptime_start_x, ptime_start_y = box[0, 0], box[0, 1]
                pt3_x, pt3_y = box[2, 0], box[2, 1]
                center_x, center_y = int((ptime_start_x + pt3_x) / 2), int((ptime_start_y + pt3_y) / 2)
                cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)

        return center_x, center_y, angle, box

    def get_available_locations(self):
        """获取可用位置(get available locations)"""
        self.get_logger().info("Querying available locations.")
        return json.dumps(position_dict, ensure_ascii=False, indent=2)

    def get_node_state(self, request, response):
        """获取节点状态(Get node state)"""
        return response

    def get_joy_up(self, request, response):
        """手柄上键回调(Joy up callback)"""
        response.success = True
        self.reach_goal = True
        self.get_logger().info(f"Reach_goal:{self.reach_goal}")
        return response

    def garb_pick(self, request, response):
        """垃圾拾取回调(Joy down callback)"""
        response.success = True
        self.obj_pick_start = True
        self.get_logger().info(f"obj_pick_start:{self.obj_pick_start}")
        return response

    def garb_place(self, request, response):
        """垃圾放置回调(Joy up callback)"""
        response.success = True
        self.obj_place_start = True
        self.get_logger().info(f"obj_place_start:{self.obj_place_start}")
        return response

    def init_process(self):
        """初始化过程(Initialization process)"""
        self.timer.cancel()
        self._wait_for_services()

        # 设置模型
        msg = SetModel.Request()
        msg.model_type = 'llm_tools'
        msg.model = self.llm_model
        msg.api_key = api_key 
        msg.base_url = base_url
        self.send_request(self.set_model_client, msg)

        # 设置提示词(Set prompt)
        msg = SetString.Request()
        msg.data = content_string
        self.send_request(self.set_prompt_client, msg)

        # 设置工具(Set tools)
        tools_json = [json.dumps(tool, ensure_ascii=False) for tool in tools]
        msg = SetTools.Request()
        msg.tools = tools_json
        self.send_request(self.set_tool_client, msg)
        
        # 初始化机械臂(Init arm)
        self.controller.run_action('init')

        # 启动处理线程(Start processing thread)
        threading.Thread(target=self.process, daemon=True).start()
        threading.Thread(target=self.display_thread, daemon=True).start()
        threading.Thread(target=self.object_track_thread, daemon=True).start()

        # 启动功能开启语音提示(Start function opening voice prompt)
        speech.play_audio(start_audio_path)
        
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def send_request(self, client, msg):
        """发送请求并等待响应(Send request and wait for response)"""
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done():
                try:
                    return future.result()
                except Exception as e:
                    self.get_logger().error(f"Service call failed: {str(e)}")
                    return None
            time.sleep(0.01)
        return None

    def tools_callback(self, msg):
        """工具回调(Tools callback)"""
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}AI Decision-Making:{LogColors.RESET}')
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}Tools id [{msg.id}]:{LogColors.RESET}')
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}Tools name [{msg.name}]:{LogColors.RESET}')
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}Tools data [{msg.data}]:{LogColors.RESET}')
        
        self.tools = [msg.id, msg.name, json.loads(msg.data)]

    def play_audio_finish_callback(self, msg):
        """音频播放完成回调(Audio playback finished callback)"""
        if msg.data:
            self.play_audio_finish = True
            awake_msg = SetBool.Request()
            awake_msg.data = True
            self.send_request(self.awake_client, awake_msg)

    def wakeup_callback(self, msg):
        """唤醒回调(Wakeup callback)"""
        if msg.data and self.is_task_running:
            self.get_logger().info('唤醒中断(Wakeup interrupt)')
            
            # 停止巡线(Stop line following)
            if self.line_following_start:
                request = SetBool.Request()
                request.data = False
                self.line_follower_start_client.call_async(request)
                self.line_following_start = False

            # 停止颜色追踪(Stop color tracking)
            if self.color_track_status:
                request = SetBool.Request()
                request.data = False
                self.object_tracker_start_client.call_async(request)
                self.color_track_status = False

            # 停止运动(Stop movement)
            self.cmd_vel_pub.publish(Twist())
            self.track.stop()
            self.is_task_running = False
            self.draw_flag = False
            self.llm_result = ''
            
        elif msg.data:
            self.get_logger().info('Wakeup received, but no interruptible task is running.')


    def process(self):
        """主处理循环(Main processing loop)"""
        while rclpy.ok():
            if self.tools:
                tool_id, tool_name, args_dict = self.tools
                res = None

                try:
                    if tool_name == 'get_obejct_pixel':
                        content = args_dict.get('content') 
                        if content:
                            res = self.get_obejct_pixel(content)
                    elif tool_name == 'get_pixel_distance':
                        position_str = args_dict.get('pixel_position')
                        if position_str:
                            positions_list = ast.literal_eval(position_str)
                            if (isinstance(positions_list, list) and 
                                all(isinstance(p, list) and len(p) == 2 for p in positions_list)):
                                res = self.get_pixel_distance(positions_list)
                    elif tool_name == 'move_to_location':
                        destination = args_dict.get('destination')
                        if destination:
                            res = self.move_to_location(destination)
                    elif tool_name == 'describe_current_view':
                        question = args_dict.get('question')
                        if question:
                            res = self.describe_current_view(question)
                    elif tool_name == 'get_available_locations':
                        res = self.get_available_locations()
                    elif tool_name == 'get_current_location':
                        res = self.get_current_location()
                    elif tool_name == 'line_following':
                        color = args_dict.get('color')
                        if color:
                            res = self.line_following(color)
                    elif tool_name == 'color_track':
                        color = args_dict.get('color')
                        if color:
                            res = self.color_track(color)
                    elif tool_name == 'lidar_scan_detect':
                        scan_detect = args_dict.get('scan_detect')
                        if scan_detect:
                            res = self.lidar_scan_detect(scan_detect)
                    elif tool_name == 'robot_move_control':
                        if all(k in args_dict for k in ["linear_x", "linear_y", "angular_z", "duration"]):
                            res = self.robot_move_control(
                                linear_x=args_dict['linear_x'],
                                linear_y=args_dict['linear_y'],
                                angular_z=args_dict['angular_z'],
                                duration=args_dict['duration']
                        )

                    elif tool_name == 'obj_track_box_detect':
                        user_query = args_dict.get('user_query')
                        if user_query:
                            res = self.obj_track_box_detect(user_query)
                    elif tool_name == 'object_track':
                        box = args_dict.get('box')
                        if box:
                            res = self.object_track(box)

                    elif tool_name == 'obj_pick_box_detect':
                        user_query = args_dict.get('user_query')
                        if user_query:
                            res = self.obj_pick_box_detect(user_query)
                    elif tool_name == 'object_pick_function':
                        box = args_dict.get('box')
                        if box:
                            res = self.object_pick_function(box)
                    elif tool_name == 'object_place_function':
                        res = self.object_place_function()
    
                    elif tool_name == 'color_pick':
                        pick_color = args_dict.get('pick_color')
                        if pick_color:
                            res = self.color_pick(pick_color)
                    elif tool_name == 'color_place':
                        res = self.color_place()
                    # self.get_logger().info(f'{LogColors.BLUE}{LogColors.BOLD}Feedback for large model {res}:{LogColors.RESET}')

                    if res is not None:
                        self.tools_result_pub.publish(Tools(id=tool_id, name=tool_name, data=res))
                    else:
                        self.tools_result_pub.publish(Tools(id='', name='None', data='Largemodel None'))
                except Exception as e:
                    self.get_logger().error(f"Tool {tool_name} execution error: {str(e)}")
                    res = f"工具执行错误: {str(e)}(tool used failed{tool_name})"
                    
                self.tools = []
                time.sleep(2)
            else:
                time.sleep(0.02)

    def display_thread(self):
        """显示线程(show thread)"""
        while self.running:
            try:
                rgb_image, depth_image = self.image_pair_queue.get(block=True, timeout=1)
                result_image = rgb_image.copy()

                with self.draw_lock:
                    if self.draw_flag:
                        # 颜色识别绘制(color detect)
                        if self.color_detect_box is not None:
                            cv2.drawContours(result_image, [self.color_detect_box], -1, (0, 255, 255), 2)
                            cv2.circle(result_image, (self.color_detect_center[0], self.color_detect_center[1]), 5, (0, 255, 255), -1)
                        # 物体追踪绘制(object tracking)
                        if self.track_box_p1 and self.track_box_p2:
                            if self.start_track:
                                cv2.rectangle(result_image, self.track_box_p1, self.track_box_p2, (0, 255, 0), 2)
                            if not self.obj_move_finish and self.object_mode == 'pick':
                                cv2.line(result_image, (self.obj_pick_center_x, 0), (self.obj_pick_center_x, 480), (0, 255, 255), 2)
                                cv2.line(result_image, (0, self.obj_pick_center_y), (640, self.obj_pick_center_y), (0, 255, 255), 2)
                cv2.imshow("result_image", result_image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:
                    self.running = False

            except queue.Empty:
                if not self.running:
                    break
                continue
            except Exception as e:
                self.get_logger().error(f"Display thread error: {str(e)}")
                continue

    def object_track_thread(self):
        """物体追踪线程(object tracking thread)"""
        while self.running:
            try:
                image, depth_image = self.image_pair_queue.get(block=True)
                img_h, img_w, _ = image.shape

                if self.object_detect_box and self.box is not None:
                    try:
                        box = self.box
                        box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                        self.track.set_track_target(box, image)
                        self.start_track = True
                        self.object_detect_box = False
                        self.box = []
                    except (ValueError, TypeError) as e:
                        self.start_track = False
                        self.get_logger().error(f"Object track setup error: {str(e)}")

                if self.start_track:
                    if self.object_mode == "track":
                        self.data = self.track.track(image, depth_image)
                    else:
                        self.data = self.track.track(image,depth_image,use_pixel=True)
                    if not isinstance(self.data, (list, tuple)) or len(self.data) < 4:
                       self.get_logger().warn(f"Unexpected track data: {self.data}")
                       continue
                    with self.draw_lock:
                        self.draw_flag = True
                        self.track_box_p1 = self.data[2]
                        self.track_box_p2 = self.data[3]

                        self.track_box_p1 = (max(0, self.track_box_p1[0]), min(img_w-1, self.track_box_p1[1]))
                        self.track_box_p2 = (max(0, self.track_box_p2[0]), min(img_h-1, self.track_box_p2[1]))
                    
                    if self.object_mode == 'track' or self.object_mode == 'pick':
                        twist = Twist()
                        twist.linear.x, twist.angular.z = self.data[0], self.data[1]

                        if 'Acker' in self.machine_type:
                            steering_angle = common.set_range(twist.angular.z, -math.radians(40), math.radians(40))
                            if steering_angle != 0:
                                R = 0.145 / math.tan(steering_angle)
                                twist.angular.z = twist.linear.x / R

                        self.cmd_vel_pub.publish(twist)
                        if self.object_mode == 'pick':
                            center_x = self.track_box_p1[0] + (self.track_box_p2[0] - self.track_box_p1[0]) / 2
                            center_y = self.track_box_p1[1] + (self.track_box_p2[1] - self.track_box_p1[1]) / 2
                            if (center_y - self.obj_pick_center_y) < 28 and abs(center_x - self.obj_pick_center_x) < 28:
                                self.box_count += 1
                                self.get_logger().info(f'self.box_count{self.box_count}')
                                if self.box_count > 80:
                                    self.obj_pick_start = True
                                    self.cmd_vel_pub.publish(Twist())
                                    self.start_track = False
                else:
                    self.box_count = 0
                    self.track_box_p1 = None
                    self.track_box_p2 = None
                    time.sleep(0.01)
                    
            except queue.Empty:
                if not self.running:
                    break
                continue
            except Exception as e:
                time.sleep(0.01)

def main():
    rclpy.init()
    node = LLMControlMove('llm_control_move')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()
 
if __name__ == "__main__":
    main()
