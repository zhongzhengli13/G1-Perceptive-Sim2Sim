import os

_HERE = os.path.dirname(__file__)

STAND_POLICY_FILE = os.path.join(_HERE, "policy", "stand_actor.onnx")
STAND_DEPTH_ENCODER_FILE = os.path.join(_HERE, "policy", "stand_depth_encoder.onnx")
PARKOUR_POLICY_FILE = os.path.join(_HERE, "policy", "parkour_actor.onnx")
PARKOUR_DEPTH_ENCODER_FILE = os.path.join(_HERE, "policy", "parkour_depth_encoder.onnx")

UNITREE_G1_29DOF_SDK_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

POLICY_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

MJCF_FILE = os.path.join(_HERE, "g1_29dof", "scene_rough.xml")
SIM_DT = 0.005
DECIMATION = 4
DEPTH_CAMERA_NAME = "depth_camera"
DEPTH_IMAGE_SHAPE = (270, 480)
USE_SECONDARY_IMU = True

VIEWER_LOOKAT = (15.0, 0.0, 0.5)
VIEWER_DISTANCE = 32.0
VIEWER_AZIMUTH = 0.0
VIEWER_ELEVATION = -50.0

COMMAND_STEP = (0.2, 0, 0.5)
COMMAND_RANGES = {
    "lin_vel_x": (0, 0.9),
    "lin_vel_y": (0, 0),
    "ang_vel_z": (-1.0, 1.0),
}

KEY_KP_2 = 322
KEY_KP_4 = 324
KEY_KP_5 = 325
KEY_KP_6 = 326
KEY_KP_8 = 328

HISTORY_LENGTH = 8

RESIZED_DEPTH_IMAGE_SHAPE = (36, 64)
OBS_DEPTH_IMAGE_CROP_REGION = (18, 0, 16, 16)
OBS_DIMS = [3, 3, 3, 29, 29, 29]
STIFFNESS = [
    14.25062309787429,
    14.25062309787429,
    28.50124619574858,
    14.25062309787429,
    14.25062309787429,
    28.50124619574858,
    14.25062309787429,
    14.25062309787429,
    40.17923847137318,
    14.25062309787429,
    14.25062309787429,
    40.17923847137318,
    40.17923847137318,
    14.25062309787429,
    14.25062309787429,
    99.09842777666113,
    99.09842777666113,
    16.77832748089279,
    16.77832748089279,
    40.17923847137318,
    40.17923847137318,
    16.77832748089279,
    16.77832748089279,
    99.09842777666113,
    99.09842777666113,
    28.50124619574858,
    28.50124619574858,
    28.50124619574858,
    28.50124619574858,
]
DAMPING = [
    0.907222843292423,
    0.907222843292423,
    1.814445686584846,
    0.907222843292423,
    0.907222843292423,
    1.814445686584846,
    0.907222843292423,
    0.907222843292423,
    2.5578897650279457,
    0.907222843292423,
    0.907222843292423,
    2.5578897650279457,
    2.5578897650279457,
    0.907222843292423,
    0.907222843292423,
    6.3088018534966395,
    6.3088018534966395,
    1.06814150219,
    1.06814150219,
    2.5578897650279457,
    2.5578897650279457,
    1.06814150219,
    1.06814150219,
    6.3088018534966395,
    6.3088018534966395,
    1.814445686584846,
    1.814445686584846,
    1.814445686584846,
    1.814445686584846,
]
DEFAULT_JOINT_POS = [
    0.2,
    0.2,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.6,
    0.6,
    -0.312,
    -0.312,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.669,
    0.669,
    -0.363,
    -0.363,
    0.0,
    0.0,
]
ACTION_SCALES = [
    0.439,
    0.439,
    0.439,
    0.439,
    0.439,
    0.439,
    0.439,
    0.439,
    0.548,
    0.439,
    0.439,
    0.548,
    0.548,
    0.439,
    0.439,
    0.351,
    0.351,
    0.0745,
    0.0745,
    0.548,
    0.548,
    0.0745,
    0.0745,
    0.351,
    0.351,
    0.439,
    0.439,
    0.439,
    0.439,
]
TORQUE_LIMITS = [
    25,
    25,
    50,
    25,
    25,
    50,
    25,
    25,
    88,
    25,
    25,
    88,
    88,
    25,
    25,
    88,
    88,
    5,
    5,
    88,
    88,
    5,
    5,
    139,
    139,
    50,
    50,
    50,
    50,
]
JOINT_SIGNS = [1.0, 1.0, -1] * 3 + [1] * 20
