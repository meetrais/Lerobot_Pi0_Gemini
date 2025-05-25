import os
import re
import cv2
import time
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import json
import PIL
import google.generativeai as genai
import rerun as rr # Not used in the provided snippet, but kept as in original
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import os
from pydub import AudioSegment
import io
import random # Added for get_random_targets
from dotenv import load_dotenv, find_dotenv


from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.control_utils import busy_wait, log_control_info
from lerobot.common.robot_devices.robots.configs import So100RobotConfig

# --- Reimplemented functions from lerobot.common.vision_utils.gemini_perception ---

def tensor_to_pil(image_tensor):
    """Converts a PyTorch tensor image (H, W, C) to a PIL Image."""
    # Ensure tensor is on CPU and convert to numpy
    image_np = image_tensor.cpu().numpy()
    # If float32 in [0,1], convert to uint8 in [0,255]
    if image_np.dtype == np.float32 and image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    elif image_np.dtype == np.uint8:
        pass # Already in correct format
    else:
        raise ValueError(f"Unsupported image tensor dtype: {image_np.dtype}")

    return PIL.Image.fromarray(image_np)

def parse_json(json_string):
    """Parses a JSON string, handling markdown code blocks."""
    try:
        # Gemini sometimes returns markdown code blocks, so we need to extract the JSON.
        match = re.search(r"```json\n(.*)\n```", json_string, re.DOTALL)
        if match:
            json_string = match.group(1)
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw JSON string: {json_string}")
        return None

def normalize_bbox_0to1(bbox, width, height):
    """Normalizes bounding box coordinates to [0, 1]."""
    x1, y1, x2, y2 = bbox
    return [x1 / width, y1 / height, x2 / width, y2 / height]

def get_2D_bbox(image, prompt, model):
    """Sends an image and prompt to a Gemini model to get 2D bounding box detections."""
    if not isinstance(image, PIL.Image.Image):
        raise TypeError("Image must be a PIL.Image.Image object.")

    contents = [
        image,
        prompt
    ]
    response = model.generate_content(contents)
    return response.text

def get_target_bbox(image_tensor, prompt, gemini_model):
    """
    Identifies pick and place targets from an image using Gemini,
    based on a user prompt.
    """
    image_pil = tensor_to_pil(image_tensor)
    width, height = image_pil.size

    gemini_response_text = get_2D_bbox(image_pil, prompt, gemini_model)
    detected_objects = parse_json(gemini_response_text)

    if not detected_objects:
        print("No objects detected by Gemini or JSON parsing failed.")
        return [], [], [], [], [], []

    pick_boxes = []
    place_boxes = []
    norm_pick_boxes = []
    norm_place_boxes = []
    pick_labels = []
    place_labels = []

    # Extract user_task from the prompt to help identify pick/place roles
    user_task_match = re.search(r"User request: (.*?)\n", prompt)
    user_task = ""
    if user_task_match:
        user_task = user_task_match.group(1).strip()
        print(f"Extracted user_task: {user_task}")

    # Simple heuristic for identifying place targets based on keywords in label
    place_keywords_in_label = ["bin", "box", "container", "tray"]

    # Attempt to identify a specific place object mentioned in the user_task
    # e.g., "put it in the blue bin" -> "blue bin"
    target_place_label_from_task = None
    place_match_in_task = re.search(r"(?:in|to) the (.*?)(?:\s|$)", user_task, re.IGNORECASE)
    if place_match_in_task:
        target_place_label_from_task = place_match_in_task.group(1).strip().lower()
        print(f"Identified potential place target from task: '{target_place_label_from_task}'")

    for obj in detected_objects:
        label = obj.get("label", "").lower()
        bbox = obj.get("box_2d")
        if bbox is None or len(bbox) != 4:
            print(f"Skipping malformed bounding box: {obj}")
            continue

        norm_bbox = normalize_bbox_0to1(bbox, width, height)
        norm_bbox_tensor = torch.tensor(norm_bbox, dtype=torch.float32)

        is_place = False
        # Check if label contains common place keywords
        if any(keyword in label for keyword in place_keywords_in_label):
            is_place = True
        # Check if the label matches the identified place target from the task
        elif target_place_label_from_task and target_place_label_from_task in label:
            is_place = True

        if is_place:
            place_boxes.append({"label": obj["label"], "box_2d": bbox})
            norm_place_boxes.append(norm_bbox_tensor)
            place_labels.append(obj["label"])
        else:
            pick_boxes.append({"label": obj["label"], "box_2d": bbox})
            norm_pick_boxes.append(norm_bbox_tensor)
            pick_labels.append(obj["label"])

    return pick_boxes, place_boxes, norm_pick_boxes, norm_place_boxes, pick_labels, place_labels

def get_random_targets(pick_boxes, place_boxes, norm_pick_boxes, norm_place_boxes, pick_labels, place_labels):
    """
    Selects a random pick and place target from the identified lists.
    Returns the selected single targets and the original lists.
    """
    if not pick_boxes:
        print("No pick targets available.")
        return None, None, None, None, [], [], None, None, []
    if not place_boxes:
        print("No place targets available. Cannot select a pick-and-place pair.")
        return None, None, None, None, pick_boxes, norm_pick_boxes, None, None, pick_labels

    pick_idx = random.randrange(len(pick_boxes))
    pick_t = pick_boxes[pick_idx]["box_2d"]
    norm_pick_t = norm_pick_boxes[pick_idx]
    pick_target_label = pick_labels[pick_idx]

    place_idx = random.randrange(len(place_boxes))
    place_t = place_boxes[place_idx]["box_2d"]
    norm_place_t = norm_place_boxes[place_idx]
    place_target_label = place_labels[place_idx]

    return pick_t, place_t, norm_pick_t, norm_place_t, pick_boxes, norm_pick_boxes, pick_target_label, place_target_label, pick_labels

def setup_key_listener():
    events = {"exit_early": False, "rerecord_episode": False,
              "stop_recording": False, "select_new_bbox": False}
    from pynput import keyboard
    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.space:
                print("Space key pressed. Selecting new bbox...")
                events["select_new_bbox"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events

# --- Main script execution starts here ---
load_dotenv(".env")
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("GEMINI_API_KEY environment variable not set.")
    exit()

gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

cameras = {
    "primary": OpenCVCameraConfig(camera_index=2, fps=30, width=640, height=480),
    "top": OpenCVCameraConfig(camera_index=0, fps=30, width=640, height=480)
}
robot_cfg = So100RobotConfig(cameras=cameras, mock=False)

if torch.backends.mps.is_available(): device = "mps"
elif torch.cuda.is_available(): device = "cuda"
else: device = "cpu"
print(f"Using device: {device}")

inference_time_s = 100
fps = 30
SAMPLE_RATE = 16_000
DURATION = 7
output_dir = "/Users/meetr/Downloads/robot/audio" # Specific path
os.makedirs(output_dir, exist_ok=True)

model_size = "medium"
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel(model_size, device=whisper_device, compute_type="int8")

pi0_path = "lerobot/pi0"
print("Loading Policy.")
policy = PI0Policy.from_pretrained(pi0_path)
policy.to(device)
print("Policy loaded.")

user_task = "Pickup yellow lego and put into white box."
# Audio recording section commented out as in user's provided code
prompt = f"""User request: {user_task}

    Analyze the provided image and understand what the user needs. Identify all relevant objects on the desk that would help fulfill this request.
    For example:
    - If the user wants to build something (like a red lego wall or wooden tower or lego plane), find all appropriate building pieces (Focus on identifying all red lego bricks on the desk that could be used for building a wall. Or in case of wooden tower, identify all wooden blocks on the desk that could be used for building a tower or all lego pieces on the desk that could be used to build a plane.)
    - If the user wants to clear the desk, identify all objects on the desk that need to be cleared
    - If the user wants specific colored items, focus on those colors
    - If the users mentions place location, identify the location of the object on the desk
    - If user is pointing to an object, identify the object that the user is pointing to.
    Ignore the robot arm itself if visible.
    Return your findings strictly as a JSON array of bounding boxes.
    Example format: [{{"label": "red lego brick", "box_2d": [100, 200, 150, 280]}}, {{"label": "blue bin", "box_2d": [500, 600, 700, 850]}}]"""

robot = make_robot_from_config(robot_cfg)
print("Connecting main follower arm.")
print("Connecting main leader arm.")
robot.connect()
print("Activating torque on main follower arm.")

observation_raw = robot.capture_observation()
pick, place, norm_pick, norm_place, pick_labels, place_labels = get_target_bbox(observation_raw["observation.images.primary"], prompt=prompt, gemini_model=gemini_model)

print("Pick objects:")
for lbl in pick_labels: print(lbl)
print("Place location:")
for i, lbl in enumerate(place_labels): print(lbl, place[i])

pick_t, place_t, norm_pick_t, norm_place_t, _, _, pick_target_label, place_target_label, _ = \
    get_random_targets(pick, place, norm_pick, norm_place, pick_labels, place_labels)

if pick_t is None or place_t is None:
    print("Could not determine valid pick and/or place targets. Exiting.")
    if robot.is_connected: robot.disconnect()
    exit()

single_task = f"Grasp {pick_target_label} and put it in {place_target_label}"
print(f"Current task: {single_task}")

listener, events = setup_key_listener()

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()
    if events["exit_early"]: break

    observation = robot.capture_observation()
    current_obs_state = observation["observation.state"]
    # Ensure norm_pick_t and norm_place_t are 1D tensors before concatenation
    norm_pick_t_flat = norm_pick_t.flatten()
    norm_place_t_flat = norm_place_t.flatten()
    
    observation["observation.state"] = torch.cat([current_obs_state, norm_pick_t_flat, norm_place_t_flat])
    observation["task"] = single_task # Ensure task is part of the observation for the policy

    batch_for_policy = {}
    for name, value in observation.items():
        if name == "task":
            batch_for_policy[name] = [value] # Wrap in list
            continue
        if "image" in name:
            processed_image = value.type(torch.float32) / 255
            processed_image = processed_image.permute(2, 0, 1).contiguous()
            batch_for_policy[name] = processed_image.unsqueeze(0).to(device)
        elif name == "observation.state":
             batch_for_policy[name] = value.unsqueeze(0).to(device)
        else: # For any other keys that might be there from robot.capture_observation()
            if torch.is_tensor(value):
                batch_for_policy[name] = value.unsqueeze(0).to(device)
            else:
                batch_for_policy[name] = value


    # +++ START DEBUGGING PRINTS +++
    print("\n--- DEBUG INFO ---")
    print("Batch keys being passed to policy.select_action:", list(batch_for_policy.keys()))
    print("Inspecting policy.config object:")
    if hasattr(policy, 'config'):
        config_obj = policy.config
        print(f"  Type of policy.config: {type(config_obj)}")
        
        attributes_to_check = ['image_features', 'input_features', 'output_features', 
                               'env_state_feature', 'robot_state_feature', 'modalities', 
                               'n_obs_steps', 'n_action_steps']
        for attr_name in attributes_to_check:
            if hasattr(config_obj, attr_name):
                print(f"  policy.config.{attr_name}: {getattr(config_obj, attr_name)}")
            else:
                print(f"  policy.config.{attr_name}: Not found")
        
        if not hasattr(config_obj, 'modalities') and hasattr(config_obj, '__dataclass_fields__'):
            if 'modalities' in config_obj.__dataclass_fields__:
                print("  'modalities' is a dataclass field, but not directly an attribute. Value might be missing or None.")
            else:
                print("  'modalities' is not among __dataclass_fields__.")
    else:
        print("  policy.config attribute does not exist.")
    print("--- END DEBUG INFO ---\n")
    # +++ END DEBUGGING PRINTS +++

    action = policy.select_action(batch_for_policy)
    action = action.squeeze(0).to("cpu")
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

    if events["select_new_bbox"]:
        events["select_new_bbox"] = False
        print("Re-selecting new bounding box targets...")
        fresh_observation_raw = robot.capture_observation()
        pick, place, norm_pick, norm_place, pick_labels, place_labels = get_target_bbox(
            fresh_observation_raw["observation.images.primary"], prompt=prompt, gemini_model=gemini_model
        )
        pick_t_new, place_t_new, norm_pick_t_new, norm_place_t_new, _, _, pick_target_label_new, place_target_label_new, _ = \
            get_random_targets(pick, place, norm_pick, norm_place, pick_labels, place_labels)

        if pick_t_new is not None and place_t_new is not None:
            pick_t, place_t, norm_pick_t, norm_place_t = pick_t_new, place_t_new, norm_pick_t_new, norm_place_t_new
            pick_target_label, place_target_label = pick_target_label_new, place_target_label_new
            single_task = f"Grasp {pick_target_label} and put it in {place_target_label}"
            print(f"New task: {single_task}")
        else:
            print("Could not determine valid new pick and/or place targets. Continuing with previous task.")

if robot.is_connected:
    robot.disconnect()
print("Robot disconnected.")