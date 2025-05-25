import os
import re
import cv2
import matplotlib.pyplot as plt
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
        print(f"JSONDecodeError: {e}")
        print(f"Raw JSON string: {json_string}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON parsing: {e}")
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
    response_text = response.text
    if not isinstance(response_text, str):
        print(f"Gemini response is not a string: {type(response_text)}")
        return None
    return response_text

def get_target_bbox(image_tensor, prompt, gemini_model):
    """
    Identifies pick and place targets from an image using Gemini,
    based on a user prompt.
    Returns: detected_objects (raw parsed JSON), pick_boxes, place_boxes,
             norm_pick_boxes, norm_place_boxes, pick_labels, place_labels
    """
    image_pil = tensor_to_pil(image_tensor)
    width, height = image_pil.size

    gemini_response_text = get_2D_bbox(image_pil, prompt, gemini_model)
    detected_objects = parse_json(gemini_response_text)

    if not detected_objects:
        print("No objects detected by Gemini or JSON parsing failed.")
        return [], [], [], [], [], [], [] # Added empty list for detected_objects

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
    place_keywords_in_label = ["bin", "box", "container", "tray", "area", "table"] # Added "area", "table"

    # Attempt to identify a specific place object mentioned in the user_task
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

    return detected_objects, pick_boxes, place_boxes, norm_pick_boxes, norm_place_boxes, pick_labels, place_labels

# Removed get_random_targets as it's replaced by specific selection logic

# --- End of reimplemented functions ---

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

def return_bbox(boxes, idx):
    # Note: This function's normalization by 1000 is separate from the [0,1] normalization
    # used for the policy, and is likely for a specific unit conversion (e.g., mm to meters).
    pick_t = boxes[idx]['box_2d']
    norm_pick_t = [p/1000 for p in pick_t]
    return norm_pick_t, pick_t

def select_bbox(boxes):
    selection = int(input("Enter the number of the bounding box you want to select: "))

    # Validate selection
    if 0 <= selection < len(boxes):
        return return_bbox(boxes,selection)
    else:
        print(f"Invalid selection. Please choose a number between 0 and {len(boxes)-1}")

def iterate_over_bbox(boxes):
    for i in range(len(boxes)):
        norm_pick_t, pick_t = return_bbox(boxes,i)
        yield norm_pick_t, pick_t

# --- Main script execution starts here ---

# Configure Gemini API
try:
    load_dotenv(".env") # Load environment variables from .env file
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("GEMINI_API_KEY environment variable not set. Please set it before running.")
    exit()

# Initialize Gemini Model
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Create camera config using proper config objects
# IMPORTANT: Changed "laptop" to "hand" to match PI0Policy's expected image features
cameras = {
    "hand": OpenCVCameraConfig( # Changed "laptop" to "hand"
        camera_index=2,  # Built-in webcam (assuming this is your "hand" view)
        fps=30,
        width=640,
        height=480
    ),
    "top": OpenCVCameraConfig(
        camera_index=0,  # iPhone camera (assuming this is your "top" view)
        fps=30,
        width=640,
        height=480
    )
    }

robot_cfg = So100RobotConfig(
            cameras=cameras,
            mock=False,
        )

# Determine device for PyTorch
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

inference_time_s = 100
fps = 30
# audio configs (commented out as per original, but kept for context)
# SAMPLE_RATE = 16_000
# DURATION = 7

# Whisper configs (commented out as per original, but kept for context)
# model_size = "medium"
# whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
# model = WhisperModel(model_size, device=whisper_device, compute_type="int8")

# Policy paths - now using Hugging Face Hub ID for PI0Policy
act_path = "lerobot/act" # Example for ACT, if you were to use it
pi0_path = "lerobot/pi0" # Changed to Hugging Face Hub ID
pi0fast_path = "lerobot/pi0fast" # Example for PI0FAST, if you were to use it

print("Loading Policy.")
# Select policy
#policy = ACTPolicy.from_pretrained(act_path)
policy = PI0Policy.from_pretrained(pi0_path) # This will now download from Hugging Face Hub
#policy = PI0FASTPolicy.from_pretrained(pi0fast_path)
policy.to(device)
print("Policy loaded.")

# --- Audio recording and transcription (commented out as per original) ---
# user_task = "Add all wooden blocks to the blue bin" # Default hardcoded task
"""
try:
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # wait until recording is finished

    # Convert numpy array to MP3 using pydub
    audio_segment = AudioSegment(
        audio.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=audio.dtype.itemsize,
        channels=1
    )

    output_file = os.path.join(output_dir, "command1.mp3")
    audio_segment.export(output_file, format="mp3")
    print(" Saved audio to", output_file)

    segments, info = model.transcribe(output_file)
    print(f"Detected language: {info.language} — probability {info.language_probability:.2%}\n")

    transcribed_task = " ".join(s.text.strip() for s in segments)
    print("📝 Transcript:", transcribed_task)
    user_task = transcribed_task # Use transcribed task if successful
except Exception as e:
    print(f"Error during audio recording or transcription: {e}")
    print(f"Proceeding with hardcoded user_task: '{user_task}'")
"""
# Define the user's high-level task for the robot
user_task_for_robot = "Pickup yellow lego."

# Construct the prompt for Gemini to perform general object detection
gemini_vision_prompt = f"""User request: {user_task_for_robot}

You are a robot tasked with identifying objects in an image.
Identify all distinct objects in the image and their 2D bounding boxes.
For each object, provide a concise label and its bounding box coordinates [x1, y1, x2, y2].
Return your findings strictly as a JSON array of objects.
Example format: [{{ "label": "red block", "box_2d": [10, 20, 30, 40] }}, {{ "label": "blue bin", "box_2d": [50, 60, 70, 80] }}]
"""

# --- Perception: Get target bounding boxes using Gemini ---
# Create and connect robot
print("Create and connect robot")
robot = make_robot_from_config(robot_cfg)
print("Connecting main follower arm.")
print("Connecting main leader arm.")
robot.connect()
print("Activating torque on main follower arm.")

raw_observation = robot.capture_observation() # Renamed to avoid confusion
# Call get_target_bbox with the comprehensive prompt
# IMPORTANT: Changed "observation.images.laptop" to "observation.images.hand"
detected_objects_raw, pick_all, place_all, norm_pick_all, norm_place_all, pick_labels_all, place_labels_all = \
    get_target_bbox(raw_observation["observation.images.hand"], prompt=gemini_vision_prompt, gemini_model=gemini_model)

print("\n--- Gemini Detected Objects (Raw JSON) ---")
if detected_objects_raw:
    print(json.dumps(detected_objects_raw, indent=2))
else:
    print("No objects detected by Gemini.")

print("\nPick objects detected by Gemini:")
for i in range(len(pick_all)):
    print(f"- {pick_labels_all[i]} (Box: {pick_all[i]['box_2d']})")
print("\nPlace locations detected by Gemini:")
for i in range(len(place_all)):
    print(f"- {place_labels_all[i]} (Box: {place_all[i]['box_2d']})")

# --- Specific target selection for "yellow lego" and a place ---
target_pick_label_str = "yellow lego"
selected_pick_t = None
selected_norm_pick_t = None
selected_pick_target_label = None

# Find the specific pick target ("yellow lego")
for i, label in enumerate(pick_labels_all):
    if target_pick_label_str.lower() in label.lower():
        selected_pick_t = pick_all[i]["box_2d"]
        selected_norm_pick_t = norm_pick_all[i]
        selected_pick_target_label = pick_labels_all[i]
        break

if selected_pick_t is None:
    print(f"Could not find '{target_pick_label_str}' among detected objects. Exiting.")
    robot.disconnect()
    exit()

# Determine the place target
selected_place_t = None
selected_norm_place_t = None
selected_place_target_label = None

# Strategy for place target:
# 1. Look for a place target mentioned in the user_task_for_robot (e.g., "put in blue bin")
# 2. If not specified, look for generic place keywords in detected objects ("bin", "box", "container", "tray", "area", "table")
# 3. If still no place, use the first available place object detected by Gemini.

# Extract potential place target from user_task_for_robot
place_target_from_task = None
place_match_in_task = re.search(r"(?:in|to) the (.*?)(?:\s|$)", user_task_for_robot, re.IGNORECASE)
if place_match_in_task:
    place_target_from_task = place_match_in_task.group(1).strip().lower()
    print(f"Identified potential place target from task: '{place_target_from_task}'")

if place_target_from_task:
    for i, label in enumerate(place_labels_all):
        if place_target_from_task in label.lower():
            selected_place_t = place_all[i]["box_2d"]
            selected_norm_place_t = norm_place_all[i]
            selected_place_target_label = place_labels_all[i]
            break

# If no specific place target from task, try generic keywords
if selected_place_t is None:
    for i, label in enumerate(place_labels_all):
        if any(keyword in label.lower() for keyword in ["bin", "box", "container", "tray", "area", "table"]):
            selected_place_t = place_all[i]["box_2d"]
            selected_norm_place_t = norm_place_all[i]
            selected_place_target_label = place_labels_all[i]
            break

# If still no place target, take the first one if any exist
if selected_place_t is None and place_all:
    selected_place_t = place_all[0]["box_2d"]
    selected_norm_place_t = norm_place_all[0]
    selected_place_target_label = place_labels_all[0]

if selected_place_t is None:
    print("No suitable place target found. Exiting.")
    robot.disconnect()
    exit()

# Assign the selected targets for the policy
pick_t = selected_pick_t
place_t = selected_place_t
norm_pick_t = selected_norm_pick_t
norm_place_t = selected_norm_place_t
pick_target_label = selected_pick_target_label
place_target_label = selected_place_target_label

single_task = f"Grasp {pick_target_label} and put it in {place_target_label}"
print(f"\nCurrent task: {single_task}")

listener, events = setup_key_listener()

# --- Main control loop ---
for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    if events["exit_early"]:
        print("Exiting early due to key press.")
        break

    # Read the follower state and access the frames from the cameras
    raw_observation = robot.capture_observation()

    # Prepare a new dictionary for the policy input
    policy_input_batch = {}

    # Handle task
    policy_input_batch["task"] = [single_task] # Wrap in list for batch dimension

    # Handle state
    current_state = raw_observation["observation.state"]
    if not isinstance(current_state, torch.Tensor):
        current_state = torch.tensor(current_state, dtype=torch.float32)

    # Concatenate normalized pick and place targets to the observation state
    combined_state = torch.cat([current_state, norm_pick_t, norm_place_t])
    policy_input_batch["observation.state"] = combined_state.unsqueeze(0).to(device)

    # Handle images: Iterate through the camera names defined in `cameras` config
    for cam_name in cameras.keys():
        image_key_in_observation = f"observation.images.{cam_name}"
        if image_key_in_observation in raw_observation:
            image_data = raw_observation[image_key_in_observation]
            
            # Ensure image_data is a PyTorch tensor
            if not isinstance(image_data, torch.Tensor):
                # If it's a NumPy array, convert it
                image_tensor = torch.from_numpy(image_data)
            else:
                # If it's already a tensor, use it directly
                image_tensor = image_data

            # Ensure the tensor is float32 and normalized to [0, 1]
            image_tensor = image_tensor.type(torch.float32) / 255.0

            # Permute to (C, H, W) if it's (H, W, C)
            # Check current dimensions to avoid errors if already CHW
            if image_tensor.ndim == 3 and image_tensor.shape[2] in [1, 3]: # Assuming C is last dim for HWC
                image_tensor = image_tensor.permute(2, 0, 1).contiguous()
            elif image_tensor.ndim == 3 and image_tensor.shape[0] in [1, 3]: # Already CHW
                pass # Do nothing
            else:
                print(f"Warning: Unexpected image tensor shape for {image_key_in_observation}: {image_tensor.shape}. Expected (H, W, C) or (C, H, W).")
                continue # Skip this image if shape is unexpected

            # Add batch dimension and move to device
            policy_input_batch[image_key_in_observation] = image_tensor.unsqueeze(0).to(device)
        else:
            print(f"Warning: Image key '{image_key_in_observation}' not found in raw_observation.")


    # Compute the next action with the policy based on the current observation
    action = policy.select_action(policy_input_batch)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

    if events["select_new_bbox"]:
        events["select_new_bbox"] = False
        print("Re-selecting new bounding box targets...")
        # Capture a fresh observation for new perception
        fresh_raw_observation = robot.capture_observation()
        # IMPORTANT: Changed "observation.images.laptop" to "observation.images.hand"
        detected_objects_raw, pick_all, place_all, norm_pick_all, norm_place_all, pick_labels_all, place_labels_all = \
            get_target_bbox(fresh_raw_observation["observation.images.hand"], prompt=gemini_vision_prompt, gemini_model=gemini_model)

        print("\n--- Gemini Detected Objects (Raw JSON) after re-selection ---")
        if detected_objects_raw:
            print(json.dumps(detected_objects_raw, indent=2))
        else:
            print("No objects detected by Gemini after re-selection.")

        # Re-apply specific target selection logic
        selected_pick_t_new = None
        selected_norm_pick_t_new = None
        selected_pick_target_label_new = None

        for i, label in enumerate(pick_labels_all):
            if target_pick_label_str.lower() in label.lower():
                selected_pick_t_new = pick_all[i]["box_2d"]
                selected_norm_pick_t_new = norm_pick_all[i]
                selected_pick_target_label_new = pick_labels_all[i]
                break

        selected_place_t_new = None
        selected_norm_place_t_new = None
        selected_place_target_label_new = None

        place_target_from_task_new = None
        place_match_in_task_new = re.search(r"(?:in|to) the (.*?)(?:\s|$)", user_task_for_robot, re.IGNORECASE)
        if place_match_in_task_new:
            place_target_from_task_new = place_match_in_task_new.group(1).strip().lower()

        if selected_place_t_new is None: # Check if a place was found from task, if not, try generic
            for i, label in enumerate(place_labels_all):
                if any(keyword in label.lower() for keyword in ["bin", "box", "container", "tray", "area", "table"]):
                    selected_place_t_new = place_all[i]["box_2d"]
                    selected_norm_place_t_new = norm_place_all[i]
                    selected_place_target_label_new = place_labels_all[i]
                    break
        
        if selected_place_t_new is None and place_all: # If still no place, take the first available
            selected_place_t_new = place_all[0]["box_2d"]
            selected_norm_place_t_new = norm_place_all[0]
            selected_place_target_label_new = place_labels_all[0]

        if selected_pick_t_new is None or selected_place_t_new is None:
            print("Could not determine valid pick and/or place targets after re-selection. Continuing with previous task.")
        else:
            pick_t, place_t, norm_pick_t, norm_place_t = selected_pick_t_new, selected_place_t_new, selected_norm_pick_t_new, selected_norm_place_t_new
            pick_target_label, place_target_label = selected_pick_target_label_new, selected_place_target_label_new
            single_task = f"Grasp {pick_target_label} and put it in {place_target_label}"
            print(f"New task: {single_task}")


robot.disconnect()
print("Robot disconnected.")