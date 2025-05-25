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
        # In a real scenario, you might want to handle this differently,
        # e.g., by only performing pick actions or waiting for a place target.
        # For now, returning None for place-related items.
        return None, None, None, None, pick_boxes, norm_pick_boxes, None, None, pick_labels

    # Select a random pick target
    pick_idx = random.randrange(len(pick_boxes))
    pick_t = pick_boxes[pick_idx]["box_2d"]
    norm_pick_t = norm_pick_boxes[pick_idx]
    pick_target_label = pick_labels[pick_idx]

    # Select a random place target
    place_idx = random.randrange(len(place_boxes))
    place_t = place_boxes[place_idx]["box_2d"]
    norm_place_t = norm_place_boxes[place_idx]
    place_target_label = place_labels[place_idx]

    # Return the selected single targets and the original lists (as per original function's return signature)
    return pick_t, place_t, norm_pick_t, norm_place_t, pick_boxes, norm_pick_boxes, pick_target_label, place_target_label, pick_labels

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
cameras = {
    "laptop": OpenCVCameraConfig(
        camera_index=0,  # Built-in webcam
        fps=30,
        width=640,
        height=480
    ),
#    "top": OpenCVCameraConfig(
#        camera_index=1,  # iPhone camera - commented out as in original
#        fps=30,
#        width=640,
#        height=480
#    )
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
# audio configs
SAMPLE_RATE = 16_000   # Whisper works best on 16 kHz mono
DURATION = 7           # seconds (max length requested)

# Ensure output directory exists
#output_dir = "/Users/meetr/Downloads/robot/audio" # This path is specific to the user's system
#os.makedirs(output_dir, exist_ok=True)

# Whisper configs
#model_size = "medium"  # options: tiny, base, small, medium, large-v3
#whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
#model = WhisperModel(model_size, device=whisper_device, compute_type="int8")

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

# --- Audio recording and transcription ---
#print("🎙️ Recording for", DURATION, "seconds…")
#user_task = "Add all wooden blocks to the blue bin" # Default hardcoded task
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
user_task="Pickup yellow logo."

prompt = """User request: """ + user_task + """

    Analyze the provided image and understand what the user needs. Identify all relevant objects on the desk that would help fulfill this request.

    For example:
    - If the user wants to build something (like a red lego wall or wooden tower or lego plane), find all appropriate building pieces (Focus on identifying all red lego bricks on the desk that could be used for building a wall. Or in case of wooden tower, identify all wooden blocks on the desk that could be used for building a tower or all lego pieces on the desk that could be used to build a plane.)
    - If the user wants to clear the desk, identify all objects on the desk that need to be cleared
    - If the user wants specific colored items, focus on those colors
    - If the users mentions place location, identify the location of the object on the desk
    - If user is pointing to an object, identify the object that the user is pointing to.

    Ignore the robot arm itself if visible.
    Return your findings strictly as a JSON array of bounding boxes.
    Example format: [{"label": "red lego brick", "box_2d": [100, 200, 150, 280]}, {"label": "blue bin", "box_2d": [500, 600, 700, 850]}]"""

# --- Perception: Get target bounding boxes using Gemini ---
# Create and connect robot
print("Create and connect robot")
robot = make_robot_from_config(robot_cfg)
print("Connecting main follower arm.")
print("Connecting main leader arm.")
robot.connect()
print("Activating torque on main follower arm.")

observation = robot.capture_observation()
pick, place, norm_pick, norm_place, pick_labels, place_labels = get_target_bbox(observation["observation.images.laptop"], prompt=prompt, gemini_model=gemini_model)

print("Pick objects:")
for i in range(len(pick)):
    print(pick_labels[i])
print("Place location:")
for i in range(len(place)):
    print(place_labels[i], place[i])

# --- Select random pick and place targets ---
pick_t, place_t, norm_pick_t, norm_place_t, pick, norm_pick, pick_target_label, place_target_label, pick_labels = \
    get_random_targets(pick, place, norm_pick, norm_place, pick_labels, place_labels)

if pick_t is None or place_t is None:
    print("Could not determine valid pick and/or place targets. Exiting.")
    robot.disconnect()
    exit()

single_task = f"Grasp {pick_target_label} and put it in {place_target_label}"
print(f"Current task: {single_task}")

listener, events = setup_key_listener()

# --- Main control loop ---
for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    if events["exit_early"]:
        print("Exiting early due to key press.")
        break

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()
    observation["task"] = single_task
    # Concatenate normalized pick and place targets to the observation state
    # norm_pick_t and norm_place_t are already torch.Tensors from get_random_targets
    observation["observation.state"] = torch.cat([observation["observation.state"], norm_pick_t, norm_place_t])

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if name == "task":
            # Skip tensor operations for string values
            observation[name] = [observation[name]]  # Wrap in list to simulate batch dimension
            continue
        if "image" in name:
            # Assuming image is (H, W, C) uint8 and needs to be (C, H, W) float32 in [0,1]
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0) # Add batch dimension
        observation[name] = observation[name].to(device)

    # Compute the next action with the policy based on the current observation
    action = policy.select_action(observation)
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
        fresh_observation = robot.capture_observation()
        pick, place, norm_pick, norm_place, pick_labels, place_labels = get_target_bbox(fresh_observation["observation.images.laptop"], prompt=prompt, gemini_model=gemini_model)

        pick_t_new, place_t_new, norm_pick_t_new, norm_place_t_new, _, _, pick_target_label_new, place_target_label_new, _ = \
            get_random_targets(pick, place, norm_pick, norm_place, pick_labels, place_labels)

        if pick_t_new is None or place_t_new is None:
            print("Could not determine valid pick and/or place targets after re-selection. Continuing with previous task.")
            # If no new targets, continue with the existing task or break if desired.
            # For this example, we'll just print a message and continue with the current task.
        else:
            pick_t, place_t, norm_pick_t, norm_place_t = pick_t_new, place_t_new, norm_pick_t_new, norm_place_t_new
            pick_target_label, place_target_label = pick_target_label_new, place_target_label_new
            single_task = f"Grasp {pick_target_label} and put it in {place_target_label}"
            print(f"New task: {single_task}")


robot.disconnect()
print("Robot disconnected.")