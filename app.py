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
import rerun as rr
from faster_whisper import WhisperModel # Not used
import sounddevice as sd # Not used
from pydub import AudioSegment # Not used
import io # Not used
import random
from dotenv import load_dotenv, find_dotenv

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.control_utils import busy_wait
from lerobot.common.robot_devices.robots.configs import So100RobotConfig


def tensor_to_pil(image_tensor):
    """Converts a PyTorch tensor image (H, W, C) to a PIL Image."""
    image_np = image_tensor.cpu().numpy()
    if image_np.dtype == np.float32 and image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    elif image_np.dtype != np.uint8:
        raise ValueError(f"Unsupported image tensor dtype: {image_np.dtype}")
    return PIL.Image.fromarray(image_np)


def parse_json(json_string):
    """Parses a JSON string, handling markdown code blocks and multiple formats."""
    original_string_for_error = str(json_string)
    processed_json_string = json_string
    try:
        patterns = [r"```json\n(.*?)\n```", r"```(.*?)```", r"\[.*?\]"]
        for pattern in patterns:
            match = re.search(pattern, processed_json_string, re.DOTALL)
            if match:
                processed_json_string = match.group(0) if pattern == r"\[.*?\]" else match.group(1)
                break
        return json.loads(processed_json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw input string to parse_json: {original_string_for_error}")
        print(f"String after regex processing (attempted to parse): {processed_json_string}")
        return None
    except Exception as e:
        print(f"Unexpected error in parse_json: {e}")
        print(f"Raw input string to parse_json: {original_string_for_error}")
        return None


def normalize_bbox_0to1(bbox, width, height):
    """Normalizes bounding box coordinates to [0, 1]."""
    x1, y1, x2, y2 = bbox
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def get_2D_bbox(image, prompt, model):
    """Sends an image and prompt to a Gemini model to get 2D bounding box detections."""
    if not isinstance(image, PIL.Image.Image):
        raise TypeError("Image must be a PIL.Image.Image object.")
    contents = [image, prompt]
    response = model.generate_content(contents)
    return response.text


def get_target_bbox(image_tensor, prompt, gemini_model):
    """Identifies pick and place targets from an image using Gemini."""
    image_pil = tensor_to_pil(image_tensor)
    width, height = image_pil.size
    gemini_response_text = get_2D_bbox(image_pil, prompt, gemini_model)
    print(f"Gemini raw response text:\n---\n{gemini_response_text}\n---")
    detected_objects = parse_json(gemini_response_text)

    if not detected_objects:
        print("No objects detected by Gemini or JSON parsing failed.")
        return [], [], [], [], [], []

    pick_boxes, place_boxes, norm_pick_boxes, norm_place_boxes, pick_labels, place_labels = [], [], [], [], [], []
    user_task_match = re.search(r"User request: (.*?)\n", prompt)
    user_task = user_task_match.group(1).strip() if user_task_match else ""
    if user_task: print(f"Extracted user_task: {user_task}")

    place_keywords_in_label = ["bin", "box", "container", "tray", "bowl", "basket"]
    target_place_label_from_task = None
    place_match_in_task = re.search(r"(?:in|to|into) the (.*?)(?:\s|$)", user_task, re.IGNORECASE)
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
        is_place = any(keyword in label for keyword in place_keywords_in_label) or \
                   (target_place_label_from_task and target_place_label_from_task in label)
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
    """Selects a random pick and place target."""
    if not pick_boxes:
        print("No pick targets available.")
        return None, None, None, None, [], [], None, None, []
    if not place_boxes:
        print("No place targets available.")
        return None, None, None, None, pick_boxes, norm_pick_boxes, None, None, pick_labels

    pick_idx = random.randrange(len(pick_boxes))
    place_idx = random.randrange(len(place_boxes))
    return (pick_boxes[pick_idx]["box_2d"], place_boxes[place_idx]["box_2d"],
            norm_pick_boxes[pick_idx], norm_place_boxes[place_idx],
            pick_boxes, norm_pick_boxes, pick_labels[pick_idx], place_labels[place_idx], pick_labels)


def setup_key_listener():
    """Sets up keyboard listener."""
    events = {"exit_early": False, "select_new_bbox": False}
    try:
        from pynput import keyboard
        def on_press(key):
            try:
                if key == keyboard.Key.right: events["exit_early"] = True; print("Exit key pressed.")
                elif key == keyboard.Key.space: events["select_new_bbox"] = True; print("New bbox key pressed.")
            except Exception as e: print(f"Error handling key press: {e}")
        listener = keyboard.Listener(on_press=on_press); listener.start()
        return listener, events
    except ImportError:
        print("pynput not available. Key listener disabled.")
        return None, events


def load_policy(model_name_or_path, device, policy_type="act"):
    """Loads a policy of the specified type (act, pi0, pi0fast)."""
    print(f"Attempting to load {policy_type.upper()} policy from: {model_name_or_path}")
    policy_class = None
    if policy_type.lower() == "act":
        policy_class = ACTPolicy
    elif policy_type.lower() == "pi0":
        policy_class = PI0Policy
    elif policy_type.lower() == "pi0fast":
        policy_class = PI0FASTPolicy
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    try:
        policy = policy_class.from_pretrained(model_name_or_path)
        print(f"Successfully loaded {policy_type.upper()} policy from: {model_name_or_path}")
    except Exception as e:
        print(f"Failed to load {policy_type.upper()} policy from {model_name_or_path}: {e}")
        # Attempt fallback to a known default if the provided path fails
        # Ensure these default paths are correct and exist on Hugging Face Hub
        default_models = {
            "act": "lerobot/act_aloha_sim_transfer_cube_human",
            "pi0": "lerobot/pi0_aloha_sim_transfer_cube_human",  # Verify this exact model name
            # "pi0fast": "lerobot/pi0fast_aloha_sim_transfer_cube_human" # Verify this exact model name
        }
        if policy_type.lower() in default_models and model_name_or_path != default_models[policy_type.lower()]:
            fallback_model_path = default_models[policy_type.lower()]
            print(f"Retrying with default {policy_type.upper()} model: {fallback_model_path}")
            try:
                policy = policy_class.from_pretrained(fallback_model_path)
                print(f"Successfully loaded {policy_type.upper()} policy from fallback: {fallback_model_path}")
            except Exception as e_fallback:
                raise ValueError(f"Could not load {policy_type.upper()} policy from {model_name_or_path} or fallback {fallback_model_path}: {e_fallback}")
        else:
            # If it was already the default or no default for this type, re-raise original error or a new one
            raise ValueError(f"Could not load {policy_type.upper()} policy from {model_name_or_path}. Original error: {e}")


    policy.to(device)
    print(f"{policy_type.upper()} policy loaded and moved to device.")
    if hasattr(policy, 'config'):
        print(f"\n=== {policy_type.upper()} Policy Configuration ===")
        config = policy.config
        for attr in ['input_features', 'output_features', 'n_obs_steps', 'n_action_steps', 'chunk_size']: # Common config attributes
            if hasattr(config, attr) and getattr(config, attr) is not None:
                print(f"{attr}: {getattr(config, attr)}")
        print("=============================\n")
    return policy


prepare_observation_for_policy_map_print_count = 0

def prepare_observation_for_policy(observation, task, device, policy_config):
    """
    Prepare observation batch for policy inference.
    Assumes the policy expects a 14D dual-arm proprioceptive state.
    Does NOT use bounding boxes within this state vector.
    """
    global prepare_observation_for_policy_map_print_count

    current_robot_proprio = observation["observation.state"].to(device) 
    
    gripper_follower_state = torch.tensor([0.0], device=device) # Placeholder for 1 DoF gripper (0.0 often means 'open')

    if current_robot_proprio.shape[0] == 6: # Assuming 6DoF joint state
        follower_arm_full_state = torch.cat([current_robot_proprio, gripper_follower_state]) # Now 7D
    elif current_robot_proprio.shape[0] == 7: # Assuming 6DoF joints + 1DoF gripper already provided
        follower_arm_full_state = current_robot_proprio
    else:
        raise ValueError(f"Unexpected robot proprioception state dimension: {current_robot_proprio.shape[0]}. Expected 6 or 7.")

    leader_arm_q_placeholder = torch.zeros(6, device=device)
    leader_arm_g_placeholder = torch.tensor([0.0], device=device) 
    leader_arm_full_state_placeholder = torch.cat([leader_arm_q_placeholder, leader_arm_g_placeholder]) # 7D

    # Construct the 14D state vector: [follower_arm_state (7D), leader_arm_state (7D)]
    # The order (follower first vs leader first) must match the policy's training.
    # Aloha typically has right arm (often leader) then left arm (often follower).
    # If your 'follower' is, say, left arm, and policy expects right then left:
    # augmented_state = torch.cat([leader_arm_full_state_placeholder, follower_arm_full_state])
    # For now, assuming follower then leader:
    augmented_state = torch.cat([follower_arm_full_state, leader_arm_full_state_placeholder])


    expected_state_dim_from_config = policy_config.input_features['observation.state'].shape[0]
    if augmented_state.shape[0] != expected_state_dim_from_config:
        raise ValueError(
            f"Constructed augmented_state has dimension {augmented_state.shape[0]}, "
            f"but policy config expects {expected_state_dim_from_config}."
        )

    batch_for_policy = {}
    expected_image_keys = [k for k in policy_config.input_features.keys() if 'image' in k]

    for name, value in observation.items():
        if name == "task": continue
        elif "image" in name:
            target_key = name
            if name == "observation.images.primary" and "observation.images.top" in expected_image_keys:
                target_key = "observation.images.top"
                if name != target_key and prepare_observation_for_policy_map_print_count < 5:
                    print(f"Mapping {name} -> {target_key} (will show 5 times then stop)")
                    prepare_observation_for_policy_map_print_count += 1
            
            processed_image = value.type(torch.float32) / 255.0
            processed_image = processed_image.permute(2, 0, 1).contiguous()
            batch_for_policy[target_key] = processed_image.unsqueeze(0).to(device)
        elif name == "observation.state":
            batch_for_policy[name] = augmented_state.unsqueeze(0)
        elif torch.is_tensor(value) and name not in batch_for_policy: 
            batch_for_policy[name] = value.to(device).unsqueeze(0)
        elif name not in batch_for_policy: 
            batch_for_policy[name] = value

    for expected_key in expected_image_keys:
        if expected_key not in batch_for_policy:
            print(f"Warning: Expected image key {expected_key} not found, using zeros.")
            expected_shape = policy_config.input_features[expected_key].shape
            batch_for_policy[expected_key] = torch.zeros(1, *expected_shape, device=device)
    
    batch_for_policy["task"] = [task]
    return batch_for_policy


def main():
    load_dotenv(".env")
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except KeyError: print("GEMINI_API_KEY environment variable not set."); return
    
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    cameras = {"primary": OpenCVCameraConfig(camera_index=2, fps=30, width=640, height=480),
               "top": OpenCVCameraConfig(camera_index=0, fps=30, width=640, height=480)}
    robot_cfg = So100RobotConfig(cameras=cameras, mock=False)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    inference_time_s = 100; fps = 10

    # --- Select Policy Type and Model ---
    # policy_type_to_load = "act"
    # model_to_load = "lerobot/act_aloha_sim_transfer_cube_human"

    # Example for PI0 (ensure the model name is correct on Hugging Face Hub)
    # policy_type_to_load = "pi0"
    # model_to_load = "lerobot/pi0_aloha_sim_transfer_cube_human" # Verify this exact name

    # For this run, let's stick to the ACT policy we've been working with
    policy_type_to_load = "act"
    model_to_load = "lerobot/act_aloha_sim_transfer_cube_human"
    # --- End Select Policy ---

    try:
        policy = load_policy(model_to_load, device, policy_type=policy_type_to_load)
    except Exception as e: print(f"Failed to load policy: {e}"); return

    user_task = "Pickup yellow lego and put into white box."
    prompt = f"""User request: {user_task}
Analyze the provided image and understand what the user needs. Identify all relevant objects on the desk that would help fulfill this request.
Ignore the robot arm itself if visible.
Return your findings strictly as a JSON array of bounding boxes.
Example format: [{{"label": "red lego brick", "box_2d": [100, 200, 150, 280]}}, {{"label": "blue bin", "box_2d": [500, 600, 700, 850]}}]"""

    robot = make_robot_from_config(robot_cfg)
    print("Connecting robot..."); robot.connect(); print("Robot connected.")
    listener = None

    try:
        observation_raw = robot.capture_observation()
        camera_key = next((k for k in ["observation.images.top", "observation.images.primary"] if k in observation_raw),
                          next((k for k in observation_raw if "image" in k), None))
        if not camera_key: print("No camera images found!"); return
        print(f"Using camera: {camera_key}")

        print("\n=== Initial Observation Debug ===");
        for k, v_obs in observation_raw.items(): print(f"  {k}: {v_obs.shape if torch.is_tensor(v_obs) else type(v_obs)}")
        if "observation.state" in observation_raw and hasattr(policy, 'config') and 'observation.state' in policy.config.input_features:
            s_shape = observation_raw["observation.state"].shape[0]
            exp_s_dim = policy.config.input_features['observation.state'].shape[0]
            print(f"Original robot proprio state dim: {s_shape}. Policy expects {exp_s_dim}D state (will be constructed).")
        print("=========================\n")

        pick_boxes, place_boxes, norm_pick_boxes, norm_place_boxes, pick_labels, place_labels = get_target_bbox(
            observation_raw[camera_key], prompt=prompt, gemini_model=gemini_model)
        print("Pick objects:"); [print(f"  - {lbl}") for lbl in pick_labels]
        print("Place locations:"); [print(f"  - {lbl}: {p['box_2d'] if p else 'N/A'}") for lbl, p in zip(place_labels, place_boxes)]

        _, _, _, _, _, _, pick_target_label, place_target_label, _ = \
            get_random_targets(pick_boxes, place_boxes, norm_pick_boxes, norm_place_boxes, pick_labels, place_labels)
        
        if pick_target_label is None or place_target_label is None:
            print("Could not determine valid pick and/or place target labels for task string. Exiting.")
            return
        
        single_task = f"Grasp {pick_target_label} and put it in {place_target_label}"
        print(f"Current task: {single_task}")
        listener, events = setup_key_listener()
        print("Starting control loop...")

        for step in range(inference_time_s * fps):
            start_time = time.perf_counter()
            if events["exit_early"]: print("Early exit."); break
            
            observation = robot.capture_observation()
            try:
                batch_for_policy = prepare_observation_for_policy(
                    observation, single_task, device, policy.config
                )

                action_chunk = policy.select_action(batch_for_policy)
                
                if action_chunk.ndim == 3: 
                    current_action = action_chunk[0, 0, :].to("cpu")
                elif action_chunk.ndim == 2: 
                    current_action = action_chunk[0, :].to("cpu") 
                else:
                    raise ValueError(f"Unexpected action_chunk dimension: {action_chunk.ndim}. Shape: {action_chunk.shape}")

                if step % fps == 0:
                    print(f"\n--- Step {step} ---")
                    print(f"  action_chunk.shape from policy: {action_chunk.shape}")
                    print(f"  Current_action to send (shape {current_action.shape}):")
                    action_list_rounded = [round(x, 4) for x in current_action.tolist()]
                    print(f"    Values: {action_list_rounded}")
                    
                    raw_robot_state = observation["observation.state"].cpu().numpy()
                    raw_robot_state_rounded = [round(x, 4) for x in raw_robot_state.tolist()]
                    print(f"  Raw robot state from obs (len {len(raw_robot_state_rounded)}): {raw_robot_state_rounded}")

                    augmented_state_for_log = batch_for_policy['observation.state'].squeeze().cpu().numpy()
                    augmented_state_rounded = [round(x, 4) for x in augmented_state_for_log.tolist()]
                    print(f"  State sent to policy (len {len(augmented_state_rounded)}): {augmented_state_rounded}")

                robot.send_action(current_action)
            except Exception as e:
                print(f"Error in control step {step}: {e}"); import traceback; traceback.print_exc(); break

            if events["select_new_bbox"]:
                events["select_new_bbox"] = False; print("Re-selecting targets for task string...")
                obs_raw_fresh = robot.capture_observation()
                pick_f, place_f, norm_pick_f, norm_place_f, p_labels_f, pl_labels_f = get_target_bbox(
                    obs_raw_fresh[camera_key], prompt=prompt, gemini_model=gemini_model)
                _, _, _, _, _, _, ptl_f, pltl_f, _ = get_random_targets(
                    pick_f, place_f, norm_pick_f, norm_place_f, p_labels_f, pl_labels_f)

                if ptl_f is not None and pltl_f is not None:
                    pick_target_label, place_target_label = ptl_f, pltl_f
                    single_task = f"Grasp {pick_target_label} and put it in {place_target_label}"
                    print(f"New task string: {single_task}")
                else:
                    print("Could not get new valid target labels. Continuing with previous task string.")
            
            busy_wait(1/fps - (time.perf_counter() - start_time))

    except KeyboardInterrupt: print("\nInterrupted by user.")
    except Exception as e: print(f"Unexpected error in main: {e}"); import traceback; traceback.print_exc()
    finally:
        if robot and robot.is_connected: robot.disconnect()
        print("Robot disconnected.")
        if listener: listener.stop()

if __name__ == "__main__":
    main()