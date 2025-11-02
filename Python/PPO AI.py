import time
import numpy as np
import mmap
import struct

# Optional: uncomment if you have SB3 installed and a trained model file
# from stable_baselines3 import PPO

# ------------------------
# Configuration (tweak these)
# ------------------------
slots_config = {
    'ray_distances': (0, 8),   # slots 0-7
    'ray_hits': (8, 16),       # slots 8-15
    'reward': (16, 17),        # slot 16
    'done': (17, 18),          # slot 17
    'actions': (18, 20),       # slots 18-19: speed, steering
    'speed': (20, 21)          # slot 20: current speed (read-only)
}
slot_count = 21  # total slots
slot_size = 4    # float32 per slot

# How often we let the policy choose a new action (seconds)
decision_period = 0.05  # 20 Hz decision frequency (model runs at this cadence)

# MINIMUM time an action must be held before allowing a new change (seconds).
# This enforces the "key held" fairness; set to match human input responsiveness.
min_action_duration = 0.12  # e.g., ~120 ms; tune to match Unity Input sensitivity if needed

# -------------------------
# Memory map setup (auto-create)
# -------------------------
size_bytes = slot_count * slot_size
mm = mmap.mmap(-1, size_bytes, tagname='unity_ram', access=mmap.ACCESS_WRITE)

# -------------------------
# Helpers: read/write float32 slots
# -------------------------
def read_slots(start, end):
    mm.seek(start * slot_size)
    vals = []
    for _ in range(end - start):
        b = mm.read(slot_size)
        vals.append(struct.unpack('<f', b)[0])
    return vals

def write_slots(start, end, values):
    # write float32 values into consecutive slots
    mm.seek(start * slot_size)
    for v in values:
        mm.write(struct.pack('<f', float(v)))
    mm.flush()

# -------------------------
# Action mapping utilities
# -------------------------
# Action values that will be written to Unity: -1, 0, +1
mapping = [-1.0, 0.0, 1.0]

def action_index_to_cmd(index):
    """Convert discrete index 0/1/2 -> -1/0/+1"""
    return mapping[int(index)]

# -------------------------
# (Optional) Load a trained SB3 policy here
# -------------------------
# If you have a saved model, uncomment and change path:
# model = PPO.load("path_to_your_model.zip")
# The environment action_space used during training should be MultiDiscrete([3,3])
#
# When using a model: call
#   action, _states = model.predict(obs, deterministic=False)
# where action will be an array-like of two ints in {0,1,2}

# -------------------------
# Example fallback policy: random (replace with model.predict)
# -------------------------
import random
def sample_random_action():
    return (random.randrange(3), random.randrange(3))  # (speed_idx, steer_idx)

# -------------------------
# Main loop
# -------------------------
try:
    # store last action indices and timestamps (start neutral)
    last_speed_idx = 1   # index 1 -> mapping[1] == 0.0 (neutral)
    last_steer_idx = 1
    last_action_time = time.time()

    next_decision_time = time.time()

    while True:
        # Read state from Unity
        ray_distances = read_slots(*slots_config['ray_distances'])
        ray_hits = read_slots(*slots_config['ray_hits'])
        reward = read_slots(*slots_config['reward'])[0]
        done = bool(read_slots(*slots_config['done'])[0])
        current_speed = read_slots(*slots_config['speed'])[0]

        now = time.time()

        # Only ask model for a new action at decision_period intervals
        if now >= next_decision_time:
            next_decision_time = now + decision_period

            # --- Replace the following line with model.predict when using SB3 ---
            # Example:
            # obs = build_observation(ray_distances, ray_hits, current_speed, ...)
            # action_arr, _states = model.predict(obs, deterministic=False)
            # speed_idx, steer_idx = int(action_arr[0]), int(action_arr[1])

            # Fallback random (for testing) — replace this with your model.predict
            speed_idx, steer_idx = sample_random_action()

            # Enforce min_action_duration: don't allow immediate flip if too soon
            time_since_last = now - last_action_time
            if time_since_last < min_action_duration:
                # keep previous actions (hold)
                speed_idx = last_speed_idx
                steer_idx = last_steer_idx
            else:
                # Accept the new action and update last_action_time
                # (If you want to restrict only certain transitions, add logic here)
                last_action_time = now
                last_speed_idx = speed_idx
                last_steer_idx = steer_idx

            # Map discrete indices 0/1/2 -> -1/0/1
            speed_cmd = action_index_to_cmd(speed_idx)
            steer_cmd = action_index_to_cmd(steer_idx)

            # Write actions to shared memory
            actions = [speed_cmd, steer_cmd]
            write_slots(*slots_config['actions'], values=actions)

        # Debug print (throttle down frequency)
        # You can print less often or log to file to avoid slowing loop
        # Print current status at some interval:
        print(f"[{time.strftime('%H:%M:%S')}] Reward:{reward:.3f} Done:{done} Speed:{current_speed:.3f} Actions:{[last_speed_idx,last_steer_idx]} -> {action_index_to_cmd(last_speed_idx)},{action_index_to_cmd(last_steer_idx)} Rays:{ray_distances} Hits:{ray_hits}")
        # Sleep briefly to avoid busy loop; decision cadence controls model calls
        time.sleep(0.01)

except KeyboardInterrupt:
    mm.close()
