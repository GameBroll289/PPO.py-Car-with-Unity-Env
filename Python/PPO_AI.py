#!/usr/bin/env python3
"""
unity_sb3_runner.py

- Uses a shared memory layout (unity_ram) to talk with Unity.
- Action space: MultiDiscrete([3,3])  -> indices {0,1,2} mapped to [-1,0,1]
- Two primary modes:
    * infer  : loads a saved SB3 model and runs policy, writing actions to Unity
    * train  : wraps the Unity memory as a Gym Env and runs SB3.PPO.learn(...)
"""
import time
import numpy as np
import mmap
import struct
import os

# Shared-memory configuration (match your Unity layout)
# ------------------------
slots_config = {
    'ray_distances': (0, 8),   # slots 0-7
    'ray_hits': (8, 16),       # slots 8-15
    'reward': (16, 17),        # slot 16
    'done': (17, 18),          # slot 17
    'actions': (18, 20),       # slots 18-19: speed, steering (write)
    'speed': (20, 21),          # slot 20: current speed (read-only)
    'Wall_distances': (21, 29)   # slots 0-7
}

#25 OBS
slot_count = 29
slot_size = 4  # float32
size_bytes = slot_count * slot_size
TAGNAME = 'unity_ram'  # mmap tag used by Unity too

# Mapping discrete indices -> -1,0,1
INDEX_TO_CMD = [-1.0, 0.0, 1.0]

global Episode, step_wait, episode_steps
Episode=0

# -------------------------
# Memory map helpers
# -------------------------
def open_mmap():
    mm = mmap.mmap(-1, size_bytes, tagname=TAGNAME, access=mmap.ACCESS_WRITE)
    return mm

def read_slots(mm, start, end):
    mm.seek(start * slot_size)
    vals = []
    for _ in range(end - start):
        b = mm.read(slot_size)
        vals.append(struct.unpack('<f', b)[0])
    return vals

def write_slots(mm, start, end, values):
    mm.seek(start * slot_size)
    for v in values:
        mm.write(struct.pack('<f', float(v)))
    mm.flush()

# -------------------------
def Start(mm):
    episode_steps = 0

def Read_obs():
    Hitrays = read_slots(mm, *slots_config['ray_distances'])      # 8 floats
    Wallrays = read_slots(mm, *slots_config['Wall_distances'])      # 8 floats
    hits = read_slots(mm, *slots_config['ray_hits'])          # 8 floats
    speed = read_slots(mm, *slots_config['speed'])[0]         # 1 float
    obs = np.array(list(Wallrays) + list(Hitrays) + list(hits) + [speed], dtype=np.float32)
    
        # Sanity checks: replace any NaN/Inf with a safe fallback
    if np.isnan(obs).any() or np.isinf(obs).any():
        print("WARNING: invalid observation detected, replacing NaN/Inf with large value. obs:", obs)
        obs = np.nan_to_num(obs, nan=1.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    return obs

def reset():
    # Optionally: write neutral actions and wait a short bit
    write_slots(mm, *slots_config['actions'], values=[0.0, 0.0])  # neutral
    time.sleep(step_wait)
    episode_steps = 0
    obs = Read_obs()
    return obs

def step(action):
    """
    action: array-like of two ints (0..2)
    """
    episode_steps += 1

    speed_idx = int(action[0])
    steer_idx = int(action[1])
    speed_cmd = INDEX_TO_CMD[speed_idx]
    steer_cmd = INDEX_TO_CMD[steer_idx]

    # Write actions
    write_slots(mm, *slots_config['actions'], values=[speed_cmd, steer_cmd])

    # Allow Unity to step forward
    time.sleep(step_wait)

    # Read obs/reward/done
    obs = Read_obs()
    reward = read_slots(mm, *slots_config['reward'])[0]
    done = bool(read_slots(mm, *slots_config['done'])[0])
    if done:
        global Episode
        Episode+=1
        print("Episode: ", Episode)

    info = {}
    #print(f"Step: {self.episode_steps} | Speed cmd: {speed_cmd}, Steer cmd: {steer_cmd} | Reward: {reward:.3f} | Done: {done}")
    # print("Action indices:", action)
    return obs, float(reward), done, info

# -------------------------
# Inference loop: load model & run
# -------------------------
def load_model(mm, model_path):
    """
    Continuously read obs, call model.predict, and write actions to Unity.
    - deterministic: whether to use deterministic policy (True) or stochastic (False)
    - sleep_between_steps: small sleep to avoid 100% busy loop (seconds)
    """
    # Build a dummy observation of the correct shape for predict call
    # Model was trained with obs shape (17,)
    # We read obs directly from the map.
    model = load(model_path)

    print("Loaded model:", model_path)
    try:
        while True:
            obs = np.array(read_slots(mm, *slots_config['Wall_distances']) +
                           read_slots(mm, *slots_config['ray_distances']) +
                           read_slots(mm, *slots_config['ray_hits']) +
                           [read_slots(mm, *slots_config['speed'])[0]], dtype=np.float32)
            # reshaped to (n,) or (1, n) depending on model expectations
            # SB3 accepts 1D obs
            action, _states = model.predict(obs)
            # action should be array-like [speed_idx, steer_idx]
            speed_idx = int(action[0])
            steer_idx = int(action[1])
            speed_cmd = INDEX_TO_CMD[speed_idx]
            steer_cmd = INDEX_TO_CMD[steer_idx]

            write_slots(mm, *slots_config['actions'], values=[speed_cmd, steer_cmd])

            # optional debug print (comment out if noisy)
            reward = read_slots(mm, *slots_config['reward'])[0]
            done = bool(read_slots(mm, *slots_config['done'])[0])
            current_speed = read_slots(mm, *slots_config['speed'])[0]
            print(f"Obs speed={current_speed:.3f} | action idx=({speed_idx},{steer_idx}) -> cmds=({speed_cmd},{steer_cmd}) | reward={reward:.3f} done={done}")

    except KeyboardInterrupt:
        print("Inference interrupted by user.")


# -------------------------
def train_model(mm, model_path):

    #Make Model

    print("Starting adaptive training...")
    model.fit()
    model.save(model_path)
    print("Training complete. Best model saved.")
# -------------------------
def main():
    # Relative model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "car_agent")

    # Mode and other configs hardcoded
    print("(T)rain or (I)nfer?",end=' ')
    if input().lower().startswith('t'):
        mode = "train"
    else:
        mode = "load"
    stepwait = 0.04
    # open mmap
    mm = open_mmap()

    if mode == "load":
        load_model(mm, model_path)
    elif mode == "train":
        train_model(mm, model_path, step_wait=stepwait)

    mm.close()


if __name__ == "__main__":
    main()




