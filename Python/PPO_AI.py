#!/usr/bin/env python3
"""
unity_sb3_runner.py

- Uses a shared memory layout (unity_ram) to talk with Unity.
- Action space: MultiDiscrete([3,3])  -> indices {0,1,2} mapped to [-1,0,1]
- Two primary modes:
    * infer  : loads a saved SB3 model and runs policy, writing actions to Unity
    * train  : wraps the Unity memory as a Gym Env and runs SB3.PPO.learn(...)
"""

import argparse
import time
import numpy as np
import mmap
import struct
import os

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces

# ------------------------
# Shared-memory configuration (match your Unity layout)
# ------------------------
slots_config = {
    'ray_distances': (0, 8),   # slots 0-7
    'ray_hits': (8, 16),       # slots 8-15
    'reward': (16, 17),        # slot 16
    'done': (17, 18),          # slot 17
    'actions': (18, 20),       # slots 18-19: speed, steering (write)
    'speed': (20, 21)          # slot 20: current speed (read-only)
}
slot_count = 21
slot_size = 4  # float32
size_bytes = slot_count * slot_size
TAGNAME = 'unity_ram'  # mmap tag used by Unity too

# Mapping discrete indices -> -1,0,1
INDEX_TO_CMD = [-1.0, 0.0, 1.0]

global Episode
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
# Simple Gym Env wrapper using shared memory
# -------------------------
class UnityRAMEnv(gym.Env):
    """
    Minimal Gym wrapper that interacts with Unity via shared memory.

    Observation:
        - ray_distances (8)
        - ray_hits (8)
        - current_speed (1)
        => total obs shape = (17,)
    Action:
        MultiDiscrete([3,3])  # speed_idx, steer_idx
    Step semantics:
        - write action into slots_config['actions']
        - sleep for step_wait seconds (allow Unity to simulate)
        - read new obs/reward/done
    NOTE: Depending on your Unity setup you may need to adapt reset() and step() logic.
    """
    def __init__(self, mm, step_wait=0.04):
        super().__init__()
        self.mm = mm
        self.step_wait = step_wait
        self.episode_steps = 0

        # action and observation spaces
        self.action_space = spaces.MultiDiscrete([3, 3])
        # bounds: rays probably >=0; use wide bounds to be safe
        low = np.array([-1000.0] * 17, dtype=np.float32)
        high = np.array([1000.0] * 17, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _read_obs(self):
        rays = read_slots(self.mm, *slots_config['ray_distances'])      # 8 floats
        hits = read_slots(self.mm, *slots_config['ray_hits'])          # 8 floats
        speed = read_slots(self.mm, *slots_config['speed'])[0]         # 1 float
        obs = np.array(list(rays) + list(hits) + [speed], dtype=np.float32)
        return obs

    def reset(self):
        # Optionally: write neutral actions and wait a short bit
        write_slots(self.mm, *slots_config['actions'], values=[0.0, 0.0])  # neutral
        time.sleep(self.step_wait)
        self.episode_steps = 0
        obs = self._read_obs()
        return obs

    def step(self, action):
        """
        action: array-like of two ints (0..2)
        """
        self.episode_steps += 1
        

        speed_idx = int(action[0])
        steer_idx = int(action[1])
        speed_cmd = INDEX_TO_CMD[speed_idx]
        steer_cmd = INDEX_TO_CMD[steer_idx]

        # Write actions
        write_slots(self.mm, *slots_config['actions'], values=[speed_cmd, steer_cmd])

        # Allow Unity to step forward
        time.sleep(self.step_wait)

        # Read obs/reward/done
        obs = self._read_obs()
        reward = read_slots(self.mm, *slots_config['reward'])[0]
        done = bool(read_slots(self.mm, *slots_config['done'])[0])
        if done:
            global Episode
            Episode+=1
            print("Episode: ", Episode)


        info = {}
        print(f"Step: {self.episode_steps} | Speed cmd: {speed_cmd}, Steer cmd: {steer_cmd} | Reward: {reward:.3f} | Done: {done}")
        # print("Action indices:", action)
        return obs, float(reward), done, info


# -------------------------
# Inference loop: load model & run
# -------------------------
def run_inference(mm, model_path, deterministic=False):
    """
    Continuously read obs, call model.predict, and write actions to Unity.
    - deterministic: whether to use deterministic policy (True) or stochastic (False)
    - sleep_between_steps: small sleep to avoid 100% busy loop (seconds)
    """
    # Build a dummy observation of the correct shape for predict call
    # Model was trained with obs shape (17,)
    # We read obs directly from the map.
    model = PPO.load(model_path)

    print("Loaded model:", model_path)
    try:
        while True:
            obs = np.array(read_slots(mm, *slots_config['ray_distances']) +
                           read_slots(mm, *slots_config['ray_hits']) +
                           [read_slots(mm, *slots_config['speed'])[0]], dtype=np.float32)
            # reshaped to (n,) or (1, n) depending on model expectations
            # SB3 accepts 1D obs
            action, _states = model.predict(obs, deterministic=deterministic)
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
# Training helper: train PPO with Unity env
# -------------------------
def train_model(mm, model_path, step_wait=0.04):
    def make_env():
        return UnityRAMEnv(mm, step_wait=step_wait)

    venv = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", venv, verbose=2, policy_kwargs=dict(net_arch=[256, 256]))


    print("Starting adaptive training...")
    model.learn(total_timesteps=int(1e10))
    model.save(model_path)
    print("Training complete. Best model saved.")

# -------------------------
# CLI entrypoint
# -------------------------
def main():
    import os

    # Relative model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "car_agent")

    # Mode and other configs hardcoded
    print("(T)rain or (I)nfer?",end=' ')
    if input().lower().startswith('t'):
        mode = "train"
    else:
        mode = "infer"
    deterministic = True
    stepwait = 0.04
    timesteps = 10000#20000

    # open mmap
    mm = open_mmap()

    if mode == "infer":
        run_inference(mm, model_path, deterministic=deterministic)
    elif mode == "train":
        train_model(mm, model_path, step_wait=stepwait)

    mm.close()


if __name__ == "__main__":
    main()




