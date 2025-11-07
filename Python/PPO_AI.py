#!/usr/bin/env python3
"""
unity_sb3_runner_episode.py

Episode-based version of unity_sb3_runner.py

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
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics

# ------------------------
# Shared-memory configuration (match your Unity layout)
# -----------------------
slots_config = {
    'ray_distances': (0, 8),   # slots 0-7
    'ray_hits': (8, 16),       # slots 8-15
    'reward': (16, 17),        # slot 16
    'done': (17, 18),          # slot 17
    'actions': (18, 20),       # slots 18-19: speed, steering (write)
    'speed': (20, 21),          # slot 20: current speed (read-only)
    'Wall_distances': (21, 29)   # slots 21-28
}
slot_count = 29
slot_size = 4  # float32
size_bytes = slot_count * slot_size
TAGNAME = 'unity_ram'  # mmap tag used by Unity too

# Mapping discrete indices -> -1,0,1
INDEX_TO_CMD = [-1.0, 0.0, 1.0]

# -------------------------
from stable_baselines3.common.callbacks import BaseCallback

class UnityEpisodeCallback(BaseCallback):
    """
    Tracks episodes using Unity's 'done' slot (reads shared memory directly).
    Works with DummyVecEnv where each env in vec_env.envs is your UnityRAMEnv instance.
    Prints when an episode ends: episode number, steps, cumulative reward.
    """
    def __init__(self, slots_config, verbose=1):
        super().__init__(verbose)
        self.slots_config = slots_config
        self.episode_counts = None    # list, one per env
        self.episode_steps = None
        self.episode_rewards = None
        self.prev_done = None

    def _get_mm_from_wrapped(self, env):
        # support wrapper layers: env, unwrapped, etc.
        if hasattr(env, "mm"):
            return env.mm
        inner = getattr(env, "env", None)
        if inner is not None and hasattr(inner, "mm"):
            return inner.mm
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None and hasattr(unwrapped, "mm"):
            return unwrapped.mm
        return None

    def _on_training_start(self):
        # Called once when training begins. Initialize arrays sized to number of envs.
        vec_env = self.model.get_env()
        real_envs = getattr(vec_env, "envs", None)
        n = len(real_envs) if real_envs is not None else 1
        self.episode_counts = [0] * n
        self.episode_steps = [0] * n
        self.episode_rewards = [0.0] * n
        self.prev_done = [False] * n

    def _on_step(self) -> bool:
        vec_env = self.model.get_env()
        real_envs = getattr(vec_env, "envs", None)
        if real_envs is None:
            return True

        for idx, env in enumerate(real_envs):
            self.episode_steps[idx] += 1

            mm = self._get_mm_from_wrapped(env)
            if mm is None:
                continue

            # Read current reward from Unity
            try:
                r = float(read_slots(mm, *self.slots_config['reward'])[0])
            except Exception:
                r = 0.0
            self.episode_rewards[idx] += r

            # Read done flag from Unity
            try:
                done_flag = bool(read_slots(mm, *self.slots_config['done'])[0])
            except Exception:
                done_flag = False

            # # Debug prints
            # if self.episode_steps[idx] % 100 == 0:
            #     print(f"[UnityEpisodeCallback] Env#{idx} Step: {self.episode_steps[idx]} | Done: {done_flag} | Prev Done: {self.prev_done[idx]}")

            # Detect rising edge
            if done_flag and not self.prev_done[idx]:
                self.episode_counts[idx] += 1
                ep = self.episode_counts[idx]
                ep_r = self.episode_rewards[idx]
                ep_l = self.episode_steps[idx]
                print(f"[UnityEpisodeCallback1] Env#{idx} Episode {ep} ended | reward={ep_r:.3f} | steps={ep_l}")
                self.episode_steps[idx] = 0
                self.episode_rewards[idx] = 0.0
                self.prev_done[idx] = True
            elif not done_flag:
                self.prev_done[idx] = False

        return True


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
# Simple Gym Env wrapper using shared memory (episode-based)
# -------------------------
    """
    Minimal Gym wrapper that interacts with Unity via shared memory.

    Observation:
        - wall_distances (8)
        - ray_distances (8)
        - ray_hits (8)
        - current_speed (1)
        => total obs shape = (25,)

    Action:
        MultiDiscrete([3,3])  # speed_idx, steer_idx

    Step semantics:
        - write action into slots_config['actions']
        - sleep for step_wait seconds (allow Unity to simulate)
        - read new obs/reward/done

    Reset semantics (episode-based):
        - write neutral actions and wait until Unity clears its `done` flag
          (within a timeout) and then read environment observation.
    """
class UnityRAMEnv(gym.Env):
    def __init__(self, mm, step_wait=0.04):
        super().__init__()
        self.mm = mm
        self.step_wait = step_wait
        self.action_space = gym.spaces.MultiDiscrete([3, 3])
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(25,), dtype=np.float32)
        self._ep_reward = 0.0
        self._ep_steps = 0

    def _read_obs(self):
        Hitrays = read_slots(self.mm, *slots_config['ray_distances'])
        Wallrays = read_slots(self.mm, *slots_config['Wall_distances'])
        hits = read_slots(self.mm, *slots_config['ray_hits'])
        speed = read_slots(self.mm, *slots_config['speed'])[0]
        obs = np.array(list(Wallrays) + list(Hitrays) + list(hits) + [speed], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=1.0, posinf=1.0, neginf=-1.0)
        return obs

    # follow Gymnasium API: reset returns (obs, info)
    def reset(self, *, seed=None, options=None):
        print("Unity environment reset triggered by SB3.")
        write_slots(self.mm, *slots_config['actions'], values=[0.0, 0.0])
        start = time.time()
        while bool(read_slots(self.mm, *slots_config['done'])[0]):
            if (time.time() - start) > 5.0:
                print("Reset: timeout waiting for Unity to clear done flag; proceeding.")
                break
            time.sleep(0.01)
        obs = self._read_obs()
        self._ep_reward = 0.0
        self._ep_steps = 0
        return obs, {}

    # follow Gymnasium API: step returns (obs, reward, terminated, truncated, info)
    def step(self, action):
        speed_cmd = INDEX_TO_CMD[int(action[0])]
        steer_cmd = INDEX_TO_CMD[int(action[1])]
        write_slots(self.mm, *slots_config['actions'], values=[speed_cmd, steer_cmd])
        time.sleep(self.step_wait)

        obs = self._read_obs()
        reward = float(read_slots(self.mm, *slots_config['reward'])[0])
        done = bool(read_slots(self.mm, *slots_config['done'])[0])

        self._ep_reward += reward
        self._ep_steps += 1

        terminated = done
        truncated = False
        info = {}

        if terminated or truncated:
            # This tells SB3 that the episode is finished
            info["episode"] = {"r": self._ep_reward, "l": self._ep_steps}
            print(f"[Env] Episode ended (from step): reward={self._ep_reward:.3f}, steps={self._ep_steps}")
            self._ep_reward = 0.0
            self._ep_steps = 0

        return obs, reward, terminated, truncated, info



# -------------------------
# Inference loop: load model & run (episode-aware)
# -------------------------
def run_inference(mm, model_path, deterministic=False, step_wait=0.04):
    """
    Uses UnityRAMEnv so inference is episode-aware.
    On episode completion, prints episode summary and waits for Unity to clear `done`.
    """
    env = UnityRAMEnv(mm, step_wait=step_wait)
    # load model
    model = PPO.load(model_path)
    print("Loaded model:", model_path)

    # handle both gym and gymnasium reset return styles
    res = env.reset()
    if isinstance(res, tuple):
        obs = res[0]
    else:
        obs = res

    episode_reward = 0.0
    episode_len = 0
    episode_count = 0

    try:
        while True:
            # Model expects 1D obs
            action, _states = model.predict(obs, deterministic=deterministic)
            step_ret = env.step(action)

            # support both old (obs, reward, done, info) and gymnasium (obs, reward, terminated, truncated, info)
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_ret

            episode_reward += float(reward)
            episode_len += 1

            # Optional debugging output every N steps (comment out if noisy)
            current_speed = read_slots(mm, *slots_config['speed'])[0]
            print(f"Infer | Ep#{episode_count+1} Step:{episode_len} | speed={current_speed:.3f} | action={action} | r={reward:.3f} done={done}")

            if done:
                episode_count += 1
                # Print summary (if env provided episode info)
                ep_info = info.get('episode', {'r': episode_reward, 'l': episode_len})
                print(f"=== Episode {episode_count} ended: reward={ep_info['r']:.3f} length={ep_info['l']} ===")

                # Wait for Unity to clear done flag before starting next episode (similar to reset logic)
                start = time.time()
                while True:
                    done_flag = bool(read_slots(mm, *slots_config['done'])[0])
                    if not done_flag:
                        break
                    if (time.time() - start) > 5.0:
                        print("Inference: timeout waiting for Unity to clear done flag; proceeding.")
                        break
                    time.sleep(0.01)

                # reset local counters and call env.reset to sync (handle both return styles)
                episode_reward = 0.0
                episode_len = 0
                res = env.reset()
                obs = res[0] if isinstance(res, tuple) else res

    except KeyboardInterrupt:
        print("Inference interrupted by user.")
    finally:
        # neutralize actions before exit
        write_slots(mm, *slots_config['actions'], values=[0.0, 0.0])
# -------------------------
# Training helper: train PPO with Unity env (episode-based)
# -------------------------
def train_model(mm, model_path, step_wait=0.04):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
   
    # -------------------------------
    # Training env (SB3 expects a callable)
    # -------------------------------
    train_env = DummyVecEnv([
    lambda: RecordEpisodeStatistics(UnityRAMEnv(mm, step_wait=step_wait))
])


    # -------------------------------
    # Model
    # -------------------------------
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=2,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    # -------------------------------
    # Eval env: separate instance
    # -------------------------------
    eval_env = DummyVecEnv([
        lambda: RecordEpisodeStatistics(UnityRAMEnv(mm, step_wait=step_wait))
    ])

    from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
    from stable_baselines3.common.callbacks import CallbackList

    stop_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=8,
    min_evals=5,
    verbose=2
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path='./best_model/',
        log_path='./logs/',
        eval_freq=5000,
        callback_after_eval=stop_callback
    )

    unity_episode_cb = UnityEpisodeCallback(slots_config=slots_config, verbose=1)
    callback = CallbackList([eval_callback, unity_episode_cb])

    print("Starting adaptive training (episode-based)...")
    model.learn(total_timesteps=int(1e10), callback=callback)

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
    print("(T)rain or (I)nfer?", end=' ')
    if input().lower().startswith('t'):
        mode = "train"
    else:
        mode = "infer"
    deterministic = True
    stepwait = 0.04

    # open mmap
    mm = open_mmap()

    if mode == "infer":
        run_inference(mm, model_path, deterministic=deterministic, step_wait=stepwait)
    elif mode == "train":
        train_model(mm, model_path, step_wait=stepwait)

    mm.close()


if __name__ == "__main__":
    main()
