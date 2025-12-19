import time
import numpy as np
import mmap
import struct
import os

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Shared-memory configuration (match your Unity layout)
# ------------------------
slots_config = {
    'ray_distances': (0, 8),   # slots 0-7
    'ray_hits': (8, 16),       # slots 8-15
    'reward': (16, 17),        # slot 16
    'done': (17, 18),          # slot 17
    'actions': (18, 20),       # slots 18-19: speed, steering (write)
    'speed': (20, 21),          # slot 20: current speed (read-only)
    'Wall_distances': (21, 29),   # slots 21-28
    'Time_Remaining': (28, 29)   # slot 28-29
}

#25 OBS
slot_count = 30
slot_size = 4  # float32
size_bytes = slot_count * slot_size
TAGNAME = 'unity_ram'  # mmap tag used by Unity too

# Mapping discrete indices -> -1,0,1
INDEX_TO_CMD = [-1.0, 0.0, 1.0]

global Episode, Episodes_Per_Batch, step_wait, episode_steps, Epochs, MINI_BATCH_SIZE, mm, BATCH_SIZE_TARGET
BATCH_SIZE_TARGET=1200
Episode=0
episode_steps=0
Epochs=10
MINI_BATCH_SIZE = 64 # Or another power of 2, often 64 or 128


#Hyper parameters
gamma=0.9 #How much later rewards are worth, Goes from 0.95=< to >=0.99. With higher values, the agent will consider future rewards more strongly.
gae_lambda=0.95
entropy_coefficient=0.01
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
def Read_obs():
    Hitrays = read_slots(mm, *slots_config['ray_distances'])      # 8 floats
    Wallrays = read_slots(mm, *slots_config['Wall_distances'])      # 8 floats
    hits = read_slots(mm, *slots_config['ray_hits'])          # 8 floats
    speed = read_slots(mm, *slots_config['speed'])[0]         # 1 float
    Time_Remaining = read_slots(mm, *slots_config['Time_Remaining'])[0]  # 1 float
    obs = np.array(list(Wallrays) + list(Hitrays) + list(hits) + [speed] + [Time_Remaining], dtype=np.float32)
    
        # Sanity checks: replace any NaN/Inf with a safe fallback
    if np.isnan(obs).any() or np.isinf(obs).any():
        print("WARNING: invalid observation detected, replacing NaN/Inf with large value. obs:", obs)
        obs = np.nan_to_num(obs, nan=1.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    return obs

def reset():
    global Episode, episode_steps
    # Optionally: write neutral actions and wait a short bit
    write_slots(mm, *slots_config['actions'], values=[0.0, 0.0])  # neutral
    time.sleep(step_wait)
    episode_steps = 0
    print(f"Starting Episode: {Episode}")

def step(action_logits,value):
    global episode_steps, Episode
    episode_steps += 1
    
    # 1. Split the 6 logits into Speed (3) and Steer (3)
    speed_logits, steer_logits = tf.split(action_logits, num_or_size_splits=2, axis=0)
    
    # 2. Create distributions and sample INDEPENDENTLY
    speed_dist = tfp.distributions.Categorical(logits=speed_logits)
    steer_dist = tfp.distributions.Categorical(logits=steer_logits)
    
    speed_action = speed_dist.sample()
    steer_action = steer_dist.sample()
    
    # 3. Get Log Probs (We need both!)
    speed_log = speed_dist.log_prob(speed_action)
    steer_log = steer_dist.log_prob(steer_action)
    total_log_prob = speed_log + steer_log
    
    # 4. Convert to Python integers for the mmap
    speed_idx = int(speed_action)
    steer_idx = int(steer_action)
    
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
    
    # Save the combined action [speed, steer] for the training buffer
    final_action_indices = [speed_idx, steer_idx]
    
    trajectories(obs, final_action_indices, total_log_prob, reward, done, value)

    if done:
        print(f"Episode {Episode} ended after {episode_steps} steps.")

    return obs, float(reward), done

def trajectories(obs, action, log_prob, reward, done, value):
    all_obs.append(obs)
    all_actions.append(action)
    all_log_probs.append(log_prob)
    all_rewards.append(reward)
    all_dones.append(done)
    all_values.append(value)
    
    
# -------------------------
# Inference loop: load actor & run
# -------------------------
def load_model(mm, model_path):
    """
    Inference loop: Load raw SavedModel & run.
    Uses tf.saved_model.load to avoid Keras 3 format errors.
    """
    print(f"Loading SavedModel from: {model_path}")
    
    # FIX 1: Use low-level loader (Works with Keras 3 and SavedModel folders)
    imported_model = tf.saved_model.load(model_path)
    inference_func = imported_model.signatures["serving_default"]

    print("Model loaded successfully. Starting inference...")
    try:
        while True:
            # 1. Read Observations
            obs = np.array(read_slots(mm, *slots_config['Wall_distances']) +
                           read_slots(mm, *slots_config['ray_distances']) +
                           read_slots(mm, *slots_config['ray_hits']) +
                           [read_slots(mm, *slots_config['speed'])[0]] + 
                           [read_slots(mm, *slots_config['Time_Remaining'])[0]], dtype=np.float32) # Added Time_Remaining
            
            # 2. Prepare Input Tensor (Add batch dimension: Shape (1, 26))
            input_tensor = tf.constant(obs[None, :], dtype=tf.float32)

            # 3. Run Prediction
            # Output is a dictionary, we grab the first tensor (the logits)
            predictions = inference_func(input_tensor)
            logits = list(predictions.values())[0] # Shape (1, 6)
            
            # FIX 2: Convert Logits to Action Indices
            # The model outputs 6 raw numbers. We must split them and find the highest one.
            speed_logits = logits[0, :3] # First 3 numbers
            steer_logits = logits[0, 3:] # Last 3 numbers
            
            # Use argmax to pick the best action (Deterministic/Greedy)
            speed_idx = int(tf.argmax(speed_logits))
            steer_idx = int(tf.argmax(steer_logits))

            # 4. Convert Index to Command (-1, 0, 1)
            speed_cmd = INDEX_TO_CMD[speed_idx]
            steer_cmd = INDEX_TO_CMD[steer_idx]

            # 5. Write to Unity
            write_slots(mm, *slots_config['actions'], values=[speed_cmd, steer_cmd])

            # Debug Print
            current_speed = read_slots(mm, *slots_config['speed'])[0]
            # print(f"Speed: {current_speed:.1f} | Action: {speed_cmd}, {steer_cmd}")

            # Small sleep to prevent freezing
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Inference interrupted by user.")

# -------------------------
def train_model(mm, model_path):
    global Episode
    globals()['mm'] = mm
    globals()['step_wait'] = 0.04
    # Create Optimizers
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    clip_ratio = 0.2

    # FIXED ACTOR: Output 6 logits (3 for speed, 3 for steering). 
    # No 'softmax' here! We want raw logits for PPO stability.
    actor = Sequential([
        tf.keras.layers.Input(shape=(26,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(6)  # [Speed_Logits(3), Steer_Logits(3)]
    ])

    # CRITIC: Outputs 1 value estimate
    critic = Sequential([
        tf.keras.layers.Input(shape=(26,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    

    print("Starting adaptive training...")

    Episode=1
    Loop=1
    done=0

    global all_obs, all_actions, all_log_probs, all_rewards, all_dones, all_values, all_rtg
    all_obs=[]
    all_actions=[]
    all_log_probs=[]
    all_rewards=[]
    all_dones=[]
    all_values=[]
    
    reset()
    current_obs = Read_obs()
    while True:
        # Get the total number of steps collected (T)
        steps_collected = len(all_obs)
        if done and steps_collected >= BATCH_SIZE_TARGET:
            reset()
            print("Updating")
            all_obs_tf=tf.stack(all_obs,axis=0)
            all_actions_tf=tf.stack(all_actions,axis=0)
            all_log_probs_tf=tf.stack(all_log_probs,axis=0)
            all_rewards_tf=tf.stack(all_rewards,axis=0)
            all_dones_tf=tf.cast(tf.stack(all_dones,axis=0),tf.float32)
            all_values_tf=tf.stack(all_values,axis=0)

            # Get the total number of steps collected (T)
            T = tf.shape(all_rewards_tf)[0]
            sum_rewards = tf.reduce_sum(all_rewards_tf)
            print("\n========================================")
            print(f"Total Rewards in Batch: {sum_rewards.numpy():.2f}") # Use .numpy() to see the number!
            print("The number of steps: ", T, "\nBatches: ", T % MINI_BATCH_SIZE)
            print("========================================\n")
        
            # Get Bootstrapping Value ---
            # We need the value for the state that the batch ended on (current_obs)
            # Assuming 'current_obs' is the observation from the final step's 'step()' call
            final_obs_tf = tf.convert_to_tensor(current_obs[None, :], dtype=tf.float32)
            next_value = critic(final_obs_tf)[0, 0] # Get the single scalar value

            # --- GAE Backward Sweep ---

            advantages = tf.TensorArray(dtype=tf.float32, size=T, dynamic_size=False)
            lastgaelam = tf.constant(0.0, dtype=tf.float32)

            # Note: Using a standard Python loop for reversed iteration is often simpler
            # and acceptable when operating on pre-collected data outside a core tf.function graph.
            for t in reversed(range(T.numpy())):
                
                # Check if this is the last collected step
                if t == T.numpy() - 1:
                    # Logic for the last step (bootstrapping)
                    # We need the 'next_done' flag that came with the final observation
                    # Assuming 'next_done' (boolean or float 0/1) is available after the last step() call.
                    
                    # nextnonterminal = 1.0 - next_done 
                    # (Assuming next_done is the flag associated with the state *after* the final obs in the batch)
                    nextnonterminal = 1.0 - (1.0 if done else 0.0) # Using 'done' flag from the last step() call
                    nextvalues = next_value
                else:
                    # Logic for all other steps
                    nextnonterminal = 1.0 - all_dones_tf[t + 1]
                    nextvalues = all_values_tf[t + 1]

                # TD Error (Delta)
                delta = all_rewards_tf[t] + gamma * nextvalues * nextnonterminal - all_values_tf[t]
                
                # GAE Recursive Update
                current_advantage = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                
                # Store and update lastgaelam for the next iteration
                advantages = advantages.write(t, current_advantage)
                lastgaelam = current_advantage

            advantages_tf = advantages.stack()
            returns_tf = advantages_tf + all_values_tf


            # ----------------------------------------------------
            # STEP 1: Zip the input Tensors into a single Dataset
            # ----------------------------------------------------
            # This packages all trajectories data together for simultaneous processing
            ppo_dataset = tf.data.Dataset.from_tensor_slices((
                tf.stack(all_obs),      # Your current observations
                tf.stack(all_actions),  # Actions taken
                returns_tf,             # The calculated Returns-to-Go
                advantages_tf,          # The calculated Advantages
                tf.stack(all_log_probs) # The old log probabilities
            ))

            # ----------------------------------------------------
            # STEP 2: Shuffle, Batch, and Pre-fetch the Dataset
            # ----------------------------------------------------
            ppo_dataset = ppo_dataset.shuffle(
                # Set buffer_size = T to ensure maximum randomness (shuffle the entire batch)
                buffer_size=tf.cast(T, tf.int64), 
                reshuffle_each_iteration=True # Important for PPO: ensures a new shuffle for each Epoch
            )
            ppo_dataset = ppo_dataset.batch(MINI_BATCH_SIZE)
            ppo_dataset = ppo_dataset.prefetch(tf.data.AUTOTUNE) # Optimization: loads next batch while current batch is being processed

            # ----------------------------------------------------
            # STEP 3: Run the PPO Epochs
            # ----------------------------------------------------
            for epoch in range(Epochs):
                print(f"  Epoch {epoch+1}/{Epochs}")
                
                for mini_batch in ppo_dataset:
                    m_obs, m_actions, m_returns, m_advantages, m_old_log_probs = mini_batch
                    
                    # Open a GradientTape to record operations for Automatic Differentiation
                    with tf.GradientTape(persistent=True) as tape:
                        # 1. Get CURRENT predictions
                        current_logits = actor(m_obs, training=True) # Shape: (batch, 6)
                        current_values = critic(m_obs, training=True)
                        
                        # 2. Split logits for Speed (0-2) and Steering (3-5)d
                        speed_logits, steer_logits = tf.split(current_logits, num_or_size_splits=2, axis=1)
                        
                        # 3. Create Distributions
                        speed_dist = tfp.distributions.Categorical(logits=speed_logits)
                        steer_dist = tfp.distributions.Categorical(logits=steer_logits)
                        
                        # 4. Calculate New Log Probs for the actions we took
                        # m_actions column 0 is speed, column 1 is steer
                        new_log_probs_speed = speed_dist.log_prob(m_actions[:, 0])
                        new_log_probs_steer = steer_dist.log_prob(m_actions[:, 1])
                        
                        # Total Log Prob = Sum of independent log probs
                        new_log_probs = new_log_probs_speed + new_log_probs_steer
                        
                        # 5. Calculate Ratio
                        # ratio = exp(new_log - old_log)
                        log_ratio = new_log_probs - m_old_log_probs
                        ratio = tf.exp(log_ratio)
                        
                        # 6. PPO CLIP LOSS (The famous formula)
                        surr1 = ratio * m_advantages
                        surr2 = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * m_advantages
                        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                        
                        # 7. VALUE LOSS (MSE)
                        # We want the critic to predict the Returns (Real Reward + Future Reward)
                        value_loss = tf.reduce_mean(tf.square(m_returns - tf.squeeze(current_values)))
                        # huber = tf.keras.losses.Huber()
                        # value_loss = huber(m_returns, tf.squeeze(current_values))

                        
                        # 8. ENTROPY BONUS (To encourage exploration)
                        entropy = tf.reduce_mean(speed_dist.entropy() + steer_dist.entropy())
                        
                        # Total Loss
                        total_actor_loss = policy_loss - (entropy_coefficient * entropy)
                        total_critic_loss = value_loss

                    # 9. APPLY GRADIENTS
                    # Calculate Actor Gradients
                    actor_grads = tape.gradient(total_actor_loss, actor.trainable_variables)
                    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
                    
                    # Calculate Critic Gradients
                    critic_grads = tape.gradient(total_critic_loss, critic.trainable_variables)
                    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
                    
                    del tape # Free up memory
            
            # Clear buffers for next batch
            all_obs.clear()
            all_actions.clear()
            all_log_probs.clear()
            all_rewards.clear()
            all_dones.clear()
            all_values.clear()
            print(f"Completed PPO update after {Episode} Episodes. In Loop {Loop}.")
            Episode=1
            Loop+=1
            reset()
            
        # ----------------------------------------------------
        # STEP 4: Collect Data (The Fix)
        # ----------------------------------------------------
        # 1. Prepare the input: Add batch dimension (25,) -> (1, 25)
        obs_tensor = tf.convert_to_tensor(current_obs[None, :], dtype=tf.float32)
        
        # 2. Run the networks
        logits = actor(obs_tensor)  # Get raw action scores
        value = critic(obs_tensor)  # Get value estimate
        
        # 3. Step the environment
        # We use [0] to remove the batch dimension, passing just the data
        new_obs, reward, done = step(logits[0], value[0])
        
        # 4. Update state for next loop
        current_obs = new_obs
        
        if done:
            print(f"Episode {Episode} finished. Resetting environment.")
            Episode += 1
            reset()
            current_obs = Read_obs() # Get fresh observation for new episode
                    
    print("Training complete. Best actor saved.")
# -------------------------
def main():
    # Relative actor path
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
        train_model(mm, model_path)

    mm.close()


if __name__ == "__main__":
    main()




