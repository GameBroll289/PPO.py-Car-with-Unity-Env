import time
import numpy as np
import mmap
import struct
import os

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random

# Define a specific number (can be anything, 42 is tradition)
SEED_VALUE = 42

# 1. Lock Python's internal random generator
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)

# 2. Lock NumPy (used for processing obs)
np.random.seed(SEED_VALUE)

# 3. Lock TensorFlow (used for weights and action sampling)
tf.random.set_seed(SEED_VALUE)

print(f"Random Seed set to: {SEED_VALUE}")
# ------------------------
# Shared-memory configuration (match your Unity layout)
# ------------------------
slots_config = {
    'actions': (0, 2),       # slots 0-1: vertical, horizontal (write)
    'player_position': (2, 4), # slots 2-3: x,y (read)
    'goal_position': (4, 6),   # slots 4-5: x,y (read)
    'ray_distances': (6, 10), # slots 6-9: 4 ray distances (read)
    'reward': (10, 11),     # slot 10: reward (read)
    'done': (11, 12),       # slot 11: done flag (read)
}

#10 OBS
slot_count = 12
slot_size = 4  # float32
size_bytes = slot_count * slot_size
TAGNAME = 'unity_ram2'  # mmap tag used by Unity too

# Mapping discrete indices -> -1,0,1
INDEX_TO_CMD = [-1.0, 0.0, 1.0]

global Episode, Episodes_Per_Batch, step_wait, episode_steps, Epochs, MINI_BATCH_SIZE, mm, BATCH_SIZE_TARGET, start_saving_after_loop, SIGNAL_CODE
SIGNAL_CODE = -999.0
BATCH_SIZE_TARGET=2400
Episode=0
episode_steps=0
Epochs=8
MINI_BATCH_SIZE = 64 # Or another power of 2, often 64 or 128
start_saving_after_loop=20


#Hyper parameters
gamma=0.99 #How much later rewards are worth, Goes from 0.95=< to >=0.99. With higher values, the agent will consider future rewards more strongly.
gae_lambda=0.95# "How much do I trust my specific memories vs. my general intuition?"
entropy_coefficient=0.01 #The "Curiosity" Knob: Higher values encourage more exploration by adding an entropy bonus to the loss function. Between 0.001 and 0.01 and 0.1 usually.
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
def Read_obs():      # 8 floats
    Wallrays = read_slots(mm, *slots_config['ray_distances'])      # 4 floats
    Player_pos = read_slots(mm, *slots_config['player_position'])   # 2 floats
    Goal_pos = read_slots(mm, *slots_config['goal_position'])       # 2 floats
    obs = np.array(list(Wallrays) + list(Player_pos) + list(Goal_pos), dtype=np.float32)
    
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

    # 2. WAIT for Unity to finish the frame (Handshake)
    while True:
        current_val = read_slots(mm, *slots_config['actions'])[0]
        if current_val == SIGNAL_CODE:
            break # Unity has processed the action
        time.sleep(0.001)  # Sleep briefly to avoid busy-waiting
            

    # Read obs/reward/done
    obs = Read_obs()
    reward = read_slots(mm, *slots_config['reward'])[0]
    done = bool(read_slots(mm, *slots_config['done'])[0])

    if reward>=1 or reward<=-1:
        print("Non Passive: ", reward,"\n")
    
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
    globals()['mm'] = mm
    globals()['step_wait'] = 0.04
    
    
    print(f"Loading SavedModel from: {model_path}")
    
    imported_model = tf.saved_model.load(model_path)
    inference_func = imported_model.signatures["serving_default"]

    print("Model loaded successfully. Starting inference...")
    
    # Reset environment once to align state
    reset() 
    
    while True:
        # 1. Read Observations using the SAFE function (handles NaNs)
        obs = Read_obs()
        
        # 2. Prepare Input Tensor
        input_tensor = tf.constant(obs[None, :], dtype=tf.float32)

        # 3. Run Prediction
        predictions = inference_func(input_tensor)
        
        # Safe extraction: usually the key is the output layer name, 
        # but values()[0] is acceptable if you only have one output.
        logits = list(predictions.values())[0] 
        
        # 4. Decode Actions (Deterministic/Greedy for Inference)
        speed_logits = logits[0, :3]
        steer_logits = logits[0, 3:]
        
        speed_idx = int(tf.argmax(speed_logits))
        steer_idx = int(tf.argmax(steer_logits))

        speed_cmd = INDEX_TO_CMD[speed_idx]
        steer_cmd = INDEX_TO_CMD[steer_idx]

        # 5. Write to Unity
        write_slots(mm, *slots_config['actions'], values=[speed_cmd, steer_cmd])

        # 6. CRITICAL FIX: Wait for Unity Handshake
        # This ensures we don't overwrite the action before Unity sees it,
        # and we don't read the same observation twice.
        while True:
            # Read the first float of the action slot
            current_val = read_slots(mm, *slots_config['actions'])[0]
            if current_val == SIGNAL_CODE:
                break 
            # Busy wait is fine here, or very short sleep
        time.sleep(0.001) 
# -------------------------
def train_model(mm, model_path):
    global Episode
    globals()['mm'] = mm
    globals()['step_wait'] = 0.04

    # Define Initializers
    # Gain 1.414 is standard for ReLU layers in PPO
    init_hidden = tf.keras.initializers.Orthogonal(gain=1.414)
    # Gain 0.01 makes the final output very small (near 0) -> Equal probabilities
    init_final = tf.keras.initializers.Orthogonal(gain=0.01)

    # Create Optimizers
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=0.5)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=0.5)
    clip_ratio = 0.2

    actor = Sequential([
        tf.keras.layers.Input(shape=(26,)),
        Dense(128, activation='relu', kernel_initializer=init_hidden),
        Dense(128, activation='relu', kernel_initializer=init_hidden),
        Dense(6, kernel_initializer=init_final)  # [Speed_Logits(3), Steer_Logits(3)]
    ])

    # CRITIC: Outputs 1 value estimate
    critic = Sequential([
        tf.keras.layers.Input(shape=(26,)),
        Dense(128, activation='relu', kernel_initializer=init_hidden),
        Dense(128, activation='relu', kernel_initializer=init_hidden),
        Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0))
    ])
    

    print("Starting adaptive training...")

    # ### NEW CODE: Setup Saving Variables ###
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "checkpoints")
    
    # Create the folder if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    best_batch_reward = -float('inf')
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

            current_batch_reward = float(sum_rewards.numpy())
            print("\n========================================")
            print(f"Total Rewards in Batch: {current_batch_reward:.2f}") # Use .numpy() to see the number!
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
                    nextnonterminal = 1.0 - all_dones_tf[t]
                    nextvalues = all_values_tf[t + 1]

                # TD Error (Delta)
                delta = all_rewards_tf[t] + gamma * nextvalues * nextnonterminal - all_values_tf[t]
                
                # GAE Recursive Update
                current_advantage = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                
                # Store and update lastgaelam for the next iteration
                advantages = advantages.write(t, current_advantage)
                lastgaelam = current_advantage

            advantages_tf = advantages.stack()
            
            # Normalize advantages (Critical for PPO stability)
            adv_mean = tf.math.reduce_mean(advantages_tf)
            adv_std = tf.math.reduce_std(advantages_tf)
            advantages_tf = (advantages_tf - adv_mean) / (adv_std + 1e-8)
            # ----------------------

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
                tf.stack(all_log_probs), # The old log probabilities
                all_values_tf
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
                    m_obs, m_actions, m_returns, m_advantages, m_old_log_probs, m_old_values = mini_batch
                    
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
                        # 7. VALUE LOSS (Clipped)
                        v_pred = tf.squeeze(current_values)

                        # Clip the difference between old and new value predictions
                        # This prevents the critic from moving too far from its previous prediction
                        v_pred_clipped = m_old_values + tf.clip_by_value(v_pred - m_old_values, -clip_ratio, clip_ratio)

                        # Calculate both losses
                        loss_v_unclipped = tf.square(v_pred - m_returns)
                        loss_v_clipped = tf.square(v_pred_clipped - m_returns)

                        # Take the maximum (pessimistic bound)
                        value_loss = 0.5 * tf.reduce_mean(tf.maximum(loss_v_unclipped, loss_v_clipped))

                        
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
                
            if Loop >= start_saving_after_loop:
                if current_batch_reward > best_batch_reward:
                    print(f"New Record ({current_batch_reward:.2f} > {best_batch_reward:.2f})! Saving Checkpoint...")
                    best_batch_reward = current_batch_reward
                    
                    # Create a specific name like: Loop_25_Reward_450.0
                    save_name = f"Loop_{Loop}_Reward_{int(best_batch_reward)}"
                    save_path = os.path.join(checkpoint_dir, save_name)
                    
                    # Save the Actor model in SavedModel format (Using low-level TF save)
                    tf.saved_model.save(actor, save_path)
                    
                    # Also update the main 'car_agent' folder
                    tf.saved_model.save(actor, model_path)
            else:
                print(f"Warming up... (Loop {Loop}/{start_saving_after_loop})")
            
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
# -------------------------
def main():
    # Relative actor path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "C:\\Unity Projects\\PPO.py_Car_with_UnityEnv\\Python\\checkpoints\\Loop_20_Reward_611")

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




