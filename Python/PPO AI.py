import numpy as np
import mmap
import struct
import time

# -------------------------
# Configuration
# -------------------------
slots_config = {
    'ray_distances': (0, 8),   # slots 0-7
    'ray_hits': (8, 16),       # slots 8-15
    'reward': (16, 17),        # slot 16
    'done': (17, 18),          # slot 17
    'actions': (18, 20),       # slots 18-19: speed, steering
    'speed': (20, 21)          # slot 20: current speed (read-only)
}
slot_count = 21  # total slots (added 2 for actions)
slot_size = 4    # float32

# -------------------------
# Pure RAM mapping (auto-create, ACCESS_WRITE allows both read/write)
# -------------------------
size_bytes = slot_count * slot_size

# Directly create or attach; ACCESS_WRITE is sufficient for read/write
mm = mmap.mmap(-1, size_bytes, tagname='unity_ram', access=mmap.ACCESS_WRITE)

# -------------------------
# Helper to read/write float32 slots
# -------------------------
def read_slots(start, end):
    mm.seek(start * slot_size)
    return [struct.unpack('<f', mm.read(slot_size))[0] for _ in range(end - start)]

def write_slots(start, end, values):
    mm.seek(start * slot_size)
    for v in values:
        mm.write(struct.pack('<f', v))

# -------------------------
# Main loop: continuously read Unity data and send actions
# -------------------------
try:
    while True:
        # Read state from Unity
        ray_distances = read_slots(*slots_config['ray_distances'])
        ray_hits = read_slots(*slots_config['ray_hits'])
        reward = read_slots(*slots_config['reward'])[0]
        done = bool(read_slots(*slots_config['done'])[0])
        current_speed = read_slots(*slots_config['speed'])[0]

        ray_distances_np = np.array(ray_distances, dtype=np.float32)
        ray_hits_np = np.array(ray_hits, dtype=np.float32)

        # Example actions (speed=3, steering=4)
        actions = [3.0, 4.0]
        write_slots(*slots_config['actions'], values=actions)

        # Debug print
        print('Reward:', reward, 'Done:', done, 'Speed: ', current_speed, 'Actions sent:', actions,'Rays:', ray_distances_np, 'Hits:', ray_hits_np, )
        time.sleep(0.5)

except KeyboardInterrupt: #Brr
    mm.close()
