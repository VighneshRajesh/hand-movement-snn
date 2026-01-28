# ===============================
# main.py
# DVS → Region Pooling → Spike Generation
# ===============================

from read_dvs_aedat import read_aedat31
import json

# -------------------------------
# 1. Dataset file path
# -------------------------------
aedat_file = "//home/asus/Desktop/snn/dataset/DvsGesture/user10/user10_fluorescent_led.aedat"

# -------------------------------
# 2. Read DVS events
# -------------------------------
events = read_aedat31(aedat_file, max_events=20000)

print("Total events read:", len(events))

# -------------------------------
# 3. 8-region spatial pooling
# -------------------------------
def map_to_region(x, y):
    """
    Divide 128x128 sensor into 2x4 grid (8 regions)
    """
    col = x // 32      # 0–3
    row = y // 64      # 0–1
    return row * 4 + col


# Count events per region (for verification)
region_counts = [0] * 8
for x, y, t, p in events:
    region = map_to_region(x, y)
    region_counts[region] += 1

print("\nEvent count per region:")
for i, count in enumerate(region_counts):
    print(f"Region {i}: {count} events")

# -------------------------------
# 4. Temporal spike generation
# -------------------------------
TIME_WINDOW = 10_000    # 10 ms (microseconds)
THRESHOLD = 50          # events required to fire a spike
NUM_REGIONS = 8

# Sort events by time
events.sort(key=lambda e: e[2])

# Initialize spike trains
spike_trains = [[] for _ in range(NUM_REGIONS)]

window_start = events[0][2]
window_end = window_start + TIME_WINDOW
region_event_counts = [0] * NUM_REGIONS

for x, y, t, p in events:
    if t < window_end:
        region = map_to_region(x, y)
        region_event_counts[region] += 1
    else:
        # Generate spikes for this window
        for r in range(NUM_REGIONS):
            spike = 1 if region_event_counts[r] >= THRESHOLD else 0
            spike_trains[r].append(spike)

        # Reset for next window
        region_event_counts = [0] * NUM_REGIONS
        window_start = window_end
        window_end += TIME_WINDOW

# -------------------------------
# 5. Inspect spike trains
# -------------------------------
print("\nSpike trains (first 20 time windows):")
for r in range(NUM_REGIONS):
    print(f"Region {r}: {spike_trains[r][:20]}")

# -------------------------------
# 6. Save spike trains (handover to SNN)
# -------------------------------
with open("region_spike_trains.json", "w") as f:
    json.dump(spike_trains, f)

print("\nSpike trains saved to region_spike_trains.json")
print("This file is the INPUT to the SNN module.")
