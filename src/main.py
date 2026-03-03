# ===============================
# main.py
# DVS → Region Pooling → Spike Generation
# ===============================

from read_dvs_aedat import read_aedat31
import json

# -------------------------------
# 1. Dataset file path
# -------------------------------
aedat_file = "/home/asus/Desktop/snn/dataset/DvsGesture/user02/user02_led.aedat"

# -------------------------------
# 2. Read DVS events
# -------------------------------
events = read_aedat31(aedat_file, max_events=10000000)
print("Total events read:", len(events))

if not events:
    print("No events found in file.")
    exit()

# Normalize timestamps
t0 = events[0][2]
events = [(x, y, t - t0, p) for (x, y, t, p) in events]

# -------------------------------
# 3. Filter ONLY right & left wave
# -------------------------------

# Relative timestamps (Class 2 & 3)
START2 = 7252462
END2   = 13009312

START3 = 15184538
END3   = 22010249

events = [
    e for e in events
    if (START2 <= e[2] <= END2) or
       (START3 <= e[2] <= END3)
]

print("Events after wave filtering:", len(events))

if not events:
    print("No events found for wave gestures.")
    exit()

# -------------------------------
# 4. 8-region spatial pooling
# -------------------------------

def map_to_region(x, y):
    """
    Divide 128x128 sensor into 2x4 grid (8 regions)
    """
    col = x // 32      # 0–3
    row = y // 64      # 0–1
    return row * 4 + col


region_counts = [0] * 8
for x, y, t, p in events:
    region = map_to_region(x, y)
    region_counts[region] += 1

print("\nEvent count per region:")
for i, count in enumerate(region_counts):
    print(f"Region {i}: {count} events")

# -------------------------------
# 5. Temporal spike generation (FIXED)
# -------------------------------

TIME_WINDOW = 500000   # 500 ms
THRESHOLD = 10
NUM_REGIONS = 8

events.sort(key=lambda e: e[2])

spike_trains = [[] for _ in range(NUM_REGIONS)]

start_time = events[0][2]
end_time = events[-1][2]

current_time = start_time

while current_time < end_time:

    window_end = current_time + TIME_WINDOW
    region_event_counts = [0] * NUM_REGIONS

    # Count events inside this window
    for x, y, t, p in events:
        if current_time <= t < window_end:
            region = map_to_region(x, y)
            region_event_counts[region] += 1

    # Generate spikes
    for r in range(NUM_REGIONS):
        spike = 1 if region_event_counts[r] >= THRESHOLD else 0
        spike_trains[r].append(spike)

    current_time = window_end

# -------------------------------
# 6. Inspect spike trains
# -------------------------------

print("\nSpike trains (first 20 time windows):")
for r in range(NUM_REGIONS):
    print(f"Region {r}: {spike_trains[r][:20]}")

# -------------------------------
# 7. Save spike trains
# -------------------------------

with open("region_spike_trains.json", "w") as f:
    json.dump(spike_trains, f)

print("\nSpike trains saved to region_spike_trains.json")
print("This file is the INPUT to the SNN module.")