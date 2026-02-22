import json
from lif_neuro import LIFNeuron

# ===============================
# Load spike trains
# ===============================
with open("region_spike_trains.json") as f:
    spike_trains = json.load(f)

NUM_REGIONS = 8
inputs = [LIFNeuron() for _ in range(NUM_REGIONS)]
output = LIFNeuron(threshold=3)

movement_detected = False

# ===============================
# Time Loop
# ===============================
for t in range(len(spike_trains[0])):

    total = 0
    for i in range(NUM_REGIONS):
        total += inputs[i].step(spike_trains[i][t])

    out = output.step(total)

    if out == 1:
        movement_detected = True

# ===============================
# Final Decision
# ===============================
if movement_detected:
    print("✅ Hand movement detected")
else:
    print("❌ No significant movement detected")