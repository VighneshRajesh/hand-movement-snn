import json
from lif_neuro import LIFNeuron

# load spike trains from dataset
with open("region_spike_trains.json") as f:
    spike_trains = json.load(f)

NUM_REGIONS = 8

inputs = [LIFNeuron() for _ in range(NUM_REGIONS)]
output = LIFNeuron(threshold=3)

# time loop
for t in range(len(spike_trains[0])):
    total = 0
    for i in range(NUM_REGIONS):
        total += inputs[i].step(spike_trains[i][t])

    out = output.step(total)
    print(f"time {t} | output spike: {out}")

    if out == 1:
        print("Movement detected")
