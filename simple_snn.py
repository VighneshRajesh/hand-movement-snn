from lif_neuro import LIFNeuron

# 5 input neurons → 1 output neuron
inputs = [LIFNeuron() for _ in range(5)]
output = LIFNeuron(threshold=2.5)

# fake spike matrix (time × neurons)
spike_train = [
    [0,1,0,0,1],
    [1,0,0,1,0],
    [1,1,0,0,1],
    [0,0,0,0,0],
]

for t, spikes in enumerate(spike_train):
    total = 0
    for i, s in enumerate(spikes):
        total += inputs[i].step(s)

    out = output.step(total)
    print(f"time {t} | output spike: {out}")
    if out == 1:
       print("Movement detected")


