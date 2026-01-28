from lif_neuro import LIFNeuron

neuron = LIFNeuron(threshold=1.0, leak=0.9)

# fake spike train (simulating motion)
spikes = [0,0,1,0,1,1,0,0,1,0]

for t, s in enumerate(spikes):
    out = neuron.step(s)
    print(f"time {t} | input {s} | output {out}")