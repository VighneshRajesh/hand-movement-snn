import torch
from feature_extractor import extract_features
from lif_snn_model import GestureSNN

MAX_TIMESTEPS = 200

def pad_or_trim(spikes):

    spikes = list(zip(*spikes))

    if len(spikes) > MAX_TIMESTEPS:
        spikes = spikes[:MAX_TIMESTEPS]

    while len(spikes) < MAX_TIMESTEPS:
        spikes.append([0] * 8)

    return spikes


model = GestureSNN()

model.load_state_dict(torch.load("gesture_snn.pth"))

model.eval()

print("Model loaded")


aedat_file = "/home/asus/Desktop/snn/testfile/user20_led.aedat"

start_time = 96032665
end_time = 100793005


spikes = extract_features(aedat_file, start_time, end_time)

if spikes is None:
    print("Feature extraction failed")
    exit()

features = pad_or_trim(spikes)

x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():

    output = model(x)

prob = output.item()

print("Gesture probability:", prob)

if prob > 0.30:
    print("Hand waving detected")
else:
    print("No hand gesture")