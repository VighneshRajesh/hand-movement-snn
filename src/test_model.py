import torch
import csv
import random
from feature_extractor import extract_features
from lif_snn_model import GestureSNN

model = GestureSNN()

model.load_state_dict(torch.load("gesture_snn.pth"))

model.eval()

print("Model loaded")

aedat_file = "/home/asus/Desktop/snn/testfile/user20_led.aedat"
csv_file = "/home/asus/Desktop/snn/testfile/user20_led_labels.csv"

gestures = []

with open(csv_file) as f:

    reader = csv.DictReader(f)

    for row in reader:

        gesture_class = int(row["class"])

        start = int(row["startTime_usec"])
        end = int(row["endTime_usec"])

        gestures.append((gesture_class,start,end))


gesture_class,start_time,end_time = random.choice(gestures)

print("\nRandom gesture selected")
print("True class:",gesture_class)

features = extract_features(aedat_file,start_time,end_time)

if features is None:
    print("Feature extraction failed")
    exit()

x = torch.tensor(features,dtype=torch.float32).unsqueeze(0)

with torch.no_grad():

    prob = model(x).item()

print("Gesture probability:",prob)

if prob > 0.5:
    prediction = "Wave detected"
else:
    prediction = "No wave"

print("Prediction:",prediction)

if gesture_class in [2,3]:
    print("Actual gesture type: Wave")
else:
    print("Actual gesture type: Not wave")