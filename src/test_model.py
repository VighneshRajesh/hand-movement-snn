import torch
import csv
import os
from feature_extractor import extract_features
from lif_snn_model import GestureSNN


DATASET_PATH = "/home/asus/Desktop/snn/testfile"


model = GestureSNN()
model.load_state_dict(torch.load("gesture_snn.pth"))
model.eval()

print("SNN Model loaded\n")


# ---------- list available AEDAT files ----------

files = []

for f in os.listdir(DATASET_PATH):
    if f.endswith(".aedat"):
        files.append(f)

print("Available recordings:\n")

for i,f in enumerate(files):
    print(i, ":", f)


file_index = int(input("\nSelect recording index: "))

selected_file = files[file_index]

aedat_file = os.path.join(DATASET_PATH, selected_file)
csv_file = os.path.join(DATASET_PATH, selected_file.replace(".aedat","_labels.csv"))

print("\nSelected file:", selected_file)


# ---------- read gestures from csv ----------

gestures = []

with open(csv_file) as f:

    reader = csv.DictReader(f)

    for i,row in enumerate(reader):

        gesture_class = int(row["class"])
        start = int(row["startTime_usec"])
        end = int(row["endTime_usec"])

        gestures.append((gesture_class,start,end))


print("\nGestures in this recording:\n")

for i,g in enumerate(gestures):
    print(i,"| class:",g[0]," start:",g[1]," end:",g[2])


gesture_index = int(input("\nSelect gesture index: "))

gesture_class,start_time,end_time = gestures[gesture_index]


print("\nRunning SNN on gesture...\n")
print("True class:",gesture_class)


# ---------- extract features ----------

features = extract_features(aedat_file,start_time,end_time)

if features is None:
    print("Feature extraction failed")
    exit()


x = torch.tensor(features,dtype=torch.float32).unsqueeze(0)


# ---------- run model ----------

with torch.no_grad():
    prob = model(x).item()


print("Gesture probability:",prob)


if prob > 0.5:
    prediction = "Wave detected"
else:
    prediction = "No wave"


print("Prediction:",prediction)


if gesture_class in [2,3]:
    actual = "Wave"
else:
    actual = "Not wave"


print("Actual gesture type:",actual)