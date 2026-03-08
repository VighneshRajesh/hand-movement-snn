import os
import json
import csv
from feature_extractor import extract_features

DATASET_PATH = "/home/asus/Desktop/snn/dataset/DvsGesture"

MAX_TIMESTEPS = 200

X = []
y = []


def pad_or_trim(spikes):

    spikes = list(zip(*spikes))  # convert to [time, regions]

    if len(spikes) > MAX_TIMESTEPS:
        spikes = spikes[:MAX_TIMESTEPS]

    while len(spikes) < MAX_TIMESTEPS:
        spikes.append([0] * 8)

    return spikes


print("Scanning dataset folder:", DATASET_PATH)

for file in os.listdir(DATASET_PATH):

    if file.endswith("_labels.csv"):

        csv_path = os.path.join(DATASET_PATH, file)
        aedat_file = file.replace("_labels.csv", ".aedat")
        aedat_path = os.path.join(DATASET_PATH, aedat_file)

        print("Processing:", csv_path)

        with open(csv_path) as f:

            reader = csv.DictReader(f)

            for row in reader:

                gesture_class = int(row["class"])
                start = int(row["startTime_usec"])
                end = int(row["endTime_usec"])

                spikes = extract_features(aedat_path, start, end)

                if spikes is None:
                    continue

                features = pad_or_trim(spikes)

                if gesture_class == 2 or gesture_class == 3:
                    label = 1
                else:
                    label = 0

                X.append(features)
                y.append(label)

print("Total samples:", len(X))

with open("dataset.json", "w") as f:
    json.dump({"X": X, "y": y}, f)

print("Dataset built successfully")