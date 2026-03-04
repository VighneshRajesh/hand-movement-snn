import csv
import os
import json
from feature_extractor import extract_features

DATASET_FOLDER = "/home/asus/Desktop/snn/dataset/DvsGesture"

X = []
y = []

print("Scanning dataset folder:", DATASET_FOLDER)

for root, dirs, files in os.walk(DATASET_FOLDER):

    for file in files:

        print("Checking file:", file)

        if file.endswith("_labels.csv"):

            label_path = os.path.join(root, file)
            aedat_path = label_path.replace("_labels.csv", ".aedat")

            print("Processing:", label_path)

            with open(label_path) as f:

                reader = csv.DictReader(f)

                for row in reader:

                    class_id = int(row["class"])
                    start = int(row["startTime_usec"])
                    end = int(row["endTime_usec"])

                    features = extract_features(aedat_path, start, end)

                    if features is None:
                        continue

                    # Wave gestures
                    if class_id == 2 or class_id == 3:
                        label = 1
                    else:
                        label = 0

                    X.append(features)
                    y.append(label)

print("Total samples:", len(X))

if len(X) > 0:
    print("Example feature:", X[0])

with open("dataset.json", "w") as f:
    json.dump({"X": X, "y": y}, f)

print("Dataset built successfully")