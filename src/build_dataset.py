import os
import csv
import pickle

from feature_extractor import extract_features

# dataset location
DATASET_FOLDER = r"C:\Users\USER\hand-movement-snn\dataset\DvsGesture"

dataset = []
labels = []

print("Scanning dataset folder:", DATASET_FOLDER)

# loop through files
for file in os.listdir(DATASET_FOLDER):

    if file.endswith(".aedat"):

        aedat_path = os.path.join(DATASET_FOLDER, file)

        # find corresponding label file
        label_file = file.replace(".aedat", "_labels.csv")
        label_path = os.path.join(DATASET_FOLDER, label_file)

        print("Processing:", file)

        if not os.path.exists(label_path):
            print("Label file missing:", label_file)
            continue

        # open label file
        with open(label_path, "r") as f:

            reader = csv.reader(f)

            # skip CSV header
            next(reader)

            for row in reader:

                gesture = int(row[0])
                start_time = int(row[1])
                end_time = int(row[2])

                # convert 11 gestures → binary
                if gesture in [2, 3]:
                    label = 1
                else:
                    label = 0

                features = extract_features(
                    aedat_path,
                    start_time,
                    end_time
                )

                if features is None:
                    continue

                dataset.append(features)
                labels.append(label)

print("\nDataset statistics")
print("------------------")
print("Total samples:", len(dataset))
print("Positive (hand):", sum(labels))
print("Negative (no hand):", len(labels) - sum(labels))

# save dataset
with open("gesture_dataset.pkl", "wb") as f:
    pickle.dump((dataset, labels), f)

print("\nDataset saved as: gesture_dataset.pkl")