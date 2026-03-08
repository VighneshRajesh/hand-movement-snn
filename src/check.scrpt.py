import pickle

with open("gesture_dataset.pkl","rb") as f:
    X,y = pickle.load(f)

for i in range(10):
    print(X[i], " sum=", sum(X[i]))