import torch
import torch.nn as nn
from feature_extractor import extract_features


class GestureNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(8,16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,1)

    def forward(self,x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


model = GestureNet()

model.load_state_dict(torch.load("gesture_model.pth"))

model.eval()

print("Model loaded successfully")


aedat_file = "/home/asus/Desktop/snn/testfile/user20_led.aedat"

start_time = 52731986
end_time = 58159600


features = extract_features(aedat_file,start_time,end_time)

if features is None:

    print("Could not extract features")

    exit()


print("Extracted features:",features)

x = torch.tensor(features,dtype=torch.float32)

with torch.no_grad():

    output = model(x)


print("Model probability:",output.item())


prediction = 1 if output.item() > 0.35 else 0


if prediction == 1:

    print("✅ Hand waving detected")

else:

    print("❌ No hand detected")