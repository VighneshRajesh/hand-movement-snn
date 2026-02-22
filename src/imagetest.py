from read_dvs_aedat import read_aedat31
import numpy as np
import matplotlib.pyplot as plt

aedat_file = "/home/asus/Desktop/snn/dataset/DvsGesture/user02/user02_led.aedat"
events = read_aedat31(aedat_file, max_events=100000)

frame = np.zeros((128, 128))

for x, y, t, p in events:
    frame[y, x] += 1

frame = np.log1p(frame)        # boost low intensities
frame = frame / frame.max()    # normalize

plt.imshow(frame, cmap="hot")
plt.colorbar()
plt.title("Event Accumulation Image")
plt.savefig("event_image.png")
print("Image saved as event_image.png")