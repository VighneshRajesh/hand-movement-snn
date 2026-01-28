from read_dvs_aedat import read_aedat31

aedat_file = "/home/asus/Downloads/DVS  Gesture dataset/DvsGesture/user10_fluorescent_led.aedat"

events = read_aedat31(aedat_file, max_events=20000)

print("Total events read:", len(events))
print("First event:", events[0])
print("Last event:", events[-1])
