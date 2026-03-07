from read_dvs_aedat import read_aedat31

TIME_WINDOW = 50000
THRESHOLD = 1
NUM_REGIONS = 8


def map_to_region(x, y):
    col = x // 32
    row = y // 64
    return row * 4 + col


def extract_features(aedat_file, start_time, end_time):

    events = read_aedat31(aedat_file)

    if not events:
        return None

    # Filter gesture window
    events = [e for e in events if start_time <= e[2] <= end_time]

    print("Filtered events:", len(events))

    if not events:
        return None

    events.sort(key=lambda e: e[2])

    spike_trains = [[] for _ in range(NUM_REGIONS)]

    start = events[0][2]
    end = events[-1][2]

    current = start

    while current < end:

        window_end = current + TIME_WINDOW
        counts = [0] * NUM_REGIONS

        for x, y, t, p in events:
            if current <= t < window_end:
                region = map_to_region(x, y)
                counts[region] += 1

        for r in range(NUM_REGIONS):
            spike = 1 if counts[r] >= THRESHOLD else 0
            spike_trains[r].append(spike)

        current = window_end

    # Convert spike trains → features
    features = []

    for region in spike_trains:
        total = sum(region)
        first_half = sum(region[:len(region)//2])
        second_half = sum(region[len(region)//2:])

        features.append(total)
        features.append(first_half - second_half)

    # keep only first 8 values
    features = features[:8]

    if len(features) != 8:
        return None

    return features