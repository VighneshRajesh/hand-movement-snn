from read_dvs_aedat import read_aedat31

NUM_REGIONS = 4

def map_to_region(x, y):

    col = x // 64
    row = y // 64

    return row * 2 + col


def extract_features(aedat_file, start_time, end_time):

    events = read_aedat31(aedat_file)

    if not events:
        return None

    filtered = []

    for x, y, p, t in events:
        if start_time <= t <= end_time:
            filtered.append((x, y, p, t))

    if len(filtered) == 0:
        return None

    on_counts = [0]*NUM_REGIONS
    off_counts = [0]*NUM_REGIONS

    for x, y, p, t in filtered:

        region = map_to_region(x, y)

        if region < 0 or region >= NUM_REGIONS:
            continue

        if p == 1:
            on_counts[region] += 1
        else:
            off_counts[region] += 1

    features = on_counts + off_counts

    total = sum(features)

    if total == 0:
        return None

    features = [f/total for f in features]

    return features