import struct

def read_aedat31(filepath, max_events=50000):
    events = []

    with open(filepath, "rb") as f:
        while True:
            header = f.read(28)  # AEDAT3 header size
            if len(header) < 28:
                break

            (
                eventType,
                eventSource,
                eventSize,
                eventTSOffset,
                eventTSOverflow,
                eventCapacity,
                eventNumber,
                eventValid
            ) = struct.unpack("HHIIIIII", header)

            # Read event block
            for _ in range(eventValid):
                raw = f.read(8)
                if len(raw) < 8:
                    break

                data, timestamp = struct.unpack("II", raw)
                x = ((data >> 17) & 0x1FFF) & 0x7F
                y = ((data >> 2)  & 0x1FFF) & 0x7F
                p = (data >> 1) & 0x1
                
                events.append((x, y, timestamp, p))

                if len(events) >= max_events:
                    return events

    return events
