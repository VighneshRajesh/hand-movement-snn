class LIFNeuron:
    def __init__(self, threshold=1.0, leak=0.9):
        self.v = 0.0
        self.threshold = threshold
        self.leak = leak

    def step(self, spike):
        # leak
        self.v *= self.leak
        # integrate
        self.v += spike

        if self.v >= self.threshold:
            self.v = 0.0
            return 1  # output spike
        return 0