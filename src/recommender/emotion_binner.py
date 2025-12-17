class EmotionBinner:
    def __init__(self, bins: int = 5):
        self.bins = bins

    def bin(self, valence: float, arousal: float):
        v = int((valence + 1) / 2 * self.bins)
        a = int((arousal + 1) / 2 * self.bins)

        v = min(max(v, 0), self.bins - 1)
        a = min(max(a, 0), self.bins - 1)

        return (v, a)
