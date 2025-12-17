class CandidateSelector:
    """
    Selects candidate songs.
    Uses top-artist filtering, but relaxes it for extreme emotions.
    """

    def __init__(self, user_top_artists, arousal_threshold=0.75):
        self.user_top_artists = user_top_artists
        self.arousal_threshold = arousal_threshold

    def select(self, user_id, songs, query_va=None):
        """
        If arousal is high, return full catalog (emotion-driven exploration).
        Otherwise, filter by top artist with fallback.
        """
        if query_va is not None:
            arousal = abs(query_va[1].item())
            if arousal >= self.arousal_threshold:
                return songs

        top_artist = self.user_top_artists.get(user_id)
        if not top_artist:
            return songs

        filtered = [
            s for s in songs
            if s["artist"].lower() == top_artist.lower()
        ]

        return filtered if filtered else songs
