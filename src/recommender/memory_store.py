import pickle
from pathlib import Path

class MemoryStore:
    def __init__(
        self,
        user_emotion_path: str,
        emotion_artist_path: str
    ):
        self.user_emotion = self._load_pickle(user_emotion_path)
        self.emotion_artist = self._load_pickle(emotion_artist_path)

    @staticmethod
    def _load_pickle(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Memory table not found: {path}")

        with open(path, "rb") as f:
            return pickle.load(f)

    def mem_ue(self, user_id, emotion_bin):
        return self.user_emotion.get(user_id, {}).get(emotion_bin, 0.0)

    def mem_ea(self, artist, emotion_bin):
        return self.emotion_artist.get(artist, {}).get(emotion_bin, 0.0)
