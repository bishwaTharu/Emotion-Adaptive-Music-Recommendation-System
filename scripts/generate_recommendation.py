import sys
sys.path.append("src")

import torch
import pandas as pd
import requests
import logging

from src.recommender.memory_store import MemoryStore
from src.recommender.candidate_selector import CandidateSelector
from src.recommender.recommender import EmotionRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OnlineRecommenderService:
    """
    Online emotion-aware music recommender
    (paper-faithful implementation)
    """

    def __init__(
        self,
        api_url="http://localhost:8000",
        song_parquet="data/processed/merged_data.parquet",
        user_emotion_path="data/processed/user_emotion_table.pkl",
        emotion_artist_path="data/processed/emotion_artist_table.pkl",
        user_top_artists=None,
    ):
        self.api_url = api_url
        self.songs = self._load_songs(song_parquet)
        self.memory_store = MemoryStore(
            user_emotion_path=user_emotion_path,
            emotion_artist_path=emotion_artist_path
        )
        self.candidate_selector = CandidateSelector(
            user_top_artists=user_top_artists or {}
        )
        self.recommender = EmotionRecommender(
            memory_store=self.memory_store,
            candidate_selector=self.candidate_selector,
            bins=5
        )

        logger.info("Online recommender initialized successfully")

    def _load_songs(self, parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            df = df.drop_duplicates(subset=["song", "artist"])

            songs = [
                {
                    "song": row["song"],
                    "artist": row["artist"],
                    "va": torch.tensor(
                        [row["V"], row["A"]],
                        dtype=torch.float32
                    )
                }
                for _, row in df.iterrows()
            ]

            logger.info(
                f"Loaded {len(songs)} unique songs "
                f"(deduplicated) from {parquet_path}"
            )
            return songs

        except FileNotFoundError:
            logger.error(
                f"Song VA database not found at {parquet_path}. "
                "Build Spotify VA first."
            )
            raise

    def infer_va(self, text: str) -> torch.Tensor:
        response = requests.post(
            f"{self.api_url}/infer",
            json={"text": text},
            timeout=5
        )

        if response.status_code != 200:
            raise RuntimeError("Emotion inference API failed")

        data = response.json()
        va = torch.tensor(
            [data["valence"], data["arousal"]],
            dtype=torch.float32
        )

        logger.info(
            f"Inferred VA | V={va[0]:.3f}, A={va[1]:.3f} | text='{text}'"
        )
        return va

    def recommend(self, query_va, user_id, top_k=5):
        return self.recommender.recommend(
            query_va=query_va,
            user_id=user_id,
            songs=self.songs,
            top_k=top_k
        )


def main():
    user_top_artists = {
        "4fea8ee3745dada8a8fa2a2d26514bc1232c3a15": "the beatles"
    }

    service = OnlineRecommenderService(
        api_url="http://localhost:8000",
        song_parquet="data/processed/merged_data.parquet",
        user_emotion_path="data/processed/user_emotion_table.pkl",
        emotion_artist_path="data/processed/emotion_artist_table.pkl",
        user_top_artists=user_top_artists
    )

    user_id = "4a47c65bf7894a80761acaa525295e54832d2057"
    query = "My heart is racing, I feel panicked and overstimulated, like I can’t slow down."

    query_va = service.infer_va(query)

    recommendations = service.recommend(
        query_va=query_va,
        user_id=user_id,
        top_k=5
    )

    print("\nTop Recommendations:")
    for r in recommendations:
        print(
            f"{r['song']} - {r['artist']} | "
            f"V={r['valence']:.2f}, "
            f"A={r['arousal']:.2f}, "
            f"Score={r['score']:.3f}"
        )


if __name__ == "__main__":
    main()