import torch
from typing import List, Dict, Any

from src.recommender.emotion_binner import EmotionBinner
from src.recommender.memory_store import MemoryStore
from src.recommender.candidate_selector import CandidateSelector


class EmotionRecommender:
    """
    Emotion-aware music recommender (paper-faithful + exploration fix)

    score = -||qVA - sVA|| + mem_ue + mem_ea
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        candidate_selector: CandidateSelector,
        bins: int = 5,
        memory_decay: float = 0.3,
        arousal_threshold: float = 0.75,
    ):
        self.memory_store = memory_store
        self.candidate_selector = candidate_selector
        self.binner = EmotionBinner(bins)

        self.memory_decay = memory_decay
        self.arousal_threshold = arousal_threshold

    @staticmethod
    def _emotional_similarity(query_va, song_va):
        return -torch.norm(query_va - song_va, p=2)

    def recommend(
        self,
        query_va: torch.Tensor,
        user_id: str,
        songs: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:

        candidates = self.candidate_selector.select(
            user_id=user_id,
            songs=songs,
            query_va=query_va
        )

        emotion_bin = self.binner.bin(
            query_va[0].item(),
            query_va[1].item()
        )

        high_arousal = abs(query_va[1].item()) >= self.arousal_threshold

        scored = []


        for song in candidates:
            sim = self._emotional_similarity(query_va, song["va"])

            mem_ue = self.memory_store.mem_ue(user_id, emotion_bin)
            mem_ea = self.memory_store.mem_ea(song["artist"], emotion_bin)

            if high_arousal:
                mem_ue *= self.memory_decay
                mem_ea *= self.memory_decay

            score = sim + mem_ue + mem_ea

            scored.append({
                "song": song["song"],
                "artist": song["artist"],
                "valence": song["va"][0].item(),
                "arousal": song["va"][1].item(),
                "score": score.item()
            })


        scored.sort(key=lambda x: x["score"], reverse=True)

    
        seen = set()
        unique = []

        for r in scored:
            key = (r["song"], r["artist"])
            if key not in seen:
                unique.append(r)
                seen.add(key)
            if len(unique) == top_k:
                break

        return unique
