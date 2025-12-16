from .scoring import score_song

def recommend(query_va, user, songs, ue_table, ea_table, top_k=5):
    scored = []

    for song in songs:
        score = score_song(
            query_va,
            song["va"],
            user,
            song["artist"],
            ue_table,
            ea_table
        )
        scored.append((score.item(), song))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]
