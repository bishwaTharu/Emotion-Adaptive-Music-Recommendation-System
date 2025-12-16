from collections import defaultdict
from ..data.va_binning import va_to_bin

def build_user_emotion_table(play_logs):
    table = defaultdict(lambda: defaultdict(float))

    for row in play_logs:
        user = row["user"]
        v, a = va_to_bin(row["V"], row["A"])
        table[user][(v, a)] += 1.0

    for user in table:
        total = sum(table[user].values())
        for e in table[user]:
            table[user][e] /= total

    return table


def build_emotion_artist_table(play_logs):
    table = defaultdict(lambda: defaultdict(float))

    for row in play_logs:
        artist = row["artist"]
        v, a = va_to_bin(row["V"], row["A"])
        table[artist][(v, a)] += 1.0

    for artist in table:
        total = sum(table[artist].values())
        for e in table[artist]:
            table[artist][e] /= total

    return table
