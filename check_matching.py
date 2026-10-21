
import pandas as pd
import re
import ast
import unicodedata
from pathlib import Path
from collections import Counter

# 파일 찾기
lastfm_files = list(Path(".").glob("userid-timestamp*.tsv"))
spotify_files = list(Path(".").glob("spotify_tracks*.csv"))

if not lastfm_files or not spotify_files:
    print("데이터 파일을 찾을 수 없습니다.")
    raise SystemExit

lastfm_file = lastfm_files[0]
spotify_file = spotify_files[0]

# 아티스트 이름 정리
def clean_name(name):
    if pd.isna(name):
        return ""

    name = unicodedata.normalize("NFKC", str(name))
    name = name.casefold().strip()
    name = re.sub(r"\s+", " ", name)

    return name

# Spotify의 아티스트 목록 처리
def get_artists(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass

    return [value] if value else []

# 1. Last.fm 청취 기록 집계
listens = Counter()
total_events = 0

for chunk in pd.read_csv(
    lastfm_file,
    sep="\t",
    header=None,
    usecols=[3],
    names=["artist_name"],
    dtype=str,
    chunksize=200_000,
    on_bad_lines="skip"
):
    chunk["artist_name"] = chunk["artist_name"].map(clean_name)
    chunk = chunk[chunk["artist_name"] != ""]

    total_events += len(chunk)
    listens.update(chunk["artist_name"])

lastfm_artists = set(listens.keys())

# 2. Spotify 아티스트 추출
spotify = pd.read_csv(spotify_file, dtype={"artists": str})

if "artists" not in spotify.columns:
    print("Spotify에 artists 컬럼이 없습니다.")
    raise SystemExit

spotify_artists = set()

for value in spotify["artists"].dropna():
    for artist in get_artists(value):
        name = clean_name(artist)

        if name:
            spotify_artists.add(name)

# 3. 매칭 결과
matched = lastfm_artists & spotify_artists
unmatched = lastfm_artists - spotify_artists

matched_events = sum(listens[a] for a in matched)

artist_rate = (
    len(matched) / len(lastfm_artists) * 100
    if lastfm_artists else 0
)

event_rate = (
    matched_events / total_events * 100
    if total_events else 0
)

print("\n아티스트 매칭 결과")
print(f"Last.fm 청취 기록 수: {total_events:,}")
print(f"Last.fm 고유 아티스트 수: {len(lastfm_artists):,}")
print(f"Spotify 고유 아티스트 수: {len(spotify_artists):,}")
print(f"공통 아티스트 수: {len(matched):,}")
print(f"미매칭 아티스트 수: {len(unmatched):,}")
print(f"고유 아티스트 매칭률: {artist_rate:.2f}%")
print(f"청취 기록 커버리지: {event_rate:.2f}%")

print("\n 매칭된 아티스트 예시")
for artist in sorted(matched)[:20]:
    print(artist)

print("\n매칭되지 않은 아티스트 예시")
for artist in sorted(unmatched)[:20]:
    print(artist)

print("\n 청취 기록이 많은 매칭 아티스트 TOP 10")
top_matched = [
    (artist, count)
    for artist, count in listens.most_common()
    if artist in matched
][:10]

for artist, count in top_matched:
    print(f"{artist}: {count:,}")


# 사용자별 청취 기록 커버리지 확인
user_stats = {}

for chunk in pd.read_csv(
    lastfm_file,
    sep="\t",
    header=None,
    usecols=[0, 3],
    names=["user_id", "artist_name"],
    dtype=str,
    chunksize=200_000,
    on_bad_lines="skip"
):
    chunk["artist_name"] = chunk["artist_name"].map(clean_name)
    chunk = chunk[chunk["artist_name"] != ""]

    chunk["matched"] = chunk["artist_name"].isin(matched)

    stats = chunk.groupby("user_id")["matched"].agg(
        total="size",
        matched="sum"
    )

    for user, row in stats.iterrows():
        if user not in user_stats:
            user_stats[user] = [0, 0]

        user_stats[user][0] += int(row["total"])
        user_stats[user][1] += int(row["matched"])

users = pd.DataFrame.from_dict(
    user_stats,
    orient="index",
    columns=["total", "matched"]
)

users["coverage"] = users["matched"] / users["total"] * 100

print("\n===== 사용자별 매칭 커버리지 =====")
print(users["coverage"].describe().round(2))

print("\n매칭된 청취 기록이 100건 이상인 사용자:",
      (users["matched"] >= 100).sum())

print("전체 사용자 수:", len(users))
print("커버리지 50% 이상인 사용자:",
      (users["coverage"] >= 50).sum())
