"""
AI Hub 감정 분류 대화 음성 데이터셋 → data/emotions/<감정>/*.wav 전처리 스크립트

3개 데이터셋(4차년도, 5차년도, 5차년도_2차)의 CSV에서 '상황' 컬럼을 읽어
감정별 폴더로 WAV 파일을 복사합니다.
"""

import pandas as pd
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent / "감정 분류를 위한 대화 음성 데이터셋"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "emotions"

# 데이터셋별 CSV 경로와 WAV 디렉토리
DATASETS = {
    "4차년도": {
        "csv": BASE_DIR / "4차년도.csv",
        "wav_dir": BASE_DIR / "4차년도",
    },
    "5차년도": {
        "csv": BASE_DIR / "5차년도.csv",
        "wav_dir": BASE_DIR / "5차년도" / "5차_wav",
    },
    "5차년도_2차": {
        "csv": BASE_DIR / "5차년도_2차.csv",
        "wav_dir": BASE_DIR / "5차년도_2차",
    },
}

# ── 감정 라벨 정규화 맵 ───────────────────────────────────
LABEL_MAP = {
    "anger": "angry",
    "sad": "sadness",
    # 나머지는 이미 통일: angry, sadness, fear, disgust, neutral, happiness, surprise
}


def load_dataset(name: str, info: dict) -> pd.DataFrame:
    """CSV 로드 후 wav_path 컬럼 추가"""
    df = pd.read_csv(info["csv"], encoding="cp949")
    # '상황' 컬럼이 감정 라벨
    df["emotion"] = df["상황"].str.strip().str.lower().replace(LABEL_MAP)
    df["wav_path"] = df["wav_id"].apply(lambda x: info["wav_dir"] / f"{x}.wav")
    df["source"] = name
    return df[["wav_id", "emotion", "wav_path", "source"]]


def main():
    print("=" * 60)
    print("AI Hub 감정 데이터셋 전처리")
    print("=" * 60)

    # 1. 3개 CSV 로드 & 병합
    frames = []
    for name, info in DATASETS.items():
        df = load_dataset(name, info)
        print(f"  [{name}] {len(df):,}개 로드  감정 분포: {dict(Counter(df['emotion']))}")
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    print(f"\n전체: {len(all_df):,}개")

    # 2. 중복 wav_id 체크
    dup = all_df[all_df.duplicated(subset="wav_id", keep=False)]
    if len(dup):
        print(f"[경고] 중복 wav_id {len(dup)}건 → 첫 번째만 유지")
        all_df = all_df.drop_duplicates(subset="wav_id", keep="first")

    # 3. 감정별 폴더 생성
    emotions = sorted(all_df["emotion"].unique())
    print(f"\n감정 클래스({len(emotions)}개): {emotions}")
    for emotion in emotions:
        (OUTPUT_DIR / emotion).mkdir(parents=True, exist_ok=True)

    # 4. WAV 복사
    missing = 0
    copied = 0
    emotion_counts = Counter()

    print(f"\n출력 경로: {OUTPUT_DIR}")
    print("WAV 파일 복사 중...\n")

    for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="복사"):
        src = Path(row["wav_path"])
        if not src.exists():
            missing += 1
            continue
        dst = OUTPUT_DIR / row["emotion"] / f"{row['wav_id']}.wav"
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1
        emotion_counts[row["emotion"]] += 1

    # 5. 통계 출력
    print("\n" + "=" * 60)
    print("완료 통계")
    print("=" * 60)
    print(f"{'감정':<12} {'파일수':>8}")
    print("-" * 22)
    for emotion in sorted(emotion_counts):
        print(f"{emotion:<12} {emotion_counts[emotion]:>8,}")
    print("-" * 22)
    print(f"{'합계':<12} {copied:>8,}")
    if missing:
        print(f"\n[경고] WAV 파일 누락: {missing}건 (스킵됨)")
    print("=" * 60)


if __name__ == "__main__":
    main()
