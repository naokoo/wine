"""替え歌のかな歌詞＋メロディを LeapSinger の入力（音素・持続長・F0）に変換するフロントエンド。

LeapSinger（https://github.com/wavtechyukky/LeapSinger）は音響モデル単体で、
入力は「音素列・音素ごとのフレーム数・フレームごとのF0」。
譜面（音符＋歌詞）からその3つを作る部分は本体に含まれないため、ここで用意する。

    かな歌詞 ── モーラ分割 ─→ 音素列（dict/kana2phonemes.table）
    音符列  ── 音長割当   ─→ 音素ごとのフレーム数
             └ 音高      ─→ F0カーブ（ポルタメント＋ビブラート）

注意: kana2phonemes.table はカタカナ＝無声化母音（ボ→ b O）、ひらがな＝有声母音（ぼ→ b o）
という UTAU 系の慣習。歌唱では有声が必要なので、必ずひらがなに正規化してから引くこと。
"""
from __future__ import annotations

import math
import re
import sys

SR = 44100
HOP = 256
FPS = SR / HOP  # 172.27 frames/sec

# ── かな正規化 ────────────────────────────────────────────────────────────
_KATA2HIRA = {chr(c): chr(c - 0x60) for c in range(0x30A1, 0x30F7)}  # ァ..ヶ → ぁ..ゖ


def to_hiragana(text: str) -> str:
    """カタカナをひらがなへ。長音符「ー」と促音はそのまま残す。"""
    return "".join(_KATA2HIRA.get(ch, ch) for ch in text)


def load_kana_table(path: str) -> dict[str, list[str]]:
    """dict/kana2phonemes.table を読み、ひらがな見出しだけの表にする。"""
    table: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            grapheme, phonemes = parts[0], parts[1:]
            hira = to_hiragana(grapheme)
            # ひらがな見出し（有声母音）を優先。カタカナ由来（無声化）は既出なら上書きしない。
            if grapheme == hira or hira not in table:
                table[hira] = phonemes
    return table


_MORA_RE_CACHE: dict[int, re.Pattern] = {}


def split_mora(text: str, table: dict[str, list[str]]) -> list[str]:
    """かな文字列をモーラへ分割（「きゃ」等の拗音は最長一致）。空白は区切りとして落とす。"""
    key = id(table)
    if key not in _MORA_RE_CACHE:
        keys = sorted((k for k in table if k not in ("pau", "sil", "br", "R")),
                      key=len, reverse=True)
        _MORA_RE_CACHE[key] = re.compile("|".join(re.escape(k) for k in keys) + "|ー|.")
    text = to_hiragana(text).replace("・", "")
    return [m for m in _MORA_RE_CACHE[key].findall(text) if not m.isspace()]


# ── 音素・持続長 ─────────────────────────────────────────────────────────
CONSONANT_SEC = {  # 子音に割り当てる長さ（秒）。破裂音は短く、摩擦音は長めに。
    "k": 0.045, "ky": 0.050, "g": 0.035, "gy": 0.040, "t": 0.045, "ty": 0.050,
    "d": 0.035, "dy": 0.040, "p": 0.045, "py": 0.050, "b": 0.035, "by": 0.040,
    "s": 0.075, "sh": 0.080, "z": 0.055, "j": 0.055, "ts": 0.070, "ch": 0.065,
    "h": 0.055, "hy": 0.055, "f": 0.065, "fy": 0.065, "m": 0.045, "my": 0.050,
    "n": 0.040, "ny": 0.045, "r": 0.030, "ry": 0.035, "y": 0.030, "w": 0.030,
    "v": 0.045,
}
VOWELS = set("aiueo")
MIN_VOWEL_SEC = 0.05
SMALL_VOWELS = {"ぁ": "あ", "ぃ": "い", "ぅ": "う", "ぇ": "え", "ぉ": "お"}  # 単独で現れた小書き母音
# かな表にはあるが ja.phonemes に無い音素の代替（LeapSinger 本体の export/openutau_assets.py と同じ）
PHONEME_FALLBACK = {"vy": "v"}


def fit_notes(moras: list[str], notes: list[tuple[float, float]]) -> list[list[tuple[int, float]]]:
    """モーラ列と音符列（(Hz, 秒)）を対応づけ、モーラごとの [(フレーム数, Hz), ...] を返す。

    - モーラ数＝音符数 … 1対1
    - モーラ数＜音符数 … 余った音符は最後のモーラが引き受ける（語尾を伸ばすメリスマ）
    - モーラ数＞音符数 … 1音符に複数モーラを詰める（同じ音高を分け合う）
    """
    if not moras or not notes:
        raise ValueError("モーラまたは音符が空です")
    note_frames = [(max(1, round(sec * FPS)), hz) for hz, sec in notes]

    if len(moras) <= len(note_frames):
        per_mora: list[list[tuple[int, float]]] = [[nf] for nf in note_frames[:len(moras)]]
        for extra in note_frames[len(moras):]:
            per_mora[-1].append(extra)            # 余りは末尾モーラに足す
        return per_mora

    # モーラが多い: 各音符に何モーラ乗せるかを均等に配る
    n_notes = len(note_frames)
    share = [len(moras) // n_notes] * n_notes
    for i in range(len(moras) % n_notes):
        share[i] += 1
    per_mora = []
    for (frames, hz), k in zip(note_frames, share):
        base, rest = divmod(frames, k)
        for j in range(k):
            per_mora.append([(max(1, base + (1 if j < rest else 0)), hz)])
    return per_mora


def resolve_mora(mora: str, table: dict[str, list[str]]) -> list[str] | None:
    """モーラ1つを音素列に。辞書にない小書き母音は母音として扱い、
    かな表にあって音素表に無いものは代替に置き換える。引けなければ None。"""
    units = table.get(mora) or table.get(SMALL_VOWELS.get(mora, ""))
    if units is None:
        return None
    return [PHONEME_FALLBACK.get(u, u) for u in units]


def notes_to_phonemes(moras: list[str], per_mora: list[list[tuple[int, float]]],
                      table: dict[str, list[str]],
                      *, lead_sec: float = 0.12, tail_sec: float = 0.20):
    """モーラ列と（fit_notes が返した）モーラごとの音符から、音素列・音素ごとのフレーム数・
    音高スパンを作る。

    - 「ー」は直前の母音を延ばす（音素は増やさない）
    - 「っ」は cl（閉鎖）としてその音符を消費する

    返り値: (phonemes, frames, pitch_spans) — pitch_spans は [(start, end, hz), ...]。
    """
    if len(moras) != len(per_mora):
        raise ValueError(f"モーラ数 {len(moras)} と割当 {len(per_mora)} が一致しません")

    lead = max(1, round(lead_sec * FPS))
    phonemes: list[str] = ["pau"]
    frames: list[int] = [lead]
    pitch_spans: list[tuple[int, int, float]] = []
    cursor = lead
    for mora, segments in zip(moras, per_mora):
        for seg_frames, hz in segments:
            pitch_spans.append((cursor, cursor + seg_frames, hz))
            cursor += seg_frames
        n_frames = sum(f for f, _ in segments)
        if mora == "ー":
            if phonemes[-1] in VOWELS or phonemes[-1] == "N":
                frames[-1] += n_frames          # 直前の母音を延ばすだけ
                continue
            mora = "あ"                          # 直前が母音でない異常系のフォールバック
        units = resolve_mora(mora, table)
        if units is None:
            # 辞書に無い文字（英字・漢字など）。音符ぶんは直前の音素を延ばして飛ばす。
            print(f"警告: 辞書にないかな {mora!r} を読み飛ばしました", file=sys.stderr)
            frames[-1] += n_frames
            continue
        min_vowel = max(1, round(MIN_VOWEL_SEC * FPS))
        head_units, tail_unit = units[:-1], units[-1]
        used = 0
        for cons in head_units:
            c_frames = max(1, round(CONSONANT_SEC.get(cons, 0.04) * FPS))
            c_frames = max(1, min(c_frames, n_frames - used - min_vowel))
            phonemes.append(cons)
            frames.append(c_frames)
            used += c_frames
        phonemes.append(tail_unit)
        frames.append(max(1, n_frames - used))
    phonemes.append("pau")
    frames.append(max(1, round(tail_sec * FPS)))
    return phonemes, frames, pitch_spans


# ── F0 ───────────────────────────────────────────────────────────────────
def build_f0(pitch_spans: list[tuple[int, int, float]], total_frames: int,
             *, portamento_sec: float = 0.05, vibrato_hz: float = 5.5,
             vibrato_cents: float = 22.0, vibrato_onset_sec: float = 0.35):
    """音高スパンからフレーム単位の F0（Hz）を作る。音符境界はポルタメントで繋ぎ、
    長い音符の後半に軽くビブラートを掛ける。返り値は [T] の Hz 配列（無音区間も値を持つ）。"""
    import numpy as np

    f0 = np.zeros(int(total_frames), dtype=np.float32)
    for start, end, hz in pitch_spans:
        f0[start:min(end, total_frames)] = hz
    f0[:pitch_spans[0][0]] = pitch_spans[0][2]              # 先頭 pau
    f0[pitch_spans[-1][1]:] = pitch_spans[-1][2]            # 末尾 pau

    port = max(2, round(portamento_sec * FPS))
    for start, _end, _hz in pitch_spans[1:]:
        a, b = max(0, start - port // 2), min(len(f0), start + port // 2)
        if b - a < 2:
            continue
        lo, hi = math.log2(float(f0[a])), math.log2(float(f0[b - 1]))
        if abs(hi - lo) < 1e-6:
            continue
        for t in range(b - a):
            w = 0.5 - 0.5 * math.cos(math.pi * t / (b - a - 1))
            f0[a + t] = 2.0 ** (lo + (hi - lo) * w)

    onset = round(vibrato_onset_sec * FPS)
    ramp = max(1, round(0.2 * FPS))
    for start, end, _hz in pitch_spans:
        if end - start <= onset:
            continue
        for t in range(start + onset, min(end, total_frames)):
            phase = 2 * math.pi * vibrato_hz * (t - start - onset) / FPS
            depth = min(1.0, (t - start - onset) / ramp)
            f0[t] *= 2.0 ** (vibrato_cents * depth * math.sin(phase) / 1200.0)
    return f0
