"""index.html の替え歌（歌詞＋メロディ）を LeapSinger で歌声にするCLI。

使い方:

    # 1) 歌詞とメロディが噛み合うかだけ確認する（LeapSinger も学習済みモデルも不要）
    python3 tools/leapsinger/render_verses.py --check

    # 2) 実際に合成する（LeapSinger のクローンと Release のチェックポイントが必要）
    python3 tools/leapsinger/render_verses.py \
        --leapsinger ~/src/LeapSinger \
        --ckpt ~/src/LeapSinger/3speaker_gan2d.pth \
        --song twinkle --speaker 2 --out audio/twinkle

LeapSinger: https://github.com/wavtechyukky/LeapSinger （コードは MIT。学習済みモデルと
歌声DBは各DBの規約に従うこと。配布物の CREDITS.txt を必ず確認する）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from leap_frontend import (build_f0, fit_notes, load_kana_table,  # noqa: E402
                           notes_to_phonemes, resolve_mora, split_mora)


# ── index.html から譜面を取り出す ────────────────────────────────────────
def parse_index(path: str) -> dict:
    """index.html の NOTE / MELODY_* / SONGS から、曲・番・行の歌詞と音符列を取り出す。"""
    src = open(path, encoding="utf-8").read()

    notes = {k: float(v) for k, v in re.findall(r"(\w+):(\d+\.\d+)",
                                                re.search(r"const NOTE=\{(.*?)\};", src, re.S).group(1))}
    qtr = float(re.search(r"const QTR=([\d.]+)", src).group(1))
    last = float(re.search(r"LASTNOTE=([\d.]+)", src).group(1))

    def melody(name: str) -> list[list[str]]:
        block = re.search(rf"const {name}=\[(.*?)\n\];", src, re.S).group(1)
        return [re.findall(r'"(\w+)"', row) for row in re.findall(r"\[(.*?)\]", block, re.S)]

    melodies = {"twinkle": melody("MELODY_TWINKLE"), "clarinet": melody("MELODY_CLAR")}

    songs_block = re.search(r"const SONGS=\{(.*?)\n\};", src, re.S).group(1)
    songs: dict[str, list[dict]] = {}
    for key in ("twinkle", "clarinet"):
        m = re.search(rf"\n  {key}:\{{(.*?)\n  \}},", songs_block, re.S)
        if not m:
            continue
        verses: list[dict] = []
        for token, value in re.findall(r'(?<![A-Za-z_])(title|t):"([^"]*)"', m.group(1)):
            if token == "title":
                verses.append({"title": value, "lines": []})
            elif verses:
                verses[-1]["lines"].append(value)
        songs[key] = verses
    return {"notes": notes, "qtr": qtr, "last": last, "melodies": melodies, "songs": songs}


def phrase_notes(phrase: list[str], notes: dict[str, float], qtr: float, last: float,
                 *, tempo: float = 1.0, transpose: float = 0.0) -> list[tuple[float, float]]:
    """1フレーズの音符列を [(Hz, 秒), ...] で返す。末尾だけ長い、という index.html の再生と同じ規則。"""
    ratio = 2.0 ** (transpose / 12.0)
    return [(notes[name] * ratio, (last if i == len(phrase) - 1 else qtr) * tempo)
            for i, name in enumerate(phrase)]


# ── 検査 ─────────────────────────────────────────────────────────────────
def check(data: dict, table: dict[str, list[str]]) -> int:
    """歌詞のかなが辞書で引けるか、モーラ数と音符数がどれだけずれるかを一覧する。"""
    unknown_total = 0
    for song, verses in data["songs"].items():
        phrases = data["melodies"][song]
        print(f"\n=== {song}: {len(verses)}番 / メロディ {len(phrases)}フレーズ ===")
        for vi, verse in enumerate(verses, 1):
            notes, cram, hold, bad = [], 0, 0, []
            for li, line in enumerate(verse["lines"]):
                moras = split_mora(line, table)
                n_notes = len(phrases[li]) if li < len(phrases) else 0
                unknown = [m for m in moras
                           if m != "ー" and resolve_mora(m, table) is None]
                if unknown:
                    bad.append(f"  行{li + 1}: 辞書にないかな {unknown} — {line}")
                    unknown_total += len(unknown)
                notes.append((len(moras), n_notes))
                if len(moras) > n_notes:
                    cram += len(moras) - n_notes
                elif len(moras) < n_notes:
                    hold += n_notes - len(moras)
            fit = "1対1" if cram == 0 and hold == 0 else f"詰め込み{cram} / 伸ばし{hold}"
            print(f"{vi:2d}. {verse['title'][:26]:<28} {fit}")
            for b in bad:
                print(b)
    print(f"\n辞書にないかな: {unknown_total} 個"
          f"（「詰め込み」「伸ばし」は fit_notes が自動で吸収する。多い曲は手当てが要る）")
    return unknown_total


# ── 合成 ─────────────────────────────────────────────────────────────────
def render(data: dict, table, args) -> None:
    sys.path.insert(0, args.leapsinger)
    import numpy as np
    import soundfile as sf
    from infer import build_item, infer_mel, load_acoustic, load_vocoder, mel_to_wav  # noqa: E402

    model, cfg = load_acoustic(args.ckpt)
    voc = load_vocoder(os.path.join(args.leapsinger, "checkpoints", "nhv_v3.onnx"))
    phrases = data["melodies"][args.song]
    os.makedirs(args.out, exist_ok=True)

    for vi, verse in enumerate(data["songs"][args.song], 1):
        chunks = []
        for li, line in enumerate(verse["lines"]):
            moras = split_mora(line, table)
            notes = phrase_notes(phrases[li], data["notes"], data["qtr"], data["last"],
                                 tempo=args.tempo, transpose=args.transpose)
            per_mora = fit_notes(moras, notes)
            phonemes, frames, pitch_spans = notes_to_phonemes(moras, per_mora, table)
            f0 = build_f0(pitch_spans, sum(frames))
            item = build_item(phonemes, frames, f0, spk_id=args.speaker,
                              vuv_mode="f0" if cfg.get("use_uv", False) else "all")
            mel = infer_mel(model, item, num_steps=cfg.get("num_steps", 1))
            chunks.append(mel_to_wav(voc, mel, item["f0_logf0"], item["uv"]))
        wav = np.concatenate(chunks)
        path = os.path.join(args.out, f"verse{vi:02d}.wav")
        sf.write(path, wav, 44100)
        print(f"{path}  {len(wav) / 44100:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=os.path.join(REPO, "index.html"))
    ap.add_argument("--leapsinger", default=os.environ.get("LEAPSINGER_DIR", ""),
                    help="LeapSinger のクローン先")
    ap.add_argument("--ckpt", help="学習済み音響モデル（Release の .pth）")
    ap.add_argument("--song", default="twinkle", choices=["twinkle", "clarinet"])
    ap.add_argument("--speaker", type=int, default=2, help="話者ID（3話者モデル: 0/1/2）")
    ap.add_argument("--transpose", type=float, default=0.0, help="半音単位の移調")
    ap.add_argument("--tempo", type=float, default=1.0, help="音長の倍率（大きいほど遅い）")
    ap.add_argument("--out", default=os.path.join(REPO, "audio", "twinkle"))
    ap.add_argument("--check", action="store_true", help="歌詞と音符の対応だけ検査する")
    args = ap.parse_args()

    dict_path = (os.path.join(args.leapsinger, "dict", "kana2phonemes.table")
                 if args.leapsinger else os.path.join(HERE, "kana2phonemes.table"))
    table = load_kana_table(dict_path)
    data = parse_index(args.index)

    if args.check or not args.ckpt:
        if not args.check:
            print("--ckpt が無いので検査のみ実行します\n", file=sys.stderr)
        return 1 if check(data, table) else 0
    render(data, table, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
