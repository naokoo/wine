# LeapSinger 用ツール

替え歌（`index.html` の `SONGS`）の歌詞とメロディを、歌声合成モデル
[LeapSinger](https://github.com/wavtechyukky/LeapSinger) に食わせる形へ変換して
`audio/<曲>/verseNN.wav` を作るためのスクリプト。

| ファイル | 中身 |
|---|---|
| `leap_frontend.py` | かな歌詞＋音符 → 音素・音素ごとのフレーム数・F0カーブ |
| `render_verses.py` | `index.html` から譜面を読み、検査／合成を行うCLI |
| `kana2phonemes.table` | かな→音素表（LeapSinger からの複製。下記参照） |

## 使い方

歌詞と音符の対応だけ見る（LeapSinger もモデルも不要）:

    python3 tools/leapsinger/render_verses.py --check

実際に合成する:

    git clone https://github.com/wavtechyukky/LeapSinger ~/src/LeapSinger
    cd ~/src/LeapSinger && pip install -e .        # PyTorch は環境に合わせて別途
    # 学習済みモデル（3speaker_gan2d.pth）を Release からダウンロードして置く

    python3 tools/leapsinger/render_verses.py \
      --leapsinger ~/src/LeapSinger \
      --ckpt ~/src/LeapSinger/3speaker_gan2d.pth \
      --song twinkle --speaker 2 --out audio/twinkle

二次対策の替え歌（`index2.html` の「暗記ソング」タブ）を合成する場合:

    python3 tools/leapsinger/render_verses.py --index index2.html --check
    python3 tools/leapsinger/render_verses.py --index index2.html \
      --leapsinger ~/src/LeapSinger --ckpt ~/src/LeapSinger/3speaker_gan2d.pth \
      --song taste --speaker 2 --out audio/taste

`--speaker` は3話者モデルの話者ID（0=御丹宮くるみ / 1=夏目悠李 / 2=波音リツ）。
`--transpose` で半音単位の移調、`--tempo` で音長の倍率を変えられる。

## 音符の割り当て

1モーラ＝1音符が基本。数が合わない行は `fit_notes()` が自動で吸収する。

- モーラ＜音符（キラキラ星版に多い）… 余った音符を最後のモーラが引き受ける（語尾を伸ばす）
- モーラ＞音符（クラリネット版は全番これ）… 1音符に複数モーラを詰めて等分する

クラリネット版は1行あたり3〜5モーラ余るので、自動割当では早口になる。
気になる場合は歌詞側を音符数に合わせるか、行ごとの割当を手で与える必要がある。
`index2.html` の二次対策ソングは1行7モーラで作ってあり、11番中10番が1対1で収まる。

## 注意

- `kana2phonemes.table` はカタカナ＝無声化母音（`ボ` → `b O`）、ひらがな＝有声母音（`ぼ` → `b o`）
  という UTAU 系の慣習。歌わせるときは必ずひらがなに正規化する（`to_hiragana()` が行う）。
- 歌詞中の英字・漢字（`PN`、`AOC`、`上` など）は辞書を引けないので警告を出して読み飛ばす。
  読ませたい場合は歌詞をかなにする。

## ライセンス

`kana2phonemes.table` は LeapSinger（MIT License, Copyright (c) 2026 wavtechyukky）
に含まれるファイルの複製。MIT の許諾表示は LeapSinger リポジトリの `LICENSE` を参照。

LeapSinger の**学習済みモデル**と、その学習に使われた歌声データベース
（御丹宮くるみ／夏目悠李／波音リツ）は MIT の対象外で、それぞれの利用規約に従う。
生成した音声を公開・配布する前に、モデル配布物の `CREDITS.txt` と
「夏目悠李の出力音声に関する利用規約」を必ず確認すること。
