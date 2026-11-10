# PointCloudConverter

フォルダ内の `.las` ファイルをまとめて、八分木（Octree）形式の点群ファイル `.opc` に変換する Python スクリプトです。
出力した `.opc` は [PointCloudViewer](https://github.com/UpaDotti/PointCloudViewer)（Unity）で表示します。

## 必要なもの

- Python 3.10 以上（3.12 で動作確認）
- `numpy`、`laspy`、`tqdm`

```bash
pip install -r requirements.txt
```

## 使い方

リポジトリの直下で、入力フォルダと出力フォルダを指定して実行します。

```bash
python -m PointCloudConverter.OctreePointCloud.convert_opc <LASのフォルダ> <出力フォルダ>
```

- 入力フォルダの中の `*.las` を、サブフォルダも含めてすべて読み込み、1つの点群として変換します
- 出力フォルダに `sample.opc` を書き出します。途中で作る中間ファイルは最後に削除します
- 中間ファイルを書くため、出力フォルダには入力の点群と同じくらいの空き容量が必要です
- 出力フォルダ内の `nodes` フォルダは中間ファイル用として最後に丸ごと削除します。出力フォルダには空のフォルダか専用のフォルダを指定してください

### 入力できる LAS

- 色（RGB）を持つ LAS が必要です。色の無い LAS（点フォーマット 0 / 1 / 4 など）を渡すと `no field of name red` のエラーで止まります
- 圧縮形式の `.laz` は読み込み対象外です（検索するのは `*.las` だけです）

### Git Bash でヘルプが文字化けするとき

Git Bash では Python の出力が Shift_JIS になり、日本語が文字化けすることがあります。`PYTHONUTF8=1` を付けて実行してください。PowerShell ではそのまま表示されます。

```bash
PYTHONUTF8=1 python -m PointCloudConverter.OctreePointCloud.convert_opc --help
```

## 変換の流れ

1. 点群を8分割のキャッシュへ振り分ける
2. 八分木を作る
3. 葉ノードの LOD を作る
4. 枝ノードの代表点を作る
5. `.opc` を書き出す
6. 中間ファイルを削除する

## 座標と色の扱い

- LAS の Y と Z を入れ替えて、Unity の座標系（Y が上）に合わせます
- 座標はノードごとの範囲で 16bit に量子化します（量子化＝細かい値を決まった段階に丸めること）
- 色は 16bit から 8bit に変換します

## OPC形式（1.0.0）

すべてリトルエンディアンです。

| 部分 | 内容 |
| --- | --- |
| ヘッダ | マジック `OPC\0`、バージョン（3byte）、全体の大きさと範囲、枝と葉の数、元の点数、1点のバイト数 |
| 枝の索引 | 深さ、八分木上の位置（path）、範囲、点データの開始位置、代表点の数 |
| 葉の索引 | 番号、深さ、path、範囲、点データの開始位置、点の数 |
| 点データ | 1点 9byte（位置 `u16`×3、色 `u8`×3） |

細かい並びは `PointCloudConverter/OctreePointCloud/opc_writer.py` が正本です。

## 設定

八分木の深さや1ノードの最大点数、並列処理のプロセス数は `PointCloudConverter/OctreePointCloud/config.py` の `Options` で変えます。

## ライセンス

[MIT](LICENSE)
