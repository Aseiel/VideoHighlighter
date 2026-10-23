<p align="center">
  <img src="assets/icon.png" alt="VideoHighlighter" width="160">
</p>

<p align="center">
  <a href="https://discord.gg/cUPJqPAMmm"><img src="https://img.shields.io/discord/1471915944652832788?logo=discord&logoColor=white&label=Discord&color=brightgreen" alt="Discord"></a>
</p>

# VideoHighlighter

<!-- hy-mt2-i18n:start -->
[English](./README.md) | [中文](./README_zh-CN.md) | **日本語** | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->

**アップロードしたくない映像から大事な瞬間を見つけ、その理由を示し、手元のマシンで切り出します。**

ローカルで動くデスクトップアプリです。撮ったままの動画を入れると、シーン・動き・音声・物体・動作・
文字起こしから見どころを採点し、各瞬間が*なぜ*選ばれたのかをシグナルタイムラインとレポートで示したうえで、
ハイライト動画と個別のクリップを書き出します。解析のために何かがアップロードされることはなく、
基本機能に API キーは不要です。

無料のオープンソース（AGPL-3.0）。Windows・macOS・Linux 対応。Intel・AMD・NVIDIA の GPU に対応。

**アプリの表示言語**（About → Interface language）：English、简体中文、Español、
Português (Brasil)、Français、日本語、Deutsch、한국어、Bahasa Indonesia、Türkçe、Italiano、
Polski、Tiếng Việt、Nederlands。この README は英語版が基準で、上の翻訳版は
中文 / 日本語 / Español に対応しています。

> **無料です。** 今後の新しいリリースを見逃さないよう、ページ上部の「やる気ボタン」⭐ を押してください。
> 私たちが受け取る中で一番安い支払い方法です。

## どんな用途に使われているか

- **4 時間の Twitch や YouTube のアーカイブ → 数分のハイライト。** 観客の歓声、自分のリアクション、
  シーンの切り替わりがすべて点数になるので、盛り上がった部分がシークバーを動かさなくても浮かび上がります。
- **一晩分の防犯カメラやトレイルカメラの映像 → 何かが動いた数分間。** 何も起きない何時間分もの映像に対して、
  物体検出と動き検出をローカルで行います。映っているのが自分の家なら、これは大事なことです。
- **GoPro やドローンの SD カード → 完成した作品。** **Auto** タブがカードを見つけ、素材をコピーし、
  ハイライトを切り出し、まとめた動画を作ってビートに合わせて音楽を乗せます。全体が中断・再開できる
  ひとつのジョブです。
- **試合 → ゴールシーン。** 「ボールがネットの中」というルールを書けば、行動モデルが見た目で判別してくれるのを
  期待するのではなく、物の位置関係からそのイベントを採点します。
- **長いインタビュー、講義、ポッドキャスト → チャプターと字幕。** ローカルの Whisper による文字起こし、
  チャプター分けした動画、任意でローカル翻訳。
- **ドラレコやボディカメラの確認 → 人に渡せる記録。** レポートは、測定したこと、言われただけのこと、
  判断できなかったことを分けて書きます。

どの検出器が自分の映像に合うかわからないときは、
[docs/DETECTION-GUIDE.md](docs/DETECTION-GUIDE.md) にそれぞれの得意なことと苦手なことをまとめてあります。

## ほかと何が違うか

**[毎回の実行が自分で説明します。](docs/REPORTS.md)** レポートは、残った各瞬間の裏にある計算そのものです。
シグナルごとの得点の内訳、何が反応したか、点数は高かったのに選ばれなかったもの。文字起こしから来ていて
一度も測定されていない主張は、そのまま採点せずに、そうだと明記します。
**[実際のレポートを開く →](https://aseiel.github.io/VideoHighlighter-site/example-report.html)**

**[組み合わせルール。](docs/DETECTION-GUIDE.md#4-the-composition-engine)** 検出結果の*組み合わせ*が
自分の映像で何を意味するかを指定できます。あるクラスが別のクラスの中にある、数を数える、一定時間続く、
といった条件が、自分で名前を付けたイベントになります。ルールはすでにある検出結果を読み直すだけなので、
編集して再実行してもかかるのは数ミリ秒です。

## プレビュー

![VideoHighlighter](assets/Highlighter.png)

**タイムラインビューアー**

![タイムラインビューアー](assets/TimelineViewer.png)

**デモ**

https://github.com/user-attachments/assets/5c85af94-9228-4537-926a-1ed7a91fa5ee

## インストール

[Releases](https://github.com/Aseiel/VideoHighlighter/releases) からビルドを入手してください：

- **Windows** —
  [`00-VideoHighlighter-Windows-Setup.exe`](https://github.com/Aseiel/VideoHighlighter/releases/latest/download/00-VideoHighlighter-Windows-Setup.exe)
  を実行します。ユーザー単位のインストールで、管理者権限は不要です。まだコード署名されていないため、
  *詳細情報 → 実行* をクリックしてください。
- **GitHub が遅い、またはつながらない場合**は、同じファイルを
  [dl.videohighlighter.com](https://dl.videohighlighter.com/downloads/latest/index.html) からも入手できます。
- **macOS** — `.dmg` をアプリケーションフォルダにドラッグし、隔離フラグを一度だけ外します：
  `xattr -dr com.apple.quarantine /Applications/VideoHighlighter.app`
  （[理由](docs/INSTALL.md#macos)）。
- **Linux / ソースから** — `pip install -r requirements.txt && python main.py`。FFmpeg も一緒に入ります。

ポータブル版、GPU の設定、書き込み先、大きすぎる UI の直し方：
**[docs/INSTALL.md](docs/INSTALL.md)**。

## ドキュメント

| | |
| --- | --- |
| [検出器の選び方](docs/DETECTION-GUIDE.md) | 物体、動作、CLIP 検索、組み合わせルール――それぞれの用途 |
| [なぜこれらの瞬間なのか](docs/REPORTS.md) | レポートの中身と、そう作られている理由 |
| [Auto パイプライン](docs/AUTO-PIPELINE.md) | カード → 取り込み → 台本 → 音楽 → ハイライト動画、中断・再開可能 |
| [インストール](docs/INSTALL.md) | 全プラットフォーム、GPU バックエンド、設定 |
| [モデルの学習](docs/CUSTOM-MODEL-TRAINING.md) | 自分のクラスにラベルを付けて学習する |
| [コミュニティモデル](docs/COMMUNITY-MODELS.md) | 他の人が学習したモデルを入れる、自分のモデルを公開する |
| [Intel GPU](docs/INTEL-GPU.md) · [AMD GPU](docs/AMD-GPU.md) · [Apple GPU](docs/MAC-GPU.md) | メーカー別の高速化 |
| [リモートの ollama](docs/OLLAMA-REMOTE.md) | LAN 内の別のマシンでローカル LLM を動かす |

## Pro 版

**VideoHighlighter（このリポジトリ）は AGPL-3.0 のフリーソフトウェアで、これからもそうです。**
オフライン解析、リアルタイムの顔検出、VR のサイドバイサイド再生、CLIP 検索、組み合わせエンジン、
モデルの学習（プレーヤー上で枠をひとつ描くだけでモデルに教えられ、その後もアプリが空いている間に
改善し続けます）、そしてモデルハブを含みます。

**[VideoHighlighter Pro](https://aseiel.github.io/VideoHighlighter-site/)** は別の有料版で、
その上にリアルタイムの機能を加えます：再生中の物体・動作のライブ表示、再生しながらのライブカテゴリの学習、
「似たものをもっと探す」検索、オープンボキャブラリ検出、カウンター／スコアボードの検出。

説明の機能はそこには含まれません。レポート、所見、アドバイザーはどちらの版でもまったく同じです。
クラウドのツールが渡すのはボタンと、問いただせない結果です。「なぜ」にローカルで答えること、
それがこのアプリの役割です。

## コミュニティ

VideoHighlighter は、ときどき映像について思うところがあります。そんなときは
[Discord に参加して](https://discord.gg/cUPJqPAMmm) #support で声をかけてください。たいていいます。
バグやアイデアは [Issues](https://github.com/Aseiel/VideoHighlighter/issues) へどうぞ。

**よくある質問。** [FAQ を読む](docs/FAQ/faq.md)：
- 検出器の選び方と採点の調整
- CPU / GPU でのパフォーマンスのヒント
- よくある問題の解決

## ライセンス

Copyright (C) 2026 Przemysław Kreft and Meric Donmezer.

GNU Affero General Public License v3.0 で公開しています。自由に使用・改変・再配布できますが、
改変版（ネットワーク越しに提供するものを含む）は、同じライセンスで完全なソースコードを公開する
必要があります。全文は [LICENSE](LICENSE)、著作権表示は [COPYRIGHT](COPYRIGHT) にあります。

貢献者は自分の作品の著作権を保持します。[CONTRIBUTING.md](CONTRIBUTING.md) と [CLA.md](CLA.md)
を参照してください。VideoHighlighter は、著作権者による別の商用ライセンスでも提供されています。

## コントリビュート

コードの品質、テスト、良いコントリビューションとは何かについては
[CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。バグ報告、小さな修正、機能の提案を歓迎します。
最初の PR の前に CLA を確認してください。

![スター履歴](assets/star-history-2026630.png)
