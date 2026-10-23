<p align="center">
  <img src="assets/icon.png" alt="VideoHighlighter" width="160">
</p>

<p align="center">
  <a href="https://discord.gg/cUPJqPAMmm"><img src="https://img.shields.io/discord/1471915944652832788?logo=discord&logoColor=white&label=Discord&color=brightgreen" alt="Discord"></a>
</p>

# VideoHighlighter

<!-- hy-mt2-i18n:start -->
[English](./README.md) | **中文** | [日本語](./README_ja.md) | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->

致来自抖音的朋友们 👋
谢谢大家的关注和评论，我们都看到了！
如果程序无法正常运行，请在 Issues 或 Discord 告诉我们具体情况（系统、显卡、报错信息），我们会尽快修复。

**在你不愿上传的视频里，找出关键时刻并说明原因——然后在你自己的电脑上导出剪辑。**

一款本地桌面应用。放入原始视频，它会根据场景、运动、音频、物体、动作和字幕为精彩时刻打分，
在信号时间轴和报告中告诉你每个时刻*为什么*得分，然后导出精彩集锦和单独的片段。
分析过程中不会上传任何内容，基础流程也不需要 API 密钥。

免费开源（AGPL-3.0）。支持 Windows、macOS、Linux，以及 Intel、AMD 和 NVIDIA 显卡。

**应用界面语言**（关于 → 界面语言）：English、简体中文、Español、Português (Brasil)、
Français、日本語、Deutsch、한국어、Bahasa Indonesia、Türkçe、Italiano、Polski、Tiếng Việt、
Nederlands。本 README 以英文版为准；上方的翻译版本覆盖中文 / 日本語 / Español。

> **本项目免费。** 为了以后不错过新版本，请点击页面顶部的“动力按钮” ⭐。
> 这是我们接受的最便宜的付款方式。

## 大家用它来做什么

- **4 小时的 Twitch 或 YouTube 直播回放 → 几分钟的精彩片段。** 观众的欢呼、你自己的反应和场景切换都会计分，
  热闹的部分自然浮现，无需来回拖动进度条。
- **一整夜的监控或野外相机画面 → 有东西在动的那几分钟。** 物体检测加运动检测，在几小时的空镜头里本地完成——
  当画面拍的是你自己的家时，这一点很重要。
- **一张 GoPro 或无人机存储卡 → 一部完整的影片。** **Auto** 标签页会找到存储卡、把素材拷出来、剪出精彩片段、
  合成集锦并让音乐卡点，整个过程是一个可以中断后继续的任务。
- **一场比赛 → 所有进球。** 为“球在球网内”写一条规则，它会根据物体的位置为这个事件打分，
  而不是指望动作模型从画面外观上分辨出来。
- **一段长访谈、讲座或播客 → 章节和字幕。** 本地 Whisper 生成字幕，按章节切分视频，可选本地翻译。
- **行车记录仪或执法记录仪回看 → 一份可以交给别人的说明。** 报告会写明哪些是测量得到的、哪些只是听到的，
  以及哪些无法确定。

不确定哪种检测器适合你的视频？
[docs/DETECTION-GUIDE.md](docs/DETECTION-GUIDE.md) 介绍了每种检测器擅长什么、在哪里会失效。

## 它有什么不同

**[每次运行都会说明自己。](docs/REPORTS.md)** 报告就是每个保留时刻背后的计算过程——各信号的得分明细、
触发了什么、哪些得分不错却仍未入选。来自字幕、从未被测量过的说法会被单独标出，而不是照样计分。
**[打开一份真实的报告 →](https://aseiel.github.io/VideoHighlighter-site/example-report.html)**

**[组合规则。](docs/DETECTION-GUIDE.md#4-the-composition-engine)** 你可以定义多个检测结果的*组合*对你的视频意味着什么——
一个类别在另一个类别里面、数量、在一段时间内持续——它就会成为一个由你命名的事件。
因为规则只是重新读取已有的检测结果，修改一条规则后重新运行只需几毫秒。

## 预览

![VideoHighlighter](assets/Highlighter.png)

**时间轴查看器**

![时间轴查看器](assets/TimelineViewer.png)

**演示**

https://github.com/user-attachments/assets/5c85af94-9228-4537-926a-1ed7a91fa5ee

## 安装

从 [Releases](https://github.com/Aseiel/VideoHighlighter/releases) 下载：

- **Windows** — 运行
  [`00-VideoHighlighter-Windows-Setup.exe`](https://github.com/Aseiel/VideoHighlighter/releases/latest/download/00-VideoHighlighter-Windows-Setup.exe)。
  按用户安装，无需管理员权限。安装程序尚未进行代码签名，所以请点击*更多信息 → 仍要运行*。
- **中国大陆用户：无需梯子，** 直接从镜像下载同样的文件：
  [dl.videohighlighter.com](https://dl.videohighlighter.com/downloads/latest/index.html)。
- **macOS** — 把 `.dmg` 拖入“应用程序”，然后清除一次隔离标记：
  `xattr -dr com.apple.quarantine /Applications/VideoHighlighter.app`
  （[原因](docs/INSTALL.md#macos)）。
- **Linux / 从源码运行** — `pip install -r requirements.txt && python main.py`。FFmpeg 会随之安装。

便携版、GPU 设置、程序写入文件的位置，以及界面过大的解决方法：
**[docs/INSTALL.md](docs/INSTALL.md)**。

## 文档

| | |
| --- | --- |
| [选择检测器](docs/DETECTION-GUIDE.md) | 物体、动作、CLIP 搜索、组合规则——各自的用途 |
| [为什么是这些时刻](docs/REPORTS.md) | 报告包含什么，以及为什么这样设计 |
| [Auto 流程](docs/AUTO-PIPELINE.md) | 存储卡 → 导入 → 脚本 → 音乐 → 集锦，可中断后继续 |
| [安装](docs/INSTALL.md) | 所有平台、GPU 后端、设置 |
| [训练模型](docs/CUSTOM-MODEL-TRAINING.md) | 标注你自己的类别并训练 |
| [社区模型](docs/COMMUNITY-MODELS.md) | 安装别人训练的模型，发布你自己的模型 |
| [Intel GPU](docs/INTEL-GPU.md) · [AMD GPU](docs/AMD-GPU.md) · [Apple GPU](docs/MAC-GPU.md) | 各厂商的加速说明 |
| [远程 ollama](docs/OLLAMA-REMOTE.md) | 在局域网内另一台机器上运行本地 LLM |

## Pro 版本

**VideoHighlighter——也就是本仓库——是基于 AGPL-3.0 的自由软件，并将一直如此。**
它包含离线分析、实时人脸检测、VR 左右并排播放、CLIP 搜索、组合引擎、模型训练——
包括在播放器里画一个框就能教会模型，之后模型会在应用空闲时继续改进——以及模型中心。

**[VideoHighlighter Pro](https://aseiel.github.io/VideoHighlighter-site/)** 是一个单独的付费版本，
在此基础上增加实时功能：播放时实时叠加物体和动作、在播放中途教会一个实时类别、
“查找更多类似内容”搜索、开放词汇检测，以及计数器 / 计分板检测。

“解释”不在其中。报告、发现和顾问在两个版本中完全相同——云端工具给你一个按钮和一个你无法追问的结果；
而在本地回答“为什么”，正是本项目所做的。

## 社区

VideoHighlighter 偶尔会对你的视频有自己的“想法”。遇到这种情况时：
[加入 Discord](https://discord.gg/cUPJqPAMmm)，在 #support 频道里喊一声，我通常都在。
欢迎在 [Issues](https://github.com/Aseiel/VideoHighlighter/issues) 中提交问题和想法。

**常见问题。** [阅读 FAQ](docs/FAQ/faq.md)，获取以下方面的帮助：
- 选择检测器和调整评分
- CPU / GPU 上的性能建议
- 常见问题排查

## 许可证

版权所有 (C) 2026 Przemysław Kreft 与 Meric Donmezer。

依据 GNU Affero General Public License v3.0 发布——可以自由使用、修改和分发，前提是修改后的版本
（包括通过网络提供的版本）必须以相同许可证公开其完整源代码。完整文本见 [LICENSE](LICENSE)；
版权声明见 [COPYRIGHT](COPYRIGHT)。

贡献者保留其作品的版权——参见 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [CLA.md](CLA.md)。
版权持有人也以单独的商业许可提供 VideoHighlighter。

## 参与贡献

关于代码质量、测试以及怎样才算一个好的贡献，请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。
欢迎提交错误报告、小修复和功能建议——第一次提交 PR 之前请先阅读 CLA。

![星标历史](assets/star-history-2026630.png)
