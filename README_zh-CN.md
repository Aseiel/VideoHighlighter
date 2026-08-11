<p align="center">
  <img src="assets/icon.png" alt="VideoHighlighter" width="160">
</p>

<!-- hy-mt2-i18n:start -->
[English](./README.md) | **中文** | [日本語](./README_ja.md) | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->

VideoHighlighter（免费版）

一款基于 Python 的工具，能够通过场景检测、动作检测、音频峰值分析、物体检测、动作识别以及字幕分析，自动从视频中生成精彩片段。

> **本项目免费。** 为了以后不错过新版本，请点击页面顶部的“动力按钮” ⭐。
> 这是我们接受的最便宜的付款方式。


功能特性

检测功能：
- 使用 OpenCV 检测场景。
- 检测动作峰值与场景变化。
- 检测物体。
- 检测动作。
- 检测音频峰值。

通过 OpenAI Whisper 生成字幕。  
筛选出得分最高的片段并将其合并为精彩集锦视频。  
支持全面自定义：帧跳过间隔、高光时长及关键词。  
还提供可选的图形界面，便于操作。

不确定该选用哪种检测器？请查看
[docs/DETECTION-GUIDE.md](docs/DETECTION-GUIDE.md)，了解物体识别、动作识别、CLIP搜索以及合成引擎各自的优点与不足。

> **需要实时检测吗？** 上述所有功能均为事后离线处理。
> [VideoHighlighter Pro](#pro-edition) 则可在播放过程中添加实时的物体与动作叠加显示、通过示例学习分类、开放词汇表检测以及反欺诈检测功能。[查看差异 →](#pro-edition)


## 预览

![VideoHighlighter](assets/Highlighter.png)

## 时间轴查看器
![时间轴查看器](assets/TimelineViewer.png)

## 动作识别
![动作识别](assets/power_rangers_actions_annotated.gif)

## 工作流程阶段
![Workflow Stages](assets/workflow_stages.png)

## Pro版本

该版本已包含实时人脸检测、VR并排播放与渲染功能、离线分析、CLIP搜索、组合引擎以及训练脚本。

[VideoHighlighter Pro](https://aseiel.github.io/VideoHighlighter-site/) 增加了以下功能：

- **实时物体与动作叠加**——在播放过程中实现实时检测，包括在并排显示的VR视频中也能进行检测。
- **通过指向来标注类别**——只需在任何对象周围画一个框并为其命名，系统就会立即开始对该对象进行评分，无需任何数据集或训练过程。
- **查找类似内容**——选定某一帧中的某个区域，即可在整个视频中搜索与之相同的内容。
- **开放词汇检测**——直接输入普通单词即可找到对应内容，无需使用预先训练好的模型。
- **计数器/计分板检测**——如果视频中存在屏幕上的计数器，每次数值变化都代表一次事件发生，因此Pro版本能够显示检测器遗漏了哪些真实时刻。

该版本依然免费，并采用 AGPL-3.0 许可协议。

## 安装

### Windows（推荐）
从[Releases](https://github.com/Aseiel/VideoHighlighter/releases)下载最新的`.exe`文件——无需安装Python或任何依赖项。

### Linux / 从源码构建
1. **Python & FFmpeg**
   系统中必须已安装 FFmpeg，且其路径需在系统 PATH 中。

## 使用方法
Linux：python main.py
Windows：运行 Videohighlighter.exe
Mac：目前似乎无法使用，未来会修复。仍会生成 DMG 文件

## Discord
VideoHighlighter 有时会对你的视频内容发表“意见”。当它这么做时：
[加入 Discord 社区](https://discord.gg/cUPJqPAMmm)，在 #support 频道里发消息，我通常会在那里。


## 备注

OpenAI Whisper采用MIT许可证，可自由使用。

Google Translate API 是可选的。如果使用非官方库（googletrans），则无需 API 密钥，但若 Google 更改接口地址，功能可能会出现异常。

该项目不包含任何付费 API 密钥。若要使用官方服务，用户需自行提供相关密钥。


## 许可证

本仓库遵循 GNU Affero General Public License v3.0（AGPLv3）发布。您可自由使用、修改及分发该代码，前提是所有修改后的版本，包括通过网络提供的版本，都必须以相同许可证公开其完整的源代码。


## 项目背景

这个项目最初是作为一款个人工具开发的，旨在为我7岁的小儿子自动为视频生成字幕。后来，它逐渐发展成了可用于电影、体育赛事及个人视频的高光片段生成器。

该项目的首要目标依然很实际：加快视频分析速度，自动生成精彩片段，并自动创建可用的字幕。

![星星历史记录](assets/star-history-2026630.png)
