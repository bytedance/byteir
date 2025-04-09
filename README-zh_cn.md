# ByteIR项目

[English](README.md) | 中文 | [日本語](README-ja_jp.md)

ByteIR项目是字节跳动的模型编译解决方案。ByteIR包括编译器、运行时和前端，并提供端到端的模型编译解决方案。
尽管所有的ByteIR组件（编译器/运行时/前端）一起提供端到端的解决方案，并且都在同一个代码库下，但每个组件在技术上都可以独立运行。

## ByteIR名字由来

ByteIR这个名称是个公司内部的历史遗留物。ByteIR项目并不是一个IR规范定义项目。相反，在大多数情况下，ByteIR直接使用上游的几个MLIR方言和Google Mhlo。大多数ByteIR编译器的pass与所选的上游MLIR方言和Google Mhlo兼容。

## 为何选择使用ByteIR
* **最先进的模型:**
ByteIR会负责维护前端模型转换到Stablehlo，并且提供模型库（近期开放）方便科研和测试。
* **方便好用:**
ByteIR直接使用上游MLIR方言和Google Mhlo，为编译器提供兼容的passes和基础设施。允许混合使用passes去建构编译器，这包括ByteIR，上游MLIR方言，Mhlo或是自己写的passes。  
* **新硬件支持:**
ByteIR提供大量Mhlo和Linalg方言的图优化，Loop优化，或者张量优化，新硬件编译器可以复用，大大化简编译器开发。

## 项目状态

ByteIR目前仍处于早期阶段。在这个阶段，我们的目标是为广泛的深度学习加速器以及通用CPU和GPU提供定义明确、必要的构建块和基础架构支持，以进行模型编译。因此，并没有将针对特定架构的高度调优的kernel代码作为优先考虑。当然，欢迎任何有关优先考虑特定架构的贡献。

## [编译器](compiler/README.md)

ByteIR编译器是一个基于MLIR的，用于CPU/GPU/ASIC的编译器。

## [Runtime](runtime/README.md)

ByteIR Runtime是一个通用、轻量级的runtime，能够接入现成的kernel库和ByteIR编译器生成的kernel。

## [前端](frontends/README.md)

ByteIR前端支持Tensorflow，PyTorch，和ONNX。

## 各组件的交互接口

每个ByteIR组件在技术上都可以独立运行。组件之间有预定义的交互接口。

### 前端和编译器之间使用Stablehlo

ByteIR前端和ByteIR编译器通过Stablehlo方言进行交互（注意在开发过程中Stablehlo的版本可能会更新）。这也意味着，任何生成兼容版本Stablehlo的前端都可以与ByteIR编译器交互，并且任何使用兼容版本Stablehlo的编译器都可以与ByteIR前端交互。

### 编译器和runtime之间使用ByRE

ByteIR编译器和ByteIR Runtime通过ByRE格式进行交互，其版本可能在开发过程中更新。ByRE方言在ByteIR编译器中被定义为一种ByRE格式，目前支持生成文本形式或者带有版本控制的字节码形式。其他ByRE格式正在开发中。

## 出版与引用

ByteIR是许多字节跳动的杰出研究人员和实习生共同努力的成果。以下是我们的公开演讲列表：
* [Linalg is All You Need to Optimize Attention](talks/c4ml23_poster.pdf) -- C4ML'23
* [ByteIR: 迈向端到端的AI编译](talks/ChinaSoftCon-ByteIR.pdf) -- China SoftCon'23

如果您认为ByteIR有用，请考虑引用。
``` 
@misc{byteir2023,
title = {{ByteIR}},
author = {Cao, Honghua and Chang, Li-Wen and Chen, Chongsong and Jiang, Chengquan and Jiang, Ziheng and Liu, Liyang and Liu, Yuan and Liu, Yuanqiang and Shen, Chao and Wang, Haoran and Xiao, Jianzhe and Yao, Chengji and Yuan, Hangjian and Zhang, Fucheng and Zhang, Ru and Zhang, Xuanrun and Zhang, Zhekun and Zhang, Zhiwei and Zhu, Hongyu and Liu, Xin},
url = {https://github.com//bytedance/byteir},
year = {2023}
}
```

## [License](LICENSE)

ByteIR项目采用Apache License v2.0。
