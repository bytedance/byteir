# ByteIRプロジェクト

英語 | [中国語](README-zh_cn.md) | 日本語

ByteIRプロジェクトは、ByteDanceのモデルコンパイルソリューションです。
ByteIRには、コンパイラ、ランタイム、フロントエンドが含まれており、エンドツーエンドのモデルコンパイルソリューションを提供します。

すべてのByteIRコンポーネント（コンパイラ/ランタイム/フロントエンド）が一緒にエンドツーエンドのソリューションを提供し、すべてがこのリポジトリの同じ傘下にあるにもかかわらず、各コンポーネントは技術的には独立して機能することができます。

## ByteIRの名前の由来
ByteIRという名前は、社内の遺産から来ています。
ByteIRプロジェクトは、IR仕様定義プロジェクトではありません。
代わりに、ほとんどのシナリオで、ByteIRはいくつかの上流のMLIR方言とGoogle Mhloを直接使用します。
ByteIRコンパイラのほとんどのパスは、選択された上流のMLIR方言とGoogle Mhloと互換性があります。

## ByteIRを選ぶ理由
* **最先端のモデルを楽しむ：**
ByteIRは、多くのSOTAモデルをStablehloに落とし込むための人気のあるフロントエンドを維持し、研究やベンチマーク目的でモデル動物園（近日公開予定）も提供します。
* **使いやすさ：**
ByteIRは、上流のMLIR方言とGoogle Mhloを採用し、すべてのコンパイラビルダーが上流のMLIRを使用するための互換性のあるパス、ユーティリティ、およびインフラストラクチャを提供します。ByteIRパスを上流のMLIRやMhloパスと混在させて使用することができます。
* **独自のアーキテクチャを持ち込む：**
ByteIRは、MhloおよびLinalgの豊富な一般的なグラフ、ループ、テンソルレベルの最適化を提供し、DL ASICコンパイラが再利用し、バックエンドの最後のマイルにのみ焦点を当てることができます。

## プロジェクトの状況
ByteIRはまだ初期段階にあります。
この段階では、幅広いディープラーニングアクセラレーターおよび汎用CPUおよびGPUでのモデルコンパイルのための明確に定義された、必要なビルディングブロックおよびインフラストラクチャサポートを提供することを目指しています。
したがって、特定のアーキテクチャの高度にチューニングされたカーネルコードは優先されていません。
もちろん、特定のアーキテクチャを優先するためのフィードバックや対応する貢献を歓迎します。

## [コンパイラ](compiler/README.md)

ByteIRコンパイラは、CPU/GPU/ASIC用のMLIRベースのコンパイラです。

## [ランタイム](runtime/README.md)

ByteIRランタイムは、既存のカーネルとByteIRコンパイラが生成したカーネルの両方を提供するための一般的で軽量なランタイムです。

## [フロントエンド](frontends/README.md)

ByteIRフロントエンドには、Tensorflow、PyTorch、およびONNXが含まれます。

## コンポーネント間の通信インターフェース
各ByteIRコンポーネントは、技術的には独立して機能することができます。
コンポーネント間には、事前に定義された通信インターフェースがあります。

### フロントエンドとコンパイラ間のStablehlo
ByteIRフロントエンドとByteIRコンパイラは、Stablehlo方言を介して通信します。このバージョンは、開発中に更新される可能性があります。

これはまた、互換性のあるバージョンのStablehloを生成するフロントエンドであれば、ByteIRコンパイラと連携できることを意味します。また、互換性のあるバージョンのStablehloを使用するコンパイラであれば、ByteIRフロントエンドと連携できます。

### コンパイラとランタイム間のByRE

ByteIRコンパイラとByteIRランタイムは、ByRE形式を介して通信します。このバージョンは、開発中に更新される可能性があります。
ByRE方言は、ByteIRコンパイラ内でByRE形式の一種として定義されており、現在、ByteIRコンパイラとランタイムのためにバージョニングされたテキスト形式またはバイトコード形式を生成することができます。

他のByRE形式は開発中です。

## 出版物と引用

ByteIRは、ByteDanceの多くの優れた研究者とインターンによる成果です。以下は、公開されたトークのリストです。

* [Linalg is All You Need to Optimize Attention](talks/c4ml23_poster.pdf) -- C4ML'23
* [ByteIR: Towards End-to-End AI Compilation](talks/ChinaSoftCon-ByteIR.pdf) -- China SoftCon'23

ByteIRが役立つと思われる場合は、引用を検討してください。
``` 
@misc{byteir2023,
title = {{ByteIR}},
author = {Cao, Honghua and Chang, Li-Wen and Chen, Chongsong and Jiang, Chengquan and Jiang, Ziheng and Liu, Liyang and Liu, Yuan and Liu, Yuanqiang and Shen, Chao and Wang, Haoran and Xiao, Jianzhe and Yao, Chengji and Yuan, Hangjian and Zhang, Fucheng and Zhang, Ru and Zhang, Xuanrun and Zhang, Zhekun and Zhang, Zhiwei and Zhu, Hongyu and Liu, Xin},
url = {https://github.com//bytedance/byteir},
year = {2023}
}
```

## [ライセンス](LICENSE)

ByteIRプロジェクトは、Apache License v2.0の下にあります。
