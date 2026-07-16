# 服务器 MVP 准备设计

_状态：已口头批准，等待书面复核 | 日期：2026-07-10_

---

## 🎯 目标与边界

本阶段只建立可复现推理环境、下载 `Qwen2.5-VL-3B-Instruct` 并完成图像推理冒烟测试。模型官方仓库文件约 7.52 GB。[^1]

包括：独立环境、目录与缓存规划、3B模型下载、基础图像推理、结构化输出测试、资源记录。

不包括：7B模型、数据集下载、QLoRA训练、SAM2、正式实验和大规模评估。

## 📁 目录与存储

- 代码和Conda环境使用系统盘
- 模型、数据、缓存、日志和输出统一放在 `/root/autodl-tmp/EviAgriDiag/`
- 首批目录为 `models/`、`datasets/`、`cache/`、`outputs/`、`logs/`
- 数据盘剩余空间预计低于20 GB前通知用户扩容；未经确认不下载7B或多个完整数据集

## ⚙️ 环境设计

- 新建独立Conda环境，不修改base环境
- 初始版本矩阵固定为Python 3.10、PyTorch 2.5.1+cu121、TorchVision 0.20.1+cu121、Transformers 4.51.3、Qwen-VL-Utils 0.0.8和Accelerate 1.6.0
- 首轮不安装FlashAttention、vLLM或训练依赖，推理统一使用BF16与SDPA
- 安装Qwen图像推理所需的最小依赖；Qwen2.5-VL已由Transformers提供原生模型类支持[^2]
- 冒烟测试通过后导出完整版本清单，作为后续实验环境基线

## 📦 下载与验证

1. 优先使用ModelScope下载到固定模型目录；Qwen官方建议中国大陆环境使用ModelScope下载检查点[^3]
2. 检查模型分片、配置、处理器和分词器文件是否齐全
3. 运行官方风格的单图描述测试
4. 运行农业诊断结构化提示测试：证据存在性、bbox、属性、诊断、可靠性
5. 记录峰值显存、推理耗时、磁盘占用、模型输出和依赖版本

## ✅ 验收标准

- CUDA和模型导入正常，3B模型可在单卡上完整加载
- 单张图像能生成非空、可读回答
- 结构化提示能返回可保存的文本；现阶段不要求诊断或bbox正确
- 测试过程无OOM，资源记录完整
- 完成后数据盘仍保留至少20 GB可用空间

## ⚠️ 失败处理

- 下载中断：使用断点续传，不重复创建模型副本
- 依赖冲突：停止安装并记录冲突；只有查明原因后才调整单个版本，只重建独立环境，不修改base
- 显存不足：降低输入像素和生成长度，使用BF16与SDPA
- 磁盘不足：立即停止新增下载并通知用户扩容
- Hugging Face不可达：继续使用ModelScope，不切换到不明镜像源

## 🔗 阶段结束后的决策

通过验收后，再根据本地Age数据位置和公开数据可得性，决定首个数据分支。默认优先比较“Age/IP102害虫分支”和“PlantVillageVQA病害分支”，不在本阶段预先下载。

## 🔗 参考资料

[^1]: Qwen. “Qwen2.5-VL-3B-Instruct model files.” Hugging Face. https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main

[^2]: Hugging Face. “Qwen2.5-VL model documentation.” Transformers. https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_5_vl.md

[^3]: QwenLM. “Qwen vision-language model repository.” GitHub. https://github.com/QwenLM/Qwen2.5-VL
