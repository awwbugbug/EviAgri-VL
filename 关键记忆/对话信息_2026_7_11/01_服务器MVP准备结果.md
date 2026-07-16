# 服务器 MVP 准备结果

_日期：2026-07-11 | 状态：成功_

---

## 📊 硬结果

- 环境：Python 3.10、PyTorch 2.5.1+cu121、Transformers 4.51.3、Accelerate 1.6.0、ModelScope 1.25.0、Pillow 11.2.1
- 模型：Qwen2.5-VL-3B-Instruct，`MODEL_OK`，2个分片，共7,520,919,303字节
- 冒烟测试：普通描述2.22秒；结构化输出1.60秒；峰值显存7.40 GB；无Warning、OOM或异常
- 结构化结果可解析；非农业演示图返回 `evidence_present=false`、`diagnosis=uncertain`
- 存储：模型约7.1 GB；独立环境约5.4 GB；数据盘剩余约43 GB；系统盘剩余约24 GB
- base环境前后完全一致；未下载7B或任何数据集

## ⚠️ 问题与决策

- PyTorch官方源直连慢；网络加速适合大型wheel，但会拖慢普通依赖
- 最终采用已校验wheel复用 + 清华PyPI补齐普通依赖，版本未改变
- 当前无需扩容；下一步先做数据资产审计和首个数据分支选择
- 本次只证明环境、模型和结构化推理链路可用，不代表农业诊断性能

## 🔗 证据文件

- [模型校验](../../artifacts/server_mvp/model_verify.json)
- [冒烟结果](../../artifacts/server_mvp/smoke_results.json)
- [环境冻结](../../artifacts/server_mvp/environment.freeze.txt)
- [资源记录](../../artifacts/server_mvp/resource_after.txt)
- [SHA256清单](../../artifacts/server_mvp/artifact.sha256)
