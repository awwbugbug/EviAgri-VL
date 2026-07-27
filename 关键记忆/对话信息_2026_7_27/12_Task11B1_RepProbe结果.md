# Task11B.1 RepProbe 结果

- 协议：`protocol_v1`；attempt_01完成1290次冻结前向，attempt_02仅补齐CPU审计报告；无训练、无optimizer、未读Task8。
- 单变量：同一Qwen2.5-VL-3B、320个IP102正样本、640个synthetic stress、40个PlantDoc null、290个PlantSeg damage-null；只比较vision-pooled与`hidden_states[27]` query-token。
- 完整性：代码/输入/输出SHA全部通过；同一split、LogisticRegression、3 seeds、温度与阈值规则；三seed结果逐项一致。
- V→Q：Accuracy 0.825→0.325；Macro-F1 0.8094→0.2790；coverage 0.8125→0.375。
- null V→Q：blank FPR 0→1.0；blur 0.10→0.6875；shuffle 0.0708→0.65；PlantDoc 0→0.725；PlantSeg 0.0828→0.3897。
- 配对证据：Macro-F1差-0.5304，95%CI[-0.6401,-0.4182]；PlantSeg FPR差+0.3069，95%CI[+0.2448,+0.3724]。
- 校准法医：Q的temperature=0.0183157（达到`exp(-4)`搜索下界），threshold=0.15；说明表示的类别/拒答几何严重失配，不是门槛附近波动。
- 决策：FAIL；不授权evidence head或更大训练。Task11B.0只证明Q表示对像素敏感，Task11B.1证明“敏感”不等于“可判别/可靠”。
- 下一单变量：Local-Crop/Mask微实验；保持vision-pooled与全部评测不变，只把IP102 bbox/PlantSeg mask产生的局部像素crop作为输入。先冻结设计和极小可行性，不叠加query-token/VEP/VIB/动态模块。
- 产物：`artifacts/2026-07-27_task11/11B1_repprobe_result/`。
