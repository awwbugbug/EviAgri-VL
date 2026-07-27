# Task14B 独立 Oracle 复现结果

- 数据门禁：全新 IP102 64 train / 32 test，16 类严格 4+2；class 17 恰好剩 6 个 component；全新 PlantSeg 32；family overlap=0。
- attempt_01：协议和 128 份特征完成并通过 SHA；评估因残留 `range(144)` 在算指标前 fail-closed，科学数据未损坏。
- attempt_02：仅修复行数断言，复用并核对 attempt_01 特征；`scientific_protocol_changed=false`。
- 主结果 GR-GG：Accuracy -0.03125；Macro-F1 -0.02083；true-prob +0.00790，CI [0.00636,0.00979]。
- 可靠性：PlantSeg confidence +0.00269，CI [0.00133,0.00426]；AUROC +0.00781，CI 跨 0。
- 决策：`H2_MEAN_REGION_RETIRED`；两批均“更自信但不更正确，并放大病斑伪证据”。不授权 selector/大训练/Task8。
- 证据：`artifacts/2026-07-27_task14/14B_independent_token_oracle_result`；feature rows SHA `06d9f43d...`。
- 下一步：先冻结 Task15A 配对分辨率法医，区分 crop 放大/重编码与 within-frame token selection；仅诊断，不算独立确认。
