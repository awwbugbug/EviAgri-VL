# MVP 子集上传验收

_日期：2026-07-12 | 状态：通过_

- 全量 SCP 实测约0.3MB/s；Age预计近2小时，因此改为先传可复现MVP子集
- 仅取val；固定种子 `eviagri-mvp-20260712-v1`；按类别分层并以路径哈希排序
- Age每个非空联合类最多3图：602图；IP102每类5图：510图；共1,112图/41,364,983图像字节
- 用途仅限流程验证与zero-shot probe，不作为论文最终成绩
- 服务器路径：`/root/autodl-tmp/EviAgriDiag/datasets/mvp/mvp_subset_v1`
- 验收：归档SHA256一致、tar路径安全、1,112张逐文件大小与SHA256一致，`MVP_DATA_OK`
- 第三方HF分类镜像下载已停止；保留部分文件，未混入MVP
- 全量Age/IP102后续采用分片并行或夜间上传；IP102 bbox仍待获取
