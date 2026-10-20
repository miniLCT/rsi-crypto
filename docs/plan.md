# 主任务计划

> 任务以 [`docs/spec.md`](spec.md) 的需求 ID 为验收依据。
> 任务格式：`- [x|~| ] T<里程碑>.<序号> 描述 — 需求 ID — 完成标准（DoD）`，`[x]` 完成、`[~]` 进行中、`[ ]` 未开始。
> 每完成一个任务：更新本文件的勾选状态、spec 中对应需求的 `status`，并在 commit message 中引用任务号。

## 里程碑总览

| 里程碑 | 目标 | 依赖 | 状态 |
|---|---|---|---|
| M0 | 项目骨架、配置分层 | — | 完成 |
| M1 | 数据层：同步、入库、质量、Panel | M0 | 进行中（全量下载中） |
| M2 | 币池、特征库与因子契约（含截面性质） | M1 | 进行中 |
| M3 | 标签、切分、模型 | M2 | 基本完成，待真实数据验证 |
| M4 | 策略、风控、成本、统一引擎 | M3 | 基本完成，缺硬止损与两级回撤 |
| M5 | Harness：spec、runner、ledger、门槛、搜索、筛选、agent 协议 | M4 | 进行中 |
| M6 | Baseline 与首轮 RSI 迭代 | M1–M5 | 未开始 |
| M7 | 实盘：paper → testnet | M5 | 未开始 |
| M8 | 线上闭环：parity、监控、重训 | M7 | 未开始 |
| M9 | 主网小资金上线 | M6 通过 holdout，M7 跑满 14 天，M8 | 未开始（需人工批准与 API 密钥） |

---

## M0 骨架
- [x] T0.1 `pyproject.toml`、uv 虚拟环境、`.gitignore` — CONV-04 — `uv pip install -e ".[dev]"` 成功
- [x] T0.2 `configs/data.yaml`、`configs/harness.yaml` — CONV-04、DEC-08 — 能被 `config.py` 加载
- [ ] T0.3 ledger 记录 `harness.yaml` 内容哈希，version 不变但内容变化时报错 — CONV-04、HRN-03 — 有测试

## M1 数据层
- [x] T1.1 S3 列表、币种过滤、带校验的并发下载 — DATA-01 — 小规模同步 `ok` 334 个文件
- [~] T1.2 全量同步与失败重试（最多 3 轮） — DATA-01 — `logs/sync_retry.log` 最后一轮 `fail` 为 0
- [x] T1.3 zip 入库为 parquet — DATA-02 — 实现完成
- [ ] T1.4 全量入库后跑质量报告，确定质量门槛 — DATA-03 — 报告产出，阈值写入 `data.yaml`
- [x] T1.5 Panel 构建与缓存 — DATA-04
- [ ] T1.6 资金费率当月 REST 补齐 — DATA-05 — 与次月归档逐条一致
- [ ] T1.7（可选）5m 辅助数据 — DATA-06
- [ ] T1.8（可选）全市场币种 Panel — DATA-07

## M2 币池与特征
- [ ] T2.1 币池改为默认 `size=100`，新增 `keep_rank=120`、`min_history_days=30` — UNI-01、DEC-02 — 测试覆盖日切更新、缓冲区、新币
- [x] T2.2 特征注册与声明式 lookback — FEAT-01
- [x] T2.3 因果性与 lookback 契约检查 — FEAT-02 — `rsi feat check` 全部通过
- [ ] T2.4 截面性质硬门槛：价格单位不变性、成交量单位、广播类特征禁止截面变换 — FEAT-03 — 属性测试纳入 `pytest`
- [ ] T2.5 截面变换补齐：覆盖率不足置 NaN、MAD 缩尾、可选中性化 — FEAT-04
- [ ] T2.6 按专项规则审计 48 个因子（波动率调整、资金费按小时、相对自身历史） — FEAT-05 — 审计表写入 `docs/factors.md`
- [ ] T2.7 `rsi screen` 输出完整报告项 — FEAT-03、HRN-06

## M3 标签、切分、模型
- [x] T3.1 标签 — LBL-01
- [x] T3.2 walk-forward + purge/embargo — SPLIT-01
- [x] T3.3 holdout 物理隔离与次数预算 — SPLIT-02
- [x] T3.4 lgbm / ridge / factor 模型 — MDL-01
- [x] T3.5 模型产物与 manifest 校验 — MDL-02
- [ ] T3.6 用真实数据验证可复现性（同 spec 重跑指标一致） — HRN-02

## M4 策略、风控、引擎
- [x] T4.1 `xs_longshort` 与 `ts_directional` — STR-01、STR-02
- [ ] T4.2 打分平滑、置信门槛、beta 中性 — STR-03 — 默认关闭时结果不变
- [x] T4.3 下单前约束、软止损、冷却 — RISK-01、RISK-02
- [ ] T4.4 硬止损（4–5 倍持仓周期波动，高低价模拟）、移动止盈（实验参数）、持仓时间上限 — RISK-02、DEC-05、DEC-09
- [ ] T4.5 按盈亏的熔断改为默认关闭，保留为实验参数 — RISK-03、DEC-07
- [ ] T4.7 杠杆与保证金：`max_gross` / `exchange_leverage` 上限 10 的校验、保证金占用约束、回测强平模拟 — RISK-05、DEC-06 — 测试覆盖 1x/5x/10x 与强平场景
- [ ] T4.8 门槛在固定波动率目标下评估 — RISK-05、HRN-04
- [ ] T4.9 灾难阈值标定（分块自助抽样）写入 manifest — RISK-06、DEC-07
- [x] T4.6 统一步进引擎、成本模型、资金费、下架处理 — ENG-01..03

## M5 Harness
- [x] T5.1 实验 spec、runner、ledger、门槛、搜索、holdout、导出 — HRN-01..05、SPLIT-02、MDL-02
- [ ] T5.2 `AGENTS.md` 研究 agent 协议 — HRN-07
- [ ] T5.3 `docs/rsi_log.md` 模板及 run 结束时自动追加摘要 — HRN-08
- [ ] T5.4 baseline 实验 spec：截面多空（前 100）、BTC/ETH 单币时序、纯因子规则策略 — HRN-01

## M6 Baseline 与首轮 RSI
- [ ] T6.1 全量 Panel + 特征缓存，记录耗时（目标：单次 `rsi run` ≤ 10 分钟） — HRN-02
- [ ] T6.2 跑 3 个 baseline，结果写入 ledger 与 `rsi_log.md`
- [ ] T6.3 `rsi search` 首轮（每个 baseline 20–30 次试验）
- [ ] T6.4 研究 agent 按 AGENTS.md 迭代至少 5 轮（新因子或策略结构）
- [ ] T6.5 冠军跑 holdout；通过后 `rsi export`

## M7 实盘：paper → testnet
- [ ] T7.1 REST 行情 feed 与 Panel 窗口 — LIVE-02
- [ ] T7.2 Broker 接口与 `PaperBroker` — LIVE-04
- [ ] T7.3 实盘主循环与触发时间线 — LIVE-01
- [ ] T7.4 下单逻辑（死区、取整、幂等、对账、异常处理） — LIVE-03
- [ ] T7.5 状态持久化与重启恢复 — LIVE-05
- [ ] T7.6 决策日志 — LIVE-06
- [ ] T7.7 运维风控 — RISK-04
- [ ] T7.8 `BinanceBroker`（testnet），含交易所端硬止损 — LIVE-04、RISK-02 — testnet 上完成开仓、止损、平仓全流程
- [ ] T7.10 实盘杠杆：按杠杆档位取实际杠杆、每根 bar 检查保证金率并减仓、读取 manifest 中的灾难阈值 — RISK-05、RISK-06
- [ ] T7.9 paper 连续运行 14 天

## M8 线上闭环
- [ ] T8.1 `rsi parity` — MON-01 — paper 日志重放全部一致
- [ ] T8.2 成本与成交监控 — MON-02
- [ ] T8.3 特征漂移 PSI — MON-03
- [ ] T8.4 定期重训与冠军轮换 — MON-04
- [ ] T8.5 报警通道 — MON-05

## M9 主网
- [ ] T9.1 主网配置与小资金上线（人工批准，API 密钥放 `secrets/`）
- [ ] T9.2 逐级放大规则（基于 MON-02 的成本偏差和实盘 Sharpe）

---

## 待确认事项
- 无。DEC-06（杠杆 1–10 倍可配）、DEC-07（灾难阈值按分布标定，机械熔断默认关闭）、DEC-09（宽硬止损）已确认。

## 近期执行顺序
1. T1.2 → T1.4（数据下完、入库、质量报告）
2. T2.1、T2.4、T2.5（币池对齐 DEC-02，截面性质硬门槛）
3. T5.4 → T6.1 → T6.2（baseline 跑通）
4. T4.4、T4.5、T4.7、T4.8、T4.9（止损、熔断、杠杆与保证金、灾难阈值）
5. T5.2、T5.3 → T6.3、T6.4（进入 RSI 迭代）
6. M7 → M8
