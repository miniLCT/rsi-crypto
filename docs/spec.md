# rsi-crypto 产品与技术规格（PRD / Spec）

> 本文件是系统行为的唯一规格来源。代码、配置、测试与本文件冲突时，以本文件为准，并修正冲突方。
> 面向读者：开发者、研究 agent（LLM）、harness 自身。
>
> **阅读约定（机器可读）**
> - 每条需求以 `### <ID> <标题>` 开头，ID 稳定、不复用。分类前缀：`CONV` 约定、`DEC` 产品决策、`DATA`、`UNI`、`FEAT`、`LBL`、`SPLIT`、`MDL`、`STR`、`RISK`、`ENG`、`EVAL`、`HRN`、`LIVE`、`MON`。
> - 需求体使用固定字段：`status`、`module`、`config`、`input`、`output`、`acceptance`。
> - `status` 取值：`done`（已实现且有测试）、`partial`（部分实现，缺口写在 `gap`）、`todo`（未实现）。
> - `acceptance` 中每一条都必须能用测试或命令验证；任务计划见 [`docs/plan.md`](plan.md)，任务以需求 ID 引用本文件。

---

## 1. 产品定义

### 1.1 目标
在币安 USDT 本位永续合约上，提供一站式、可扩展的量化系统：
1. 数据：可复现地获取、校验、存储全量历史行情（含已下架合约）。
2. 研究：因子、标签、模型、策略、风控在统一 harness 下评估，结果全部扣除真实成本，样本外、可比较、可追溯。
3. 迭代（RSI，Recursive Self-Improvement）：自动搜索与研究 agent 持续提出改进，harness 统一评估、按门槛晋级，评估规则对迭代者不可修改。
4. 交易：同一套代码驱动回测、模拟盘、测试网、实盘；支持做多做空、单币与截面两种模式。
5. 闭环：线上持续核对训推一致性、成本偏差、特征漂移，驱动重训与冠军轮换。

### 1.2 非目标（当前版本不做）
做市/挂单策略、盘口与逐笔数据、币本位合约、跨交易所套利、高频（< 15m）决策。

### 1.3 成功标准
- 研究侧：存在通过全部 `gates` 且通过 holdout 检验的策略。
- 一致性：`rsi parity` 对实盘日志重放，特征/打分/目标仓位逐 bar 一致（容差见 MON-01）。
- 交易侧：模拟盘连续运行 ≥ 14 天无人工干预；实际滑点与成本模型偏差 ≤ 30%。

---

## 2. 全局约定

### CONV-01 时间
- status: done
- module: `rsi/utils/timeutil.py`
- 规则：
  - 内部时间一律为 UTC epoch 毫秒（int64）。展示层可转换时区，计算层禁止使用本地时区。
  - bar 以**开盘时间**标识；bar `t` 的数据在 `open_time[t] + interval` 之后才可用。
  - 在 bar `t` 收盘时决策，以 `open[t+1]` 成交。回测与实盘都遵守此约定。
  - UTC 日切（00:00）为 universe 更新、日内亏损重置的唯一边界。

### CONV-02 数组与缺失值
- status: done
- 规则：
  - 所有横截面数据都是 `float64[T, S]` 矩阵，行是时间，列是币种，列顺序与 `Panel.symbols` 一致。
  - 缺失值统一为 `NaN`。禁止用 0 填充表示"未知"。不在 universe 内的位置，打分与权重为 `NaN`/0。
  - 布尔掩码为 `bool[T, S]`。

### CONV-03 单位
- status: done
- 规则：金额一律为 USDT；收益为对数收益或小数收益（字段名注明）；成本用 bps（1bp = 0.01%）；资金费率在特征层统一换算为**每小时**费率。

### CONV-04 配置分层与不可变性
- status: partial
- module: `configs/`
- 规则：
  - `configs/data.yaml`：数据契约。
  - `configs/harness.yaml`：评估契约（时间切分、成本、门槛）。**研究迭代者不可修改**；修改时必须提升 `version`，ledger 按 version 隔离比较。
  - `experiments/*.yaml`：实验 spec，迭代者可修改。
  - `configs/live/*.yaml`：实盘运行配置。
  - 密钥放在 `secrets/`（git 忽略），禁止写入配置文件。
- gap: ledger 尚未记录 `harness.yaml` 内容哈希；运行时未校验"version 不变但内容变了"。

---

## 3. 产品决策（已确认 / 默认值）

| ID | 决策 | 状态 |
|---|---|---|
| DEC-01 | 决策频率 30m；5m/1m 仅作为可选辅助数据（K 线内部特征、止损成交模拟） | 已确认 |
| DEC-02 | 交易币池：point-in-time 前 **100** 名（30 天日均成交额），已持有的币排名在 **120** 以内可继续持有，上线满 30 天 | 已确认 |
| DEC-03 | "算特征的币"与"交易的币"分离：全市场指标可用全部币计算，交易只在 DEC-02 币池内 | 已确认 |
| DEC-04 | v1 下单全部用市价单（吃单），与成本模型一致 | 默认 |
| DEC-05 | 默认不设固定止盈；移动止盈作为实验参数 | 默认 |
| DEC-06 | 总杠杆 1 倍；交易所端硬止损 3 倍持仓周期波动 | 默认（待确认） |
| DEC-07 | 当日亏损 3% 全平停到次日；回撤 15% 杠杆减半，25% 停机等待人工 | 默认（待确认） |
| DEC-08 | 历史数据：2020-01 起；研究期 2020-07-01 至 2025-10-01；holdout 2025-10-01 至 2026-10-01 | 已确认 |

---

## 4. 数据流总览

```
data.binance.vision ─► raw zip ─► lake parquet ─► Panel[T,S]
                                                  │
  Universe mask ◄─────────────────────────────────┤
  Features[T,S] ◄─────────────────────────────────┤
  Labels[T,S]   ◄─────────────────────────────────┘
        │
        ▼
  walk-forward 训练 ─► OOS score[T,S] ─► TargetBuilder ─► Risk ─► Portfolio.step ─► 指标 ─► gates ─► ledger
                                                                     ▲
  实盘：REST bar ─► Panel 窗口 ─► 同一套 Features/Model/TargetBuilder/Risk ─┘ ─► Broker(paper/testnet/mainnet)
```

---

## 5. 接口契约（I/O 类型）

| 类型 | 形状 / 结构 | 生产者 | 消费者 |
|---|---|---|---|
| `Panel` | `times: int64[T]`，`symbols: list[S]`，`data[field]: float64[T,S]`；字段见 DATA-04 | `build_panel`、实盘 feed | 所有下游 |
| `UniverseMask` | `bool[T,S]` | `compute_universe` | 特征截面变换、标签、策略 |
| `FeatureCols` | `dict[name, float64[T,S]]`；截面排名列名前缀 `r:` | `featureset.build_batch` / `build_last` | 模型 |
| `Label` | `float64[T,S]`，未来未知处为 NaN | `make_label` | 训练 |
| `Fold` | `train=[i0,i1)`、`test=[j0,j1)`，二者间隔 ≥ label span + embargo | `walk_forward` | runner |
| `Score` | `float64[T,S]`，仅样本外区间有值 | `oos_scores` / 实盘模型 | `TargetBuilder` |
| `TargetWeights` | `float64[S]`（每个 bar），`notional / equity`，正多负空 | `TargetBuilder.step` | `Portfolio` |
| `TradeRecord` | 时间、币、方向、数量、价格、成本 bps、原因（rebalance / stop / delist / halt …） | `Portfolio.fill` | 指标、报告、监控 |
| `RunRecord` | run_id、spec、spec_hash、git sha、数据指纹、harness version、metrics、gates、状态、父 run | runner | ledger |
| `ModelArtifact` | `model.*` + `manifest.json`（特征列表与代码哈希、lookback、universe/strategy/risk 配置、数据指纹、git sha） | `rsi export` | 实盘、parity |

---

## 6. 需求

### 6.1 数据

### DATA-01 历史数据同步
- status: done
- module: `rsi/data/vision.py`，命令 `rsi data sync`
- config: `configs/data.yaml`
- input: S3 列表（data.binance.vision）、`exchangeInfo`
- output: `data/raw/...`（与 bucket 路径一致的 zip 文件），每个成功文件旁边一个 `.ok` 标记；`data/meta/symbols.json`、`exchange_info.json`
- acceptance:
  - 币种来自 S3 列表，包含已下架合约（无幸存者偏差）；按 `symbol_filter` 剔除交割合约、稳定币对、指数类合约、非 COIN 标的。
  - 每个文件通过 SHA256 校验后才写入；重复执行是幂等的，只补缺失或失败的文件。
  - 月度文件覆盖到最近完整月；当月用日度文件补齐（资金费率除外，见 DATA-05）。
  - 结果统计包含 `ok / skip / missing / fail`；`fail` 可通过重跑清零。

### DATA-02 数据湖入库
- status: done
- module: `rsi/data/lake.py`，命令 `rsi data ingest`
- input: `data/raw/`
- output:
  - `data/lake/um/klines/30m/{SYM}.parquet`：`open_time, open, high, low, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume`
  - `data/lake/um/premium/30m/{SYM}.parquet`：`open_time, premium_open/high/low/close`
  - `data/lake/um/funding/{SYM}.parquet`：`funding_time, funding_rate, interval_hours`
- acceptance: 兼容有表头和无表头的 CSV；兼容微秒时间戳；`open_time` 唯一且升序。

### DATA-03 数据质量报告
- status: partial
- module: `lake.quality_report`，命令 `rsi data quality`
- output: 每个币的 bar 数、首尾时间、缺失 bar 数、OHLC 异常数、零成交占比、成交额中位数
- gap: 缺少质量门槛（例如缺失率 > 5% 的币在缺失段不进入 universe）和对应测试。

### DATA-04 Panel
- status: done
- module: `rsi/data/panel.py`，命令 `rsi data panel`
- input: lake、`start`、`end`、币种列表（默认 `liquid_symbols(top_k=150)`）
- output: `Panel`，字段 `open, high, low, close, volume, quote_volume, trades, taker_buy_quote_volume, premium, funding_rate, funding_interval_h`；缓存于 `cache/panel/{hash}/`
- acceptance:
  - 时间网格连续（步长 = interval），缺失 bar 为 NaN。
  - 资金费率放在 `open_time == floor(funding_time)` 的 bar 上，其余为 NaN。
  - 历史与实盘都通过 `frames_to_panel` 构建同一类型。
  - `top_k ≥ 120`，以覆盖 DEC-02 的持有缓冲区。

### DATA-05 资金费率尾部补齐
- status: todo
- 说明: 资金费率没有日度归档，当月数据通过 REST `/fapi/v1/fundingRate` 补齐。
- acceptance: 补齐后与次月发布的月度归档逐条一致。

### DATA-06 5m 辅助数据（可选）
- status: todo
- 说明: 为前约 150 个流动性币下载 5m K 线，用于 K 线内部特征和止损成交模拟（DEC-01）。
- acceptance: 5m 聚合到 30m 后与 30m 归档的 OHLCV 完全一致。

### DATA-07 全市场币种 Panel（可选）
- status: todo
- 说明: 为 DEC-03 的全市场指标加载全部币种（不做 top_k 预筛）。
- acceptance: 全市场特征只依赖当时已上线的币；交易币池不受影响。

### 6.2 币池

### UNI-01 Point-in-time 流动性币池
- status: partial
- module: `rsi/universe.py`
- config: 实验 spec 中的 `universe`
- input: `Panel`、`UniverseSpec`
- output: `UniverseMask bool[T,S]`
- acceptance:
  - 每个 UTC 日切，按前一日结束时的 `adv_days`（30 天）日均成交额排名更新，日内不变。
  - 只使用决策时点之前的数据。
  - 入池条件：排名 ≤ `size`（100），且上线满 `min_history_days`（30 天）。
  - 持有缓冲：已持有的币排名 ≤ `keep_rank`（120）时仍允许持有，不能新开或加仓。
  - 早期币种不足 100 个时，取全部符合条件的币。
- gap: 默认 `size=50`，需改为 100；缺少 `keep_rank`、`min_history_days` 字段及测试。

### UNI-02 固定币池 / 单币模式
- status: done
- 说明: `kind: fixed, symbols: [BTCUSDT, ...]`
- acceptance: 掩码只在指定币有价格时为 True。

### 6.3 特征

### FEAT-01 注册与声明式 lookback
- status: done
- module: `rsi/features/registry.py`、`library.py`、`custom/`
- 规则: 每个特征通过 `@register(name, lookback, group)` 注册；实现只能使用有限窗口的运算（rolling、shift），禁止使用 EWM 等无限记忆运算、全样本统计量和负向 shift。
- output: `Feature`（含源码哈希 `src_hash`，用于缓存和 manifest 校验）

### FEAT-02 因果性与 lookback 契约
- status: done
- module: `rsi/features/contract.py`，命令 `rsi feat check`
- acceptance:
  - 无未来函数：在随机截断点 c，`f(data[:c])` 的末段等于 `f(data)` 的对应段。
  - lookback 足够：只用最近 `lookback` 根 bar 计算出的末行，等于用全历史计算出的同一行。
  - 结果按 `src_hash` 缓存；新特征或源码有改动时，runner 自动复检，不通过就拒绝运行。

### FEAT-03 截面数学性质检查
- status: todo
- module: `rsi/features/contract.py`（扩展），`rsi screen`
- acceptance（硬门槛，不通过不能用于截面）：
  - **价格单位不变性**：价格 ×1000、数量 ÷1000 后，特征值不变（模拟 PEPE 和 1000PEPE）。
  - **成交量单位**：只能使用 `quote_volume` 或比值，不能使用 base volume 的水平值。
  - 广播类特征（同一时点所有币取值相同，如 `calendar`、`market_ctx`）禁止做截面变换。
- acceptance（报告项，`rsi screen` 输出）：覆盖率 ≥ 80%；截面并列占比 ≤ 30%；与波动率、规模的截面相关性绝对值 ≤ 0.7（否则标记为伪装因子）；排名自相关；多周期 Rank IC/ICIR（Newey-West t 值）；分年 IC 方向一致率；五分组单调性；多空腿拆解；扣成本后的多空价差；与已入库因子的最大相关性 ≤ 0.8。

### FEAT-04 截面变换
- status: partial
- module: `rsi/features/engine.py`
- 规则:
  - 默认 `rank`：只在 universe 内对有值的币排名，映射到 [-0.5, 0.5]，并列取平均名次。
  - 可选 `zscore`：用 MAD 缩尾到 ±3。
  - 某时点有值的币少于 universe 的 50% 时，该时点整列置 NaN。
  - 可选中性化：每个时点对"波动率 + 对数规模"做截面回归，取残差。
- gap: 覆盖率不足时置 NaN 的规则、MAD 缩尾、中性化尚未实现。

### FEAT-05 首批因子库
- status: partial
- module: `rsi/features/library.py`
- 内容（当前 48 个）：momentum 10、volatility 10、flow 9、carry 6、candle 4、calendar 4、market 3、market_ctx 2。
- acceptance: 每个因子都通过 FEAT-02 和 FEAT-03。专项规则：
  - 收益和均线偏离类因子用于截面时必须按波动率调整。
  - 资金费率按 `funding_interval_h` 换算为每小时费率。
  - 主动买入占比、单笔成交规模、Amihud 必须先相对自身历史归一化，再做截面。
  - 成交额水平只作为规模、中性化或容量变量，不作为信号。
- gap: 需按上述规则逐个审计（FEAT-03 落地后自动化）。

### FEAT-06 在线特征计算
- status: done
- module: `featureset.build_last`
- acceptance: 对最近 `lookback` 根 bar 的窗口调用，输出与批量计算在同一时点的值一致（相对误差 ≤ 1e-9）。

### 6.4 标签与切分

### LBL-01 标签
- status: done
- module: `rsi/labels.py`
- 定义: `y[t] = log(open[t+1+h] / open[t+1])`；可选截面去均值、除以 `rv × √h` 做波动率归一化、按 `clip` 个标准差截尾。`span = h + 1`。
- acceptance: 只有掩码内的币有标签；标签在 `t + span` 之后才可知，训练时自动剔除。

### SPLIT-01 Walk-forward 切分
- status: done
- module: `rsi/splits.py`
- config: `harness.yaml → walk_forward`
- acceptance:
  - 第一个测试窗口从 `first_test` 开始，每窗 `test_months` 个月，滚动推进到 `research_end`。
  - 训练窗口为测试开始之前 `train_months` 个月（null 表示扩展窗口）。
  - 训练结束与测试开始之间的间隔 ≥ `label span + embargo_bars`（purge + embargo）。
  - 测试窗口互不重叠，拼接后覆盖全部 OOS 区间。

### SPLIT-02 Holdout 隔离
- status: done
- module: `harness/runner.research_panel`、`harness/release.holdout`，命令 `rsi holdout <run>`
- acceptance:
  - 研究运行加载的 Panel 在 `research_end` 处截断，物理上看不到 holdout 数据。
  - 每次查看 holdout 都写入 ledger；30 天内超过 `max_looks_per_30d` 次时拒绝（`--force` 需留痕）。

### 6.5 模型

### MDL-01 模型
- status: done
- module: `rsi/models/core.py`
- kind: `lgbm`（默认）、`ridge`、`factor`（给定权重的线性组合，不训练）
- acceptance: 每折使用训练窗口尾部 `valid_frac` 做 early stopping；`train_stride` 控制训练采样步长；固定 `seed` 时结果可复现。

### MDL-02 模型产物与校验
- status: done
- module: `harness/release.export`、`verify_manifest`，命令 `rsi export <run>`
- acceptance: manifest 包含特征列表与每个特征的 `src_hash`、lookback、universe/strategy/risk 配置、数据指纹、git sha；加载时任意 `src_hash` 与当前代码不一致就拒绝启动。

### 6.6 策略

### STR-01 截面多空 `xs_longshort`
- status: done
- module: `rsi/strategy.py`
- input: `score[S]`、`mask[S]`、`vol[S]`（每个调仓 bar）
- output: `TargetWeights[S]`
- 规则: 每 `rebalance_bars` 根 bar 调仓一次（按绝对时钟）；做多排名前 `long_frac`，做空排名后 `short_frac`；已持有的头寸在越过入场线 `exit_buffer` 之前继续保留（滞后带）；加权方式 `equal | score | inv_vol`；可用币数少于 `min_names` 时空仓；总敞口为 `gross`。

### STR-02 单币时序 `ts_directional`
- status: done
- 规则: 标准化打分 > `entry_z` 做多，< −`entry_z` 做空；回到 ±`exit_z` 以内平仓；每个币权重 = `gross / max_names`。

### STR-03 信号增强
- status: todo
- 内容: 打分平滑（最近 k 次的均值）；置信门槛 `min_abs_score`；beta 中性的权重调整。
- acceptance: 均为实验参数，默认关闭时结果与 STR-01 完全一致。

### 6.7 风控（回测与实盘共用同一实现）

### RISK-01 下单前约束
- status: done
- module: `rsi/engine/portfolio.py`（`RiskSpec`）
- 规则: `max_gross`、`max_weight`（单币）、`max_net`；`adv_cap`（单币名义金额 ≤ ADV × 比例）；`min_trade_frac` 死区（变化量小于目标的一定比例时不调仓）；`min_notional_usd`。

### RISK-02 持仓级止损止盈
- status: partial
- 规则:
  - 软止损：收盘浮亏超过 `stop_loss_vol` 倍日波动时，下一根 bar 开盘平仓。（已实现）
  - 止损后冷却 `cooldown_bars` 根 bar，冷却期内同币不开仓。（已实现）
  - 硬止损：开仓后在交易所挂 STOP_MARKET（reduceOnly）；回测用 bar 的 high/low 判断是否触发，成交价取止损价与开盘价中较差者，再加滑点。（todo）
  - 移动止盈（可选）：浮盈超过 x 倍波动后启动，回撤 y 倍波动平仓；同一 bar 内止损与止盈都被触及时，先按止损处理。（todo）
  - 持仓时间上限 `max_hold_bars`。（todo）
- gap: 当前默认 `stop_loss_vol=4.0`，与 DEC-06 不一致，待确认后统一。

### RISK-03 组合级熔断
- status: partial
- 规则: 波动率目标（`vol_target`，按 `vol_lookback` 实际波动缩放，限制在 `[vol_scale_min, vol_scale_max]`）；当日亏损达到 `daily_loss_halt` 全平，停到下一个 UTC 日；回撤达到 `dd_halt` 全平，停 `dd_halt_bars` 根 bar。
- gap: DEC-07 的两级回撤（15% 降杠杆、25% 停机）尚未实现；当前默认 `daily_loss_halt=0.06`、`dd_halt=0.25`。

### RISK-04 运维风控（实盘）
- status: todo
- 规则: universe 内数据覆盖率 < 98% 时本 bar 不开新仓；数据连续 2 根 bar 过期时冻结开仓，保留交易所端止损单；币的状态不再是 TRADING（预告下架）时平仓并拉黑；下单拒绝率超过阈值时熔断并报警。

### 6.8 执行引擎与成本

### ENG-01 统一步进引擎
- status: done
- module: `rsi/engine/portfolio.py`、`backtest.py`
- 每根 bar 的流程: 盯市（按 `open[t+1]`）→ 结算资金费 → 止损检查 → 若为调仓 bar：`TargetBuilder` → 风控 → 目标数量 → 成交（加成本）→ 记录。
- acceptance: 回测与实盘调用同一个 `Portfolio`，差别只在成交来源（模拟 / broker）；`state_dict` / `load_state` 可以无损恢复。

### ENG-02 成本模型
- status: done
- config: `harness.yaml → costs`
- 公式:
  ```
  cost_bps = taker_fee_bps
           + clip(spread_coef_bps / sqrt(ADV_usd / 1e6), min_half_spread_bps, max_half_spread_bps)
           + impact_eta * daily_vol * 1e4 * sqrt(|notional| / ADV_usd)
  ```
  资金费：持仓在结算时点支付 `qty × price × funding_rate`（做多为正费率时付费）。下架：按最后价格强制平仓，并额外扣 `delist_penalty_bps`。

### ENG-03 回测输出
- status: done
- output: `runs/{run_id}/` 下的 `equity.parquet`（逐 bar 权益、多空贡献、成本、资金费）、`trades.parquet`、`metrics.json`、`report.md`

### 6.9 评估

### EVAL-01 指标定义
- status: done
- module: `rsi/eval/metrics.py`
- 定义（`r` 为逐 bar 收益，`P = bars_per_year = 17520`）：
  ```
  sharpe      = mean(r) / std(r) * sqrt(P)
  sortino     = mean(r) / std(r[r<0]) * sqrt(P)
  max_drawdown= max_t (1 - E_t / max_{s<=t} E_s)
  turnover    = 0.5 * sum_t sum_s |Δw_{t,s}| / years        (单边，年化)
  psr(SR*)    = Φ( (SR - SR*) * sqrt(n-1) / sqrt(1 - γ3*SR + (γ4-1)/4 * SR^2) )   (日收益)
  dsr         = psr(SR0)，SR0 = sqrt(Var[SR_trials]) * ((1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(N·e)))，γ≈0.5772
  ```
  其中 N 为 ledger 中同一 harness version 下的试验次数。
- 还需输出：年化收益、Calmar、胜率、平均持仓时长、多空各自贡献、成本占比、交易笔数、逐折 Sharpe、正收益折占比。

### 6.10 Harness（RSI 核心）

### HRN-01 实验 spec
- status: done
- module: `rsi/harness/spec.py`
- 结构（`experiments/*.yaml`）：
  ```yaml
  name: str                 # 唯一
  hypothesis: str           # 必填，一句话说明改了什么、为什么
  parent: run_id | null     # 从哪个 run 演化而来
  universe: UniverseSpec    # UNI-01/02
  features: {include: [selector...], transform: raw|rank|both}
  label: LabelSpec          # LBL-01
  model: ModelSpec          # MDL-01
  strategy: StrategySpec    # STR-01/02
  risk: RiskSpec            # RISK-01..03
  test_months: int | null   # 仅用于加速试验；门槛仍然适用
  search_space: {dotted.path: [候选值...]}   # HRN-05
  ```
- acceptance: `spec_hash` 只由内容决定（不含 name、hypothesis）；支持 `--set a.b=v` 覆盖参数。

### HRN-02 运行
- status: done
- module: `harness/runner.run`，命令 `rsi run <spec> [--set k=v] [--note ...]`
- 流程: 研究期 Panel → universe → 特征（契约检查 + 缓存）→ 标签 → walk-forward 训练 → OOS 打分 → 回测 → 指标 → 门槛 → 写入 ledger 和 artifact。
- acceptance: 相同 spec、代码和数据重跑，指标完全一致。

### HRN-03 Ledger
- status: partial
- module: `harness/ledger.py`（SQLite `runs/ledger.sqlite`），命令 `rsi ledger ls|show`
- 字段: run_id、created、name、spec、spec_hash、git sha、数据指纹、harness version、metrics、gates、status（`pass | fail | champion | discarded`）、parent、note。
- gap: 需加入 `harness.yaml` 内容哈希（CONV-04）。

### HRN-04 门槛与晋级
- status: done
- module: `harness/gates.py`
- config: `harness.yaml → gates`
- 规则:
  ```
  pass     : sharpe >= 1.0, dsr >= 0.90, max_dd <= 30%, 正收益折 >= 60%（≥20 天的折）, n_trades >= 200
  champion : pass，且 sharpe - 冠军 sharpe >= 0.15，且回撤不显著变差
  holdout  : holdout sharpe >= max(0.3, 0.4 * 研究期 sharpe)，且 holdout max_dd <= 30%
  paper    : 模拟盘 >= 14 天，实际成本与模型偏差 <= 30%，parity 通过
  live     : 小资金上线；逐级放大需人工批准
  ```

### HRN-05 自动搜索
- status: done
- module: `harness/search.py`，命令 `rsi search <spec> -n N`
- 规则: 在 `search_space` 内做随机采样和局部变异；每次试验都写入 ledger，并计入 DSR 的试验次数 N。

### HRN-06 因子筛选
- status: partial
- module: `harness/screen.py`，命令 `rsi screen <selectors>`
- gap: 补齐 FEAT-03 的完整报告项。

### HRN-07 研究 agent 协议
- status: todo
- 交付: 仓库根目录的 `AGENTS.md`
- 内容:
  - **允许修改**：`src/rsi/features/custom/`、`src/rsi/features/library.py`（只能新增）、`experiments/`、`docs/rsi_log.md`。
  - **禁止修改**：`configs/harness.yaml`、`splits.py`、`labels.py`、`eval/`、`engine/` 中的成本部分、`harness/gates.py`、`data/`。
  - **循环**：`rsi ledger ls` → 写假设 → 实现 → `rsi feat check` → `rsi run` → 在 `docs/rsi_log.md` 记录结论 → 保留或丢弃 → 提交（commit message 引用 run_id）。
  - holdout 只能由人触发，或在冠军晋级时由流程触发。

### HRN-08 研究日志
- status: todo
- 交付: `docs/rsi_log.md`，每次迭代一条：日期、run_id、父 run、假设、改动、关键指标、结论（保留 / 丢弃）及原因。

### 6.11 实盘

### LIVE-01 触发时间线
- status: todo
- module: `rsi/live/runner.py`，命令 `rsi live -c configs/live/<mode>.yaml`
- 每根 30m bar 的流程（T = UTC :00 / :30 收盘）：
  ```
  T+3s   REST 拉取刚收盘的 bar（确认 closeTime < now），缺失的重试；覆盖率不足时只做风控（RISK-04）
  T+10s  组装 Panel 窗口 → universe（仅日切更新）→ 特征 → 打分 → TargetBuilder → 风控
  T+15s  下单：先平仓或减仓（reduceOnly），后开仓或加仓
  T+30s  确认成交 → 挂或更新交易所端硬止损 → 对账 → 写决策日志
  截止   T+120s 仍未发出的开仓指令作废
  ```
  非调仓 bar 只做止损检查、持仓时间检查、熔断检查和对账。

### LIVE-02 行情 feed
- status: todo
- 规则: 启动时用 REST 回填 ≥ `max(feature lookback, universe lookback) + 余量` 根 bar（K 线、溢价指数、资金费率）；之后每根 bar 用 REST 增量拉取（与历史归档同源）；输出与 DATA-04 同类型的 Panel 窗口。

### LIVE-03 下单
- status: todo
- 规则:
  - 当前持仓以交易所 `positionRisk` 为准，本地状态只是缓存。
  - 交易量 = 目标 − 当前；死区与 RISK-01 一致；数量按 stepSize 向下取整，检查最小名义金额与 maxQty。
  - 单向持仓模式、全仓保证金；启动时设置每个币的杠杆。
  - `clientOrderId = {strategy}-{bar_ts}-{symbol}-{seq}`；超时先按 ID 查询订单，再决定是否重发。
  - 记录成交均价、手续费，以及相对 `open[t+1]` 的实际滑点。
  - 被拒单时调整数量重试一次，仍失败则拉黑该币并报警；部分成交在下根 bar 对账时补齐；交易所上出现未知持仓时报警（可配置为自动平仓）。

### LIVE-04 Broker
- status: todo
- module: `rsi/live/broker.py`
- 实现: `PaperBroker`（真实行情、按成本模型模拟成交，无需密钥）、`BinanceBroker`（`testnet` = demo-fapi，`mainnet`）；签名支持 HMAC 与 Ed25519。
- acceptance: 三种 broker 实现同一接口；paper 模式与同期回测重放的结果一致（MON-01）。

### LIVE-05 状态与恢复
- status: todo
- 规则: 每根 bar 结束时持久化 `Portfolio.state_dict()` 和运行状态到 `live_state/`；重启后先与交易所对账，再恢复运行。
- 单个持仓状态机：空仓 → 开仓中 → 持仓（已挂止损单）→ 平仓中 → 空仓；每次状态转换都写日志。

### LIVE-06 决策日志
- status: todo
- output: `logs/decisions/{date}.parquet`，每根 bar 每个币一行：特征、打分、掩码、目标权重、风控调整、订单、成交、成本。

### 6.12 线上闭环

### MON-01 训推一致性重放
- status: todo（CLI 入口已存在）
- 命令: `rsi parity <model_dir> --bars N`
- acceptance: 用历史数据重放决策日志对应的时段，特征相对误差 ≤ 1e-9、打分 ≤ 1e-9、目标权重完全一致；不一致项逐条列出。

### MON-02 成本与成交监控
- status: todo
- output: 实际滑点、手续费、资金费与 ENG-02 模型的偏差（按币和流动性分层）；偏差超过 30% 时报警，并作为更新成本模型的依据。

### MON-03 特征漂移
- status: todo
- output: 每个特征的 PSI（以近期 vs 训练期）；PSI > 0.25 时报警。

### MON-04 重训与冠军轮换
- status: todo
- 规则: 定期（默认每周）用冠军 spec 在最新数据上重训；新模型在最近窗口上不劣于现役模型且通过门槛后才能替换；每次替换写入 ledger。

### MON-05 报警
- status: todo
- 场景: 下单失败或拒单、止损触发、熔断、数据过期、parity 不一致、成本偏差、进程存活。

---

## 7. 目录结构

```
configs/            data.yaml · harness.yaml(锁定) · live/*.yaml
experiments/        实验 spec（agent 可修改）
src/rsi/
  data/             vision(下载) · lake(入库) · panel
  universe.py · labels.py · splits.py · strategy.py
  features/         registry · library · custom/ · engine · featureset · contract
  models/           core
  engine/           portfolio(风控+成交+成本) · backtest
  eval/             metrics
  harness/          spec · runner · ledger · gates · search · screen · release
  live/             runner · feed · broker · client · parity      (todo)
  cli.py
tests/
docs/               spec.md · plan.md · rsi_log.md
data/ cache/ runs/ live_state/ logs/ secrets/   (git 忽略)
```

## 8. 命令总览

| 命令 | 需求 |
|---|---|
| `rsi data sync / ingest / quality / panel` | DATA-01..04 |
| `rsi feat list / check` | FEAT-01..02 |
| `rsi screen <selectors>` | FEAT-03, HRN-06 |
| `rsi run <spec> [--set k=v]` | HRN-02 |
| `rsi search <spec> -n N` | HRN-05 |
| `rsi ledger ls / show <run>` | HRN-03 |
| `rsi holdout <run>` | SPLIT-02 |
| `rsi export <run>` | MDL-02 |
| `rsi live -c <config> [--once]` | LIVE-01..06 |
| `rsi parity <model_dir>` | MON-01 |
