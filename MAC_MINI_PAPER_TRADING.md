# Mac mini 纸面实盘部署说明

这个包用于每天收盘后生成 A 股策略纸面实盘信号，不会自动向券商下单。

## 1. 安装

在 Mac mini 上进入项目目录：

```bash
cd /path/to/quant
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config/paper_trading_config.example.json config/paper_trading_config.json
chmod +x scripts/run_paper_trading.sh
```

确认必要数据存在：

```text
data/processed/csi1000_fundamental_small_200.parquet
data/processed/csi1000_industry_map_200.csv
```

如果这两个文件没有同步到 Mac mini，先从主机器复制过去。当前脚本默认使用本地缓存数据生成信号。

## 2. 手动跑一次

```bash
scripts/run_paper_trading.sh
```

输出目录：

```text
paper_trading/YYYY-MM-DD/run_summary.json
paper_trading/YYYY-MM-DD/target_weights.csv
paper_trading/YYYY-MM-DD/orders.csv
paper_trading/YYYY-MM-DD/validation_candidates_top50.csv
logs/paper_trading_YYYYMMDD.log
```

`orders.csv` 是第二天开盘纸面模拟要执行的订单。执行或模拟成交后，把当前持仓更新到：

```text
paper_trading/positions.csv
```

格式：

```csv
symbol,shares
000001,100
600000,200
```

## 3. 配置

编辑：

```text
config/paper_trading_config.json
```

常用字段：

- `capital`：纸面账户资金，例如 `100000`。
- `max_price_for_new_buy`：新买入股票最高价格，资金较小时建议限制高价股。
- `min_order_value`：低于这个金额的订单忽略。
- `positions_path`：当前纸面持仓文件。
- `score_mode`：建议使用 `sharpe`，不要先追年化。

## 4. 设置 macOS 定时任务

生成 launchd plist：

```bash
REPO_PATH="$(pwd)"
sed "s#__REPO_PATH__#${REPO_PATH}#g" \
  launchd/com.hzy.quant.paper-trading.plist.template \
  > ~/Library/LaunchAgents/com.hzy.quant.paper-trading.plist
```

加载任务：

```bash
launchctl unload ~/Library/LaunchAgents/com.hzy.quant.paper-trading.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.hzy.quant.paper-trading.plist
```

手动触发一次：

```bash
launchctl start com.hzy.quant.paper-trading
```

查看日志：

```bash
tail -f logs/launchd_paper_trading.out.log
tail -f logs/launchd_paper_trading.err.log
```

默认每天 15:30 运行。Mac mini 不需要整天开着，但需要在 15:30 左右处于开机、联网、没有睡死的状态。

## 5. 纸面实盘流程

每天：

1. 15:30 自动生成信号和订单。
2. 第二天开盘后，用行情软件记录这些订单是否能成交，以及实际成交价格。
3. 更新 `paper_trading/positions.csv`。
4. 保留每日 `orders.csv` 和 `run_summary.json`，用于一个月后评估真实可执行表现。

建议至少运行 1-3 个月，不急着接自动交易。

## 6. 当前限制

- 当前脚本默认用本地缓存数据。如果要真正每日更新，需要在 Mac mini 上接入稳定数据源，并在运行前刷新 `data/processed/csi1000_fundamental_small_200.parquet`。
- 当前股票池仍有历史研究阶段的未来流动性选择偏差。实盘前应改成 point-in-time 中证1000成分或滚动流动性股票池。
- 当前程序只生成纸面订单，不负责券商下单。
