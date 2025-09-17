# 小丑牌概率与最优弃牌模拟器 · Balatro Probability & Optimal Discard Simulator

---

## 简介 · Overview

* **这是什么 | What is this**
  一组用于 *[小丑牌](https://baike.baidu.com/item/%E5%B0%8F%E4%B8%91%E7%89%8C/65126743)* 的**解析概率**与**最优弃牌策略**模拟脚本，支持多牌型、多轮弃牌、52 张（标准卡组）与去掉J、Q、K的 40 张牌组（废弃卡组）。
  A set of scripts for *[Balatro](https://en.wikipedia.org/wiki/Balatro)* that use **closed-form combinatorics** and an **optimal per-round discard planner**, supporting multi-type, multi-round discards for both 52-card and face-less 40-card decks.

* **核心思路 | Core idea**
  内层用概率（超几何 + 按点数 DP + 包含–排除）评估“本轮弃牌 k 张后立刻成型”的概率，选择最优方案；外层通过模拟统计“第 t 轮达成”的成功率。
  The inner planner computes “succeed after discarding k now” probabilities (hypergeometric + rank-DP + inclusion–exclusion), then chooses the best plan; the outer loop simulates trials to measure success at round *t*.

---

## 特性 · Features

* **目标牌型 | Target Hands**
    * `Flush`（同花）
    * `Straight`（顺子，A 可高/低）
    * `Two Pair`（两对）
    * `Full House`（葫芦）
* **牌组 | Deck**
    * 标准 **52** 张 - Standard **52**-card deck
    * **40** 张（移除 J, Q, K） - **40**-card deck (J, Q, K removed) (A, 2..10)
* **抽牌模型 | Draw Model**
    * 从**剩余牌库**不放回抽取 - Drawing without replacement from the **remaining deck**
* **策略 | Strategy**
    * 每轮选择**当前最优**弃牌方案（考虑所有听牌可能），并执行抽牌 - Each round, select the **currently optimal** discard strategy (considering all outs) and execute the draw.
* **精确计算 | Exact Calculation**
    * 同花：超几何尾和 - `Flush`: Hypergeometric tail sum.
    * 两对/葫芦：按牌点的多元超几何 **DP** - `Two Pair` / `Full House`: Multivariate hypergeometric **DP** by rank.
    * 顺子：窗口并集的**包含–排除** + “所需牌点至少 1 张”的 **DP** - `Straight`: **Inclusion-Exclusion** for the union of rank-windows + **DP** for "at least one of each required rank".

---

## 文件说明 · Files

* `balatro_multiround_exact.py`
  多轮弃牌模拟器。外层多轮模拟；内层使用**解析概率**选最优方案并真实抽牌；输出“恰在第 t 轮成功 / 截止第 t 轮累计成功”的百分比。
  Multi-round simulator. Uses exact inner planning and real draws; reports *exact* and *cumulative* success by round.

* `result.txt`
  示例结果。
  Result example.


---

## 环境与安装 · Requirements

* **Python 3.9+**
* 仅用标准库；无需额外依赖
  Uses Python standard library only; no extra dependencies.

---

## 快速开始 · Quick Start

```bash
# 1) 多轮弃牌：52 张，起手 8 张，目标“顺子”，允许每轮弃 0..5 张，5 万次试验
python balatro_multiround_exact.py --deck 52 --hand-size 8 \
  --trials 50000 --min-discard 0 --max-discard 5 --target Straight

# 2) 多轮弃牌：40 张（无 JQK），目标“葫芦”，起手 9 张，最多 3 轮
python balatro_multiround_exact.py --deck 40 --hand-size 9 \
  --trials 30000 --min-discard 0 --max-discard 3 --target "Full House"

# 3) 单轮：统计期望弃牌张数（解析法）
python balatro_expected_discards_exact.py --deck 52 --hand-size 8 --trials 50000
```

**输出解释 | Output**（以 `balatro_multiround_exact.py` 为例）

* `Exact Success%`：**恰好**在第 *t* 轮达成目标的比例
* `Cumulative Success%`：**截止**第 *t* 轮累计达成的比例
* `Failed within allowed rounds`：在限定轮次内未成功的比例

> 若 `--min-discard=0`，脚本会在“第 0 轮”检查起手牌是否已成型（已成型则记为 0 轮成功）。
> With `--min-discard=0`, a “round 0” check marks hands already made as success at 0.

---

## 命令行参数 · CLI Options

- `--deck {52,40}`：牌组类型（52 标准；40 去人头）
  Deck type (standard 52 or face-less 40).
- `--hand-size INT`：起手牌数量 $H$（默认 8）
  Starting hand size $H$ (default 8).
- `--trials INT`：试验次数（默认 50,000）
  Number of trials (default 50k).
- `--seed INT`：随机种子（默认 2025）
  RNG seed.
- `--min-discard INT` / `--max-discard INT`：**每轮**最少/最多弃牌数；同时决定报告的轮次范围
  Per-round min/max discards; also define the reported rounds.
- `--target {All, Full House,Straight,Flush,Two Pair}`：目标牌型（默认所有）
  Target hand type (default all).

---

## 致谢 · Acknowledgements

- 感谢 ChatGPT 在数学建模与代码实现上的帮助。
  Thanks to ChatGPT for assistance with mathematical modeling and code implementation.

---
