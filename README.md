# 小丑牌概率与最优弃牌模拟器 · Balatro Probability & Optimal Discard Simulator

---

## 简介 · Overview

* **这是什么 | What is this**
  一组用于 *[小丑牌](https://baike.baidu.com/item/%E5%B0%8F%E4%B8%91%E7%89%8C/65126743)* 的**解析概率**与**最优弃牌策略**模拟脚本，支持多牌型、多轮弃牌、52 张（标准卡组）与去掉 J/Q/K 的 40 张牌组（废弃卡组）。
  A set of scripts for *[Balatro](https://en.wikipedia.org/wiki/Balatro)* that use **closed-form combinatorics** and an **optimal per-round discard planner**, supporting multi-type, multi-round discards for both 52-card and face-less 40-card decks.

* **试图定量回答三个实战问题 | The three practical questions we aim to answer quantitatively**

  1. **废弃牌组打什么更合适？**直觉上“顺子”和“两对”更常见、“同花”更难，但到底差多少？。
     **Which brand is the most suitable one for the 40-card face-less deck?** Intuition says *Straights* and *Two Pairs* are easier while *Flushes* are rarer—**how much** rarer or easier?
  2. **手牌上限对“大牌型”的增幅有多大？**经验上，起手手数越多，顺子/葫芦等的成功率增长显著。试图给出具体数字而非感觉。
     **How much does hand-size $H$ boost “big hands”?** We try to turning intuition into numbers.
  3. **“同花数值偏低”到底对不对？**看看同花的概率比其他大牌型到底高了多少。
     **Is Flush “undervalued”?** See how much higher the probability of flush than other big brand.

* **核心思路 | Core idea**
  内层计算最优弃牌策略；外层通过多轮模拟统计第 t 轮达成的成功率。
  The inner planner computes “succeed after discarding $k$ now” probabilities and chooses the best plan; the outer loop simulates rounds to measure success at round *t*.

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
* **计算 | Calculation**

  * 同花：超几何尾和 - `Flush`: Hypergeometric tail sum.
  * 两对/葫芦：按牌点的多元超几何 **DP** - `Two Pair` / `Full House`: Multivariate hypergeometric **DP** by rank.
  * 顺子：窗口并集的**包含–排除** + “所需牌点至少 1 张”的 **DP** - `Straight`: **Inclusion-Exclusion** for the union of rank-windows + **DP** for "at least one of each required rank".

---

## 文件说明 · Files

* `balatro_multiround_exact.py`
  多轮弃牌模拟器。外层多轮模拟；内层使用**解析概率**选最优方案并真实抽牌；输出“恰在第 t 轮成功 / 截止第 t 轮累计成功”的百分比。
  *Multi-round simulator.* Uses exact inner planning and real draws; reports *exact* and *cumulative* success by round.

* `result.txt`
  示例结果。
  *Result example.*

---

## 环境与安装 · Requirements

* **Python 3.9+**
* 仅用标准库；无需额外依赖
  Uses Python standard library only; no extra dependencies.

---

## 示例 · Quick Start

```bash
# 1) 52 张卡组，起手 8 张，目标“顺子”，允许每轮弃 0..5 张，5 万次试验
python balatro_multiround_exact.py --deck 52 --hand-size 8 \
  --trials 50000 --min-discard 0 --max-discard 5 --target Straight

# 2) 40 张（无 JQK）卡组，目标所有牌型，起手 9 张，最多 3 轮弃牌
python balatro_multiround_exact.py --deck 40 --hand-size 9 \
  --trials 30000 --min-discard 0 --max-discard 3 --target "All"
```

**输出 | Output**

* `Exact Success%`：**恰好**在第 *t* 轮达成目标的比例
* `Cumulative Success%`：**截止**第 *t* 轮累计达成的比例
* `Failed within allowed rounds`：在限定轮次内未成功的比例

> 若 `--min-discard=0`，会在“第 0 轮”检查起手牌是否已成型（已成型则记为 0 轮成功）。
> With `--min-discard=0`, a “round 0” check marks hands already made as success at 0.

---

## 命令行参数 · CLI Options

* `--deck {52,40}`：牌组类型（52 标准；40 去人头）
  Deck type (standard 52 or face-less 40).
* `--hand-size INT`：起手牌数量 $H$（默认 8）
  Starting hand size $H$ (default 8).
* `--trials INT`：试验次数（默认 50,000）
  Number of trials (default 50k).
* `--seed INT`：随机种子（默认 2025）
  RNG seed.
* `--min-discard INT` / `--max-discard INT`：**每轮**最少/最多弃牌数；同时决定报告的轮次范围
  Per-round min/max discards; also define the reported rounds.
* `--target {All, Full House,Straight,Flush,Two Pair}`：目标牌型（默认所有）
  Target hand type (default all).

---

## 致谢 · Acknowledgements

* 感谢 **ChatGPT** 在数学建模、代码实现与文档制作上的帮助。
  Thanks to **ChatGPT** for assistance with mathematical modeling, code implementation, and document production.
