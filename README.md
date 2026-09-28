# HMM Forge · 隐马尔可夫模型锻造炉

> A single-file, zero-dependency HMM workbench: hand-written Forward / Backward / Viterbi / Baum-Welch
> in pure JavaScript, all in **log-space with logSumExp**, with a built-in one-click invariant self-check
> panel and a headless Node test suite (19/19 green).
>
> 单文件、零外部依赖的 HMM 实验台：纯 JS 手写前向 / 后向 / Viterbi / Baum-Welch，
> 全程 log-space + logSumExp 防下溢，内置一键不变量自检面板，并配 Node 无头测试（19/19 全绿）。

**作者 / Author:** 晨星 / Chenxing　·　**License:** MIT

---

## 这是什么 / What it is

`index.html` 是一个可以直接双击打开、也可以离线放进 U 盘的完整 HMM 实验台：

* **算法全部手写**，不依赖任何库；概率运算一律走 log-space，任意"对概率求和"都经过 `logSumExp`，
  因此 T=500 的长序列也不会出现 `-Infinity` 或 `NaN`。
* **引擎可整体抽出**：核心算法写在 `<script id="engine">` 块里，不含任何 `document` / `window` /
  `fetch` 引用，`engine.test.mjs` 会把它从 HTML 里正则抽出、塞进 Node 的 `vm` 沙箱执行并跑全部用例 ——
  这是"引擎能当独立 JS 模块用"的机械证明，不是口头声明。
* **不变量是真的在算**，不是写在文档里的漂亮话：前向 vs 后向、前向 vs 暴力枚举、Viterbi vs 暴力最优、
  Baum-Welch 单调性、归一化、长序列数值稳定、PRNG 确定性 —— 每一条都在页面上打印实测数值。

Everything (CSS, JS, SVG rendering, tests) is inline in one file. No CDN, no web font, no network request
of any kind — verifiable by the purity test `N3`.

---

## 快速开始 / Quick start

### 浏览器 / Browser

```bash
# 直接打开，不需要任何服务器或构建步骤
open index.html          # macOS
start index.html         # Windows
xdg-open index.html      # Linux
```

打开后：

1. 点 **「🎲 随机生成模型」** —— 按 seed 生成 λ = (A, B, π)
2. 点 **「🎯 生成观测序列」** —— 从模型采样一条观测序列 O
3. 直接在 π / A / B 表格里改数字（失焦即重算，自动行归一化）
4. 点 **「▶ 训练 Baum-Welch（实时曲线）」** —— 看对数似然曲线逐点上升
5. 点 **「🧪 运行全部不变量自检」** —— 逐条看 ✅ 与实测数值

页面默认 seed = `20260929`，同一个 seed 的结果逐位可复现。

### Node 无头测试 / Headless tests

```bash
node engine.test.mjs     # 19 个用例，退出码 0 = 全绿
node _build/smoke.mjs    # 界面层冒烟（DOM shim，15 项）
```

要求 Node ≥ 18（用到 `node:vm` ESM）。

---

## 算法清单 / Algorithms

| 算法 | 入口 | 复杂度 | 说明 |
|---|---|---|---|
| 前向 Forward | `HMM.forward(model, obs)` | O(T·N²) | log-space；同时返回未缩放 `logAlpha`、缩放后 `logAlphaHat`、每步缩放因子 `logScale`、累计 `logCumScale` |
| 前向（概率空间·逐步缩放） | `HMM.forwardScaled(model, obs)` | O(T·N²) | 教科书版独立实现，**专门用于交叉验证** log-space 版本，避免"自己证自己" |
| 后向 Backward | `HMM.backward(model, obs)` | O(T·N²) | log-space β |
| Viterbi 解码 | `HMM.viterbi(model, obs)` | O(T·N²) | log-space δ/ψ + 回溯，返回最优路径与 `logProb` |
| 暴力枚举 | `HMM.bruteForce(model, obs)` | O(T·N^T) | 穷举全部隐藏路径，给出 ground-truth 的 Σ 与 argmax（小规模基准） |
| 后验 Posterior | `HMM.posteriorGamma(model, obs)` | O(T·N²) | γ_t(i) = P(q_t = i \| O, λ) |
| Baum-Welch (EM) 单步 | `HMM.baumWelchStep(model, seqs)` | O(T·N²) | E 步 γ/ξ 全程 log-space 累积，M 步闭式重估计 + 显式行归一化 |
| Baum-Welch (EM) 迭代 | `HMM.baumWelch(seqs, model, opts)` | — | 支持多条序列，返回 `history`（每轮 log P(O\|λ)） |
| 确定性 PRNG | `HMM.mulberry32(seed)` | O(1)/次 | 32-bit 状态，同 seed 逐位可复现 |
| 采样 | `HMM.sampleSequence(model, T, rng)` | O(T·N) | 从模型采样 (states, observations) |

符号约定：`N` = 隐藏状态数，`M` = 观测符号数，`T` = 序列长度。

**防下溢纪律（两条铁律）**：
1. 所有概率存 log；
2. 任何"概率求和"必须 `logSumExp`（先减最大值再 `exp`），禁止 `exp` 后相加。

---

## 可验证不变量 / Verifiable invariants

下列数值是 `node engine.test.mjs` 在本仓库代码上的**真实输出**，不是示例。

| # | 不变量 Invariant | 判据 Threshold | 实测 Measured | 结果 |
|---|---|---|---|---|
| T1 | logSumExp 抗下溢 `log(Σ 1e-300·k)` | relerr < 1e-12 | `-688.983768429` vs 真值，**relerr = 0** | ✅ |
| T2 | 前向 vs 后向 `log P(O\|λ)` | relerr < 1e-10 | `-26.4652094697` vs `-26.4652094697`，**abs = 7.105e-15, relerr = 2.685e-16** | ✅ |
| T3 | 前向 vs 暴力枚举（T=6, N=3, 729 条路径） | relerr < 1e-10 | `-4.1743660802` vs `-4.1743660802`，**relerr = 0** | ✅ |
| T4 | α 缩放一致性：Σ log c_t ≡ logSumExp(α_T-1) | relerr < 1e-10 | **relerr = 3.472e-16**；max\|logSumExp(α_t) − Σlog c\| = **1.421e-14**；max\|logSumExp(α̂_t)\| = **3.553e-15** | ✅ |
| T5 | Viterbi vs 暴力最优（值与路径） | relerr < 1e-10 且路径相同 | `-11.4462535559` vs `-11.4462535559`，**relerr = 0**，路径 `[1,1,1,1,1,1]` 完全一致 | ✅ |
| T6 | 归一化：A 行和 / B 行和 / π 和为 1 | < 1e-12 | 随机模型 **0 / 0 / 0**；1 次 EM 后 **1.110e-16 / 0 / 0**，worst = **1.110e-16** | ✅ |
| T7 | Baum-Welch 单调性（对数似然不减） | min Δ ≥ −1e-9 | 41 轮，LL `-277.7834925258 → -274.1115612522`，**min Δ = +0.0158451601** | ✅ |
| T8 | 后验归一化 Σ_i γ_t(i) = 1 | < 1e-12 | T=50，max 偏差 **3.286e-14** | ✅ |
| T9 | 长序列数值稳定（T=300, N=5） | 无 −Infinity / NaN | 扫描 **4549** 个数，非有限 **0**；forward LL = `-415.3691923679` | ✅ |
| T10 | 确定性：同 seed 结果一致 | 逐位相同 | seed 42 两次 **bit-identical**；seed 43 不同；端到端最终 LL `-76.2650309041` 逐位相同 | ✅ |
| T11 | EM 有效性：训练后 LL 提升 | LL_last > LL_0 | `-225.3151962676 → -216.5683559178`，增益 **8.7468403498**（真值模型 LL = `-216.8146827328`） | ✅ |
| N6 | 5 组随机模型三路交叉（fwd/bwd/scaled-fwd） | relerr < 1e-10 | worst **1.465e-15** | ✅ |
| N7 | Viterbi vs 暴力（N=2..3, T=5..7，6 组） | relerr < 1e-10 且路径全同 | worst **2.668e-16**，6/6 路径一致 | ✅ |
| N8 | 极端长度 T=500, N=6 | 无 −Infinity / NaN | 扫描 **6004** 个数，非有限 **0**；relerr(fwd,bwd) = **7.147e-16** | ✅ |
| N2/N3 | 零依赖纯净性（引擎无宿主对象；HTML 无 CDN/字体/外链/fetch） | 0 命中 | 全部 **0 命中** | ✅ |
| N5 | 内联引擎与 `_build/engine.src.js` 无漂移 | 逐字节相同 | **30101 chars 完全一致** | ✅ |

**合计 19/19 PASS**。`T1–T11` 由浏览器自检面板与 Node 测试**共用同一套代码**（`HMM.runSelfTests()`），
所以页面上看到的数字和命令行里的数字是同一份。

---

## 目录结构 / Layout

```
hmm-forge/
├── index.html          # 主文件：全部内容（界面 + 引擎 + 自检），单文件零依赖
├── engine.test.mjs     # Node 无头测试：从 index.html 抽出引擎并跑 19 个用例
├── README.md           # 本文件
├── LICENSE             # MIT, Copyright (c) 2026 晨星 / Chenxing
├── .gitignore
└── _build/             # 生成 index.html 的源码与脚本（可选阅读）
    ├── engine.src.js   #   引擎的独立纯 JS 模块形态（与内联块逐字节一致）
    ├── shell.html      #   index.html 的模板，含 /*__ENGINE_INLINE__*/ 插入标记
    ├── build.mjs       #   内联构建 + 防漂移校验：node _build/build.mjs
    └── smoke.mjs       #   界面层冒烟测试（DOM shim）：node _build/smoke.mjs
```

`_build/` 存在的原因：引擎在仓库里有两份存在形式（独立模块 / HTML 内联块），
`build.mjs` 在每次构建时强制校验二者逐字节一致，杜绝源码漂移；`engine.test.mjs` 的用例 `N5`
也会再校验一次。

---

## 从 HTML 里抽出引擎 / Extracting the engine

```js
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const html = readFileSync('index.html', 'utf8');
const src  = html.match(/<script id="engine">([\s\S]*?)<\/script>/)[1];

const sandbox = { console };
vm.createContext(sandbox);
vm.runInContext(src, sandbox);

const HMM = sandbox.HMM;                       // 纯 JS 模块，无 DOM 依赖
const m = HMM.randomModel(3, 4, HMM.mulberry32(42));
const o = HMM.sampleSequence(m, 60, HMM.mulberry32(43)).observations;
console.log(HMM.forward(m, o).logLikelihood);  // -73.13416363065106（实测）
console.log(HMM.viterbi(m, o).path);           // 2,2,0,0,0,0,2,2,1,2,...（实测）
const r = HMM.baumWelch(o, HMM.randomModel(3, 4, HMM.mulberry32(44)), { maxIter: 50 });
console.log(r.history);                        // -82.0140 -72.8115 -72.6150 ... -65.0258（实测单调不减）
```

也可以直接复制 `_build/engine.src.js`，或把 `<script id="engine">` 整块粘到任何 JS 运行时里。

---

## 已知限制 / Known limitations

* `bruteForce()` 是 O(N^T) 的穷举，内置上限 2,000,000 条路径；仅用于小规模交叉验证（T ≤ 7 左右）。
* 界面上 N、M 限制在 1–8，T 限制在 1–400：这是**渲染与交互体验**的限制，算法本身对 N/T 无此约束
  （测试 N8 已在 N=6、T=500 上验证）。
* Baum-Welch 是无监督的 EM，只能收敛到局部极大值；不同的随机初值会落到不同的局部解 —— 这属于算法性质，不是缺陷。
* 训练过程会为每一轮 EM 额外做一次前向以绘制曲线点，属于可视化开销，不影响算法正确性。

---

## License

MIT © 2026 晨星 / Chenxing
