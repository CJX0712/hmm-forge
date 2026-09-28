/* =====================================================================
 * HMM Forge — 算法引擎 (Algorithm Engine)
 * 隐马尔可夫模型 / Hidden Markov Model — pure JavaScript, zero dependencies
 *
 * Author : 晨星 / Chenxing
 * License: MIT
 *
 * Design rules / 设计纪律:
 *   1. 所有概率一律在 log-space 中运算 (all arithmetic in log-space)。
 *   2. 任何"对概率求和"都必须经过 logSumExp — 禁止 exp 后再相加。
 *   3. 引擎不得引用 document / window / fetch / 任何全局宿主对象，
 *      因此可以原样抽出为独立 JS 模块，在 Node 中无头运行。
 *
 * Algorithms / 算法清单:
 *   forward()        前向算法 (log-space, per-step scaling factors)
 *   backward()       后向算法 (log-space)
 *   viterbi()       Viterbi 解码 (log-space dynamic programming)
 *   bruteForce()     暴力枚举全部隐藏路径 (仅小规模, 用于交叉验证)
 *   baumWelchStep()  Baum-Welch (EM) 单步
 *   baumWelch()      Baum-Welch (EM) 迭代训练
 *   posteriorGamma() 后验状态概率 gamma_t(i) = P(q_t = i | O, λ)
 * ===================================================================== */
(function (root) {
  'use strict';

  var NEG_INF = -Infinity;

  /* ------------------------------------------------------------------
   * 0. 基础数值工具 / numeric primitives
   * ---------------------------------------------------------------- */

  /** log(x)，并把 0 映射为 -Infinity 而不是 NaN。 */
  function safeLog(x) {
    if (x === 0) return NEG_INF;
    if (x < 0 || x !== x) return NaN;
    return Math.log(x);
  }

  /**
   * logSumExp: log( Σ_i exp(v_i) )，数值稳定版。
   * 先减去最大值再 exp，杜绝 overflow / underflow。
   */
  function logSumExp(values) {
    var n = values.length;
    if (n === 0) return NEG_INF;
    var max = NEG_INF;
    for (var i = 0; i < n; i++) {
      var v = values[i];
      if (v !== v) return NaN;          // NaN 传染
      if (v === Infinity) return Infinity;
      if (v > max) max = v;
    }
    if (max === NEG_INF) return NEG_INF;
    var acc = 0;
    for (var k = 0; k < n; k++) acc += Math.exp(values[k] - max);
    return max + Math.log(acc);
  }

  /** 两个对数概率相加: log(exp(a) + exp(b))，比通用版更快。 */
  function logAdd(a, b) {
    if (a === NEG_INF) return b;
    if (b === NEG_INF) return a;
    if (a >= b) return a + Math.log1p(Math.exp(b - a));
    return b + Math.log1p(Math.exp(a - b));
  }

  function zeros(rows, cols, fill) {
    var m = new Array(rows);
    for (var i = 0; i < rows; i++) {
      m[i] = new Array(cols);
      for (var j = 0; j < cols; j++) m[i][j] = fill;
    }
    return m;
  }

  /** 每行归一化，返回新的二维普通概率矩阵。 */
  function normalizeRows(m) {
    return m.map(function (row) {
      var s = 0, i;
      for (i = 0; i < row.length; i++) s += row[i];
      if (!(s > 0)) return row.map(function () { return 1 / row.length; });
      var out = row.map(function (v) { return v / s; });
      // 二次归一，抵消浮点残差，保证行和误差 < 1e-15
      var s2 = 0;
      for (i = 0; i < out.length; i++) s2 += out[i];
      return out.map(function (v) { return v / s2; });
    });
  }

  /** 向量归一化。 */
  function normalizeVector(v) {
    var s = 0;
    for (var i = 0; i < v.length; i++) s += v[i];
    if (!(s > 0)) return v.map(function () { return 1 / v.length; });
    var out = v.map(function (x) { return x / s; });
    var s2 = 0;
    for (i = 0; i < out.length; i++) s2 += out[i];
    return out.map(function (x) { return x / s2; });
  }

  /** 行和与 1 的最大偏差 (用于归一化不变量检查)。 */
  function rowSumError(m) {
    var worst = 0;
    for (var i = 0; i < m.length; i++) {
      var s = 0;
      for (var j = 0; j < m[i].length; j++) s += m[i][j];
      var d = Math.abs(s - 1);
      if (d > worst) worst = d;
    }
    return worst;
  }

  /* ------------------------------------------------------------------
   * 1. 模型构造 / model construction
   *     λ = (A, B, π)
   *     A : N×N 状态转移矩阵   B : N×M 发射矩阵   π : N 初始分布
   * ---------------------------------------------------------------- */

  function createModel(A, B, pi) {
    A = normalizeRows(A);
    B = normalizeRows(B);
    pi = normalizeVector(pi);
    var N = pi.length, M = B[0].length;
    return {
      N: N,
      M: M,
      A: A, B: B, pi: pi,
      logA: A.map(function (r) { return r.map(safeLog); }),
      logB: B.map(function (r) { return r.map(safeLog); }),
      logPi: pi.map(safeLog)
    };
  }

  function cloneModel(m) {
    return createModel(
      m.A.map(function (r) { return r.slice(); }),
      m.B.map(function (r) { return r.slice(); }),
      m.pi.slice()
    );
  }

  /* ------------------------------------------------------------------
   * 2. 前向算法 / Forward algorithm
   *    α_t(j) = P(O_0..O_t, q_t = j | λ)
   *    同时给出等价的"逐步缩放"分解:
   *      logCumScale[t] = log P(O_0..O_t) = Σ_{s≤t} log c_s
   *      logScale[t]    = log c_t
   *      logAlphaHat[t] = log α_t − logCumScale[t]   (归一化后每行 logSumExp = 0)
   *    log P(O|λ) = logCumScale[T-1] = logSumExp_i(log α_{T-1}(i))。
   *    缩放因子本身由 forwardScaled() 独立复算并交叉验证 (见自检 T4)。
   * ---------------------------------------------------------------- */

  function forward(model, obs) {
    var logA = model.logA, logB = model.logB, logPi = model.logPi;
    var N = model.N, T = obs.length;
    if (T === 0) return {
      logAlpha: [], logAlphaHat: [], logScale: [], logCumScale: [],
      logLikelihood: 0, T: 0, N: N
    };

    var logAlpha = new Array(T);
    var logAlphaHat = new Array(T);
    var logScale = new Array(T);
    var logCumScale = new Array(T);
    var prev = new Array(N), i, j, t;
    for (i = 0; i < N; i++) prev[i] = logPi[i] + logB[i][obs[0]];
    logAlpha[0] = prev.slice();
    logCumScale[0] = logSumExp(prev);
    logScale[0] = logCumScale[0];

    for (t = 1; t < T; t++) {
      var cur = new Array(N);
      for (j = 0; j < N; j++) {
        var terms = new Array(N);
        for (i = 0; i < N; i++) terms[i] = prev[i] + logA[i][j];
        cur[j] = logSumExp(terms) + logB[j][obs[t]];
      }
      logAlpha[t] = cur.slice();
      logCumScale[t] = logSumExp(cur);
      logScale[t] = logCumScale[t] - logCumScale[t - 1];
      prev = cur;
    }

    for (t = 0; t < T; t++) {
      var hat = new Array(N);
      for (i = 0; i < N; i++) hat[i] = logAlpha[t][i] - logCumScale[t];
      logAlphaHat[t] = hat;
    }

    var logLik = logCumScale[T - 1];
    return {
      logAlpha: logAlpha, logAlphaHat: logAlphaHat,
      logScale: logScale, logCumScale: logCumScale,
      logLikelihood: logLik, T: T, N: N
    };
  }

  /* ------------------------------------------------------------------
   * 2b. 前向算法 (概率空间 + 逐步缩放) / Forward, probability space, scaled
   *     经典教科书实现: 每步把 α̂ 归一化，累乘缩放因子 c_t。
   *     这是一条与 2a 完全独立的代码路径（不经过 log / logSumExp），
   *     专门用于交叉验证 log-space 版本，避免"自己证自己"。
   * ---------------------------------------------------------------- */

  function forwardScaled(model, obs) {
    var A = model.A, B = model.B, pi = model.pi;
    var N = model.N, T = obs.length, i, j, t;
    if (T === 0) return { alphaHat: [], logC: [], logCumScale: [], logLikelihood: 0, T: 0, N: N };

    var alphaHat = new Array(T), logC = new Array(T), logCumScale = new Array(T);
    var cur = new Array(N), s = 0;
    for (i = 0; i < N; i++) { cur[i] = pi[i] * B[i][obs[0]]; s += cur[i]; }
    for (i = 0; i < N; i++) cur[i] /= s;
    alphaHat[0] = cur.slice();
    logC[0] = Math.log(s);
    logCumScale[0] = logC[0];

    for (t = 1; t < T; t++) {
      var nxt = new Array(N);
      s = 0;
      for (j = 0; j < N; j++) {
        var acc = 0;
        for (i = 0; i < N; i++) acc += cur[i] * A[i][j];
        nxt[j] = acc * B[j][obs[t]];
        s += nxt[j];
      }
      for (j = 0; j < N; j++) nxt[j] /= s;
      alphaHat[t] = nxt.slice();
      logC[t] = Math.log(s);
      logCumScale[t] = logCumScale[t - 1] + logC[t];
      cur = nxt;
    }

    var ll = 0;
    for (t = 0; t < T; t++) ll += logC[t];
    return { alphaHat: alphaHat, logC: logC, logCumScale: logCumScale, logLikelihood: ll, T: T, N: N };
  }

  /* ------------------------------------------------------------------
   * 3. 后向算法 / Backward algorithm
   *    β_t(i) = P(O_{t+1}..O_{T-1} | q_t = i, λ)
   *    log P(O|λ) = logSumExp_i( log π_i + log b_i(O_0) + log β_0(i) )
   * ---------------------------------------------------------------- */

  function backward(model, obs) {
    var logA = model.logA, logB = model.logB, logPi = model.logPi;
    var N = model.N, T = obs.length, i, j, t;
    if (T === 0) return { logBeta: [], logLikelihood: 0, T: 0, N: N };

    var logBeta = new Array(T);
    var next = new Array(N);
    for (i = 0; i < N; i++) next[i] = 0;      // log β_{T-1}(i) = log 1 = 0
    logBeta[T - 1] = next.slice();

    for (t = T - 2; t >= 0; t--) {
      var cur = new Array(N);
      for (i = 0; i < N; i++) {
        var terms = new Array(N);
        for (j = 0; j < N; j++) terms[j] = logA[i][j] + logB[j][obs[t + 1]] + next[j];
        cur[i] = logSumExp(terms);
      }
      logBeta[t] = cur;
      next = cur;
    }

    var init = new Array(N);
    for (i = 0; i < N; i++) init[i] = logPi[i] + logB[i][obs[0]] + logBeta[0][i];
    return { logBeta: logBeta, logLikelihood: logSumExp(init), T: T, N: N };
  }

  /* ------------------------------------------------------------------
   * 4. Viterbi 解码 / Viterbi decoding
   *    δ_t(j) = max_{q_0..q_{t-1}} P(q_0..q_t = j, O_0..O_t | λ)
   * ---------------------------------------------------------------- */

  function viterbi(model, obs) {
    var logA = model.logA, logB = model.logB, logPi = model.logPi;
    var N = model.N, T = obs.length, i, j, t;
    if (T === 0) return { path: [], logProb: 0, delta: [], psi: [] };

    var delta = new Array(T), psi = new Array(T);
    var d0 = new Array(N), p0 = new Array(N);
    for (i = 0; i < N; i++) { d0[i] = logPi[i] + logB[i][obs[0]]; p0[i] = -1; }
    delta[0] = d0; psi[0] = p0;

    for (t = 1; t < T; t++) {
      var d = new Array(N), p = new Array(N);
      for (j = 0; j < N; j++) {
        var best = NEG_INF, arg = 0;
        for (i = 0; i < N; i++) {
          var v = delta[t - 1][i] + logA[i][j];
          if (v > best) { best = v; arg = i; }
        }
        d[j] = best + logB[j][obs[t]];
        p[j] = arg;
      }
      delta[t] = d; psi[t] = p;
    }

    var bestLast = NEG_INF, last = 0;
    for (i = 0; i < N; i++) if (delta[T - 1][i] > bestLast) { bestLast = delta[T - 1][i]; last = i; }

    var path = new Array(T);
    path[T - 1] = last;
    for (t = T - 1; t > 0; t--) path[t - 1] = psi[t][path[t]];
    return { path: path, logProb: bestLast, delta: delta, psi: psi };
  }

  /* ------------------------------------------------------------------
   * 5. 暴力枚举 / Brute force (ground truth, 仅小规模)
   *    穷举 N^T 条隐藏路径，给出:
   *      logLikelihood = log Σ_Q P(O, Q | λ)      (应与前向一致)
   *      bestLogProb   = max_Q P(O, Q | λ)        (应与 Viterbi 一致)
   *      bestPath                                  (应与 Viterbi 一致)
   * ---------------------------------------------------------------- */

  function bruteForce(model, obs, limit) {
    var logA = model.logA, logB = model.logB, logPi = model.logPi;
    var N = model.N, T = obs.length;
    if (T === 0) return { logLikelihood: 0, bestLogProb: 0, bestPath: [], count: 0 };
    var cap = limit || 2000000;
    var total = Math.pow(N, T);
    if (!(total <= cap)) throw new Error('bruteForce: N^T = ' + total + ' 超过上限 ' + cap);

    var logs = new Array(total);
    var best = NEG_INF, bestIdx = 0;
    var q = new Array(T);
    for (var c = 0; c < total; c++) {
      var x = c;
      for (var t = T - 1; t >= 0; t--) { q[t] = x % N; x = (x - q[t]) / N; }
      var lp = logPi[q[0]] + logB[q[0]][obs[0]];
      for (t = 1; t < T; t++) lp += logA[q[t - 1]][q[t]] + logB[q[t]][obs[t]];
      logs[c] = lp;
      if (lp > best) { best = lp; bestIdx = c; }
    }
    var bestPath = new Array(T);
    var y = bestIdx;
    for (t = T - 1; t >= 0; t--) { bestPath[t] = y % N; y = (y - bestPath[t]) / N; }
    return { logLikelihood: logSumExp(logs), bestLogProb: best, bestPath: bestPath, count: total };
  }

  /* ------------------------------------------------------------------
   * 6. 后验 / posteriors
   *    gamma_t(i) = P(q_t = i | O, λ) = α_t(i) β_t(i) / P(O|λ)
   * ---------------------------------------------------------------- */

  function posteriorGamma(model, obs) {
    var f = forward(model, obs), b = backward(model, obs);
    var N = model.N, T = obs.length, t, i;
    var gamma = new Array(T);
    for (t = 0; t < T; t++) {
      gamma[t] = new Array(N);
      for (i = 0; i < N; i++) gamma[t][i] = Math.exp(f.logAlpha[t][i] + b.logBeta[t][i] - f.logLikelihood);
    }
    return { gamma: gamma, logLikelihood: f.logLikelihood, logAlpha: f.logAlpha, logBeta: b.logBeta };
  }

  /* ------------------------------------------------------------------
   * 7. Baum-Welch (EM) / 无监督训练
   *    E-step: gamma, xi 全部在 log-space 累积
   *    M-step: 闭式重估计 + 显式行归一化
   * ---------------------------------------------------------------- */

  function asSequenceList(seqs) {
    if (!seqs.length) return [];
    return Array.isArray(seqs[0]) ? seqs : [seqs];
  }

  function logLikelihood(model, seqs) {
    var list = asSequenceList(seqs), total = 0;
    for (var i = 0; i < list.length; i++) total += forward(model, list[i]).logLikelihood;
    return total;
  }

  /** 单步 EM: 由 θ_k 推出 θ_{k+1}，同时返回 θ_k 下的对数似然。 */
  function baumWelchStep(model, seqs) {
    var list = asSequenceList(seqs);
    var N = model.N, M = model.M, i, j, t;
    var logPiNum = new Array(N); for (i = 0; i < N; i++) logPiNum[i] = NEG_INF;
    var aNum = zeros(N, N, NEG_INF);
    var aDen = new Array(N); for (i = 0; i < N; i++) aDen[i] = NEG_INF;
    var bNum = zeros(N, M, NEG_INF);
    var bDen = new Array(N); for (i = 0; i < N; i++) bDen[i] = NEG_INF;
    var totalLL = 0;

    for (var s = 0; s < list.length; s++) {
      var obs = list[s];
      if (!obs.length) continue;
      var T = obs.length;
      var f = forward(model, obs), b = backward(model, obs);
      var ll = f.logLikelihood;
      totalLL += ll;

      for (i = 0; i < N; i++) {
        logPiNum[i] = logAdd(logPiNum[i], f.logAlpha[0][i] + b.logBeta[0][i] - ll);
      }
      for (t = 0; t < T; t++) {
        for (i = 0; i < N; i++) {
          var g = f.logAlpha[t][i] + b.logBeta[t][i] - ll;
          bDen[i] = logAdd(bDen[i], g);
          bNum[i][obs[t]] = logAdd(bNum[i][obs[t]], g);
        }
      }
      for (t = 0; t < T - 1; t++) {
        for (i = 0; i < N; i++) {
          for (j = 0; j < N; j++) {
            var xi = f.logAlpha[t][i] + model.logA[i][j] +
                     model.logB[j][obs[t + 1]] + b.logBeta[t + 1][j] - ll;
            aNum[i][j] = logAdd(aNum[i][j], xi);
            aDen[i] = logAdd(aDen[i], xi);
          }
        }
      }
    }

    var piDen = logSumExp(logPiNum);
    var newPi = new Array(N);
    var newA = zeros(N, N, 1 / N);
    var newB = zeros(N, M, 1 / M);
    for (i = 0; i < N; i++) {
      newPi[i] = (piDen === NEG_INF || piDen === Infinity) ? 1 / N
        : Math.min(1, Math.max(0, Math.exp(logPiNum[i] - piDen)));
      for (j = 0; j < N; j++) {
        newA[i][j] = (aDen[i] === NEG_INF || aDen[i] === Infinity) ? 1 / N
          : Math.min(1, Math.max(0, Math.exp(aNum[i][j] - aDen[i])));
      }
      for (var k = 0; k < M; k++) {
        newB[i][k] = (bDen[i] === NEG_INF || bDen[i] === Infinity) ? 1 / M
          : Math.min(1, Math.max(0, Math.exp(bNum[i][k] - bDen[i])));
      }
    }

    var next = createModel(newA, newB, newPi);
    return { model: next, logLikelihood: totalLL };
  }

  /** 迭代 EM 直到收敛，history[k] = 第 k 次迭代时的 log P(O|λ)。 */
  function baumWelch(seqs, model0, opts) {
    opts = opts || {};
    var maxIter = opts.maxIter || 50;
    var tol = (opts.tol == null) ? 1e-9 : opts.tol;
    var list = asSequenceList(seqs);
    var model = cloneModel(model0);
    var history = [];
    var prevLL = NEG_INF;

    for (var it = 0; it < maxIter; it++) {
      var ll = logLikelihood(model, list);
      history.push(ll);
      if (opts.onIteration) opts.onIteration(it, model, ll);
      if (prevLL !== NEG_INF && Math.abs(ll - prevLL) <= tol * Math.max(1, Math.abs(prevLL))) {
        prevLL = ll;
        break;
      }
      prevLL = ll;
      var step = baumWelchStep(model, list);
      model = step.model;
      if (opts.onStep) opts.onStep(it, model);
    }
    history.push(logLikelihood(model, list));
    return { model: model, history: history, iterations: history.length };
  }

  /* ------------------------------------------------------------------
   * 8. 确定性随机 / deterministic PRNG (mulberry32)
   * ---------------------------------------------------------------- */

  function mulberry32(seed) {
    var a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** 由 seed 生成随机 HMM 参数。concentration 越大分布越均匀。 */
  function randomModel(N, M, rng, concentration) {
    var c = (concentration == null) ? 1 : concentration;
    var pi = [], A = [], B = [], i, j;
    for (i = 0; i < N; i++) pi.push(Math.pow(rng(), 1 / c) + 1e-3);
    for (i = 0; i < N; i++) {
      var ra = [];
      for (j = 0; j < N; j++) ra.push(Math.pow(rng(), 1 / c) + 1e-3);
      A.push(ra);
    }
    for (i = 0; i < N; i++) {
      var rb = [];
      for (j = 0; j < M; j++) rb.push(Math.pow(rng(), 1 / c) + 1e-3);
      B.push(rb);
    }
    return createModel(A, B, pi);
  }

  /** 从模型中采样一条 (states, observations) 序列。 */
  function sampleSequence(model, T, rng) {
    var N = model.N, states = new Array(T), obs = new Array(T);
    function drawCategorical(probs) {
      var r = rng(), acc = 0;
      for (var i = 0; i < probs.length; i++) {
        acc += probs[i];
        if (r < acc) return i;
      }
      return probs.length - 1;
    }
    var s = drawCategorical(model.pi);
    for (var t = 0; t < T; t++) {
      states[t] = s;
      obs[t] = drawCategorical(model.B[s]);
      s = drawCategorical(model.A[s]);
    }
    return { states: states, observations: obs };
  }

  /* ------------------------------------------------------------------
   * 9. 自检 / self tests — 浏览器面板与 Node 无头测试共用同一套
   * ---------------------------------------------------------------- */

  function fmt(x) {
    if (x === Infinity) return '+Infinity';
    if (x === -Infinity) return '-Infinity';
    if (x !== x) return 'NaN';
    if (x === 0) return '0';
    var a = Math.abs(x);
    if (a < 1e-4 || a >= 1e7) return x.toExponential(3);
    return x.toFixed(10).replace(/0+$/, '').replace(/\.$/, '');
  }

  function relErr(a, b) {
    var d = Math.abs(a - b);
    var scale = Math.max(1, Math.abs(a), Math.abs(b));
    return d / scale;
  }

  function runSelfTests() {
    var results = [];

    function record(id, name, fn) {
      var r;
      try {
        r = fn();
      } catch (e) {
        r = { pass: false, detail: 'EXCEPTION: ' + (e && e.message ? e.message : String(e)) };
      }
      results.push({ id: id, name: name, pass: !!r.pass, detail: r.detail || '' });
    }

    /* --- T1 logSumExp 抗下溢 ------------------------------------- */
    record('T1', 'logSumExp 抗下溢 / underflow safety', function () {
      var a = Math.log(1e-300), b = Math.log(2e-300), c = Math.log(3e-300);
      var got = logSumExp([a, b, c]);
      var want = Math.log(6e-300);
      var e = relErr(got, want);
      var g2 = logSumExp([NEG_INF, NEG_INF, Math.log(7)]);
      var e2 = relErr(g2, Math.log(7));
      return {
        pass: e < 1e-12 && e2 < 1e-12,
        detail: 'log(Σe^-300) = ' + fmt(got) + ' vs ' + fmt(want) +
                ' | relerr = ' + fmt(e) +
                ' ; with -Inf entries = ' + fmt(g2) + ' | relerr = ' + fmt(e2)
      };
    });

    /* --- T2 前向 vs 后向 ----------------------------------------- */
    var m2 = randomModel(4, 3, mulberry32(20260929));
    var s2 = sampleSequence(m2, 25, mulberry32(20260930));
    var o2 = s2.observations;
    record('T2', '前向 vs 后向 log P(O|λ) 一致 / forward ≡ backward', function () {
      var f = forward(m2, o2).logLikelihood;
      var b = backward(m2, o2).logLikelihood;
      var e = relErr(f, b);
      return {
        pass: e < 1e-10,
        detail: 'forward = ' + fmt(f) + ' ; backward = ' + fmt(b) +
                ' | abs = ' + fmt(Math.abs(f - b)) + ' | relerr = ' + fmt(e)
      };
    });

    /* --- T3 前向 vs 暴力枚举 ------------------------------------- */
    record('T3', '前向 vs 暴力枚举 (T=6, N=3) / forward ≡ brute force Σ', function () {
      var m = randomModel(3, 2, mulberry32(1234));
      var o = sampleSequence(m, 6, mulberry32(5678)).observations;
      var f = forward(m, o).logLikelihood;
      var bf = bruteForce(m, o).logLikelihood;
      var e = relErr(f, bf);
      return {
        pass: e < 1e-10,
        detail: 'paths = ' + Math.pow(3, 6) + ' | forward = ' + fmt(f) +
                ' | brute = ' + fmt(bf) + ' | relerr = ' + fmt(e)
      };
    });

    /* --- T4 α 缩放一致性 ------------------------------------------ */
    record('T4', 'α 缩放一致性 / scaled-α per-step log-sum ≡ logSumExp', function () {
      var m = randomModel(4, 3, mulberry32(999));
      var o = sampleSequence(m, 40, mulberry32(1000)).observations;
      var f = forward(m, o);          // log-space 路径
      var fs = forwardScaled(m, o);   // 概率空间 + 逐步缩放 路径（独立实现）
      var lastLSE = logSumExp(f.logAlpha[o.length - 1]);
      // (a) 两条独立路径算出的 log P(O|λ) 必须一致
      var e1 = relErr(fs.logLikelihood, lastLSE);
      // (b) 每步: 缩放累计量 logCumScale[t] 必须等于 logSumExp(未缩放 α_t)
      var worstStep = 0, i, t;
      for (t = 0; t < o.length; t++) {
        worstStep = Math.max(worstStep, Math.abs(logSumExp(f.logAlpha[t]) - fs.logCumScale[t]));
      }
      // (c) 缩放后每行 logSumExp(α̂_t) 必须 = 0
      var worstScaled = 0;
      for (t = 0; t < o.length; t++) {
        worstScaled = Math.max(worstScaled, Math.abs(logSumExp(f.logAlphaHat[t])));
        var sHat = 0;
        for (i = 0; i < f.N; i++) sHat += fs.alphaHat[t][i];
        worstScaled = Math.max(worstScaled, Math.abs(sHat - 1));
      }
      return {
        pass: e1 < 1e-10 && worstStep < 1e-10 && worstScaled < 1e-10,
        detail: 'scaled-forward LL = ' + fmt(fs.logLikelihood) +
                ' | logSumExp(α_T-1) = ' + fmt(lastLSE) + ' | relerr = ' + fmt(e1) +
                ' | max|logSumExp(α_t) − Σlog c| = ' + fmt(worstStep) +
                ' | max(|logSumExp(α̂_t)|, |Σα̂_t − 1|) = ' + fmt(worstScaled)
      };
    });

    /* --- T5 Viterbi vs 暴力最优 ------------------------------------ */
    record('T5', 'Viterbi vs 暴力最优路径 / argmax path + logprob', function () {
      var m = randomModel(3, 4, mulberry32(24680));
      var o = sampleSequence(m, 6, mulberry32(13579)).observations;
      var v = viterbi(m, o);
      var bf = bruteForce(m, o);
      var e = relErr(v.logProb, bf.bestLogProb);
      var same = v.path.length === bf.bestPath.length &&
                 v.path.every(function (x, idx) { return x === bf.bestPath[idx]; });
      return {
        pass: e < 1e-10 && same,
        detail: 'viterbi logProb = ' + fmt(v.logProb) + ' | brute max = ' + fmt(bf.bestLogProb) +
                ' | relerr = ' + fmt(e) +
                ' | path viterbi = [' + v.path.join(',') + '] brute = [' + bf.bestPath.join(',') +
                '] | identical = ' + (same ? 'YES' : 'NO')
      };
    });

    /* --- T6 归一化: A/B/π 行和 = 1 -------------------------------- */
    record('T6', '归一化 / A rows, B rows, π sum to 1', function () {
      var m = randomModel(5, 4, mulberry32(31415));
      var eA = rowSumError(m.A), eB = rowSumError(m.B);
      var ePi = Math.abs(m.pi.reduce(function (a, b) { return a + b; }, 0) - 1);
      // 再检查 EM 一步之后仍然归一
      var o = sampleSequence(m, 60, mulberry32(27182)).observations;
      var after = baumWelchStep(m, o).model;
      var eA2 = rowSumError(after.A), eB2 = rowSumError(after.B);
      var ePi2 = Math.abs(after.pi.reduce(function (a, b) { return a + b; }, 0) - 1);
      var worst = Math.max(eA, eB, ePi, eA2, eB2, ePi2);
      return {
        pass: worst < 1e-12,
        detail: 'random: |ΣA-1| = ' + fmt(eA) + ' |ΣB-1| = ' + fmt(eB) + ' |Σπ-1| = ' + fmt(ePi) +
                ' ; after 1 EM step: ' + fmt(eA2) + ' / ' + fmt(eB2) + ' / ' + fmt(ePi2) +
                ' | worst = ' + fmt(worst)
      };
    });

    /* --- T7 Baum-Welch 单调性 -------------------------------------- */
    record('T7', 'Baum-Welch 单调性 / log-likelihood non-decreasing', function () {
      var m = randomModel(3, 4, mulberry32(7));
      var seqs = [
        sampleSequence(m, 120, mulberry32(11)).observations,
        sampleSequence(m, 90, mulberry32(12)).observations
      ];
      var r = baumWelch(seqs, m, { maxIter: 40, tol: 0 });
      var minDelta = Infinity, i;
      for (i = 1; i < r.history.length; i++) {
        var d = r.history[i] - r.history[i - 1];
        if (d < minDelta) minDelta = d;
      }
      return {
        pass: minDelta >= -1e-9,
        detail: 'iters = ' + r.history.length + ' | LL[0] = ' + fmt(r.history[0]) +
                ' | LL[last] = ' + fmt(r.history[r.history.length - 1]) +
                ' | gain = ' + fmt(r.history[r.history.length - 1] - r.history[0]) +
                ' | min Δ = ' + fmt(minDelta) + ' (tol -1e-9)'
      };
    });

    /* --- T8 后验 gamma 每行和 = 1 ---------------------------------- */
    record('T8', '后验归一化 / Σ_i γ_t(i) = 1 for every t', function () {
      var m = randomModel(4, 3, mulberry32(5150));
      var o = sampleSequence(m, 50, mulberry32(5151)).observations;
      var g = posteriorGamma(m, o);
      var worst = 0;
      for (var t = 0; t < o.length; t++) {
        var s = 0;
        for (var i = 0; i < m.N; i++) s += g.gamma[t][i];
        worst = Math.max(worst, Math.abs(s - 1));
      }
      return {
        pass: worst < 1e-12,
        detail: 'T = ' + o.length + ' | max|Σ_i γ_t(i) - 1| = ' + fmt(worst)
      };
    });

    /* --- T9 长序列数值稳定 ----------------------------------------- */
    record('T9', '长序列数值稳定 / T=300 no -Infinity / NaN', function () {
      var m = randomModel(5, 4, mulberry32(8080));
      var o = sampleSequence(m, 300, mulberry32(8081)).observations;
      var f = forward(m, o), b = backward(m, o), v = viterbi(m, o);
      var bw = baumWelchStep(m, o);
      var vals = [f.logLikelihood, b.logLikelihood, v.logProb, bw.logLikelihood];
      var bad = 0, finiteCount = 0, total = 0;
      function scan(arr) {
        for (var i = 0; i < arr.length; i++) {
          total++;
          if (!isFinite(arr[i])) bad++; else finiteCount++;
        }
      }
      for (var t = 0; t < o.length; t++) {
        scan(f.logAlpha[t]); scan(b.logBeta[t]); scan(v.delta[t]);
      }
      for (var i = 0; i < vals.length; i++) {
        total++;
        if (!isFinite(vals[i])) bad++; else finiteCount++;
      }
      for (i = 0; i < m.N; i++) { scan(bw.model.A[i]); scan(bw.model.B[i]); }
      return {
        pass: bad === 0,
        detail: 'T = 300, N = 5 | scanned ' + total + ' numbers | finite = ' + finiteCount +
                ' | bad = ' + bad +
                ' | forward LL = ' + fmt(f.logLikelihood) +
                ' | viterbi logProb = ' + fmt(v.logProb)
      };
    });

    /* --- T10 PRNG 确定性 ------------------------------------------- */
    record('T10', '确定性 / same seed ⇒ bit-identical results', function () {
      var r1 = mulberry32(42), r2 = mulberry32(42), r3 = mulberry32(43);
      var a = [], b = [], c = [], i;
      for (i = 0; i < 100; i++) { a.push(r1()); b.push(r2()); c.push(r3()); }
      var identical = a.every(function (x, idx) { return x === b[idx]; });
      var differs = a.some(function (x, idx) { return x !== c[idx]; });
      // 端到端: 同一 seed 训练出的最终 LL 必须逐位相同
      function pipeline(seed) {
        var mm = randomModel(3, 3, mulberry32(seed));
        var oo = sampleSequence(mm, 80, mulberry32(seed + 1)).observations;
        var rr = baumWelch(oo, randomModel(3, 3, mulberry32(seed + 2)), { maxIter: 15, tol: 0 });
        return rr.history[rr.history.length - 1];
      }
      var p1 = pipeline(2026), p2 = pipeline(2026);
      var endToEnd = (p1 === p2);
      return {
        pass: identical && differs && endToEnd,
        detail: 'seed 42 twice identical = ' + (identical ? 'YES' : 'NO') +
                ' | seed 43 differs = ' + (differs ? 'YES' : 'NO') +
                ' | end-to-end final LL = ' + fmt(p1) + ' ≡ ' + fmt(p2) +
                ' (' + (endToEnd ? 'bit-identical' : 'MISMATCH') + ')'
      };
    });

    /* --- T11 EM 提升似然 ------------------------------------------- */
    record('T11', 'EM 有效性 / trained LL > initial LL', function () {
      var truth = randomModel(3, 3, mulberry32(600));
      var o = sampleSequence(truth, 200, mulberry32(601)).observations;
      var init = randomModel(3, 3, mulberry32(602));
      var r = baumWelch(o, init, { maxIter: 60, tol: 0 });
      var ll0 = r.history[0], llN = r.history[r.history.length - 1];
      return {
        pass: llN > ll0,
        detail: 'LL[0] = ' + fmt(ll0) + ' → LL[' + (r.history.length - 1) + '] = ' + fmt(llN) +
                ' | gain = ' + fmt(llN - ll0) +
                ' | 真值模型 LL = ' + fmt(forward(truth, o).logLikelihood)
      };
    });

    return results;
  }

  /* ------------------------------------------------------------------
   * 10. 导出 / exports
   * ---------------------------------------------------------------- */

  var HMM = {
    version: '1.0.0',
    author: '晨星 / Chenxing',
    // numeric
    safeLog: safeLog, logSumExp: logSumExp, logAdd: logAdd,
    normalizeRows: normalizeRows, normalizeVector: normalizeVector, rowSumError: rowSumError,
    // model
    createModel: createModel, cloneModel: cloneModel, randomModel: randomModel,
    // inference
    forward: forward, forwardScaled: forwardScaled, backward: backward, viterbi: viterbi,
    bruteForce: bruteForce, posteriorGamma: posteriorGamma, logLikelihood: logLikelihood,
    // learning
    baumWelchStep: baumWelchStep, baumWelch: baumWelch,
    // random
    mulberry32: mulberry32, sampleSequence: sampleSequence,
    // testing
    runSelfTests: runSelfTests, fmt: fmt, relErr: relErr
  };

  root.HMM = HMM;

})(typeof globalThis !== 'undefined' ? globalThis : (typeof window !== 'undefined' ? window : this));
