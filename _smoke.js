// 无头验证：从 index.html 抽出 <script id="engine">，在 Node vm 里跑断言
// 与 index.html 导出的 API 对齐（runSelfTests / forward / backward / viterbi /
// bruteForce / posteriorGamma / sampleSequence / baumWelch）。
const fs = require('fs'), vm = require('vm'), path = require('path');
const dir = __dirname;
const html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m) { console.error('engine script not found'); process.exit(1); }
const ctx = { console };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const H = ctx.HMM;

let pass = 0, fail = 0; const fails = [];
function ok(cond, name, detail) {
  if (cond) { pass++; } else { fail++; fails.push(name + ' | ' + detail); }
}
function close(a, b, tol) { return Math.abs(a - b) <= tol; }

// --- 引擎未导出的辅助量，在这里按定义自算（不改动 index.html） ---
// 路径重打分：log P(o, q | θ) = logπ_q0 + Σ log A + Σ log B
function pathLogProb(model, obs, q) {
  let s = model.logPi[q[0]];
  for (let t = 0; t < obs.length; t++) {
    if (t > 0) s += model.logA[q[t - 1]][q[t]];
    s += model.logB[q[t]][obs[t]];
  }
  return s;
}
// 状态标签置换：π/A 按 perm 重排，B 行重排 → 对数似然不变
function permuteModel(model, perm) {
  const inv = new Array(model.N);
  for (let i = 0; i < model.N; i++) inv[perm[i]] = i;
  const A = model.A.map((row, i) => model.A[inv[i]].map((_, j) => model.A[inv[i]][inv[j]]));
  const B = model.B.map((_, i) => model.B[inv[i]].slice());
  const pi = model.pi.map((_, i) => model.pi[inv[i]]);
  return H.createModel(A, B, pi);
}
// ξ_t(i,j) = α_t(i) A_ij B_j(o_{t+1}) β_{t+1}(j) / P(o)
function xiFrom(model, obs, f, b, ll) {
  const N = model.N, T = obs.length, xi = [];
  for (let t = 0; t < T - 1; t++) {
    const row = [];
    for (let i = 0; i < N; i++) {
      const cell = [];
      for (let j = 0; j < N; j++) {
        cell.push(Math.exp(f.logAlpha[t][i] + model.logA[i][j]
          + model.logB[j][obs[t + 1]] + b.logBeta[t + 1][j] - ll));
      }
      row.push(cell);
    }
    xi.push(row);
  }
  return xi;
}

// ---------- 1) 内置不变量 ----------
const st1 = H.runSelfTests();
ok(st1.length >= 8, '内置不变量 >= 8 条', 'got=' + st1.length);
st1.forEach((r, i) => ok(r.pass, 'selfTest#' + (i + 1) + ' ' + r.name, r.detail));

// ---------- 2) 随机压力测试：前向/后向/暴力/Viterbi/γ-ξ/置换 ----------
const rng = H.mulberry32(31337);
let bfChecked = 0;
for (let trial = 0; trial < 200; trial++) {
  const N = 2 + Math.floor(rng() * 3);          // 2..4
  const M = 2 + Math.floor(rng() * 4);          // 2..5
  const T = 1 + Math.floor(rng() * 9);          // 1..9
  const mm = H.randomModel(N, M, rng);
  const obs = H.sampleSequence(mm, T, rng).observations;

  const f = H.forward(mm, obs);
  const b = H.backward(mm, obs);
  const lf = f.logLikelihood, lb = b.logLikelihood;
  ok(close(lf, lb, Math.abs(lf) * 1e-9 + 1e-12), `trial${trial} fwd==bwd`, `Δ=${Math.abs(lf - lb)}`);

  // 暴力枚举（限制规模）
  const paths = Math.pow(N, T);
  if (paths <= 60000) {
    bfChecked++;
    const bf = H.bruteForce(mm, obs);
    ok(close(lf, bf.logLikelihood, Math.abs(lf) * 1e-9 + 1e-12), `trial${trial} fwd==brute`,
       `${lf} vs ${bf.logLikelihood}`);
    const v = H.viterbi(mm, obs);
    const rel = Math.abs(v.logProb - bf.bestLogProb) / Math.max(Math.abs(bf.bestLogProb), 1e-300);
    ok(rel < 1e-9, `trial${trial} viterbi==bruteProb`, `rel=${rel}`);
    // Viterbi 最优路径概率必须等于暴力最优；路径本身允许并列最优
    const tie = Math.abs(pathLogProb(mm, obs, v.path) - bf.bestLogProb) < 1e-9;
    ok(tie, `trial${trial} viterbi==brutePath`, `${v.path} vs ${bf.bestPath}`);
    const man = pathLogProb(mm, obs, v.path);
    ok(close(man, v.logProb, 1e-9), `trial${trial} pathRescore`, `Δ=${Math.abs(man - v.logProb)}`);
  }

  // γ / ξ 一致性
  const P = H.posteriorGamma(mm, obs);
  const xi = xiFrom(mm, obs, f, b, lf);
  let eG = 0, eX = 0;
  for (let t = 0; t < T; t++) {
    let s = 0; for (let i = 0; i < N; i++) s += P.gamma[t][i];
    eG = Math.max(eG, Math.abs(s - 1));
    if (t < T - 1) for (let i = 0; i < N; i++) {
      let s2 = 0; for (let j = 0; j < N; j++) s2 += xi[t][i][j];
      eX = Math.max(eX, Math.abs(s2 - P.gamma[t][i]));
    }
  }
  ok(Math.max(eG, eX) < 1e-9, `trial${trial} gamma/xi`, `γ=${eG} ξ=${eX}`);

  // 标签置换不变
  const perm = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const mp = permuteModel(mm, perm);
  const lp = H.forward(mp, obs).logLikelihood;
  ok(close(lp, lf, Math.abs(lf) * 1e-9 + 1e-12), `trial${trial} permInvariant`, `Δ=${Math.abs(lp - lf)}`);
}
ok(bfChecked > 100, 'bruteForce 交叉验证覆盖数', 'checked=' + bfChecked);

// ---------- 3) 边界条件 ----------
{
  const mm = H.randomModel(3, 4, H.mulberry32(5));
  ok(H.forward(mm, []).logLikelihood === 0, 'T=0 前向 loglik=0', String(H.forward(mm, []).logLikelihood));
  ok(H.backward(mm, []).logLikelihood === 0, 'T=0 后向 loglik=0', String(H.backward(mm, []).logLikelihood));
  ok(H.viterbi(mm, []).path.length === 0, 'T=0 Viterbi 路径空', '');
  ok(H.bruteForce(mm, []).logLikelihood === 0, 'T=0 暴力枚举 loglik=0', String(H.bruteForce(mm, []).logLikelihood));

  const one = H.createModel([[1]], [[0.3, 0.7]], [1]);
  const o1 = [0, 1, 1, 0];
  ok(close(H.forward(one, o1).logLikelihood, Math.log(0.3 * 0.7 * 0.7 * 0.3), 1e-12), 'N=1 解析解',
     String(H.forward(one, o1).logLikelihood));
  ok(H.viterbi(one, o1).path.join(',') === '0,0,0,0', 'N=1 Viterbi 全 0', '');

  // 含 0 概率的模型：不得 NaN / Infinity
  const zero = H.createModel([[0.5, 0.5], [0, 1]], [[1, 0], [0, 1]], [1, 0]);
  const lz = H.forward(zero, [0, 1]).logLikelihood;
  ok(Number.isFinite(lz) && lz < 0, '零概率转移/发射不产生 NaN', String(lz));
  const vz = H.viterbi(zero, [0, 1]);
  ok(vz.path.every(Number.isInteger), '零概率模型 Viterbi 路径合法', vz.path.join(','));
}

// ---------- 4) 确定性：同 seed 同结果 ----------
{
  const gen = () => H.sampleSequence(H.randomModel(3, 4, H.mulberry32(9)), 50, H.mulberry32(9)).observations;
  const a = gen(), b = gen();
  ok(a.join(',') === b.join(','), '同 seed 采样确定性', a.slice(0, 8).join(','));
  const r1 = H.runSelfTests().map(r => r.detail).join('|');
  const r2 = H.runSelfTests().map(r => r.detail).join('|');
  ok(r1 === r2, 'selfTest 可复现', '');
}

// ---------- 5) Baum-Welch：单调 + 参数还原 ----------
for (let c = 0; c < 5; c++) {
  const r0 = H.mulberry32(1000 + c);
  const N = 2 + (c % 3), M = 3 + (c % 2);
  const truth = H.randomModel(N, M, r0);
  const corpus = H.sampleSequence(truth, 2000, r0).observations;
  const init = H.randomModel(N, M, r0);
  const bw = H.baumWelch(corpus, init, { iters: 40 });
  const hist = bw.history;
  let minD = Infinity;
  for (let i = 1; i < hist.length; i++) minD = Math.min(minD, hist[i] - hist[i - 1]);
  ok(minD > -1e-7, `BW#${c} 对数似然单调非减`, `minΔ=${minD}`);
  const trueLL = H.logLikelihood(truth, corpus);
  const endLL = hist[hist.length - 1];
  // EM 从随机初值只保证收敛到驻点，不保证打到全局最优；要求相对真值 LL 的差距 < 1%
  ok(endLL >= trueLL - 0.01 * Math.abs(trueLL), `BW#${c} 训练后 LL 逼近真值 LL(<1%)`, `${endLL} vs ${trueLL}`);
  // 真值模型是 EM 不动点：从真值出发 1 步，LL 不得下降
  const one = H.baumWelch(corpus, truth, { iters: 1 });
  ok(one.history[1] - one.history[0] > -1e-9, `BW#${c} 真值起点 1 步 EM 不下降`,
     `Δ=${one.history[1] - one.history[0]}`);
  // 行和为 1
  let normErr = 0;
  for (let i = 0; i < N; i++) {
    let sa = 0, sb = 0;
    for (let j = 0; j < N; j++) sa += bw.model.A[i][j];
    for (let j = 0; j < M; j++) sb += bw.model.B[i][j];
    normErr = Math.max(normErr, Math.abs(sa - 1), Math.abs(sb - 1));
  }
  ok(normErr < 1e-12, `BW#${c} 训练后矩阵行和=1`, `err=${normErr}`);
}

fs.writeFileSync(path.join(dir, '_smoke.log'),
  `PASS ${pass} / ${pass + fail}\n` + (fail ? 'FAIL:\n' + fails.join('\n') : 'ALL GREEN') + '\n');
console.log(`PASS ${pass} / ${pass + fail}`);
if (fail) { console.log('FAIL:\n' + fails.join('\n')); process.exit(1); }
console.log('ALL GREEN');
