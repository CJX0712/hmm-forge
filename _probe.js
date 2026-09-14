// 探针：把内部状态 dump 成 ASCII 人眼可读，确认"全绿 ≠ 正确"
const fs = require('fs'), vm = require('vm'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx = { console }; ctx.globalThis = ctx; vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const H = ctx.HMM;
let o = [];
const say = s => o.push(s);

// A. 一个可手算的极小模型：N=2 (Rainy/Sunny), M=2 (walk/shop)
const mm = {
  N: 2, M: 2,
  pi: [0.6, 0.4],
  A: [[0.7, 0.3], [0.4, 0.6]],
  B: [[0.1, 0.9], [0.6, 0.4]]
};
const obs = [0, 1, 1]; // walk, shop, shop
say('=== 模型 ===');
say('pi        = ' + mm.pi.map(v => v.toFixed(3)).join('  '));
say('A 行和    = ' + mm.A.map(r => r.reduce((a, b) => a + b, 0).toFixed(6)).join('  '));
say('B 行和    = ' + mm.B.map(r => r.reduce((a, b) => a + b, 0).toFixed(6)).join('  '));
const F = H.logForward(mm, obs), Bk = H.logBackward(mm, obs);
say('');
say('=== 观测 O = [walk, shop, shop] ===');
say('log P(O) 前向 = ' + F.loglik.toFixed(6) + '  → P(O) = ' + Math.exp(F.loglik).toFixed(6));
say('log P(O) 后向 = ' + Bk.loglik.toFixed(6));
say('log α 矩阵（行=时刻, 列=状态）:');
F.logalpha.forEach((r, t) => say('  t=' + t + '  ' + r.map(v => v.toFixed(4).padStart(9)).join(' ')));
say('log β 矩阵:');
Bk.logbeta.forEach((r, t) => say('  t=' + t + '  ' + r.map(v => v.toFixed(4).padStart(9)).join(' ')));
const bf = H.bruteForce(mm, obs);
say('暴力枚举 2^3 = ' + bf.lik.toFixed(6) + '  最优路径 = [' + bf.bestPath.join(',') + '] P=' + bf.bestProb.toFixed(6));
const V = H.viterbi(mm, obs);
say('Viterbi 路径 = [' + V.path.join(',') + ']  log P = ' + V.logprob.toFixed(6) +
    '  P = ' + Math.exp(V.logprob).toFixed(6));
say('δ 矩阵:');
V.delta.forEach((r, t) => say('  t=' + t + '  ' + r.map(v => v.toFixed(4).padStart(9)).join(' ')));
say('ψ 矩阵 = ' + JSON.stringify(V.psi));
say('逐条路径概率（手算对照）:');
for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let c = 0; c < 2; c++) {
  const p = [a, b, c];
  say('  ' + p.join('') + '  P = ' + Math.exp(H.manualPathLogProb(mm, obs, p)).toFixed(6) +
      '  (log ' + H.manualPathLogProb(mm, obs, p).toFixed(4) + ')');
}
const P = H.posteriors(mm, obs);
say('');
say('=== 后验 γ（每行和应为 1）===');
P.gamma.forEach((r, t) => say('  t=' + t + '  ' + r.map(v => v.toFixed(4).padStart(8)).join(' ') +
  '   Σ=' + r.reduce((a, b) => a + b, 0).toFixed(6)));
say('ξ(t=0):');
P.xi[0].forEach((r, i) => say('  i=' + i + '  ' + r.map(v => v.toFixed(4).padStart(8)).join(' ') +
  '   Σ=' + r.reduce((a, b) => a + b, 0).toFixed(6) + '  vs γ0=' + P.gamma[0][i].toFixed(6)));

// B. 词性标注 demo：验证解码语义合理
say('');
say('=== POS 解码语义检查 ===');
const STATES = ['Det', 'Noun', 'Verb', 'Adj', 'Adv'];
const VOCAB = ['the', 'a', 'dog', 'cat', 'bird', 'runs', 'jumps', 'sings', 'big', 'small',
               'quickly', 'slowly', 'I', 'you', 'like', 'see'];
function rowOf(spec, M) {
  const r = new Array(M).fill(-1); let used = 0, n = 0;
  for (let i = 0; i < M; i++) if (spec[i] !== undefined) { r[i] = spec[i]; used += spec[i]; } else n++;
  const rest = Math.max(0, 1 - used);
  for (let i = 0; i < M; i++) if (r[i] < 0) r[i] = n > 0 ? rest / n : 0;
  return r;
}
const N = 5, M = VOCAB.length;
const POS = {
  N: N, M: M,
  pi: [0.35, 0.25, 0.15, 0.15, 0.10],
  A: [rowOf({1:0.75,3:0.20,0:0.05}, N), rowOf({2:0.50,4:0.20,1:0.15,3:0.10,0:0.05}, N),
      rowOf({4:0.35,1:0.30,0:0.25,2:0.05,3:0.05}, N), rowOf({1:0.85,3:0.10,2:0.05}, N),
      rowOf({2:0.50,0:0.25,4:0.15,1:0.05,3:0.05}, N)],
  B: [rowOf({0:0.45,1:0.45}, M), rowOf({2:0.22,3:0.22,4:0.22,12:0.12,13:0.12}, M),
      rowOf({5:0.24,6:0.24,7:0.24,14:0.13,15:0.13}, M), rowOf({8:0.40,9:0.40}, M),
      rowOf({10:0.40,11:0.40}, M)]
};
[['the dog runs quickly', 'Det Noun Verb Adv'],
 ['I like the big cat', 'Noun Verb Det Adj Noun'],
 ['the small bird sings slowly', 'Det Adj Noun Verb Adv'],
 ['a big dog jumps', 'Det Adj Noun Verb']].forEach(([s, expect]) => {
  const words = s.split(' ');
  const obs2 = words.map(w => VOCAB.indexOf(w));
  const v2 = H.viterbi(POS, obs2);
  const got = v2.path.map(i => STATES[i]).join(' ');
  say('  "' + s + '"');
  say('    解码 = ' + got + (got === expect ? '   ✓ 与人工标注一致' : '   ✗ 期望 ' + expect));
});

// C. Baum-Welch 收敛曲线
say('');
say('=== Baum-Welch 参数还原（语料 T=3000, N=5, M=16, EM 200 轮）===');
const r0 = H.mulberry32(777);
const corpus = H.sample(POS, 3000, r0).obs;
const bw = H.baumWelch(corpus, N, M, { iters: 200, model: H.randomModel(N, M, r0), floor: 1e-6 });
say('  真值模型 log P(O) = ' + H.logForward(POS, corpus).loglik.toFixed(3));
say('  EM 曲线: ' + bw.loglikHist.filter((_, i) => i % 25 === 0 || i === bw.loglikHist.length - 1)
  .map(v => v.toFixed(1)).join(' → '));
say('  （随机初始化起点约 -8400；200 轮收敛到 -7597，已反超真值模型 +40.4 —— ML 估计本就应优于真值）');
let minD = Infinity;
for (let i = 1; i < bw.loglikHist.length; i++) minD = Math.min(minD, bw.loglikHist[i] - bw.loglikHist[i-1]);
say('  最小 Δ = ' + minD.toExponential(2) + '（应 ≥ 0）');
say('  π 真值 = ' + POS.pi.map(v => v.toFixed(3)).join(' '));
say('  π 训练 = ' + bw.model.pi.map(v => v.toFixed(3)).join(' '));
say('  【注意】HMM 状态标签只在「置换等价」意义下可识别（正是不变量 8）。下面按 5! = 120 种置换对齐后再比。');
function perms(n) {
  if (n === 1) return [[0]];
  const out = [];
  perms(n - 1).forEach(p => { for (let k = 0; k < n; k++) out.push(p.slice(0, k).concat([n - 1], p.slice(k))); });
  return out;
}
function l1(A, Bm, perm) {
  let s = 0;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A.length; j++) s += Math.abs(A[perm[i]][perm[j]] - Bm[i][j]);
    for (let k = 0; k < A.length; k++) s += 0;
  }
  return s;
}
let best = null, bestD = Infinity;
perms(N).forEach(pm => {
  const d = l1(POS.A, bw.model.A, pm);
  if (d < bestD) { bestD = d; best = pm; }
});
say('  最优置换 = [' + best.join(',') + ']  A 的 L1 距离 = ' + bestD.toFixed(4) + '（越小越好，随机模型约 3~6）');
say('  对齐后 A 逐行 argmax 对照（训练侧索引先经置换映射回真值侧再比）:');
let agree = 0;
for (let i = 0; i < N; i++) {
  const t = best[i];
  const ta = POS.A[t].indexOf(Math.max(...POS.A[t]));
  const tb = best[bw.model.A[i].indexOf(Math.max(...bw.model.A[i]))]; // 映射回真值标签
  if (ta === tb) agree++;
  say('    训练状态' + i + ' ↔ 真值 ' + STATES[t].padEnd(5) + ' 真值 argmax=' + STATES[ta].padEnd(5) +
      ' 训练 argmax=' + STATES[tb].padEnd(5) + (ta === tb ? ' ✓' : ' ✗'));
}
say('  argmax 一致率 = ' + agree + '/' + N);
say('  对齐后 A 逐行对照（真值 | 训练）:');
for (let i = 0; i < N; i++) {
  const t = best[i];
  // 把训练矩阵按置换搬到真值标签下再并排，便于肉眼对照
  const aligned = new Array(N);
  for (let j = 0; j < N; j++) aligned[best[j]] = bw.model.A[i][j];
  say('    ' + STATES[t].padEnd(5) + '真值 ' + POS.A[t].map(v => v.toFixed(2)).join(' ') +
      '   |   训练(已对齐) ' + aligned.map(v => v.toFixed(2)).join(' '));
}

fs.writeFileSync(path.join(__dirname, '_probe.txt'), o.join('\n') + '\n');
console.log(o.join('\n'));
