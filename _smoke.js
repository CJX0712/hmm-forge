// 无头验证：从 index.html 抽出 <script id="engine">，在 Node vm 里跑断言
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

// ---------- 1) 内置 8 条不变量 ----------
H.selfTest().forEach((r, i) => ok(r.pass, 'selfTest#' + (i + 1) + ' ' + r.name, r.detail));

// ---------- 2) 随机压力测试：前向/后向/暴力/Viterbi/γ-ξ/置换 ----------
const rng = H.mulberry32(31337);
let bfChecked = 0;
for (let trial = 0; trial < 200; trial++) {
  const N = 2 + Math.floor(rng() * 3);          // 2..4
  const M = 2 + Math.floor(rng() * 4);          // 2..5
  const T = 1 + Math.floor(rng() * 9);          // 1..9
  const mm = H.randomModel(N, M, rng);
  const obs = H.sample(mm, T, rng).obs;

  const lf = H.logForward(mm, obs).loglik;
  const lb = H.logBackward(mm, obs).loglik;
  ok(close(lf, lb, 1e-9), `trial${trial} fwd==bwd`, `Δ=${Math.abs(lf - lb)}`);

  // 暴力枚举（限制规模）
  const paths = Math.pow(N, T);
  if (paths <= 60000) {
    bfChecked++;
    const bf = H.bruteForce(mm, obs);
    const naive = H.forward(mm, obs).lik;
    ok(close(naive, bf.lik, Math.abs(bf.lik) * 1e-12 + 1e-300), `trial${trial} fwd==brute`,
       `${naive} vs ${bf.lik}`);
    const v = H.viterbi(mm, obs);
    const rel = Math.abs(Math.exp(v.logprob) - bf.bestProb) / bf.bestProb;
    ok(rel < 1e-12, `trial${trial} viterbi==bruteProb`, `rel=${rel}`);
    const samePath = v.path.join(',') === bf.bestPath.join(',');
    const tie = samePath || Math.abs(H.manualPathLogProb(mm, obs, bf.bestPath) - v.logprob) < 1e-9;
    ok(tie, `trial${trial} viterbi==brutePath`, `${v.path} vs ${bf.bestPath}`);
    const man = H.manualPathLogProb(mm, obs, v.path);
    ok(close(man, v.logprob, 1e-9), `trial${trial} pathRescore`, `Δ=${Math.abs(man - v.logprob)}`);
  }

  // γ / ξ 一致性
  const P = H.posteriors(mm, obs);
  let eG = 0, eX = 0;
  for (let t = 0; t < T; t++) {
    let s = 0; for (let i = 0; i < N; i++) s += P.gamma[t][i];
    eG = Math.max(eG, Math.abs(s - 1));
    if (t < T - 1) for (let i = 0; i < N; i++) {
      let s2 = 0; for (let j = 0; j < N; j++) s2 += P.xi[t][i][j];
      eX = Math.max(eX, Math.abs(s2 - P.gamma[t][i]));
    }
  }
  ok(Math.max(eG, eX) < 1e-9, `trial${trial} gamma/xi`, `γ=${eG} ξ=${eX}`);

  // 标签置换不变
  const perm = Array.from({ length: N }, (_, i) => i).sort(() => rng() - 0.5);
  const mp = H.permuteModel(mm, perm);
  ok(close(H.logForward(mp, obs).loglik, lf, 1e-9), `trial${trial} permInvariant`,
     `Δ=${Math.abs(H.logForward(mp, obs).loglik - lf)}`);
}
ok(bfChecked > 100, 'bruteForce 交叉验证覆盖数', 'checked=' + bfChecked);

// ---------- 3) 边界条件 ----------
{
  const mm = H.randomModel(3, 4, H.mulberry32(5));
  ok(H.logForward(mm, []).loglik === 0, 'T=0 前向 loglik=0', String(H.logForward(mm, []).loglik));
  ok(H.logBackward(mm, []).loglik === 0, 'T=0 后向 loglik=0', String(H.logBackward(mm, []).loglik));
  ok(H.viterbi(mm, []).path.length === 0, 'T=0 Viterbi 路径空', '');
  ok(H.bruteForce(mm, []).lik === 1, 'T=0 暴力枚举 lik=1', String(H.bruteForce(mm, []).lik));

  const one = { N: 1, M: 2, pi: [1], A: [[1]], B: [[0.3, 0.7]] };
  const o1 = [0, 1, 1, 0];
  ok(close(H.logForward(one, o1).loglik, Math.log(0.3 * 0.7 * 0.7 * 0.3), 1e-12), 'N=1 解析解',
     String(H.logForward(one, o1).loglik));
  ok(H.viterbi(one, o1).path.join(',') === '0,0,0,0', 'N=1 Viterbi 全 0', '');

  // 含 0 概率的模型：不得 NaN / Infinity
  const zero = { N: 2, M: 2, pi: [1, 0], A: [[0.5, 0.5], [0, 1]], B: [[1, 0], [0, 1]] };
  const lz = H.logForward(zero, [0, 1]).loglik;
  ok(Number.isFinite(lz) && lz < 0, '零概率转移/发射不产生 NaN', String(lz));
  const vz = H.viterbi(zero, [0, 1]);
  ok(vz.path.every(Number.isInteger), '零概率模型 Viterbi 路径合法', vz.path.join(','));
}

// ---------- 4) 确定性：同 seed 同结果 ----------
{
  const a = H.sample(H.randomModel(3, 4, H.mulberry32(9)), 50, H.mulberry32(9));
  const b = H.sample(H.randomModel(3, 4, H.mulberry32(9)), 50, H.mulberry32(9));
  ok(a.obs.join(',') === b.obs.join(','), '同 seed 采样确定性', a.obs.slice(0, 8).join(','));
  const r1 = H.selfTest().map(r => r.detail).join('|'), r2 = H.selfTest().map(r => r.detail).join('|');
  ok(r1 === r2, 'selfTest 可复现', '');
}

// ---------- 5) Baum-Welch：单调 + 参数还原 ----------
for (let c = 0; c < 5; c++) {
  const r0 = H.mulberry32(1000 + c);
  const N = 2 + (c % 3), M = 3 + (c % 2);
  const truth = H.randomModel(N, M, r0);
  const corpus = H.sample(truth, 2000, r0).obs;
  const init = H.randomModel(N, M, r0);
  const bw = H.baumWelch(corpus, N, M, { iters: 40, model: init, floor: 1e-6 });
  let minD = Infinity;
  for (let i = 1; i < bw.loglikHist.length; i++) minD = Math.min(minD, bw.loglikHist[i] - bw.loglikHist[i - 1]);
  ok(minD > -1e-7, `BW#${c} 对数似然单调非减`, `minΔ=${minD}`);
  const trueLL = H.logForward(truth, corpus).loglik;
  const endLL = bw.loglikHist[bw.loglikHist.length - 1];
  // EM 从随机初值只保证收敛到驻点，不保证打到全局最优；要求相对真值 LL 的差距 < 1%
  ok(endLL >= trueLL - 0.01 * Math.abs(trueLL), `BW#${c} 训练后 LL 逼近真值 LL(<1%)`, `${endLL} vs ${trueLL}`);
  // 真值模型是 EM 不动点：从真值出发 1 步，LL 不得下降
  const one = H.baumWelch(corpus, N, M, { iters: 1, model: truth, floor: 1e-12 });
  ok(one.loglikHist[1] - one.loglikHist[0] > -1e-9, `BW#${c} 真值起点 1 步 EM 不下降`,
     `Δ=${one.loglikHist[1] - one.loglikHist[0]}`);
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
