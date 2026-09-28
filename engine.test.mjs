/* =====================================================================
 * hmm-forge / engine.test.mjs
 *
 * 无头测试 (headless test runner)。做两件事:
 *   1. 从 index.html 的 <script id="engine"> 块中把算法引擎**原样抽出**，
 *      在 Node 的 vm 沙箱里执行 —— 机械证明"引擎可以从 HTML 抽出为纯 JS 模块"，
 *      且引擎本身不依赖 document / window 等浏览器宿主对象。
 *   2. 逐条跑全部不变量用例，输出 PASS / FAIL 计数。
 *
 * 运行:
 *   node engine.test.mjs
 * 退出码 0 = 全绿，1 = 有失败。
 *
 * Author: 晨星 / Chenxing
 * ===================================================================== */
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const INDEX = join(here, 'index.html');
const ENGINE_SRC = join(here, '_build', 'engine.src.js');

/* ------------------------------------------------------------------ */
/* 用例容器                                                             */
/* ------------------------------------------------------------------ */
const results = [];
function check(id, name, fn) {
  let r;
  try {
    r = fn();
  } catch (e) {
    r = { pass: false, detail: 'EXCEPTION: ' + (e && e.stack ? e.stack.split('\n')[0] : String(e)) };
  }
  results.push({ id, name, pass: !!r.pass, detail: r.detail || '' });
}

/* ------------------------------------------------------------------ */
/* 步骤 1: 抽出引擎并在沙箱里执行                                        */
/* ------------------------------------------------------------------ */
if (!existsSync(INDEX)) {
  console.error('FATAL: 找不到 ' + INDEX);
  process.exit(1);
}
const html = readFileSync(INDEX, 'utf8');

const blockMatch = html.match(/<script\s+id="engine"\s*>([\s\S]*?)<\/script>/);
if (!blockMatch) {
  console.error('FATAL: index.html 中没有 <script id="engine"> 块');
  process.exit(1);
}
const engineSource = blockMatch[1];

const sandbox = { console };
vm.createContext(sandbox);
try {
  vm.runInContext(engineSource, sandbox, { filename: 'engine.from.html.js' });
} catch (e) {
  console.error('FATAL: 引擎在 Node 沙箱中执行失败: ' + (e && e.message));
  process.exit(1);
}
const HMM = sandbox.HMM;

/* ------------------------------------------------------------------ */
/* 步骤 2: 结构 / 纯度用例                                              */
/* ------------------------------------------------------------------ */

check('N1', '引擎可从 index.html 抽出并在 Node 无头执行 / engine extractable', () => {
  const ok = !!HMM && typeof HMM.forward === 'function' && typeof HMM.runSelfTests === 'function';
  return {
    pass: ok,
    detail: '抽出 ' + engineSource.trim().split('\n').length + ' 行源码；导出符号 ' +
            (HMM ? Object.keys(HMM).length : 0) + ' 个；version = ' + (HMM ? HMM.version : 'N/A')
  };
});

check('N2', '引擎零宿主依赖 / no document / window / fetch in engine', () => {
  const forbidden = [/\bdocument\s*\./, /\bwindow\s*\./, /\bfetch\s*\(/, /XMLHttpRequest/,
                     /require\s*\(/, /import\s+/, /https?:\/\//];
  const hits = forbidden.filter((re) => re.test(engineSource)).map((re) => String(re));
  return {
    pass: hits.length === 0,
    detail: hits.length ? '命中禁用模式: ' + hits.join(', ') : 'document/window/fetch/require/import/URL 全部为 0 命中'
  };
});

check('N3', 'index.html 零外部资源 / no CDN / font / link / external URL', () => {
  const forbidden = [/<link\b/i, /@import/i, /https?:\/\//i, /\bfetch\s*\(/, /XMLHttpRequest/,
                     /<script[^>]+src=/i, /<img[^>]+src=/i, /fonts\.googleapis/i];
  const hits = forbidden.filter((re) => re.test(html)).map((re) => String(re));
  return {
    pass: hits.length === 0,
    detail: hits.length ? '命中: ' + hits.join(', ')
                        : '<link> / @import / http(s):// / fetch / 外部 script / 外部 img 全部为 0 命中'
  };
});

check('N4', '页面必需交互元素齐备 / required UI elements present', () => {
  const need = ['id="trellis"', 'id="curve"', 'id="cases"', 'id="btnTest"', 'id="btnTrain"',
                'id="tblA"', 'id="tblB"', 'id="tblPi"', 'id="seedshow"', 'id="inObs"'];
  const missing = need.filter((k) => !html.includes(k));
  return {
    pass: missing.length === 0,
    detail: missing.length ? '缺失: ' + missing.join(', ') : '10/10 个元素齐备 (trellis / curve / 自检面板 / 训练按钮 / A,B,π 表 / seed / 序列输入)'
  };
});

check('N5', '内联引擎与 _build/engine.src.js 无漂移 / no source drift', () => {
  if (!existsSync(ENGINE_SRC)) {
    return { pass: true, detail: '（_build 未随仓库分发，跳过）' };
  }
  const src = readFileSync(ENGINE_SRC, 'utf8').trim();
  const same = src === engineSource.trim();
  return {
    pass: same,
    detail: same ? '两份源码逐字节一致 (' + src.length + ' chars)'
                 : 'DRIFT! 内联块与独立模块不一致'
  };
});

/* ------------------------------------------------------------------ */
/* 步骤 3: 引擎内自带的不变量用例（浏览器面板与此处共用同一套）           */
/* ------------------------------------------------------------------ */
const engineResults = HMM.runSelfTests();
for (const r of engineResults) results.push(r);

/* ------------------------------------------------------------------ */
/* 步骤 4: Node 侧补充用例（直接用 API 交叉验证）                        */
/* ------------------------------------------------------------------ */

check('N6', 'API 级交叉验证: forward ≡ backward ≡ scaled-forward (随机 5 组)', () => {
  let worst = 0, sample = '';
  for (let s = 0; s < 5; s++) {
    const seed = 1000 + s * 37;
    const m = HMM.randomModel(2 + (s % 4), 2 + (s % 3), HMM.mulberry32(seed));
    const o = HMM.sampleSequence(m, 30 + s * 11, HMM.mulberry32(seed + 1)).observations;
    const f = HMM.forward(m, o).logLikelihood;
    const b = HMM.backward(m, o).logLikelihood;
    const fs = HMM.forwardScaled(m, o).logLikelihood;
    const e = Math.max(HMM.relErr(f, b), HMM.relErr(f, fs));
    if (e > worst) { worst = e; sample = 'N=' + m.N + ' M=' + m.M + ' T=' + o.length + ' LL=' + HMM.fmt(f); }
  }
  return {
    pass: worst < 1e-10,
    detail: '5 组随机模型 worst relerr = ' + HMM.fmt(worst) + ' | 样本: ' + sample
  };
});

check('N7', 'API 级交叉验证: Viterbi ≡ brute-force argmax (T=5..7, N=2..3)', () => {
  let worstProb = 0, allPaths = true, cases = 0;
  for (let N = 2; N <= 3; N++) {
    for (let T = 5; T <= 7; T++) {
      const m = HMM.randomModel(N, 3, HMM.mulberry32(N * 100 + T));
      const o = HMM.sampleSequence(m, T, HMM.mulberry32(N * 100 + T + 1)).observations;
      const v = HMM.viterbi(m, o);
      const bf = HMM.bruteForce(m, o);
      worstProb = Math.max(worstProb, HMM.relErr(v.logProb, bf.bestLogProb));
      if (v.path.join(',') !== bf.bestPath.join(',')) allPaths = false;
      cases++;
    }
  }
  return {
    pass: worstProb < 1e-10 && allPaths,
    detail: cases + ' 组 (N=2..3, T=5..7) | worst relerr = ' + HMM.fmt(worstProb) +
            ' | 路径全部一致 = ' + (allPaths ? 'YES' : 'NO')
  };
});

check('N8', 'API 级: 极端长度 T=500 不产生 -Infinity / NaN', () => {
  const m = HMM.randomModel(6, 5, HMM.mulberry32(555));
  const o = HMM.sampleSequence(m, 500, HMM.mulberry32(556)).observations;
  const f = HMM.forward(m, o), b = HMM.backward(m, o), v = HMM.viterbi(m, o);
  const g = HMM.posteriorGamma(m, o);
  const vals = [f.logLikelihood, b.logLikelihood, v.logProb, g.logLikelihood];
  let bad = 0, total = 0;
  const scan = (arr) => { for (const x of arr) { total++; if (!Number.isFinite(x)) bad++; } };
  for (let t = 0; t < o.length; t++) { scan(f.logAlpha[t]); scan(g.gamma[t]); }
  scan(vals);
  return {
    pass: bad === 0 && Number.isFinite(f.logLikelihood),
    detail: 'T=500, N=6 | 扫描 ' + total + ' 个数 | 非有限 = ' + bad +
            ' | forward LL = ' + HMM.fmt(f.logLikelihood) +
            ' | relerr(fwd,bwd) = ' + HMM.fmt(HMM.relErr(f.logLikelihood, b.logLikelihood))
  };
});

/* ------------------------------------------------------------------ */
/* 汇总输出                                                            */
/* ------------------------------------------------------------------ */
const pass = results.filter((r) => r.pass).length;
const fail = results.length - pass;

const line = '='.repeat(78);
console.log(line);
console.log('  HMM Forge · engine.test.mjs · headless invariant suite');
console.log('  engine v' + (HMM ? HMM.version : '?') + '  ·  author: ' + (HMM ? HMM.author : '?'));
console.log(line);
for (const r of results) {
  console.log((r.pass ? 'PASS' : 'FAIL') + '  [' + r.id + ']  ' + r.name);
  console.log('        ' + r.detail);
}
console.log(line);
console.log('  TOTAL = ' + results.length + '   PASS = ' + pass + '   FAIL = ' + fail);
console.log('  RESULT: ' + (fail === 0 ? 'ALL GREEN ✅' : 'HAS FAILURES ❌'));
console.log(line);

process.exit(fail === 0 ? 0 : 1);
