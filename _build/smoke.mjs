/* --------------------------------------------------------------------
 * hmm-forge / _build/smoke.mjs
 *
 * 冒烟测试: 用一个极简 DOM shim 在 Node 里执行 index.html 的**界面层脚本**，
 * 确认页面启动路径（建表 → 随机模型 → 采样序列 → 画 trellis → 画曲线 →
 * 跑自检面板 → 单步 EM → 训练）不会抛异常，且真的产出了 SVG / 结果 DOM。
 *
 * 这不是视觉回归测试，只验证"代码跑得通、关键节点被真实写入"。
 *
 *   node _build/smoke.mjs
 *
 * Author: 晨星 / Chenxing
 * ------------------------------------------------------------------ */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, '..', 'index.html'), 'utf8');

const engineBlock = html.match(/<script\s+id="engine"\s*>([\s\S]*?)<\/script>/);
const uiBlock = html.match(/<script\s+id="engine"\s*>[\s\S]*?<\/script>\s*<script>([\s\S]*?)<\/script>/);
if (!engineBlock || !uiBlock) throw new Error('无法从 index.html 抽出脚本块');

/* ------------------------- 极简 DOM shim ------------------------- */
const all = [];
class El {
  constructor(tag) {
    this.tagName = String(tag).toLowerCase();
    this.children = []; this.attrs = {}; this._text = ''; this._html = '';
    this.value = ''; this.className = ''; this.style = {};
    this.listeners = {};
    all.push(this);
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); }
  dispatch(ev) { (this.listeners[ev] || []).forEach((f) => f.call(this, {})); }
  set textContent(v) { this._text = String(v); }
  get textContent() { return this._text; }
  set innerHTML(v) { this._html = String(v); if (v === '') this.children = []; }
  get innerHTML() { return this._html; }
}

/* 把 HTML 里 <input id=... value=...> 的默认值灌进 shim，让冒烟更贴近真实浏览器 */
const defaults = new Map();
for (const tag of html.match(/<input\b[^>]*>/g) || []) {
  const id = (tag.match(/id="([^"]+)"/) || [])[1];
  const val = (tag.match(/value="([^"]*)"/) || [])[1];
  if (id && val != null) defaults.set(id, val);
}

const byId = new Map();
const document = {
  createElement: (t) => new El(t),
  getElementById: (id) => {
    if (!byId.has(id)) {
      const e = new El('div');
      e.attrs.id = id;
      if (defaults.has(id)) e.value = defaults.get(id);
      byId.set(id, e);
    }
    return byId.get(id);
  },
  querySelectorAll: (sel) => {
    if (sel === 'input[data-sec]') return all.filter((e) => e.tagName === 'input' && 'data-sec' in e.attrs);
    return [];
  },
  activeElement: null
};

const timers = [];
const sandbox = {
  console, document, Math, Number, Date, Array, Object, String, JSON, isFinite, parseFloat, parseInt,
  alert: (m) => { throw new Error('unexpected alert(): ' + m); },
  setTimeout: (fn) => { timers.push(fn); return timers.length; },
  clearTimeout: () => {}
};
// 引擎把 HMM 挂到 globalThis（= 沙箱全局对象），界面层读 window.HMM，这里二者同源。
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(engineBlock[1], sandbox, { filename: 'engine.js' });
vm.runInContext(uiBlock[1], sandbox, { filename: 'ui.js' });

/* 启动后 UI 用 setTimeout 挂了自检，这里同步跑掉 */
while (timers.length) {
  const fn = timers.shift();
  fn();
}

/* ------------------------------ 断言 ------------------------------ */
const checks = [];
function ck(name, cond, info) { checks.push({ name, pass: !!cond, info: info || '' }); }

const trellis = byId.get('trellis');
const curve = byId.get('curve');
const cases = byId.get('cases');
const summary = byId.get('summary');

ck('引擎已挂载 window.HMM', !!sandbox.HMM, 'v' + (sandbox.HMM && sandbox.HMM.version));
ck('seed 已显示', byId.get('seedshow') && byId.get('seedshow').textContent !== '', 'seed = ' + (byId.get('seedshow') || {}).textContent);
ck('π/A/B 输入框已创建', all.filter((e) => 'data-sec' in e.attrs).length === 4 + 4 * 4 + 4 * 3,
   'inputs = ' + all.filter((e) => 'data-sec' in e.attrs).length + ' (π4 + A16 + B12 = 32)');
ck('trellis SVG 已绘制', trellis && trellis.innerHTML.includes('<circle') && trellis.innerHTML.includes('polyline'),
   'len = ' + (trellis ? trellis.innerHTML.length : 0) + ' chars');
ck('trellis 含 Viterbi 路径高亮', trellis && trellis.innerHTML.includes('#f0abfc'));
ck('curve SVG 初始显示占位提示', curve && curve.innerHTML.includes('点击'), 'len = ' + (curve ? curve.innerHTML.length : 0));
ck('观测序列已回填输入框', byId.get('inObs') && byId.get('inObs').value.length > 0, 'O = ' + (byId.get('inObs') || {}).value);
ck('自检面板已渲染结果', cases && cases.innerHTML.includes('class="case'), 'len = ' + (cases ? cases.innerHTML.length : 0));
ck('自检全部通过', summary && summary.textContent.includes('全部通过'), (summary || {}).textContent);
ck('归一化统计已输出', byId.get('normStat') && byId.get('normStat').innerHTML.includes('kv'));
ck('路径统计已输出', byId.get('pathStat') && byId.get('pathStat').innerHTML.includes('logProb'));

/* 触发交互: 单步 EM / 训练 / 重新随机 / 采样 / 解析序列 */
function fire(id) { const e = byId.get(id); if (e) e.dispatch('click'); }
try {
  fire('btnStep');
  fire('btnSample');
  byId.get('inObs').value = 'A B C A';
  fire('btnParse');
  fire('btnRandom');
  fire('btnTrain');
  while (timers.length) { const fn = timers.shift(); fn(); }
  ck('交互链无异常 (step/sample/parse/random/train)', true, '5 个按钮全部执行完毕');
} catch (e) {
  ck('交互链无异常 (step/sample/parse/random/train)', false, e.message);
}
ck('训练后曲线有多个点', curve && (curve.innerHTML.match(/<circle/g) || []).length > 3,
   'circles = ' + ((curve ? curve.innerHTML : '').match(/<circle/g) || []).length);
ck('训练后曲线含折线 polyline', curve && curve.innerHTML.includes('<polyline'));
ck('训练后 bwStat 含单调性判定', byId.get('bwStat') && byId.get('bwStat').innerHTML.includes('单调性'),
   (byId.get('bwStat') || {}).innerHTML.replace(/<[^>]+>/g, ' ').slice(0, 160));

/* ------------------------------ 输出 ------------------------------ */
let pass = 0;
for (const c of checks) {
  console.log((c.pass ? 'PASS' : 'FAIL') + '  ' + c.name + (c.info ? '  →  ' + c.info : ''));
  if (c.pass) pass++;
}
console.log('-'.repeat(70));
console.log('SMOKE: ' + pass + ' / ' + checks.length + ' PASS');
process.exit(pass === checks.length ? 0 : 1);
