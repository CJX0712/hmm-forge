/* --------------------------------------------------------------------
 * hmm-forge / build.mjs
 * 把 _build/engine.src.js 原样内联进 _build/shell.html 的
 * <script id="engine"> 块，生成单文件 index.html。
 *
 *   node _build/build.mjs
 *
 * 引擎源码在仓库里有两份存在形式：
 *   (1) _build/engine.src.js   —— 独立可复用的纯 JS 模块
 *   (2) index.html 内联块       —— 发布形态（零外部依赖）
 * 本脚本保证二者逐字节一致（否则报错），避免"两份源码漂移"。
 *
 * Author: 晨星 / Chenxing
 * ------------------------------------------------------------------ */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const engine = readFileSync(join(here, 'engine.src.js'), 'utf8');
const shell = readFileSync(join(here, 'shell.html'), 'utf8');

const MARK = '/*__ENGINE_INLINE__*/';
if (!shell.includes(MARK)) throw new Error('shell.html 缺少引擎插入标记 ' + MARK);
if (engine.includes('</script')) throw new Error('引擎源码含 </script，会截断 HTML 脚本块');

const out = shell.replace(MARK, () => engine.trim());
writeFileSync(join(root, 'index.html'), out, 'utf8');

// 反向校验：从生成的 index.html 抽出引擎块，必须与源文件逐字节一致
const m = out.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m) throw new Error('index.html 中找不到 <script id="engine"> 块');
const extracted = m[1].trim();
if (extracted !== engine.trim()) {
  throw new Error('内联引擎与 _build/engine.src.js 不一致（漂移）');
}

const kb = (Buffer.byteLength(out, 'utf8') / 1024).toFixed(1);
console.log('[build] index.html written  (' + kb + ' KB)');
console.log('[build] engine block: ' + engine.trim().split('\n').length + ' lines, byte-identical to _build/engine.src.js');
