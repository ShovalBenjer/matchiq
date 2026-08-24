"""Shared 2026 design system (OKLCH tokens, neo-glass, motion) for the static
pages. Dependency-free: emitted inline so the pages stay single-file and deploy
straight to GitHub Pages. Encapsulates the look so the live dashboard, the
futures page, and the landing hub feel like one product.
"""

from __future__ import annotations

# Google Fonts (progressive enhancement; system stack fallback if blocked).
FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&'
    'family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
)

# Core design tokens + neo-glass + motion. OKLCH palette, perceptually uniform.
CSS = r"""
:root{
  --bg:oklch(0.15 0.025 270); --bg2:oklch(0.19 0.035 285);
  --ink:oklch(0.97 0.01 260); --muted:oklch(0.72 0.02 268);
  --line:oklch(0.72 0.03 270 / .16);
  --glass:oklch(0.27 0.035 270 / .45); --glass2:oklch(0.30 0.04 272 / .65);
  --accent:oklch(0.74 0.15 245); --accent2:oklch(0.76 0.18 330);
  --good:oklch(0.80 0.17 150); --warn:oklch(0.84 0.15 85); --bad:oklch(0.68 0.21 25);
  --r:18px; --spring:cubic-bezier(.22,1,.36,1);
  --shadow:0 1px 0 oklch(1 0 0 / .06) inset, 0 14px 40px -18px oklch(0 0 0 / .7);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 Inter,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;overflow-x:hidden;min-height:100vh}
h1,h2,h3,.brand{font-family:"Space Grotesk",Inter,sans-serif;letter-spacing:-.01em}
a{color:inherit;text-decoration:none}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:oklch(0.3 0.03 270 / .5);
  padding:1px 6px;border-radius:6px;font-size:.85em}
/* aurora background */
.aurora{position:fixed;inset:-20% -10% auto -10%;height:120vh;z-index:-1;filter:blur(70px);
  opacity:.55;pointer-events:none}
.aurora i{position:absolute;display:block;border-radius:50%;mix-blend-mode:screen;
  animation:drift 22s var(--spring) infinite alternate}
.aurora i:nth-child(1){width:46vw;height:46vw;left:-6vw;top:-8vw;
  background:radial-gradient(circle,oklch(0.6 0.2 270),transparent 60%)}
.aurora i:nth-child(2){width:42vw;height:42vw;right:-8vw;top:2vw;animation-delay:-7s;
  background:radial-gradient(circle,oklch(0.62 0.2 330),transparent 60%)}
.aurora i:nth-child(3){width:40vw;height:40vw;left:30vw;top:30vh;animation-delay:-13s;
  background:radial-gradient(circle,oklch(0.6 0.16 210),transparent 60%)}
@keyframes drift{to{transform:translate3d(6vw,5vh,0) scale(1.15)}}
/* layout */
.wrap{max-width:1200px;margin:0 auto;padding:0 22px}
header.top{position:sticky;top:0;z-index:50;backdrop-filter:blur(16px) saturate(150%);
  background:oklch(0.16 0.03 270 / .55);border-bottom:1px solid var(--line)}
.topbar{display:flex;align-items:center;gap:14px;padding:14px 0}
.brand{font-size:19px;font-weight:700;display:flex;align-items:center;gap:9px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--bad);box-shadow:0 0 0 0 var(--bad);
  animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 oklch(0.68 0.21 25 / .6)}70%{box-shadow:0 0 0 9px transparent}100%{box-shadow:0 0 0 0 transparent}}
.spacer{flex:1}
.chip{font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:999px;
  padding:5px 11px;background:var(--glass);white-space:nowrap}
/* glass card */
.glass{background:linear-gradient(180deg,var(--glass2),var(--glass));
  border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);
  backdrop-filter:blur(18px) saturate(140%);overflow:hidden}
.glass:hover{border-color:oklch(0.74 0.05 270 / .3)}
.card-h{display:flex;align-items:center;justify-content:space-between;
  padding:12px 16px;border-bottom:1px solid var(--line);color:var(--muted);
  font-size:12.5px;font-weight:600;text-transform:uppercase;letter-spacing:.06em}
/* bento hero */
.bento{display:grid;gap:14px;margin:18px 0;
  grid-template-columns:repeat(4,1fr);grid-auto-rows:minmax(96px,auto)}
.bento .tile{padding:16px 18px}
.tile.big{grid-column:span 2;grid-row:span 2}
.tile.wide{grid-column:span 2}
.tile .k{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.08em}
.tile .v{font-family:"Space Grotesk";font-size:30px;font-weight:700;margin-top:4px;line-height:1.05}
.tile .s{color:var(--muted);font-size:13px;margin-top:6px}
.grad{background:linear-gradient(135deg,oklch(0.74 0.15 245 / .18),oklch(0.76 0.18 330 / .14))}
/* nav pills */
nav.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 16px}
.tab{font:inherit;font-size:14px;color:var(--muted);background:var(--glass);
  border:1px solid var(--line);border-radius:999px;padding:9px 15px;cursor:pointer;
  transition:transform .25s var(--spring),color .2s,border-color .2s}
.tab:hover{transform:translateY(-2px);color:var(--ink)}
.tab.active{color:var(--ink);border-color:transparent;
  background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 8px 24px -12px var(--accent)}
/* tables */
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:9px 14px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
tbody tr{transition:background .2s}
tbody tr:hover{background:oklch(0.7 0.05 270 / .06)}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.team{font-weight:600}
.flag{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:6px;margin-right:8px;
  font-size:11px;font-weight:700;color:oklch(0.18 0.03 270);
  background:linear-gradient(135deg,var(--accent),var(--accent2));vertical-align:middle}
/* probability bar */
.bar{height:7px;border-radius:6px;background:oklch(0.4 0.03 270 / .35);overflow:hidden;min-width:64px}
.bar>i{display:block;height:100%;width:0;border-radius:6px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .9s var(--spring)}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
.pill.value{color:var(--good);border-color:oklch(0.8 0.17 150 / .5);background:oklch(0.8 0.17 150 / .1)}
.pill.live{color:var(--bad);border-color:oklch(0.68 0.21 25 / .5);font-weight:700}
.muted{color:var(--muted)} .good{color:var(--good)} .bad{color:var(--bad)}
.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.datehead{margin:18px 2px 8px;font:600 13px "Space Grotesk";color:var(--accent);
  display:flex;align-items:center;gap:8px}
.datehead::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--accent)}
.hidden{display:none}
section.view{animation:rise .5s var(--spring) both}
.stagger>*{animation:rise .5s var(--spring) both}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
footer{color:var(--muted);font-size:12px;padding:26px 0 48px;text-align:center}
.crosslinks{display:flex;gap:10px;justify-content:center;margin-top:10px;flex-wrap:wrap}
.crosslinks a{font-size:13px;color:var(--accent);border:1px solid var(--line);
  border-radius:999px;padding:6px 13px;transition:transform .25s var(--spring)}
.crosslinks a:hover{transform:translateY(-2px)}
::-webkit-scrollbar{height:10px;width:10px}
::-webkit-scrollbar-thumb{background:oklch(0.4 0.03 270 / .5);border-radius:8px}
::view-transition-old(root),::view-transition-new(root){animation-duration:.35s}
@media (max-width:760px){.bento{grid-template-columns:repeat(2,1fr)}.tile.big{grid-column:span 2}}
@media (prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important;scroll-behavior:auto!important}
  .bar>i{width:var(--w)!important}
}
"""

# Shared JS helpers: count-up, animated bars, View-Transition tab switch.
JS_HELPERS = r"""
const PRM=matchMedia('(prefers-reduced-motion:reduce)').matches;
const pct=x=>x==null?'—':(100*x).toFixed(1)+'%';
const el=(t,a={},h)=>{const e=document.createElement(t);for(const k in a){if(k==='class')e.className=a[k];else e.setAttribute(k,a[k]);}if(h!=null)e.innerHTML=h;return e;};
const flag=name=>{const ab=(name||'?').replace(/[^A-Za-z ]/g,'').split(' ').map(w=>w[0]).join('').slice(0,3).toUpperCase();return `<span class="flag">${ab||'?'}</span>`;};
function fillBars(root){(root||document).querySelectorAll('.bar>i[data-w]').forEach(i=>{
  const w=i.getAttribute('data-w');if(PRM){i.style.width=w;}else{requestAnimationFrame(()=>{i.style.setProperty('--w',w);i.style.width=w;});}});}
function countUp(node){const to=parseFloat(node.dataset.to);if(PRM||isNaN(to)){node.textContent=node.dataset.fmt.replace('#',to);return;}
  const dur=900,t0=performance.now();const fmt=node.dataset.fmt;
  function step(t){const k=Math.min(1,(t-t0)/dur);const e=1-Math.pow(1-k,3);
    node.textContent=fmt.replace('#',(to*e).toFixed(node.dataset.dp||0));if(k<1)requestAnimationFrame(step);}requestAnimationFrame(step);}
function switchView(name,btn){
  const go=()=>{document.querySelectorAll('nav.tabs .tab').forEach(b=>b.classList.toggle('active',b===btn));
    document.querySelectorAll('section.view').forEach(s=>s.classList.toggle('hidden',s.id!=='v-'+name));
    fillBars(document.getElementById('v-'+name));};
  if(!PRM&&document.startViewTransition)document.startViewTransition(go);else go();}
"""
