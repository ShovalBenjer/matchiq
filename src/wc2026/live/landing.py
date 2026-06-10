"""Premium landing hub (docs/index.html) — the deployed site root.

Static shell that fetches ``./live/data.json`` and ``./futures.json`` at runtime
to show live teasers (next kick-off, title favourite, Golden Boot), then routes
to the live dashboard, the futures page, and the research dossier. Reuses the
shared 2026 design system so the whole site feels like one product.
"""

from __future__ import annotations

from wc2026.live.theme import CSS, FONTS, JS_HELPERS


def render_landing() -> str:
    return (_HTML.replace("/*__CSS__*/", CSS).replace("<!--FONTS-->", FONTS)
                 .replace("/*__HELPERS__*/", JS_HELPERS))


_HTML = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>matchiq — World Cup 2026 model + market intelligence</title>
<!--FONTS-->
<style>/*__CSS__*/
.hero-title{font-size:clamp(34px,6vw,62px);font-weight:700;line-height:1.02;margin:30px 0 8px;
  background:linear-gradient(120deg,var(--ink),var(--accent) 55%,var(--accent2));
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.lead{color:var(--muted);font-size:clamp(15px,2vw,19px);max-width:680px}
.enter{padding:20px;display:flex;flex-direction:column;gap:8px;cursor:pointer;
  transition:transform .3s var(--spring),border-color .3s}
.enter:hover{transform:translateY(-4px)}
.enter .arrow{color:var(--accent);font-weight:700}
.enter h3{margin:0;font-size:18px} .enter p{margin:0;color:var(--muted);font-size:13.5px}
</style></head><body>
<div class="aurora"><i></i><i></i><i></i></div>
<header class="top"><div class="wrap topbar">
  <div class="brand">⚽ matchiq</div><div class="spacer"></div>
  <a class="chip" href="./live/">Live</a><a class="chip" href="./futures.html">Futures</a>
  <a class="chip" href="./RESEARCH.md">Research</a>
</div></header>
<main class="wrap">
  <h1 class="hero-title">World Cup 2026,<br>model meets the market.</h1>
  <p class="lead">Real fixtures, odds and standings (ESPN · DraftKings) fused with prediction-market
  crowd wisdom (Polymarket), an evidence-based model, and the golden statistical edges — champion's
  curse, ageing squads, altitude — laid bare. Honest by design: where the book and the crowd agree,
  we say so.</p>

  <section class="bento stagger" id="hero" style="margin-top:24px"></section>

  <div class="grid2" style="margin-top:18px">
    <a class="glass enter" href="./live/">
      <div class="arrow">Live →</div><h3>Schedule, odds &amp; groups</h3>
      <p>Every fixture by date with de-vigged book lines, crowd prices, value flags, live group tables and knockout seeds. Synced daily.</p></a>
    <a class="glass enter" href="./futures.html">
      <div class="arrow">Futures →</div><h3>Winner · top scorer · exact scores</h3>
      <p>Tournament winner and Golden Boot (model vs crowd vs blended), plus Dixon-Coles exact-score predictions for group fixtures.</p></a>
    <a class="glass enter" href="./RESEARCH.md">
      <div class="arrow">Dossier →</div><h3>The golden correlations</h3>
      <p>Cited research: the 4-of-6 holders' curse, ageing-squad decline, crowd-wisdom calibration, and altitude/heat/travel effects.</p></a>
  </div>
</main>
<footer><div id="disc">Educational, market-anchored — not financial advice. Bet responsibly.</div></footer>
<script>
/*__HELPERS__*/
fetch('./live/data.json').then(r=>r.json()).then(D=>{
  const fx=(D.fixtures||[]).filter(f=>f.status==='pre');const n=fx[0]||(D.fixtures||[])[0];
  const fav=(D.winner||[])[0];const boot=(D.top_scorer||[])[0];
  const h=document.getElementById('hero');
  if(n)h.append(el('div',{class:'glass tile big grad'},
    `<div class="k">Next kick-off</div><div class="v">${flag(n.home)}${n.home} <span class=muted>v</span> ${flag(n.away)}${n.away}</div><div class="s" id="cd">—</div>`));
  if(fav)h.append(el('div',{class:'glass tile grad'},`<div class="k">Title favourite</div><div class="v">${pct(fav.prob)}</div><div class="s">${flag(fav.team)}${fav.team}</div>`));
  if(boot)h.append(el('div',{class:'glass tile grad'},`<div class="k">Golden Boot</div><div class="v" style="font-size:22px">${boot.player}</div><div class="s">${pct(boot.prob)} crowd</div>`));
  h.append(el('div',{class:'glass tile wide'},`<div class="k">Sources</div><div class="v" style="font-size:20px">ESPN · DraftKings · Polymarket</div><div class="s">Synced ${new Date(D.generated_at).toUTCString().slice(5,17)}</div>`));
  if(n){const t=new Date(n.date).getTime();(function u(){const d=t-Date.now();const e=document.getElementById('cd');if(!e)return;
    if(d<=0){e.textContent='Underway';return;}const H=Math.floor(d/3.6e6),M=Math.floor(d%3.6e6/6e4);e.textContent='in '+(H?H+'h ':'')+M+'m · '+new Date(n.date).toUTCString().slice(0,16);setTimeout(u,30000);})();}
}).catch(()=>{document.getElementById('hero').innerHTML='<div class="glass tile wide"><div class="k">Live data</div><div class="v" style="font-size:18px">Open via the deployed site to load live data</div></div>';});
</script></body></html>
"""
