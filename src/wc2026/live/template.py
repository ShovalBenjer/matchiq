"""Self-contained dashboard renderer for the live data."""

from __future__ import annotations

import json


def render(data: dict) -> str:
    return _HTML.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>matchiq · World Cup 2026 live</title>
<style>
 :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--muted:#8b949e;
   --accent:#58a6ff;--good:#3fb950;--warn:#d29922;--bad:#f85149}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
 header{max-width:1180px;margin:0 auto;padding:22px 20px 6px}
 h1{margin:0;font-size:23px} .sub{color:var(--muted);font-size:13px;margin-top:2px}
 .banner{max-width:1180px;margin:10px auto;padding:9px 14px;background:#1c2128;
   border:1px solid var(--line);border-radius:8px;color:var(--muted);font-size:12.5px}
 nav{max-width:1180px;margin:14px auto 0;display:flex;gap:8px;flex-wrap:wrap;padding:0 20px}
 nav button{background:var(--panel);color:var(--fg);border:1px solid var(--line);
   padding:8px 13px;border-radius:8px;cursor:pointer;font-size:14px}
 nav button.active{border-color:var(--accent);color:var(--accent)}
 main{max-width:1180px;margin:14px auto 60px;padding:0 20px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
   overflow:hidden;margin-bottom:16px}
 .card h3{margin:0;padding:10px 14px;font-size:14px;border-bottom:1px solid var(--line);color:var(--muted)}
 table{width:100%;border-collapse:collapse;font-size:13.5px}
 th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line)}
 th{color:var(--muted);font-weight:600} tr:last-child td{border-bottom:none}
 td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
 .pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:12px;border:1px solid var(--line)}
 .value{color:var(--good);border-color:var(--good)} .muted{color:var(--muted)}
 .live{color:var(--bad);font-weight:700} .grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
 .hidden{display:none} code{background:#21262d;padding:1px 5px;border-radius:4px}
 .bar{height:6px;background:#21262d;border-radius:4px;overflow:hidden;min-width:60px}
 .bar>i{display:block;height:100%;background:var(--accent)}
 footer{max-width:1180px;margin:0 auto 40px;padding:0 20px;color:var(--muted);font-size:12px}
 .date{margin:14px 0 6px;color:var(--accent);font-size:13px;font-weight:600}
</style></head><body>
<header><h1>⚽ matchiq — World Cup 2026 <span class="live" id="livedot"></span></h1>
<div class="sub">Live fixtures · group tables · winner & top scorer · book vs crowd value. Synced <span id="gen"></span>.</div></header>
<div class="banner" id="banner"></div>
<nav>
 <button data-tab="fixtures" class="active">Next fixtures</button>
 <button data-tab="groups">Groups</button>
 <button data-tab="ko">Knockout seeds</button>
 <button data-tab="winner">Winner</button>
 <button data-tab="scorer">Top scorer</button>
</nav>
<main>
 <section id="tab-fixtures"></section>
 <section id="tab-groups" class="hidden"></section>
 <section id="tab-ko" class="hidden"></section>
 <section id="tab-winner" class="card hidden"></section>
 <section id="tab-scorer" class="card hidden"></section>
</main>
<footer id="foot"></footer>
<script>
const D=/*__DATA__*/;
const pct=x=>x==null?'—':(100*x).toFixed(1)+'%';
const el=(t,a={},h='')=>{const e=document.createElement(t);for(const k in a)e.setAttribute(k,a[k]);if(h)e.innerHTML=h;return e;};
document.getElementById('gen').textContent=D.generated_at;
document.getElementById('foot').textContent=D.disclaimer;
document.getElementById('banner').innerHTML=
 `Sources: results & odds <code>${D.sources.results_odds}</code> · crowd <code>${D.sources.crowd}</code>. `+
 `Fixture probabilities are de-vigged book lines; <b>value</b> fires when the crowd's fair price beats the book.`;

// ---- fixtures by date ----
function fixtures(){
 const s=document.getElementById('tab-fixtures');s.innerHTML='';
 const byDate={};(D.fixtures||[]).forEach(f=>{const d=(f.date||'').slice(0,10);(byDate[d]=byDate[d]||[]).push(f);});
 Object.keys(byDate).sort().forEach(d=>{
  s.append(el('div',{class:'date'},d||'TBD'));
  const card=el('div',{class:'card'});const t=el('table');
  t.innerHTML=`<thead><tr><th>Match</th><th>Status</th><th class=num>Home</th><th class=num>Draw</th>
   <th class=num>Away</th><th class=num>Crowd</th><th class=num>O/U</th><th>Value</th></tr></thead>`;
  const tb=el('tbody');
  byDate[d].forEach(f=>{
   const b=f.book||{},c=f.crowd;
   const crowd=c?`${pct(c.home)}/${pct(c.draw)}/${pct(c.away)}`:'—';
   let val='<span class=muted>—</span>';
   if(f.value&&f.value.is_value)val=`<span class="pill value">${f.value.outcome} +${pct(f.value.ev)} @${f.value.offered_odds}</span>`;
   else if(f.value)val=`<span class=muted>${f.value.outcome} ${(100*f.value.ev).toFixed(1)}%</span>`;
   const st=f.status==='in'?'<span class=live>LIVE</span>':(f.status==='post'?(f.score||'FT'):(f.date||'').slice(11,16));
   tb.append(el('tr',{}, `<td><b>${f.home}</b> v <b>${f.away}</b>${f.details?` <span class=muted>(${f.details})</span>`:''}</td>
    <td>${st}</td><td class=num>${pct(b.home)}</td><td class=num>${pct(b.draw)}</td><td class=num>${pct(b.away)}</td>
    <td class=num>${crowd}</td><td class=num>${f.over_under??'—'}</td><td>${val}</td>`));
  });
  t.append(tb);card.append(t);s.append(card);
 });
 if(!(D.fixtures||[]).length)s.innerHTML='<div class=card><h3>No fixtures in window</h3></div>';
}
// ---- groups ----
function groups(){
 const s=document.getElementById('tab-groups');s.innerHTML='';
 const g=el('div',{class:'grid2'});
 Object.keys(D.groups||{}).sort().forEach(name=>{
  const card=el('div',{class:'card'});card.append(el('h3',{},name));
  const t=el('table');t.innerHTML='<thead><tr><th>Team</th><th class=num>P</th><th class=num>W</th><th class=num>D</th><th class=num>L</th><th class=num>GD</th><th class=num>Pts</th></tr></thead>';
  const tb=el('tbody');
  (D.groups[name]||[]).forEach((r,i)=>tb.append(el('tr',{}, `<td>${i<2?'<b>'+r.team+'</b>':r.team}</td>
   <td class=num>${r.P}</td><td class=num>${r.W}</td><td class=num>${r.D}</td><td class=num>${r.L}</td>
   <td class=num>${r.GD>0?'+':''}${r.GD}</td><td class=num><b>${r.Pts}</b></td>`)));
  t.append(tb);card.append(t);g.append(card);
 });
 s.append(g);
 if(!Object.keys(D.groups||{}).length)s.innerHTML='<div class=card><h3>Standings unavailable</h3></div>';
}
// ---- knockout seeds (projected group winners from crowd) ----
function ko(){
 const s=document.getElementById('tab-ko');s.innerHTML='';
 const g=el('div',{class:'grid2'});
 Object.keys(D.group_winners||{}).sort().forEach(grp=>{
  const card=el('div',{class:'card'});card.append(el('h3',{},'Group '+grp+' — projected'));
  const t=el('table');t.innerHTML='<thead><tr><th>Team</th><th class=num>Win group</th></tr></thead>';
  const tb=el('tbody');
  (D.group_winners[grp]||[]).forEach(r=>{const tr=el('tr');
   const td=el('td',{class:'num'});const bar=el('div',{class:'bar'});const i=el('i');i.style.width=Math.max(3,100*r.prob)+'%';bar.append(i);
   td.append(bar);const sp=el('span',{class:'muted'});sp.style.marginLeft='6px';sp.textContent=pct(r.prob);td.append(sp);
   tr.append(el('td',{}, r.team));tr.append(td);tb.append(tr);});
  t.append(tb);card.append(t);g.append(card);
 });
 s.append(g);
 if(!Object.keys(D.group_winners||{}).length)s.innerHTML='<div class=card><h3>Knockout seeds available after group markets open</h3></div>';
}
// ---- ranked tables ----
function ranked(secId,rows,label){
 const s=document.getElementById(secId);s.innerHTML='';
 s.append(el('h3',{},label));
 const t=el('table');t.innerHTML='<thead><tr><th>'+(rows[0]&&rows[0].player?'Player':'Team')+'</th><th class=num>Crowd prob</th><th class=num>Fair odds</th></tr></thead>';
 const tb=el('tbody');
 (rows||[]).forEach(r=>{const tr=el('tr');tr.append(el('td',{}, r.team||r.player));
  const td=el('td',{class:'num'});const bar=el('div',{class:'bar'});const i=el('i');i.style.width=Math.max(2,100*r.prob)+'%';bar.append(i);
  td.append(bar);const sp=el('span',{class:'muted'});sp.style.marginLeft='6px';sp.textContent=pct(r.prob);td.append(sp);tr.append(td);
  tr.append(el('td',{class:'num'}, r.fair_odds??'—'));tb.append(tr);});
 t.append(tb);s.append(t);
 if(!(rows||[]).length)s.innerHTML='<h3>'+label+'</h3><div style="padding:12px" class=muted>market unavailable</div>';
}
fixtures();groups();ko();
ranked('tab-winner',D.winner,'Tournament winner (Polymarket crowd)');
ranked('tab-scorer',D.top_scorer,'Top scorer / Golden Boot (Polymarket crowd)');
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));b.classList.add('active');
 ['fixtures','groups','ko','winner','scorer'].forEach(t=>document.getElementById('tab-'+t).classList.toggle('hidden',t!==b.dataset.tab));
});
</script></body></html>
"""
