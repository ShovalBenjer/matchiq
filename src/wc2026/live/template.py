"""Premium self-contained dashboard renderer (2026 design system)."""

from __future__ import annotations

import json

from wc2026.live.theme import CSS, FONTS, JS_HELPERS


def render(data: dict) -> str:
    return _HTML.replace("/*__CSS__*/", CSS).replace("<!--FONTS-->", FONTS)\
                .replace("/*__HELPERS__*/", JS_HELPERS)\
                .replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))


_HTML = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>matchiq · World Cup 2026 — live</title>
<!--FONTS-->
<style>/*__CSS__*/</style></head><body>
<div class="aurora"><i></i><i></i><i></i></div>
<header class="top"><div class="wrap topbar">
  <div class="brand">⚽ matchiq <span class="dot" title="live"></span></div>
  <span class="chip" id="countdown">—</span>
  <div class="spacer"></div>
  <span class="chip" id="src"></span>
  <span class="chip" id="synced"></span>
</div></header>
<main class="wrap">
  <section class="bento stagger" id="hero"></section>
  <section id="chaos"></section>
  <div id="legend" class="glass" style="padding:10px 16px;margin:10px 0;font-size:12px;color:var(--muted);display:flex;gap:18px;flex-wrap:wrap"></div>
  <nav class="tabs" id="tabs"></nav>
  <section class="view" id="v-fixtures"></section>
  <section class="view hidden" id="v-groups"></section>
  <section class="view hidden" id="v-ko"></section>
  <section class="view hidden" id="v-winner"></section>
  <section class="view hidden" id="v-scorer"></section>
</main>
<footer>
  <div id="disc"></div>
  <div class="crosslinks">
    <a href="../index.html">← Hub</a>
    <a href="../futures.html">Futures · winner · exact scores</a>
    <a href="../RESEARCH.md">Research dossier</a>
  </div>
</footer>
<script>
const D=/*__DATA__*/;
/*__HELPERS__*/
document.getElementById('synced').textContent='synced '+new Date(D.generated_at).toUTCString().slice(5,22);
document.getElementById('src').textContent=D.sources.results_odds+' · '+D.sources.crowd;
document.getElementById('disc').textContent=D.disclaimer;

const fx=D.fixtures||[];
const upcoming=fx.filter(f=>f.status==='pre');
const nextM=upcoming[0]||fx[0];
const fav=(D.winner||[])[0]; const boot=(D.top_scorer||[])[0];
const valueN=fx.filter(f=>f.value&&f.value.is_value).length;

// ---- bento hero ----
function hero(){
  const h=document.getElementById('hero');h.innerHTML='';
  if(nextM){
    const b=nextM.book||{};
    h.append(el('div',{class:'glass tile big grad'},
      `<div class="k">Next kick-off</div>
       <div class="v">${flag(nextM.home)}${nextM.home}<div style="font:600 14px Inter;color:var(--muted);margin:6px 0">vs</div>${flag(nextM.away)}${nextM.away}</div>
       <div class="s" id="nextcd">—</div>
       <div style="display:flex;gap:18px;margin-top:14px">
        ${leg('Home',b.home)}${leg('Draw',b.draw)}${leg('Away',b.away)}
       </div>`));
  }
  if(fav)h.append(el('div',{class:'glass tile grad'},
    `<div class="k">Title favourite</div><div class="v"><span class="num" data-to="${(fav.prob*100).toFixed(2)}" data-dp="1" data-fmt="#%">0%</span></div>
     <div class="s">${flag(fav.team)}${fav.team} · crowd</div>`));
  if(boot)h.append(el('div',{class:'glass tile grad'},
    `<div class="k">Golden Boot</div><div class="v" style="font-size:22px">${boot.player}</div>
     <div class="s"><span class="num" data-to="${(boot.prob*100).toFixed(2)}" data-dp="1" data-fmt="#%">0%</span> · crowd</div>`));
  h.append(el('div',{class:'glass tile wide'},
    `<div class="k">Market watch</div>
     <div class="v" style="font-size:22px">${valueN?valueN+' value flag'+(valueN>1?'s':''):'Market efficient'}</div>
     <div class="s">${valueN? 'crowd beats the book on '+valueN+' line(s)':'book ≈ crowd within the vig — no free lunch in the opener'}</div>`));
  // hero stat count-ups
  h.querySelectorAll('.num').forEach(countUp);
  if(nextM)tickCountdown();
}
function leg(label,p){return `<div><div class="k">${label}</div><div style="font:700 18px 'Space Grotesk'">${pct(p)}</div></div>`;}
function tickCountdown(){
  const t=new Date(nextM.date).getTime();
  function upd(){const d=t-Date.now();const cd=document.getElementById('countdown');const nc=document.getElementById('nextcd');
    if(d<=0){if(cd)cd.textContent='⚽ underway';if(nc)nc.textContent='Kick-off';return;}
    const h=Math.floor(d/3.6e6),m=Math.floor(d%3.6e6/6e4),s=Math.floor(d%6e4/1e3);
    const txt=(h?h+'h ':'')+m+'m '+s+'s';
    if(cd)cd.textContent='next match in '+txt;
    if(nc)nc.textContent=new Date(nextM.date).toUTCString().slice(0,22)+' · in '+txt;
    setTimeout(upd,1000);}
  upd();
}

// ---- fixtures by date ----
function vFixtures(){
  const s=document.getElementById('v-fixtures');s.innerHTML='';
  const byDate={};fx.forEach(f=>{const d=(f.date||'').slice(0,10);(byDate[d]=byDate[d]||[]).push(f);});
  Object.keys(byDate).sort().forEach(d=>{
    s.append(el('div',{class:'datehead'},new Date(d).toUTCString().slice(0,16)||'TBD'));
    const card=el('div',{class:'glass'});const t=el('table');
    t.innerHTML=`<thead><tr><th>Match</th><th>Status</th><th class=num>Book H/D/A</th><th class=num>Crowd</th><th class=num title="Dixon-Coles model on real results">Model</th><th class=num title="model P(over 2.5) / book line">O/U 2.5</th><th class=num title="Transfermarkt squad value · FM26/EA FC overall (home / away)">Squad €·FM</th><th>Edge</th></tr></thead>`;
    const tb=el('tbody');
    byDate[d].forEach(f=>{
      const b=f.book||{},c=f.crowd,m=f.model,ou=f.model_ou;
      const book=b.home!=null?`${pct(b.home)} / ${pct(b.draw)} / ${pct(b.away)}`:'<span class=muted>—</span>';
      const crowd=c?`${pct(c.home)} / ${pct(c.draw)} / ${pct(c.away)}`:'<span class=muted>—</span>';
      const model=m?`<span style="color:var(--accent)">${pct(m.home)} / ${pct(m.draw)} / ${pct(m.away)}</span>`:'<span class=muted>—</span>';
      const ouCell=ou?`<span style="color:var(--accent)">O ${pct(ou.over)}</span>${f.over_under!=null?` <span class=muted>/ ${f.over_under}</span>`:''}`:(f.over_under!=null?`<span class=muted>${f.over_under}</span>`:'<span class=muted>—</span>');
      const sq=f.squad,vfmt=v=>v>=1e9?`€${(v/1e9).toFixed(1)}b`:`€${Math.round(v/1e6)}m`;
      const squadCell=sq?`<span style="font-size:11px">${vfmt(sq.home.value_eur)}·<b>${sq.home.rating}</b> <span class=muted>/</span> ${vfmt(sq.away.value_eur)}·<b>${sq.away.rating}</b></span>`:'<span class=muted>—</span>';
      let v='<span class=muted>—</span>';
      if(f.value&&f.value.is_value)v=`<span class="pill value">${f.value.outcome} +${pct(f.value.ev)} @${f.value.offered_odds}</span>`;
      else if(f.value)v=`<span class=muted>${f.value.outcome} ${(100*f.value.ev).toFixed(1)}%</span>`;
      const st=f.status==='in'?'<span class=pill live>● LIVE</span>':(f.status==='post'?('<b>'+(f.score||'FT')+'</b>'):'<span class=muted>'+(f.date||'').slice(11,16)+'Z</span>');
      tb.append(el('tr',{}, `<td class=team>${flag(f.home)}${f.home} <span class=muted>v</span> ${flag(f.away)}${f.away}${f.details?` <span class=muted style="font-size:11px">(${f.details})</span>`:''}</td>
        <td>${st}</td><td class=num style="font-size:12px">${book}</td>
        <td class=num style="font-size:12px">${crowd}</td><td class=num style="font-size:12px">${model}</td><td class=num style="font-size:12px">${ouCell}</td><td class=num>${squadCell}</td><td>${v}</td>`));
      if(f.lineup){const L=f.lineup;
        const line=s=>`ATT ${s.attack} · DEF ${s.defense} · ★${s.star} · depth ${s.depth}`;
        const out=s=>(s.absences&&s.absences.length)?` <span style="color:#ff9a9a">⚠ OUT: ${s.absences.join(', ')}</span>`:'';
        tb.append(el('tr',{}, `<td colspan=8 style="padding:2px 12px 10px;font-size:11px" class=muted>
          <span title="${(L.home.xi||[]).join(', ')}">▸ <b>${f.home}</b> XI&nbsp;${line(L.home)}${out(L.home)}</span>
          &nbsp;&nbsp;<span class=muted>|</span>&nbsp;&nbsp;
          <span title="${(L.away.xi||[]).join(', ')}"><b>${f.away}</b> XI&nbsp;${line(L.away)}${out(L.away)}</span></td>`));}
    });
    t.append(tb);card.append(t);s.append(card);
  });
  if(!fx.length)s.innerHTML='<div class="glass" style="padding:24px" >No fixtures in window.</div>';
}
// ---- groups (bento) ----
function vGroups(){
  const s=document.getElementById('v-groups');s.innerHTML='';
  const g=el('div',{class:'grid2'});
  Object.keys(D.groups||{}).sort().forEach(name=>{
    const card=el('div',{class:'glass'});card.append(el('div',{class:'card-h'},`<span>${name}</span><span>top 2 advance</span>`));
    const t=el('table');
    t.append(el('thead',{},'<tr><th>Team</th><th class=num>P</th><th class=num>W</th><th class=num>D</th><th class=num>L</th><th class=num>GD</th><th class=num>Pts</th></tr>'));
    const tb=el('tbody');
    (D.groups[name]||[]).forEach((r,i)=>tb.append(el('tr',{},
      `<td class=team>${i<2?'<span style="color:var(--good)">●</span> ':'<span class=muted>○</span> '}${flag(r.team)}${r.team}</td>
       <td class=num>${r.P}</td><td class=num>${r.W}</td><td class=num>${r.D}</td><td class=num>${r.L}</td>
       <td class=num>${r.GD>0?'+':''}${r.GD}</td><td class=num><b>${r.Pts}</b></td>`)));
    t.append(tb);card.append(t);g.append(card);
  });
  s.append(g);
  if(!Object.keys(D.groups||{}).length)s.innerHTML='<div class="glass" style="padding:24px">Standings unavailable.</div>';
}
// ---- knockout seeds ----
function vKo(){
  const s=document.getElementById('v-ko');s.innerHTML='';
  const g=el('div',{class:'grid2'});
  Object.keys(D.group_winners||{}).sort().forEach(grp=>{
    const card=el('div',{class:'glass'});card.append(el('div',{class:'card-h'},`<span>Group ${grp}</span><span>win group</span>`));
    const t=el('table');const tb=el('tbody');
    (D.group_winners[grp]||[]).forEach(r=>tb.append(el('tr',{},
      `<td class=team>${flag(r.team)}${r.team}</td>
       <td class=num style="width:46%"><div class="bar"><i data-w="${Math.max(3,100*r.prob)}%"></i></div></td>
       <td class=num>${pct(r.prob)}</td>`)));
    t.append(tb);card.append(t);g.append(card);
  });
  s.append(g);
  if(!Object.keys(D.group_winners||{}).length)s.innerHTML='<div class="glass" style="padding:24px">Knockout seeds appear once group markets open.</div>';
}
// ---- ranked tables (winner / scorer) ----
function vRanked(id,rows,title,who){
  const s=document.getElementById(id);s.innerHTML='';
  const hasModel=(rows||[]).some(r=>r.model!=null);
  const sub=hasModel?'crowd · model · blended':'crowd · fair odds';
  const card=el('div',{class:'glass'});card.append(el('div',{class:'card-h'},`<span>${title}</span><span>${sub}</span>`));
  const t=el('table');const tb=el('tbody');
  (rows||[]).forEach((r,i)=>{
    const extra=hasModel
      ? `<td class=num style="color:var(--accent)">${r.model!=null?pct(r.model):'—'}</td><td class=num><b>${r.blended!=null?pct(r.blended):'—'}</b></td>`
      : `<td class=num>${r.fair_odds??'—'}</td>`;
    tb.append(el('tr',{},
    `<td class=team><span class=muted>${i+1}</span>&nbsp; ${flag(r.team||r.player)}${r.team||r.player}</td>
     <td class=num style="width:34%"><div class="bar"><i data-w="${Math.max(2,100*r.prob)}%"></i></div></td>
     <td class=num>${pct(r.prob)}</td>${extra}`));
  });
  t.append(tb);card.append(t);s.append(card);
  if(!(rows||[]).length)s.innerHTML='<div class="glass" style="padding:24px">'+who+' market unavailable.</div>';
}

// ---- chaos banner + provenance legend ----
function vChaos(){
  const c=D.chaos; if(!c){document.getElementById('chaos').remove?.();return;}
  const cell=(k,v,s)=>`<div class="glass tile"><div class="k">${k}</div><div class="v" style="font-size:24px">${v}</div><div class="s">${s}</div></div>`;
  document.getElementById('chaos').className='bento';
  document.getElementById('chaos').innerHTML=
    cell('Chaos index',(c.chaos_index*100).toFixed(0)+'%','sensitive dependence (Lyapunov proxy '+c.lyapunov_proxy+')')+
    cell('Favourite',flag(titlecase(c.favourite))+titlecase(c.favourite)+' '+(c.favourite_prob*100).toFixed(0)+'%','fragility '+(c.favourite_fragility*100).toFixed(0)+'% — chance it does NOT win')+
    cell('Field entropy',(c.field_entropy*100).toFixed(0)+'%','1 = wide-open tournament');
}
function titlecase(s){return (s||'').replace(/_/g,' ').replace(/\b\w/g,m=>m.toUpperCase());}
function vTempo(){
  const t=D.tempo; if(!t)return;
  const cell=(k,v,s)=>`<div class="glass tile"><div class="k">${k}</div><div class="v" style="font-size:24px">${v}</div><div class="s">${s}</div></div>`;
  const colour=t.lean==='attacking'?'var(--good)':(t.lean==='defensive'?'var(--accent)':'var(--text)');
  const el2=document.getElementById('chaos');
  el2.insertAdjacentHTML('beforeend',
    cell('Tournament tempo',`<span style="color:${colour}">${t.lean.toUpperCase()}</span>`,`index ${t.index}/100 · 50 = avg (${t.baseline} g/g)`)+
    cell('Exp. goals/game',t.avg_goals,`model mean · ${(t.avg_over25*100).toFixed(0)}% of games over 2.5`));
}
function vLegend(){
  const e=D.engines||{}; const L=document.getElementById('legend');
  if(!e.model){L.remove();return;}
  L.innerHTML=`<span><b style="color:var(--text)">Book</b> ${e.market}</span>`+
    `<span><b style="color:var(--accent)">Model</b> ${e.model}</span>`+
    `<span><b style="color:var(--text)">Blended</b> ${e.blend}</span>`;
}

// ---- tabs ----
const TABS=[['fixtures','Next fixtures'],['groups','Groups'],['ko','Knockout seeds'],['winner','Winner'],['scorer','Top scorer']];
const nav=document.getElementById('tabs');
TABS.forEach(([id,label],i)=>{const b=el('button',{class:'tab'+(i?'':' active')},label);b.onclick=()=>switchView(id,b);nav.append(b);});

hero();vChaos();vLegend();vFixtures();vGroups();vKo();
vRanked('v-winner',D.winner,'Tournament winner — crowd vs model vs blended','Winner');
vRanked('v-scorer',D.top_scorer,'Golden Boot — Polymarket crowd','Top-scorer');
fillBars(document.getElementById('v-fixtures'));
fillBars(document.getElementById('v-winner'));
</script></body></html>
"""
