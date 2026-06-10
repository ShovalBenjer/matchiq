"""Generate the static "gambles" site (GitHub Pages) from the model stack.

Produces, for the current corpus:
  * Tournament winner market   — P(win), fair odds, offered/edge if available
  * Top-scorer market          — P(top scorer), P(5+ goals), expected goals
  * First group-stage fixtures — exact-score (Dixon-Coles), 1X2, value bet

Outputs two artifacts under ``docs/`` (served by GitHub Pages):
  * ``docs/data.json``  — machine-readable predictions
  * ``docs/index.html`` — self-contained page (data embedded; opens locally too)

Usage:
    python scripts/build_site.py --paths 50000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from wc2026.betting.value import devig, edges
from wc2026.config import Config
from wc2026.pipeline.orchestrator import Orchestrator

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _top_scorelines(grid: np.ndarray, k: int = 5) -> list[dict]:
    flat = np.dstack(np.unravel_index(np.argsort(grid.ravel())[::-1], grid.shape))[0]
    out = []
    for h, a in flat[:k]:
        out.append({"score": f"{int(h)}-{int(a)}", "prob": round(float(grid[h, a]), 4)})
    return out


def _first_fixture_per_group(orch: Orchestrator) -> dict[str, object]:
    groups: dict[str, object] = {}
    for m in sorted(orch.fixtures, key=lambda x: (x.date, x.match_id)):
        g = m.extra.get("group")
        if g and g not in groups:
            groups[g] = m
    return groups


def build(config: Config, paths: int) -> dict:
    orch = Orchestrator(config).fit()
    teams = {tid: t.name for tid, t in orch.teams.items()}
    player_name = {p.player_id: p.name for p in orch.players}
    player_team = {p.player_id: p.team_id for p in orch.players}

    # --- winner & top scorer (Monte Carlo + crowd-wisdom blend) -------
    sim = orch.simulate_tournament(n_paths=paths)
    model_wp = sim.get("win_prob_model", {})
    market_wp = sim.get("market_winner", {})
    winner = [
        {"team": teams.get(t, t), "team_id": t, "prob": round(p, 4),
         "model_prob": round(model_wp.get(t, 0.0), 4),
         "market_prob": round(market_wp.get(t), 4) if t in market_wp else None,
         "fair_odds": round(1.0 / p, 2) if p > 0 else None}
        for t, p in sim["win_prob"].items() if p > 0
    ][:24]

    top_scorer = []
    ts = sim.get("top_scorer", {})
    if ts:
        ranked = sorted(ts["top_scorer_prob"].items(), key=lambda kv: kv[1], reverse=True)
        for pid, p in ranked[:20]:
            if p <= 0:
                continue
            top_scorer.append({
                "player": player_name.get(pid, pid),
                "team": teams.get(player_team.get(pid, ""), ""),
                "prob": round(p, 4),
                "fair_odds": round(1.0 / p, 2) if p > 0 else None,
                "p_5plus": round(ts["p_5plus"].get(pid, 0.0), 4),
                "exp_goals": round(ts["expected_goals"].get(pid, 0.0), 2),
            })

    # --- first group fixtures: exact score + 1X2 + value --------------
    fixtures = []
    label = {"H": "home", "D": "draw", "A": "away"}
    for g, m in sorted(_first_fixture_per_group(orch).items(),
                       key=lambda kv: int(kv[0][1:])):
        pred = orch.predict(m)
        grid = orch.dixon_coles.score_matrix(m.home_id, m.away_id, m.neutral)
        scorelines = _top_scorelines(grid, 5)
        final = pred.final.as_array()
        entry = {
            "group": g,
            "match_id": m.match_id,
            "date": m.date.isoformat(),
            "home": teams.get(m.home_id, m.home_id),
            "away": teams.get(m.away_id, m.away_id),
            "prob_1x2": {"home": round(float(final[0]), 4),
                         "draw": round(float(final[1]), 4),
                         "away": round(float(final[2]), 4)},
            "most_likely_score": scorelines[0]["score"],
            "scorelines": scorelines,
            "expected_goals": [round(x, 2) for x in
                               orch.dixon_coles.expected_goals(m.home_id, m.away_id, m.neutral)],
            "pick_1x2": label[pred.final.argmax],
        }
        if m.odds is not None:
            e = edges(final, m.odds, config.betting.devig_method)
            fair = devig(m.odds, config.betting.devig_method)
            best = int(np.argmax(e))
            entry["odds"] = {"home": m.odds.home, "draw": m.odds.draw, "away": m.odds.away}
            entry["value"] = {
                "outcome": ["home", "draw", "away"][best],
                "edge": round(float(e[best]), 4),
                "is_value": bool(e[best] > config.betting.edge_threshold),
                "fair_odds": round(1.0 / fair[best], 2),
            }
        fixtures.append(entry)

    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "meta": {
            "n_matches": len(orch.matches),
            "n_teams": len(orch.teams),
            "mc_paths": sim["n_paths"],
            "tabpfn_backend": orch.tabpfn.backend_name,
            "chronos_backend": orch.chronos.backend_name,
            "news_backend": orch.news_agent.backend_name,
            "ensemble_meta": orch.ensemble.meta,
            "ensemble_weights": orch.ensemble.weights(),
            "edge_threshold": config.betting.edge_threshold,
            "kelly_fraction": config.betting.kelly_fraction,
            "corpus": "synthetic",
        },
        "winner": winner,
        "top_scorer": top_scorer,
        "fixtures": fixtures,
    }


def render_html(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    # Self-contained page: data embedded so it works on Pages and via file://.
    return _HTML_TEMPLATE.replace("/*__DATA__*/", payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=50000)
    ap.add_argument("--history", type=int, default=6)
    args = ap.parse_args()

    cfg = Config()
    cfg.data.synthetic_n_history_tournaments = args.history
    data = build(cfg, args.paths)

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (DOCS / "index.html").write_text(render_html(data), encoding="utf-8")
    print(f"Wrote {DOCS/'index.html'} and {DOCS/'data.json'}")
    print(f"  winner rows={len(data['winner'])}  "
          f"top_scorer rows={len(data['top_scorer'])}  "
          f"fixtures={len(data['fixtures'])}")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>matchiq · World Cup 2026 model gambles</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--muted:#8b949e;
        --accent:#58a6ff;--good:#3fb950;--warn:#d29922;}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       background:var(--bg);color:var(--fg)}
  header{padding:24px 20px 8px;max-width:1100px;margin:0 auto}
  h1{margin:0 0 4px;font-size:24px}
  .sub{color:var(--muted);font-size:13px}
  .banner{max-width:1100px;margin:12px auto;padding:10px 14px;background:#1c2128;
          border:1px solid var(--line);border-radius:8px;color:var(--muted);font-size:13px}
  nav{max-width:1100px;margin:16px auto 0;display:flex;gap:8px;flex-wrap:wrap;padding:0 20px}
  nav button{background:var(--panel);color:var(--fg);border:1px solid var(--line);
             padding:8px 14px;border-radius:8px;cursor:pointer;font-size:14px}
  nav button.active{border-color:var(--accent);color:var(--accent)}
  main{max-width:1100px;margin:16px auto 60px;padding:0 20px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
        padding:4px 0;overflow:hidden;margin-bottom:18px}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600;cursor:pointer;user-select:none}
  tr:last-child td{border-bottom:none}
  td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
  .bar{height:7px;background:#21262d;border-radius:4px;overflow:hidden;min-width:80px}
  .bar > i{display:block;height:100%;background:var(--accent)}
  .pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:12px;
        border:1px solid var(--line)}
  .value{color:var(--good);border-color:var(--good)}
  .muted{color:var(--muted)}
  .score{font-weight:700;font-variant-numeric:tabular-nums}
  .hidden{display:none}
  code{background:#21262d;padding:1px 5px;border-radius:4px}
  footer{max-width:1100px;margin:0 auto 40px;padding:0 20px;color:var(--muted);font-size:12px}
</style>
</head>
<body>
<header>
  <h1>⚽ matchiq — World Cup 2026 model gambles</h1>
  <div class="sub">Tournament winner · top scorer · first group-fixture exact scores. Generated <span id="gen"></span>.</div>
</header>
<div class="banner" id="banner"></div>
<nav>
  <button data-tab="fixtures" class="active">Group fixtures (exact score)</button>
  <button data-tab="winner">Tournament winner</button>
  <button data-tab="scorer">Top scorer</button>
</nav>
<main>
  <section id="tab-fixtures" class="card"></section>
  <section id="tab-winner" class="card hidden"></section>
  <section id="tab-scorer" class="card hidden"></section>
</main>
<footer>
  Educational model output — not betting advice. Probabilities are the model's, not
  the market's. On an efficient market the long-run result is break-even minus vig.
</footer>
<script>
const DATA = /*__DATA__*/;
const pct = x => (100*x).toFixed(1) + '%';
const el = (t,a={},...k)=>{const e=document.createElement(t);for(const[p,v]of Object.entries(a)){
  if(p==='class')e.className=v;else if(p==='html')e.innerHTML=v;else e.setAttribute(p,v);}
  k.forEach(c=>e.append(c));return e;};
function bar(p){const w=el('div',{class:'bar'});const i=document.createElement('i');
  i.style.width=Math.max(2,100*p)+'%';w.append(i);return w;}

document.getElementById('gen').textContent = DATA.generated_at;
const m = DATA.meta;
document.getElementById('banner').innerHTML =
  `Corpus: <code>${m.corpus}</code> · ${m.n_matches} matches / ${m.n_teams} teams · `+
  `Monte-Carlo ${m.mc_paths.toLocaleString()} paths · ensemble <code>${m.ensemble_meta}</code> · `+
  `TabPFN <code>${m.tabpfn_backend}</code>, Chronos <code>${m.chronos_backend}</code>, `+
  `news <code>${m.news_backend}</code> · edge≥${pct(m.edge_threshold)}, `+
  `${m.kelly_fraction}-Kelly`;

function tableSort(table, rows, render){
  let dir = {};
  table.querySelectorAll('th[data-key]').forEach(th=>{
    th.onclick=()=>{const k=th.dataset.key;dir[k]=!dir[k];
      rows.sort((a,b)=>{const x=a[k],y=b[k];return (dir[k]?1:-1)*((x>y)-(x<y));});
      render();};});
}

function renderFixtures(){
  const sec=document.getElementById('tab-fixtures');sec.innerHTML='';
  const t=el('table');
  t.innerHTML=`<thead><tr><th>Grp</th><th>Match</th><th class="num">xG</th>
    <th>Most likely</th><th>Top scorelines</th><th class="num">H</th><th class="num">D</th>
    <th class="num">A</th><th>Pick</th><th>Value bet</th></tr></thead>`;
  const tb=el('tbody');
  DATA.fixtures.forEach(f=>{
    const sl=f.scorelines.map(s=>`${s.score} <span class="muted">${pct(s.prob)}</span>`).join(' · ');
    let val='<span class="muted">—</span>';
    if(f.value && f.value.is_value)
      val=`<span class="pill value">${f.value.outcome} +${pct(f.value.edge)}</span>`;
    else if(f.value) val=`<span class="muted">${f.value.outcome} ${(100*f.value.edge).toFixed(1)}%</span>`;
    tb.append(el('tr',{html:
      `<td>${f.group}</td><td><b>${f.home}</b> vs <b>${f.away}</b><br>
       <span class="muted">${f.date}</span></td>
       <td class="num">${f.expected_goals[0]}–${f.expected_goals[1]}</td>
       <td class="score">${f.most_likely_score}</td>
       <td style="font-size:12px">${sl}</td>
       <td class="num">${pct(f.prob_1x2.home)}</td>
       <td class="num">${pct(f.prob_1x2.draw)}</td>
       <td class="num">${pct(f.prob_1x2.away)}</td>
       <td><span class="pill">${f.pick_1x2}</span></td>
       <td>${val}</td>`}));
  });
  t.append(tb);sec.append(el('div',{html:'<div style="padding:10px 12px" class="muted">'+
    'Exact score = most likely Dixon-Coles scoreline. 1X2 = final ensemble probabilities.</div>'}),t);
}

function renderRanked(secId, rows, cols){
  const sec=document.getElementById(secId);sec.innerHTML='';
  const t=el('table');const head='<tr>'+cols.map(c=>
    `<th class="${c.num?'num':''}" data-key="${c.key}">${c.label}</th>`).join('')+'</tr>';
  t.innerHTML='<thead>'+head+'</thead>';
  const tb=el('tbody');
  const render=()=>{tb.innerHTML='';rows.forEach((r,i)=>{
    const tr=el('tr');cols.forEach(c=>{
      let v=r[c.key];
      if(c.key==='prob'){const td=el('td',{class:'num'});td.append(bar(v));
        const s=el('span',{class:'muted'});s.style.marginLeft='8px';s.textContent=pct(v);
        td.append(s);tr.append(td);return;}
      if(v===null||v===undefined){tr.append(el('td',{class:c.num?'num':'',html:'<span class="muted">—</span>'}));return;}
      if(c.pctf) v=pct(v);
      tr.append(el('td',{class:c.num?'num':'',html:String(v)}));
    });tb.append(tr);});};
  render();t.append(tb);tableSort(t,rows,render);sec.append(t);
}

renderFixtures();
renderRanked('tab-winner', DATA.winner, [
  {key:'team',label:'Team'},{key:'prob',label:'Blended',num:true},
  {key:'model_prob',label:'Model',num:true,pctf:true},
  {key:'market_prob',label:'Crowd',num:true,pctf:true},
  {key:'fair_odds',label:'Fair odds',num:true}]);
renderRanked('tab-scorer', DATA.top_scorer, [
  {key:'player',label:'Player'},{key:'team',label:'Team'},
  {key:'prob',label:'P(top scorer)',num:true},
  {key:'p_5plus',label:'P(5+)',num:true,pctf:true},
  {key:'exp_goals',label:'xGoals',num:true},
  {key:'fair_odds',label:'Fair odds',num:true}]);

document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');
  ['fixtures','winner','scorer'].forEach(t=>
    document.getElementById('tab-'+t).classList.toggle('hidden', t!==b.dataset.tab));
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
