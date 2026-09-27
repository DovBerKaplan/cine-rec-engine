"""Build the static clickable demo (docs site): precomputed engine output.

Runs the REAL engine over the demo catalog and embeds everything into a
single self-contained index.html:

    python demo/make_web_demo.py [--dsn ...] [--out demo/site/index.html]

20 seeds x top-8-with-why, plus two REAL recommend_for_user personas
(recorded events -> weights -> seeds -> filters). Posters hotlink TMDB.
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cine_rec_engine.service import (  # noqa: E402
    ACTIVE_WEIGHTS,
    explain_features,
    feature_vector,
)

SEED_IDS = [
    155, 27205, 680, 550, 769, 77, 807, 603, 157336, 146233, 414419, 346,
    1396, 1438, 1398, 60059, 87108, 1399, 60574, 2098,
]


async def build(dsn: str) -> dict:
    import asyncpg

    from cine_rec_engine import (
        RecommendationService,
        user_stats,
        user_vector,
    )
    from cine_rec_engine.queries import (
        enrich_candidates_batch,
        get_movie_info_batch,
    )

    pool = await asyncpg.create_pool(dsn)
    rec = RecommendationService()
    await rec.initialize(pool)

    async with pool.acquire() as c:
        rows_raw = await c.fetch(
            """SELECT id, title AS name, 'movie'::text AS media,
                      EXTRACT(YEAR FROM release_date)::int AS year
               FROM tmdb_movies
               UNION ALL
               SELECT id, name, 'tv',
                      EXTRACT(YEAR FROM first_air_date)::int
               FROM tmdb_tv ORDER BY 2""")
    catalog = {r["id"]: {"id": r["id"], "title": r["name"], "year": r["year"],
                         "media": r["media"]} for r in rows_raw}
    ids = list(catalog)

    infos_all = {}
    for i in range(0, len(ids), 60):
        infos_all.update(await get_movie_info_batch(pool, ids[i:i + 60]))

    recs = {}
    for tid in ids:
        info = infos_all.get(tid)
        if not info:
            continue
        seed = dict(info)
        seed.setdefault("media_type", catalog[tid]["media"])
        try:
            results = await rec.find_similar(tid, limit=8,
                                             media_type=seed["media_type"])
        except Exception:
            continue
        enriched = await enrich_candidates_batch(
            pool, [r["tmdb_id"] for r in results])
        rows = []
        for r in results:
            cand = dict(enriched.get(r["tmdb_id"], {}))
            cand.setdefault("id", r["tmdb_id"])
            for k, v in r.items():
                cand.setdefault(k, v)
            vec = feature_vector(cand, cand, cand.get("genres", []), seed)
            why_w = explain_features(vec, ACTIVE_WEIGHTS, with_weights=True)
            rows.append({
                "title": r.get("title_en") or r.get("title"),
                "year": r.get("release_year"),
                "media": r["media_type"],
                "score": round(r["score"], 1),
                "why_w": [[lbl, round(w, 1)] for lbl, w in why_w],
                "poster": r.get("poster_path"),
            })

        def strong(row):
            w = row["why_w"]
            return len(w) > 1 or (w and w[0][0] != "genre overlap")

        recs[str(tid)] = [r for r in rows if strong(r)][:5]

    # tabs are curated: a seed earns a tab when its list convinces
    # (>= 3 strong rows). Every title stays searchable regardless.
    def tab_ok(tid):
        rows = recs.get(str(tid), [])
        return len(rows) >= 3 and bool(catalog[tid]["title"])

    seeds = [{k: catalog[t][k] for k in ("id", "title", "year", "media")}
             for t in SEED_IDS if t in catalog and tab_ok(t)]
    index = sorted(
        ({"id": c["id"], "title": c["title"], "year": c["year"],
          "media": c["media"]} for c in catalog.values()),
        key=lambda x: (x["title"] or "").lower())

    now = datetime.now(timezone.utc)
    personas = {}
    for uid, name, events, blurb in (
        (1, "nolan",
         [dict(tmdb_id=i, media_type="movie", watched_sec=8000,
               duration_sec=8000, completed=True) for i in (155, 77, 157336)],
         "watched The Dark Knight, Memento, Interstellar"),
        (2, "crime",
         [dict(tmdb_id=1396, media_type="tv", watched_sec=2700,
               duration_sec=2700, completed=True, season=1, episode=n)
          for n in (1, 2, 3)] +
         [dict(tmdb_id=1438, media_type="tv", watched_sec=2700,
               duration_sec=2700, completed=True, season=1, episode=1)],
         "watched Breaking Bad (3 eps), The Wire (1 ep)"),
    ):
        for ev in events:
            await user_stats.record_event(
                pool, {**ev, "user_id": uid, "watched_at": now - timedelta(days=1)})
        await user_stats.refresh_user_stats(pool, uid)
        await user_vector.build_user_vector(pool, uid)
        out = await rec.recommend_for_user(uid, limit=8, include_why=True)
        w_labels = out["why"]
        for r in out["results"][:6]:
            r["_why_w"] = [[lbl, 0.0] for lbl in w_labels.get(str(r["tmdb_id"]), [])]
        personas[name] = {
            "label": f"user {uid} \u2014 {blurb} (recommend_for_user)",
            "rows": [{
                "title": r.get("title_en") or r.get("title"),
                "year": r.get("release_year"),
                "media": r["media_type"],
                "score": round(r["score"], 1),
                "why_w": r.get("_why_w", []),
                "poster": r.get("poster_path"),
            } for r in out["results"][:6]],
        }
    await pool.close()
    return {"seeds": seeds, "recs": recs, "index": index,
            "personas": personas}


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>cine-rec-engine — click a title, see why</title>
<style>
 :root{--bg:#18181b;--panel:#27272a;--line:#3f3f46;--fg:#e4e4e7;--dim:#a1a1aa;--acc:#facc15}
 *{box-sizing:border-box;margin:0}
 body{background:var(--bg);color:var(--fg);font:15px/1.5 ui-monospace,'Cascadia Code',Menlo,monospace;padding:20px;max-width:880px;margin:0 auto}
 .top{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:4px}
 h1{font-size:19px} .sub{color:var(--dim);margin-bottom:14px;font-size:13.5px}
 .star{display:inline-block;background:var(--acc);color:#18181b;font-weight:700;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px;white-space:nowrap;margin-top:2px}
 a{color:var(--acc);text-decoration:none}
 .bar{display:flex;gap:8px;margin-bottom:8px}
 #q{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:8px;color:var(--fg);padding:8px 12px;font:inherit;font-size:13px}
 #q:focus{outline:none;border-color:var(--acc)}
 .tabs{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
 .tab{padding:7px 12px;background:var(--panel);border:1px solid var(--line);border-radius:8px;cursor:pointer;font-size:13px}
 .tab:hover{border-color:var(--acc)} .tab.on{border-color:var(--acc);color:var(--acc)}
 .persona{border:1px dashed var(--acc)}
 .head{display:flex;align-items:center;gap:14px;margin:10px 0}
 .head img{width:46px;height:69px;border-radius:6px;object-fit:cover;background:var(--panel)}
 .row{display:flex;align-items:center;gap:12px;padding:9px 10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;margin-bottom:8px;opacity:0;transform:translateY(6px);transition:all .25s}
 .row.in{opacity:1;transform:none}
 .row img{width:34px;height:51px;border-radius:4px;object-fit:cover;background:var(--line)}
 .t{font-weight:700} .y{color:var(--dim);font-size:12px}
 .s{margin-left:auto;color:var(--acc);font-size:13px;white-space:nowrap}
 .why{color:var(--dim);font-size:12.5px}.why b{color:var(--fg);font-weight:600}
 .sugg{position:absolute;top:38px;left:0;right:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;z-index:5;max-height:250px;overflow:auto;display:none}
 .sg{padding:8px 12px;cursor:pointer;font-size:13px}.sg:hover{background:var(--bg)}
 .none{padding:14px;color:var(--dim);font-size:13px;line-height:1.6}
 .foot{color:var(--dim);font-size:12px;margin-top:20px;border-top:1px solid var(--line);padding-top:10px;line-height:1.7}
 code{background:var(--panel);padding:2px 6px;border-radius:4px}
</style></head><body>
<div class="top"><h1>&#127916; cine-rec-engine</h1>
<a class="star" href="https://github.com/DovBerKaplan/cine-rec-engine" target="_blank">&#9733; Star on GitHub</a></div>
<div class="sub">Click a title \u2014 see the recommendations <b>and why</b> each one scored. Precomputed by the real engine from a bundled 830-title catalog (TMDB top-rated + their rec graphs), not all of TMDB.</div>
<div class="bar" style="position:relative"><input id="q" placeholder="search any of the 830 titles\u2026" autocomplete="off"><div id="sugg" class="sugg"></div></div>
<div class="tabs" id="tabs"></div>
<div class="sub" id="hint"></div>
<div id="list"></div>
<div class="foot">
Run it yourself \u2014 one command, no API key:<br>
<code>git clone https://github.com/DovBerKaplan/cine-rec-engine && cd cine-rec-engine/demo && docker compose up</code><br>
Or in Python: <code>pip install cine-rec-engine</code> \u00b7 <a href="https://github.com/DovBerKaplan/cine-rec-engine">source &amp; docs</a><br>
Demo catalog: 830 titles. The same engine runs in production on a larger private mirror.<br>
Data from <a href="https://www.themoviedb.org/" target="_blank">TMDB</a> \u2014 this product uses the TMDB API but is not endorsed or certified by TMDB.
</div>
<script>
const DATA = __DATA__;
const tabs = document.getElementById('tabs'), list = document.getElementById('list'),
      hint = document.getElementById('hint'), q = document.getElementById('q');
const PERSONAS = DATA.personas;
let on = null;
function poster(p){return p?`https://image.tmdb.org/t/p/w92${p}`:''}
function show(rows, label){
  list.innerHTML=''; hint.innerHTML = label + ' <span style="opacity:.6">\u00b7 ranked within the bundled 830-title catalog \u00b7 score = \u03a3 weight\u00d7feature</span>';
  (rows||[]).forEach((r,i)=>{
    const d=document.createElement('div');d.className='row';
    const why=(r.why_w||[]).map(([l,w])=>w>0?`<b>${l}</b> +${w}`:`<b>${l}</b>`).join(' \u00b7 ')||'\u2014';
    d.innerHTML=`<img src="${poster(r.poster)}" onerror="this.style.visibility='hidden'">
      <div><div class="t">${r.title} <span class="y">(${r.year||''} ${r.media==='tv'?'series':'film'})</span></div>
      <div class="why">${why}</div></div>
      <div class="s">${r.score}</div>`;
    list.appendChild(d);
    setTimeout(()=>d.classList.add('in'), 60+i*70);
  });
}
function select(el, rows, label){if(on)on.classList.remove('on');el.classList.add('on');on=el;show(rows,label)}
DATA.seeds.forEach(s=>{
  const b=document.createElement('div');b.className='tab';b.textContent=s.title;
  b.onclick=()=>select(b, DATA.recs[s.id], `because you watched <b>${s.title}</b> (${s.year})`);
  b.dataset.t=(s.title||'').toLowerCase(); tabs.appendChild(b);
});
[['🧠 user: Nolan fan','nolan'],['🧠 user: crime-TV','crime']].forEach(([label,key])=>{
  const b=document.createElement('div');b.className='tab persona';b.textContent=label;
  b.onclick=()=>select(b, PERSONAS[key].rows, PERSONAS[key].label);
  tabs.appendChild(b);
});
const sugg=document.getElementById('sugg');
function hideSugg(){sugg.style.display='none';sugg.innerHTML=''}
function pickSeed(id,label){
  hideSugg(); q.value=label;
  const rows=DATA.recs[String(id)]||[];
  show(rows, `because you watched <b>${label}</b>`);
  if(!rows.length) list.innerHTML='<div class="none">No convincing recommendations for this title within the demo catalog \u2014 try another.</div>';
}
q.addEventListener('input',()=>{
  const v=q.value.trim().toLowerCase();
  if(v.length<2){hideSugg();return}
  const hits=DATA.index.filter(x=>(x.title||'').toLowerCase().includes(v)).slice(0,8);
  sugg.innerHTML='';
  if(!hits.length){
    sugg.innerHTML='<div class="none">Not in the 830-title demo catalog.<br>The repo\u2019s ingest builds a full mirror from TMDB\u2019s daily exports \u2014 same engine, every title.</div>';
    sugg.style.display='block'; return;
  }
  hits.forEach(x=>{
    const d=document.createElement('div');d.className='sg';
    d.innerHTML=`${x.title} <span class="y">(${x.year||''} ${x.media==='tv'?'series':'film'})</span>`;
    d.onclick=()=>pickSeed(x.id, x.title + (x.year?` (${x.year})`:''));
    sugg.appendChild(d);
  });
  sugg.style.display='block';
});
q.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&sugg.firstChild)sugg.firstChild.click();
  if(e.key==='Escape')hideSugg();
});
document.addEventListener('click',e=>{if(!e.target.closest('.bar'))hideSugg()});
// open with a list already loaded
tabs.children[0].click();
</script></body></html>
"""


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", default="postgresql://demo:demo@localhost:54329/demo")
    p.add_argument("--out", default=str(Path(__file__).parent / "site" / "index.html"))
    args = p.parse_args()

    data = await build(args.dsn)
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, "
          f"{len(data['seeds'])} seeds)")


asyncio.run(main())
