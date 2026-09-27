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

    from cine_rec_engine import RecommendationService, user_stats, user_vector
    from cine_rec_engine.queries import (
        enrich_candidates_batch,
        get_movie_info_batch,
    )

    pool = await asyncpg.create_pool(dsn)
    rec = RecommendationService()
    await rec.initialize(pool)

    infos = await get_movie_info_batch(pool, SEED_IDS)
    seeds = []
    for sid in SEED_IDS:
        info = infos.get(sid)
        if not info:
            continue
        seeds.append({
            "id": sid,
            "title": info.get("title_en") or info.get("title"),
            "year": info.get("release_year"),
            "media": info.get("media_type", "movie"),
        })

    recs = {}
    for seed in seeds:
        results = await rec.find_similar(seed["id"], limit=8,
                                         media_type=seed["media"])
        enriched = await enrich_candidates_batch(
            pool, [r["tmdb_id"] for r in results])
        seed_info = dict(infos[seed["id"]])
        seed_info.setdefault("media_type", seed["media"])
        rows = []
        for r in results:
            cand = dict(enriched.get(r["tmdb_id"], {}))
            cand.setdefault("id", r["tmdb_id"])
            for k, v in r.items():
                cand.setdefault(k, v)
            vec = feature_vector(cand, cand, cand.get("genres", []), seed_info)
            rows.append({
                "title": r.get("title_en") or r.get("title"),
                "year": r.get("release_year"),
                "media": r["media_type"],
                "score": round(r["score"], 1),
                "why": explain_features(vec, ACTIVE_WEIGHTS),
                "poster": r.get("poster_path"),
            })
        recs[str(seed["id"])] = rows

    async with pool.acquire() as c:
        try:
            await c.execute(
                (Path(__file__).parent.parent / "docs" / "user_data.sql"
                 ).read_text())
        except asyncpg.DuplicateObjectError:
            pass

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
        personas[name] = {
            "label": f"user {uid} \u2014 {blurb} (recommend_for_user)",
            "rows": [{
                "title": r.get("title_en") or r.get("title"),
                "year": r.get("release_year"),
                "media": r["media_type"],
                "score": round(r["score"], 1),
                "why": out["why"].get(str(r["tmdb_id"]), []),
                "poster": r.get("poster_path"),
            } for r in out["results"]],
        }
    await pool.close()
    return {"seeds": seeds, "recs": recs, "personas": personas}


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>cine-rec-engine — try it</title>
<style>
 :root{--bg:#18181b;--panel:#27272a;--line:#3f3f46;--fg:#e4e4e7;--dim:#a1a1aa;--acc:#facc15}
 *{box-sizing:border-box;margin:0}
 body{background:var(--bg);color:var(--fg);font:15px/1.5 ui-monospace,'Cascadia Code',Menlo,monospace;padding:24px;max-width:860px;margin:0 auto}
 h1{font-size:19px;margin-bottom:2px} .sub{color:var(--dim);margin-bottom:18px}
 a{color:var(--acc);text-decoration:none}
 .tabs{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
 .tab{padding:7px 12px;background:var(--panel);border:1px solid var(--line);border-radius:8px;cursor:pointer;font-size:13px}
 .tab:hover{border-color:var(--acc)} .tab.on{border-color:var(--acc);color:var(--acc)}
 .persona{border:1px dashed var(--acc);border-radius:8px}
 .head{display:flex;align-items:center;gap:14px;margin:14px 0 10px}
 .head img{width:46px;height:69px;border-radius:6px;object-fit:cover;background:var(--panel)}
 .row{display:flex;align-items:center;gap:12px;padding:9px 10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;margin-bottom:8px;opacity:0;transform:translateY(6px);transition:all .25s}
 .row.in{opacity:1;transform:none}
 .row img{width:34px;height:51px;border-radius:4px;object-fit:cover;background:var(--line)}
 .t{font-weight:700} .y{color:var(--dim);font-size:12px}
 .s{margin-left:auto;color:var(--acc);font-size:13px;white-space:nowrap}
 .why{color:var(--dim);font-size:12.5px}
 .foot{color:var(--dim);font-size:12px;margin-top:22px;border-top:1px solid var(--line);padding-top:10px}
 .hint{color:var(--dim);font-size:12.5px;margin-bottom:10px}
</style></head><body>
<h1>&#127916; cine-rec-engine</h1>
<div class="sub">content-based recommendations from your own Postgres — every row explains itself. This page is <b>precomputed by the real engine</b>; no backend.</div>
<div class="tabs" id="tabs"></div>
<div class="hint" id="hint">pick a title &#8594;</div>
<div id="list"></div>
<div class="foot">engine: 22-feature learned scorer &middot; data: TMDB (en-US), attribution &middot; <a href="https://github.com/DovBerKaplan/cine-rec-engine">github</a> &middot; one-command local demo in the repo</div>
<script>
const DATA = __DATA__;
const tabs = document.getElementById('tabs'), list = document.getElementById('list'), hint = document.getElementById('hint');
const PERSONAS = DATA.personas;
function show(rows, label){
  list.innerHTML='';
  hint.textContent = label;
  rows = rows || [];
  rows.forEach((r,i)=>{
    const d=document.createElement('div');d.className='row';
    d.innerHTML=`<img src="${poster(r.poster)}" onerror="this.style.visibility='hidden'">
      <div><div class="t">${r.title} <span class="y">(${r.year||''} ${r.media==='tv'?'series':'film'})</span></div>
      <div class="why">${(r.why||[]).join(' &middot; ')||'&mdash;'}</div></div>
      <div class="s">${r.score}</div>`;
    list.appendChild(d);
    setTimeout(()=>d.classList.add('in'), 60+i*70);
  });
}
function poster(p){return p?`https://image.tmdb.org/t/p/w92${p}`:''}
let on=null;
DATA.seeds.forEach(s=>{
  const b=document.createElement('div');b.className='tab';b.textContent=`${s.title}`;
  b.onclick=()=>{if(on)on.classList.remove('on');b.classList.add('on');on=b;show(DATA.recs[s.id],`because you watched ${s.title} (${s.year})`)};
  tabs.appendChild(b);
});
const sp=document.createElement('div');sp.className='tab persona';sp.textContent='&#129488; user: Nolan fan';
sp.onclick=()=>{if(on)on.classList.remove('on');sp.classList.add('on');on=sp;show(PERSONAS.nolan.rows,PERSONAS.nolan.label)};
tabs.appendChild(sp);
const sp2=document.createElement('div');sp2.className='tab persona';sp2.textContent='&#129488; user: crime-TV';
sp2.onclick=()=>{if(on)on.classList.remove('on');sp2.classList.add('on');on=sp2;show(PERSONAS.crime.rows,PERSONAS.crime.label)};
tabs.appendChild(sp2);
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
