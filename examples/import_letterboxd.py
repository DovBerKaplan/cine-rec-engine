"""Import a Letterboxd ratings export for one user (spec §D source).

    python examples/import_letterboxd.py ratings.csv \
        --user 1 --dsn postgresql://demo:demo@localhost:54329/demo
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv_file")
    p.add_argument("--user", type=int, required=True)
    p.add_argument("--dsn", default="postgresql://demo:demo@localhost:54329/demo")
    args = p.parse_args()

    import asyncpg

    from cine_rec_engine.letterboxd import import_letterboxd

    pool = await asyncpg.create_pool(args.dsn)
    report = await import_letterboxd(
        pool, args.user, Path(args.csv_file).read_text(encoding="utf-8-sig"))
    print(f"matched: {len(report['matched'])} "
          f"(favorites={report['kinds']['favorite']}, "
          f"dislikes={report['kinds']['dislike']})")
    if report["unmatched"]:
        print("unmatched (title not in this catalog — extend it or ignore):")
        for name in report["unmatched"][:20]:
            print(f"  - {name}")
    await pool.close()


asyncio.run(main())
