# Publishing to PyPI

**Status: LIVE** — [pypi.org/project/cine-rec-engine](https://pypi.org/project/cine-rec-engine/)
(v0.4.0 uploaded 2026-09-27).

Every release:
```bash
cd cine-rec-engine
python -m build && twine upload dist/*
git tag vX.Y.Z && git push --tags
```

Token: stored in your password manager — use a **project-scoped** token
("Single project: cine-rec-engine"), not account-wide. Rotate any token
that was ever pasted in plain text (chat, tickets, logs).
