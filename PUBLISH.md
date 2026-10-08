# Publishing to PyPI

**Status: LIVE** — [pypi.org/project/cine-rec-engine](https://pypi.org/project/cine-rec-engine/)

## How releases ship (automated)

```bash
git tag vX.Y.Z && git push origin vX.Y.Z
```

That's the whole release. The `release` workflow (`.github/workflows/release.yml`)
runs the offline suite, verifies the tag matches both `pyproject.toml` and the
package version, builds, and uploads to PyPI via **Trusted Publishing (OIDC)** —
no API token is stored in GitHub, on the runner, or anywhere else.

One-time setup (per repo, do it before the first automated release):

1. **GitHub** → repo *Settings → Environments → New environment* → name it
   `pypi` (optionally add required reviewers there for a publish approval).
2. **PyPI** → *cine-rec-engine → Publishing → Add a pending publisher*:
   - PyPI Project: `cine-rec-engine`
   - Owner: `DovBerKaplan`
   - Repository: `cine-rec-engine`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

Order matters only for the first tag: until both steps are done, the
workflow fails at the publish step with an OIDC/trusted-publisher error.

## Manual upload (fallback)

Only if the automated path is broken:

```bash
cd cine-rec-engine
.venv/bin/python -m build
.venv/bin/python -m twine upload dist/cine_rec_engine-X.Y.Z*
```

Token: stored in your password manager — use a **project-scoped** token
("Single project: cine-rec-engine"), not account-wide. Rotate any token
that was ever pasted in plain text (chat, tickets, logs).
