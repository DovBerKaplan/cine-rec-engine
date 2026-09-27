# Publishing to PyPI (one manual step)

The build is verified (`python -m build` → wheel installs + imports).

First time only:
1. Create the PyPI account + API token (pypi.org → account settings).
2. `pip install twine build`

Every release:
```bash
cd cine-rec-engine
python -m build
twine upload dist/cine_rec_engine-*
```
Then tag: `git tag v0.4.0 && git push --tags`.
