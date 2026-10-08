# cine-rec-engine as a standalone service.
#   docker build -t cine-rec-engine .
#   docker run -p 8000:8000 -e DATABASE_URL=postgresql://user:pw@host/db cine-rec-engine
#
# CINE_REC_PORT overrides the in-container port (healthcheck follows);
# CINE_REC_AUTO_INIT=1 applies the schema on boot; CINE_REC_SCHEMA_MAP
# points at a mounted table-name map — see deploy/docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# metadata + packages only — no tests, demo data, or docs in the image
COPY pyproject.toml README.md LICENSE ./
COPY cine_rec_engine/ cine_rec_engine/
COPY ingest/ ingest/

RUN pip install --no-cache-dir ".[pg,serve]" \
    && useradd --create-home --shell /usr/sbin/nologin cine
USER cine

ENV CINE_REC_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request, sys; \
        sys.exit(0 if urllib.request.urlopen(\
            'http://127.0.0.1:%s/health' % os.getenv('CINE_REC_PORT', '8000'), \
            timeout=2).status == 200 else 1)"

CMD ["sh", "-c", "exec uvicorn cine_rec_engine.serve:app \
    --host 0.0.0.0 --port \"${CINE_REC_PORT:-8000}\""]
