# cine-rec-engine as a standalone service.
#   docker build -t cine-rec-engine .
#   docker run -p 8000:8000 -e DATABASE_URL=postgresql://user:pw@host/db cine-rec-engine
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

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen(\
            'http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "cine_rec_engine.serve:app", "--host", "0.0.0.0", "--port", "8000"]
