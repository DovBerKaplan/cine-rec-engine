# HTTP API

Interactive reference, generated from the app factory (`/v1/*` routes
are canonical; the bare paths are deprecated aliases — see the
[stability policy](stability.md)). Set `CINE_REC_API_KEYS` to require
an `X-API-Key` (or `Authorization: Bearer`) on every data route;
`/health` and the docs stay open.

To regenerate the schema after route changes:

```bash
python -c "import json; from cine_rec_engine.serve import create_app as c; \
    print(json.dumps(c(service=object()).openapi(), indent=1))" \
    > docs/api/openapi.json
```

<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
  window.addEventListener('load', () => {
    SwaggerUIBundle({
      url: 'api/openapi.json',
      dom_id: '#swagger-ui',
      docExpansion: 'list',
      tryItOutEnabled: false,
    });
  });
</script>
