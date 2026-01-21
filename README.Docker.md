# Docker files (init)

These files live at the repository root so you can run Docker commands from this
folder.

## What is included
- `.dockerignore` to keep images small and clean.
- `compose.yaml` with a single `app` service.

## Expected next step
`compose.yaml` assumes a `Dockerfile` exists at the repository root. Create one
that installs
your dependencies and sets a default `CMD`.

## Usage
From the repository root:

```bash
docker compose build
docker compose up
```

To stop:

```bash
docker compose down
```
