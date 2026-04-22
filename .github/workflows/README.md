# GitHub Actions CI/CD Documentation

## Overview

This repository uses GitHub Actions to build and push Docker images to GitHub Container Registry (`ghcr.io`) when you push a **version tag** matching `v*` (for example `v0.70.0` or `v0.70.5-dev`). You can also run the workflow manually from **Actions → Build and Push Docker Images → Run workflow** (optional **version** input; if omitted, the `VERSION` file on the selected branch is used).

## Main vs dev release channels

Releases are **tag-driven** and aligned with two branches:

| Branch | Version examples (in `VERSION` / tag) | GHCR image prefix | Intended visibility |
|--------|----------------------------------------|-------------------|---------------------|
| **`main`** | `0.70.5` (no `-dev` suffix) | `ghcr.io/<owner>/bastion-<service>:…` | **Public** anonymous pulls |
| **`dev`** | `0.70.5-dev` (suffix **`-dev`**) | `ghcr.io/<owner>/bastion-dev-<service>:…` | **Private** (org members / auth) |

The workflow **checks** that:

- **Production** tags (`v0.70.0`, etc.) point at a commit that is on **`origin/main`**.
- **Development** tags (`v0.70.5-dev`, etc.) point at a commit that is on **`origin/dev`**.
- **Manual runs** use **`main`** for production versions and **`dev`** for `-dev` versions (per `VERSION` / input).

### Why two image prefixes (`bastion-` vs `bastion-dev-`)?

On GitHub Container Registry, **visibility is per package**, not per tag. A single package cannot be “public for some tags and private for others.” Also, **a package that has been made public cannot be made private again.**

So dev and production **must** use **different package names** if you want private prerelease images and public release images from the same repo. Dev builds therefore publish to **`bastion-dev-<service>`**; production builds publish to **`bastion-<service>`** (unchanged for stable releases).

### Making production images public

Workflows that publish with `GITHUB_TOKEN` follow GitHub’s [default rules](https://docs.github.com/en/packages/managing-github-packages-using-github-actions-workflows/publishing-and-installing-a-package-with-github-actions#default-permissions-and-access-settings-for-packages-modified-through-workflows) for new packages. In practice:

- Keep **organization** defaults so **new packages stay private** if you need dev images to remain private.
- For **`bastion-*`** production packages, use each package’s **Package settings → Change visibility → Public** once (or your org’s documented process). After that, new pushes add tags to the same **public** package.

Dev **`bastion-dev-*`** packages should stay **private**.

## Workflow Trigger

The workflow (`build-and-push.yml`) runs when:

1. You push a git tag matching `v*`, or  
2. You use **workflow_dispatch** (and satisfy the branch rules above).

### Manual run: one image or the full matrix

In **Actions → Build and Push Docker Images → Run workflow**, use the **image** dropdown:

- **`all`** (default) — build and push every image.  
- **`postgres_only`** — only **`bastion-postgres`** and **`bastion-postgres-data`** (e.g. SQL init changes).  
- Any other option — a **single** service (e.g. **postgres** for the main DB image only).

Pushing a **`v*`** tag still builds the full matrix (no menu).

## Version Management

Version numbers are tracked in the `VERSION` file at the repository root. This file is the source of truth for in-repo tooling; **release tags** carry the authoritative version for CI.

### Version format

- **Production**: `0.70.5` → tag **`v0.70.5`** (no `-dev` in the version string).
- **Development**: `0.70.5-dev` → tag **`v0.70.5-dev`** (suffix **`-dev`** marks the dev channel).

## Image Tagging Strategy

When a tag is pushed, each image receives multiple tags:

1. **Version tag**: Exact version from the tag (e.g. `0.70.5-dev` or `0.70.5`).
2. **Latest tag**: `latest-dev` for dev versions, `latest` for production.
3. **SHA tag**: Short git SHA (e.g. `sha-abc1234`).
4. **Branch tag**: `dev` or `main` (label only; channel is determined by the version suffix).

## Images Built

The workflow builds and pushes first-party images from `docker-compose.yml`, including **`bastion-postgres`** and **`bastion-postgres-data`** (init SQL baked in; no bind mount of `backend/postgres_init` required in production). The Vite HMR image **`bastion-frontend-dev`** is **not** published from CI; build it locally with `docker compose --profile dev --build` (see `frontend/Dockerfile.dev`). **`bastion-celery-flower`** is also **not** published from CI; build it locally from `backend/Dockerfile.celery-flower` if you use the optional Flower service in Compose.

1. `bastion-postgres` / `bastion-dev-postgres` — main DB (`backend/postgres_init` in image)
2. `bastion-postgres-data` / `bastion-dev-postgres-data` — data workspace DB (`data-service/sql` in image)
3. `bastion-backend` / `bastion-dev-backend` (by channel)
4. `bastion-tools-service` / `bastion-dev-tools-service`
5. `bastion-cli-worker` / `bastion-dev-cli-worker`
6. `bastion-celery-worker` / `bastion-dev-celery-worker`
7. `bastion-celery-beat` / `bastion-dev-celery-beat`
8. `bastion-frontend` / `bastion-dev-frontend`
9. `bastion-webdav` / `bastion-dev-webdav`
10. `bastion-llm-orchestrator` / `bastion-dev-llm-orchestrator`
11. `bastion-vector-service` / `bastion-dev-vector-service`
12. `bastion-data-service` / `bastion-dev-data-service`
13. `bastion-image-vision-service` / `bastion-dev-image-vision-service`
14. `bastion-connections-service` / `bastion-dev-connections-service`
15. `bastion-voice-service` / `bastion-dev-voice-service`
16. `bastion-document-service` / `bastion-dev-document-service`
17. `bastion-crawl4ai-service` / `bastion-dev-crawl4ai-service`
18. `bastion-bbs-server` / `bastion-dev-bbs-server`

## Image Naming Convention

**Production** (example org `myorg`, version `0.70.5`):

- `ghcr.io/myorg/bastion-backend:0.70.5`
- `ghcr.io/myorg/bastion-backend:latest`
- `ghcr.io/myorg/bastion-backend:sha-abc1234`
- `ghcr.io/myorg/bastion-backend:main`

**Development** (version `0.70.5-dev`):

- `ghcr.io/myorg/bastion-dev-backend:0.70.5-dev`
- `ghcr.io/myorg/bastion-dev-backend:latest-dev`
- `ghcr.io/myorg/bastion-dev-backend:sha-abc1234`
- `ghcr.io/myorg/bastion-dev-backend:dev`

`myorg` is `${{ github.repository_owner }}` in **lowercase** (GHCR requires lowercase paths).

## Usage Workflow

### Development release (from `dev`)

1. Update `VERSION` on `dev`, e.g. `0.70.5-dev`.
2. Commit and push to **`dev`**.
3. Tag and push (note the **`v`** prefix and **`-dev`** in the tag name):

   ```bash
   git tag v0.70.5-dev
   git push origin v0.70.5-dev
   ```

Images are published under **`bastion-dev-*`** and should remain **private** at the org level.

### Production release (from `main`)

1. Merge `dev` → **`main`** when ready.
2. Set `VERSION` to a **non-`-dev`** version (e.g. `0.70.5`) on `main` if needed.
3. Tag and push:

   ```bash
   git tag v0.70.5
   git push origin v0.70.5
   ```

Images are published under **`bastion-*`**. Set GHCR package visibility to **public** for these packages (once per package) so anonymous `docker pull` works for operators.

## GitHub Repository Settings

### Required Permissions

The workflow uses `GITHUB_TOKEN` with:

- **Contents**: Read (checkout; fetch `main` / `dev` for checks)
- **Packages**: Write (push images to GHCR)

### Enabling GitHub Packages

1. Repository **Settings → Actions → General**
2. Under **Workflow permissions**, enable **Read and write permissions** if your org allows it.

## Build Performance

- **First build**: Longer when there is no cache.
- **Subsequent builds**: Faster with BuildKit cache in GitHub Actions cache.
- **Parallel builds**: Images build sequentially in one job (a matrix can be added later for speed).

## Troubleshooting

### Workflow Not Triggering

- Tag must start with `v` (e.g. `v0.70.5-dev`, not `0.70.5-dev` alone on the remote).
- Confirm the tag was pushed to the remote.

### “Production image tags must point at a commit on origin/main” (or dev variant)

- The tagged commit must be an ancestor of **`origin/main`** (production) or **`origin/dev`** (dev). Create the tag from the correct branch history.

### Authentication Errors

- Confirm Actions can write packages (`packages: write`).
- Confirm the org allows GitHub Packages.

### Image Not Found After Push

- Confirm the workflow finished successfully.
- Use the right prefix: **`bastion-dev-`** for `-dev` versions, **`bastion-`** for production.
- Private images require `docker login ghcr.io` with a token that can read the package.

## Using Images in Docker Compose

Override `image:` to the GHCR reference you need, for example:

```yaml
services:
  backend:
    image: ghcr.io/myorg/bastion-dev-backend:0.70.5-dev
```

or for production:

```yaml
services:
  backend:
    image: ghcr.io/myorg/bastion-backend:0.70.5
```

Replace `myorg` with your GitHub org or user (lowercase).

## Cache Management

The workflow uses GitHub Actions cache (GHA) for BuildKit cache (`mode=min`), which speeds rebuilds when layers are unchanged.
