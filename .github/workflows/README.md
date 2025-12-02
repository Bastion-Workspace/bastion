# GitHub Actions CI/CD Documentation

## Overview

This repository uses GitHub Actions to automatically build and push Docker images to GitHub Container Registry (ghcr.io) when git tags are created.

## Workflow Trigger

The workflow (`build-and-push.yml`) triggers automatically when you push a git tag matching the pattern `v*` (e.g., `v0.10.1-dev` or `v0.10.1`).

## Version Management

Version numbers are tracked in the `VERSION` file at the repository root. This file serves as the source of truth for the current version.

### Version Format

- **Development versions**: `0.10.1-dev` (tagged as `v0.10.1-dev`)
- **Production versions**: `0.10.1` (tagged as `v0.10.1`)

## Image Tagging Strategy

When a tag is pushed, each image receives multiple tags for flexibility:

1. **Version tag**: The exact version from the git tag (e.g., `0.10.1-dev`)
2. **Latest tag**: `latest-dev` for development tags, `latest` for production tags
3. **SHA tag**: Git commit hash for traceability (e.g., `sha-abc1234`)
4. **Branch tag**: Branch name (`dev` or `main`)

## Images Built

The workflow builds and pushes 6 Docker images:

1. `bastion-backend` - Backend API service
2. `bastion-frontend` - Frontend web UI
3. `bastion-webdav` - WebDAV server for OrgMode sync
4. `bastion-llm-orchestrator` - LLM orchestrator service
5. `bastion-vector-service` - Vector embedding service
6. `bastion-data-service` - Data workspace service

## Image Naming Convention

All images are pushed to: `ghcr.io/{GITHUB_ORG}/bastion-{service}:{tag}`

Example for backend service with version `0.10.1-dev`:
- `ghcr.io/{GITHUB_ORG}/bastion-backend:0.10.1-dev`
- `ghcr.io/{GITHUB_ORG}/bastion-backend:latest-dev`
- `ghcr.io/{GITHUB_ORG}/bastion-backend:sha-abc1234`
- `ghcr.io/{GITHUB_ORG}/bastion-backend:dev`

## Usage Workflow

### Development Release

1. Update the `VERSION` file with the new version:
   ```bash
   echo "0.10.1" > VERSION
   ```

2. Commit and push the version change:
   ```bash
   git add VERSION
   git commit -m "Bump version to 0.10.1"
   git push origin dev
   ```

3. Create and push the development tag:
   ```bash
   git tag v0.10.1-dev
   git push origin v0.10.1-dev
   ```

The GitHub Actions workflow will automatically:
- Build all 6 images
- Tag them with `0.10.1-dev`, `latest-dev`, SHA, and `dev`
- Push to GitHub Container Registry

### Production Release

1. Merge dev branch to main:
   ```bash
   git checkout main
   git merge dev
   git push origin main
   ```

2. Create and push the production tag:
   ```bash
   git tag v0.10.1
   git push origin v0.10.1
   ```

The workflow will build and tag images with:
- `0.10.1`, `latest`, SHA, and `main`

## GitHub Repository Settings

### Required Permissions

The workflow requires the following permissions (automatically granted via `GITHUB_TOKEN`):

- **Contents**: Read (to checkout code)
- **Packages**: Write (to push images to GHCR)

### Enabling GitHub Packages

1. Go to your repository settings
2. Navigate to "Actions" → "General"
3. Under "Workflow permissions", ensure "Read and write permissions" is selected
4. Under "Packages", ensure "Allow GitHub Actions to create and approve pull requests" is enabled

### Container Registry Access

Images are pushed to GitHub Container Registry (ghcr.io). By default, images are **private**. To make them public:

1. Go to your repository's "Packages" section
2. Click on a package (e.g., `bastion-backend`)
3. Click "Package settings"
4. Scroll to "Danger Zone" and click "Change visibility" → "Public"

## Build Performance

- **First build**: Takes longer as there's no cache
- **Subsequent builds**: Faster due to BuildKit cache stored in GitHub Actions cache
- **Parallel builds**: All 6 images build sequentially (can be optimized to parallel if needed)

## Troubleshooting

### Workflow Not Triggering

- Ensure the tag starts with `v` (e.g., `v0.10.1-dev`, not `0.10.1-dev`)
- Check that the tag was pushed to the remote repository
- Verify workflow file is in `.github/workflows/` directory

### Authentication Errors

- Ensure `GITHUB_TOKEN` has write permissions to packages
- Check repository settings for Actions permissions
- Verify the repository owner/organization has GitHub Packages enabled

### Build Failures

- Check the Actions logs for specific error messages
- Verify all Dockerfiles are present and valid
- Ensure all required files and dependencies are in the repository

### Image Not Found After Push

- Images are private by default - check package visibility settings
- Verify the image name matches: `ghcr.io/{ORG}/bastion-{service}:{tag}`
- Check that the workflow completed successfully

## Using Images in Docker Compose

To use the versioned images from GHCR instead of building locally, update `docker-compose.yml`:

```yaml
services:
  backend:
    image: ghcr.io/{GITHUB_ORG}/bastion-backend:0.10.1-dev
    # Remove or comment out the build section
    # build:
    #   context: .
    #   dockerfile: ./backend/Dockerfile
```

Replace `{GITHUB_ORG}` with your GitHub organization or username.

## Cache Management

The workflow uses GitHub Actions cache (GHA) for BuildKit cache. This provides:
- Faster rebuilds when dependencies haven't changed
- Automatic cache management by GitHub
- No manual cache cleanup required

Cache is stored per repository and persists across workflow runs.


