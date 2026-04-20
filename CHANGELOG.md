# Changelog

All notable changes to Bastion AI Workspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Update: GitHub Actions `build-and-push` workflow builds and pushes all first-party Compose images (tools-service, cli-worker, Celery variants, document-service, connections-service, voice-service, image-vision-service, bbs-server, frontend-dev) on version tags.

## [0.70.0] - 2026-04-20

Release 0.70.0 aggregates a large set of improvements across the web app, BBS, document pipeline, Agent Factory / agent lines, Docker stacks, tooling, and operational hardening since 0.55.0. Prior per-change entries were retired in favor of this milestone cut; see git history for granular commits.
