# AEROTICA Deployment Guide

This guide covers deployment options for the AEROTICA atmospheric kinetic energy mapping system.

## Table of Contents
- [Quick Deployments](#quick-deployments)
  - [Netlify (Dashboard)](#netlify-dashboard)
  - [Hugging Face Spaces (Interactive)](#hugging-face-spaces-interactive)
  - [PyPI (Python Package)](#pypi-python-package)
  - [ReadTheDocs (Documentation)](#readthedocs-documentation)
- [Production Deployments](#production-deployments)
  - [Docker Compose](#docker-compose)
  - [Kubernetes](#kubernetes)
  - [Cloud Providers](#cloud-providers)
- [Database Setup](#database-setup)
- [Configuration](#configuration)
- [Security](#security)
- [Backup & Recovery](#backup--recovery)
- [Scaling](#scaling)
- [Real-time Processing](#real-time-processing)

## Quick Deployments

### Netlify (Dashboard)

The AEROTICA dashboard is pre-configured for Netlify deployment.

#### Automatic Deployment

1. Connect your Git repository to Netlify
2. Use these settings:
   - Build command: `mkdocs build`
   - Publish directory: `site`
   - Environment variables: none required

3. Or use the `netlify.toml` configuration:
```toml
[build]
  command = "mkdocs build"
  publish = "site"

[build.environment]
  PYTHON_VERSION = "3.11"

[redirects]
  from = "/api/*"
  to = "https://your-api-domain.com/:splat"
  status = 200
```

Manual Deployment

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
netlify deploy --prod --dir=site
```

Live demo: https://aerotica-science.netlify.app

Hugging Face Spaces (Interactive)

Deploy an interactive version with Streamlit for real-time wind visualization.

Using Git

```bash
# Create space at huggingface.co/new-space
# Choose Streamlit SDK

git clone https://huggingface.co/spaces/aerotica/aerotica
cp -r dashboard/* aerotica-space/
cd aerotica-space

# Create README.md
cat > README.md << 'EOF'
---
title: AEROTICA
emoji: 🌪️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# AEROTICA
Real-time atmospheric kinetic energy mapping and gust pre-alerting.
