# Deployment Guide

## Docker Deployment

```bash
# Build image
docker build -t aerotica:latest .

# Run container
docker run -p 8000:8000 aerotica:latest
```

Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Kubernetes

```bash
# Apply configurations
kubectl apply -f kubernetes/

# Scale services
kubectl scale deployment aerotica-api --replicas=3
```

Configuration

Environment Variables

Variable Description Default
DB_PASSWORD Database password required
SLACK_WEBHOOK Slack webhook URL optional
PINN_DEVICE Device for inference cpu
LOG_LEVEL Logging level INFO

Production Checklist

· Set strong database password
· Enable SSL/TLS
· Configure backups
· Set up monitoring
· Configure logging
· Set resource limits
