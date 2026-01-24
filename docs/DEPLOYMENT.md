# Deployment Guide

Complete guide for deploying the RAG Microservices platform.

## Table of Contents

1. [Local Development](#local-development)
2. [Production Deployment](#production-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Security Considerations](#security-considerations)
5. [Monitoring & Logging](#monitoring--logging)
6. [Backup & Recovery](#backup--recovery)

## Local Development

### Prerequisites

- Docker Desktop 20.10+
- Docker Compose 2.0+
- 16GB RAM recommended
- Pinecone account and API key

### Setup

```bash
# Clone and navigate
cd rag-microservices

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Build and start
./scripts/setup.sh
./scripts/start.sh
```

### Development Workflow

```bash
# View logs
docker-compose logs -f rag-service

# Restart a service
docker-compose restart rag-service

# Rebuild after code changes
docker-compose up -d --build rag-service

# Access container shell
docker exec -it rag-service bash
```

## Production Deployment

### Prerequisites

- Production server with Docker
- Domain name and SSL certificates
- Production Pinecone account
- Reverse proxy (Nginx/Traefik)
- Monitoring system

### 1. Environment Configuration

Create `.env.production`:

```bash
# Pinecone
PINECONE_API_KEY=your_production_key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX=production-index

# Ollama
OLLAMA_MODEL=llama3.1
OLLAMA_HOST=http://ollama:11434

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=strong_password_here

# API
API_URL=https://api.yourdomain.com

# Security
KONG_ADMIN_TOKEN=your_secure_admin_token
```

### 2. Docker Compose for Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: always
    networks:
      - rag-network

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    restart: always
    networks:
      - rag-network

  rag-service:
    build:
      context: ./services/rag-service
      dockerfile: Dockerfile
    environment:
      - OLLAMA_HOST=${OLLAMA_HOST}
      - OLLAMA_MODEL=${OLLAMA_MODEL}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_INDEX=${PINECONE_INDEX}
      - REDIS_HOST=${REDIS_HOST}
      - REDIS_PORT=${REDIS_PORT}
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    restart: always
    networks:
      - rag-network

  kong:
    build:
      context: ./services/kong
      dockerfile: Dockerfile
    environment:
      - KONG_DATABASE=off
      - KONG_DECLARATIVE_CONFIG=/usr/local/kong/declarative/kong.yml
      - KONG_ADMIN_TOKEN=${KONG_ADMIN_TOKEN}
    restart: always
    networks:
      - rag-network

  frontend:
    build:
      context: ./services/frontend
      dockerfile: Dockerfile
      target: production
    environment:
      - NEXT_PUBLIC_API_URL=${API_URL}
      - NODE_ENV=production
    restart: always
    networks:
      - rag-network

volumes:
  redis_data:
  ollama_data:

networks:
  rag-network:
    driver: bridge
```

### 3. Nginx Reverse Proxy

Create `/etc/nginx/sites-available/rag-microservices`:

```nginx
upstream frontend {
    server localhost:3000;
}

upstream kong {
    server localhost:8000;
}

# Frontend
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/ssl/certs/your_cert.pem;
    ssl_certificate_key /etc/ssl/private/your_key.pem;

    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

# API Gateway
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/ssl/certs/your_cert.pem;
    ssl_certificate_key /etc/ssl/private/your_key.pem;

    location / {
        proxy_pass http://kong;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_cache_bypass $http_upgrade;

        # Timeouts for streaming
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

### 4. Deploy

```bash
# Load production env
export $(cat .env.production | xargs)

# Pull latest images
docker-compose -f docker-compose.prod.yml pull

# Build custom images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

## Cloud Deployment

### AWS Deployment

#### Using ECS (Elastic Container Service)

1. **Push images to ECR:**

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag rag-service:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-service:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-service:latest
```

2. **Create ECS Task Definitions**
3. **Configure Application Load Balancer**
4. **Set up Auto Scaling**
5. **Configure CloudWatch for logging**

#### Using EKS (Kubernetes)

See `docs/KUBERNETES.md` for Kubernetes deployment guide.

### Google Cloud Platform

#### Using Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/rag-service

# Deploy
gcloud run deploy rag-service \
  --image gcr.io/PROJECT_ID/rag-service \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure

#### Using Azure Container Instances

```bash
# Create resource group
az group create --name rag-microservices --location eastus

# Create container registry
az acr create --resource-group rag-microservices \
  --name ragregistry --sku Basic

# Deploy containers
az container create --resource-group rag-microservices \
  --name rag-service \
  --image ragregistry.azurecr.io/rag-service:latest \
  --dns-name-label rag-service \
  --ports 8001
```

## Security Considerations

### 1. Environment Variables

Never commit `.env` files. Use secrets management:

- **AWS**: AWS Secrets Manager
- **GCP**: Secret Manager
- **Azure**: Key Vault
- **Kubernetes**: Sealed Secrets

### 2. Kong Authentication

Add JWT or Key Auth plugin to Kong:

```yaml
# In kong.yml
plugins:
  - name: jwt
    config:
      secret_is_base64: false

  - name: key-auth
    config:
      key_names: ["apikey"]
```

### 3. Network Security

- Use private networks for internal services
- Only expose Kong Gateway publicly
- Implement firewall rules
- Use VPC/VNet isolation

### 4. SSL/TLS

- Use Let's Encrypt for free SSL
- Enable HSTS headers
- Force HTTPS redirects

### 5. Rate Limiting

Already configured in Kong. Adjust as needed:

```yaml
plugins:
  - name: rate-limiting
    config:
      minute: 100
      hour: 1000
      policy: redis
      redis_host: redis
```

## Monitoring & Logging

### Prometheus & Grafana

Kong has Prometheus plugin enabled. Set up Prometheus:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'kong'
    static_configs:
      - targets: ['kong:8001']
```

### ELK Stack

Centralized logging with Elasticsearch, Logstash, Kibana:

```yaml
# docker-compose.yml addition
elasticsearch:
  image: elasticsearch:8.11.0
  environment:
    - discovery.type=single-node

logstash:
  image: logstash:8.11.0
  volumes:
    - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

kibana:
  image: kibana:8.11.0
  ports:
    - "5601:5601"
```

### Health Checks

All services have health endpoints:

```bash
curl http://localhost:8001/health
```

Set up external monitoring (UptimeRobot, Pingdom, etc.)

## Backup & Recovery

### Redis Backup

```bash
# Manual backup
docker exec rag-redis redis-cli SAVE

# Copy backup
docker cp rag-redis:/data/dump.rdb ./backups/

# Restore
docker cp ./backups/dump.rdb rag-redis:/data/
docker-compose restart redis
```

### Automated Backups

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/redis"

docker exec rag-redis redis-cli SAVE
docker cp rag-redis:/data/dump.rdb $BACKUP_DIR/dump_$DATE.rdb

# Keep only last 7 days
find $BACKUP_DIR -name "dump_*.rdb" -mtime +7 -delete
```

Add to crontab:
```bash
0 2 * * * /path/to/backup.sh
```

### Pinecone Backup

Pinecone handles backups automatically. For additional safety:

```python
# Export vectors
import pinecone

pc = pinecone.Pinecone(api_key="your-key")
index = pc.Index("your-index")

# Fetch all vectors (paginated)
results = index.query(vector=[0]*384, top_k=10000, include_metadata=True)
# Save to file
```

## Scaling Strategies

### Horizontal Scaling

Scale specific services:

```bash
docker-compose up -d --scale rag-service=3
```

### Load Balancing

Kong handles load balancing. Add more upstreams:

```yaml
upstreams:
  - name: rag-service-upstream
    targets:
      - target: rag-service-1:8001
      - target: rag-service-2:8001
      - target: rag-service-3:8001
```

### Caching

Add Redis caching for repeated queries in RAG service.

## Performance Optimization

### 1. Ollama Model Selection

Smaller models = faster inference:
- llama3.1 (4.7GB) - Best quality
- llama2 (3.8GB) - Good balance
- mistral (4.1GB) - Fast & efficient

### 2. Vector Search Optimization

- Use `top_k` wisely (default: 5)
- Prefer similarity strategy for speed
- Use metadata filtering in Pinecone

### 3. Connection Pooling

Already configured in services. Adjust pool sizes in production.

### 4. CDN for Frontend

Use CloudFront, CloudFlare, or Fastly for static assets.

## Disaster Recovery

### Recovery Plan

1. **Restore Redis backup**
2. **Verify Pinecone connection**
3. **Restart all services**
4. **Run health checks**
5. **Validate with test queries**

### Recovery Script

```bash
#!/bin/bash
# recover.sh

echo "Starting disaster recovery..."

# Stop services
docker-compose down

# Restore Redis
docker cp ./backups/latest/dump.rdb rag-redis:/data/

# Start services
docker-compose up -d

# Wait for health
./scripts/start.sh

echo "Recovery complete!"
```

## Cost Optimization

- Use spot instances for non-critical workloads
- Scale down during low traffic
- Use smaller Ollama models
- Optimize Pinecone queries
- Implement aggressive caching

---

For questions or issues, refer to the main [README.md](../README.md) or check the [Troubleshooting Guide](TROUBLESHOOTING.md).
