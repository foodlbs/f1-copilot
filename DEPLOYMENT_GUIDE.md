# 🏗️ F1 Race Strategy Analyzer - Deployment Guide

Complete guide for deploying the F1 Race Strategy Analyzer to production environments.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [AWS Deployment](#aws-deployment)
6. [Monitoring & Logging](#monitoring--logging)
7. [Security Considerations](#security-considerations)
8. [Scaling](#scaling)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Applications                            │
│              (Web Dashboard, Mobile App, API Consumers)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        API Gateway / Load Balancer                       │
│                    (AWS ALB / Nginx / Cloud Run)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │   FastAPI       │ │   FastAPI       │ │   FastAPI       │
          │   Instance 1    │ │   Instance 2    │ │   Instance N    │
          └─────────────────┘ └─────────────────┘ └─────────────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   Pinecone    │           │   Claude AI   │           │    AWS S3     │
│  Vector DB    │           │   (Anthropic) │           │   (Models)    │
└───────────────┘           └───────────────┘           └───────────────┘
                                                                │
                                      ┌─────────────────────────┘
                                      ▼
                            ┌───────────────────┐
                            │   AWS DynamoDB    │
                            │   (Metadata)      │
                            └───────────────────┘
```

---

## Prerequisites

### Required Accounts & API Keys

| Service | Purpose | Cost |
|---------|---------|------|
| [Pinecone](https://www.pinecone.io/) | Vector database for semantic search | Free tier / $70+/mo |
| [Anthropic](https://console.anthropic.com/) | Claude AI for strategy generation | Pay per use |
| [AWS](https://aws.amazon.com/) | Cloud infrastructure (optional) | Pay per use |

### System Requirements

**Minimum (Development)**
- 4 GB RAM
- 2 CPU cores
- 10 GB disk space
- Python 3.9+

**Recommended (Production)**
- 8+ GB RAM
- 4+ CPU cores
- 50+ GB disk space
- GPU (optional, for training)

---

## Local Deployment

### Step 1: Clone and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd f1-race-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy example environment
cp .env.example .env

# Edit with your API keys
nano .env
```

Required variables:
```
PINECONE_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

### Step 3: Initialize System

```bash
# Train models (quick mode for testing)
python train_models.py --quick --init-vectordb

# Or full training
python train_models.py --start-year 2015 --end-year 2024
```

### Step 4: Start Server

```bash
# Development mode (with auto-reload)
DEBUG=true python fastapi_backend.py

# Production mode
python fastapi_backend.py
```

Access the API at http://localhost:8000

---

## Docker Deployment

### Basic Deployment

```bash
# Build and start
docker-compose up -d api

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

### Development Mode

```bash
# Start with hot reload
docker-compose --profile dev up api-dev
```

### Full Stack with Monitoring

```bash
# Start all services
docker-compose --profile full --profile monitoring up -d

# This starts:
# - API server
# - Redis cache
# - Prometheus
# - Grafana
```

### Training with Docker

```bash
# Quick training
docker-compose --profile training-quick up trainer-quick

# Full training
docker-compose --profile training up trainer
```

---

## AWS Deployment

### Option 1: EC2 Deployment

#### Step 1: Launch EC2 Instance

```bash
# Recommended: t3.large or larger
# AMI: Amazon Linux 2 or Ubuntu 22.04
# Storage: 30+ GB
```

#### Step 2: Setup Instance

```bash
# SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install Docker
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone <your-repo-url>
cd f1-race-analyzer

# Configure environment
cp .env.example .env
nano .env

# Start services
docker-compose up -d api
```

#### Step 3: Configure Security Group

```
Inbound Rules:
- SSH (22): Your IP
- HTTP (80): 0.0.0.0/0
- HTTPS (443): 0.0.0.0/0
- Custom (8000): 0.0.0.0/0 (or through ALB)
```

### Option 2: Lambda Deployment

#### Step 1: Create Lambda Functions

```bash
# Package dependencies
pip install -r requirements.txt -t ./package
cd package && zip -r ../deployment.zip .
cd .. && zip -g deployment.zip *.py
```

#### Step 2: Deploy with SAM (Serverless Application Model)

Create `template.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 30
    MemorySize: 512
    Runtime: python3.11
    Environment:
      Variables:
        PINECONE_API_KEY: !Ref PineconeApiKey
        ANTHROPIC_API_KEY: !Ref AnthropicApiKey

Parameters:
  PineconeApiKey:
    Type: String
    NoEcho: true
  AnthropicApiKey:
    Type: String
    NoEcho: true

Resources:
  DataIngestionFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: lambda_functions.race_data_ingestion_handler
      CodeUri: .
      Events:
        ScheduledIngestion:
          Type: Schedule
          Properties:
            Schedule: cron(0 6 ? * MON *)

  StrategyGenerationFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: lambda_functions.strategy_generation_handler
      CodeUri: .
      MemorySize: 1024
      Events:
        ApiGateway:
          Type: Api
          Properties:
            Path: /strategy/generate
            Method: post

  SemanticQueryFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: lambda_functions.semantic_query_handler
      CodeUri: .
      Events:
        ApiGateway:
          Type: Api
          Properties:
            Path: /query
            Method: post
```

Deploy:

```bash
sam build
sam deploy --guided
```

#### Step 3: Setup EventBridge Rules

```bash
# Create rule for weekly data refresh
aws events put-rule \
  --name "F1WeeklyRefresh" \
  --schedule-expression "cron(0 6 ? * MON *)"

# Add Lambda target
aws events put-targets \
  --rule "F1WeeklyRefresh" \
  --targets "Id"="1","Arn"="arn:aws:lambda:region:account:function:WeeklyRefresh"
```

### Option 3: ECS/Fargate Deployment

```bash
# Create ECR repository
aws ecr create-repository --repository-name f1-analyzer

# Build and push image
docker build -t f1-analyzer .
docker tag f1-analyzer:latest $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/f1-analyzer:latest
aws ecr get-login-password | docker login --username AWS --password-stdin $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker push $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/f1-analyzer:latest
```

Create ECS task definition and service via AWS Console or CLI.

---

## Monitoring & Logging

### Prometheus + Grafana (Docker)

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### CloudWatch (AWS)

```python
# Add to your Lambda functions
import boto3

cloudwatch = boto3.client('cloudwatch')

def put_metric(name, value, unit='Count'):
    cloudwatch.put_metric_data(
        Namespace='F1RaceAnalyzer',
        MetricData=[{
            'MetricName': name,
            'Value': value,
            'Unit': unit
        }]
    )
```

### Logging Configuration

```python
# In production, use structured logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()
logger.info("strategy_generated", circuit="monaco", confidence=0.85)
```

---

## Security Considerations

### API Key Management

```bash
# Use AWS Secrets Manager
aws secretsmanager create-secret \
  --name f1-analyzer/api-keys \
  --secret-string '{"PINECONE_API_KEY":"xxx","ANTHROPIC_API_KEY":"xxx"}'

# Retrieve in code
import boto3

def get_secrets():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='f1-analyzer/api-keys')
    return json.loads(response['SecretString'])
```

### HTTPS Configuration

```nginx
# nginx.conf for HTTPS termination
server {
    listen 443 ssl;
    server_name api.f1analyzer.com;
    
    ssl_certificate /etc/letsencrypt/live/api.f1analyzer.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.f1analyzer.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Rate Limiting

```python
# Add to FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/strategy/generate")
@limiter.limit("10/minute")
async def generate_strategy(request: Request, ...):
    ...
```

---

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml with replicas
services:
  api:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 2G
```

### Load Balancing with Nginx

```nginx
upstream f1_api {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://f1_api;
    }
}
```

### Caching Strategies

```python
# Redis caching for expensive operations
import redis
import json

redis_client = redis.Redis(host='redis', port=6379)

def get_cached_strategy(key):
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None

def cache_strategy(key, strategy, ttl=3600):
    redis_client.setex(key, ttl, json.dumps(strategy))
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Increase container memory
docker-compose up -d --scale api=1 --memory=4g
```

**2. Slow Cold Starts (Lambda)**
```python
# Use provisioned concurrency
# Or keep functions warm with scheduled pings
```

**3. Rate Limits**
```python
# Implement exponential backoff
import time

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait = 2 ** attempt
            time.sleep(wait)
    raise Exception("Max retries exceeded")
```

**4. Vector DB Connection Issues**
```python
# Add connection retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def connect_to_pinecone():
    return Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
```

---

## Cost Optimization

### Estimated Monthly Costs

| Component | Development | Production |
|-----------|-------------|------------|
| Pinecone | $0 (free tier) | $70-200 |
| Claude API | $20-50 | $100-300 |
| AWS EC2 (t3.large) | - | $60-80 |
| AWS Lambda | - | $10-50 |
| AWS S3 | - | $5-20 |
| **Total** | **$20-50** | **$245-650** |

### Cost Reduction Tips

1. **Cache Claude responses** for similar queries
2. **Use Pinecone starter tier** during development
3. **Implement request batching** for bulk operations
4. **Use spot instances** for training workloads
5. **Set up billing alerts** in AWS

---

## Health Checks

### API Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "components": {
    "data_collector": true,
    "vector_database": true,
    "strategy_generator": true,
    "ml_models": true
  }
}
```

### Automated Health Monitoring

```bash
# Cron job for health checks
*/5 * * * * curl -f http://localhost:8000/health || echo "F1 Analyzer unhealthy" | mail -s "Alert" admin@example.com
```

---

## Support

For issues and questions:
1. Check the logs: `docker-compose logs api`
2. Review this guide
3. Open a GitHub issue

Happy racing! 🏎️
