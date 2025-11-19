# Docker Usage Guide

## Building the Docker Image

Build the Docker image from the project root:

```bash
cd /localhome/local-jdyer/raylm
docker build -t raylm-api:latest .
```

### Build with a specific tag

```bash
docker build -t raylm-api:0.1.0 .
```

### Build with build arguments (optional)

```bash
docker build --build-arg HTTP_PROXY=http://proxy.example.com:8080 -t raylm-api:latest .
```

## Running the Container

### Basic run

```bash
docker run -d \
  --name raylm-api \
  -p 8000:8000 \
  raylm-api:latest
```

### Run with logs visible

```bash
docker run --rm \
  --name raylm-api \
  -p 8000:8000 \
  raylm-api:latest
```

### Run on a different port

```bash
# Map container port 8000 to host port 8080
docker run -d \
  --name raylm-api \
  -p 8080:8000 \
  raylm-api:latest
```

### Run with GPU support (for future ML processing)

```bash
docker run -d \
  --name raylm-api \
  --gpus all \
  -p 8000:8000 \
  raylm-api:latest
```

### Run with volume mounts (for persistent data)

```bash
docker run -d \
  --name raylm-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/extracts:/app/extracts \
  raylm-api:latest
```

## Managing the Container

### Check container status

```bash
docker ps
```

### View logs

```bash
docker logs raylm-api

# Follow logs
docker logs -f raylm-api

# Last 100 lines
docker logs --tail 100 raylm-api
```

### Stop the container

```bash
docker stop raylm-api
```

### Start a stopped container

```bash
docker start raylm-api
```

### Restart the container

```bash
docker restart raylm-api
```

### Remove the container

```bash
docker rm -f raylm-api
```

### Access container shell

```bash
docker exec -it raylm-api /bin/bash
```

## Testing the API

Once the container is running, test the endpoints:

```bash
# Check if server is running
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health

# Submit a document
curl -X POST http://localhost:8000/submit \
  -F "file=@/path/to/document.pdf"

# Access API documentation in browser
open http://localhost:8000/docs
```

## Docker Compose (Optional)

Create a `docker-compose.yml` file for easier management:

```yaml
version: '3.8'

services:
  raylm-api:
    build: .
    image: raylm-api:latest
    container_name: raylm-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./extracts:/app/extracts
    environment:
      - LOG_LEVEL=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Then run:

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Troubleshooting

### Container exits immediately

Check logs:
```bash
docker logs raylm-api
```

### Port already in use

Use a different host port:
```bash
docker run -d --name raylm-api -p 8001:8000 raylm-api:latest
```

### Permission denied errors

Ensure the non-root user has proper permissions:
```bash
docker run -d \
  --name raylm-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:rw \
  raylm-api:latest
```

### Check container resource usage

```bash
docker stats raylm-api
```

## Production Deployment

For production, consider:

1. **Using a reverse proxy (nginx)**
2. **Setting up SSL/TLS certificates**
3. **Implementing rate limiting**
4. **Using environment variables for configuration**
5. **Setting up monitoring and logging**

### Example with environment variables

```bash
docker run -d \
  --name raylm-api \
  -p 8000:8000 \
  -e LOG_LEVEL=warning \
  -e MAX_UPLOAD_SIZE=50MB \
  raylm-api:latest
```

## Cleanup

Remove all stopped containers:
```bash
docker container prune
```

Remove unused images:
```bash
docker image prune -a
```

Remove everything (careful!):
```bash
docker system prune -a --volumes
```

