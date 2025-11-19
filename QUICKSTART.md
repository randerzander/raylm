# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Option 1: Docker (Recommended for Production)

```bash
# 1. Build the image
docker build -t raylm-api:latest .

# 2. Run the container
docker run -d --name raylm-api -p 8000:8000 raylm-api:latest

# 3. Test it
curl http://localhost:8000/health
```

### Option 2: Docker Compose (Easiest)

```bash
# 1. Start everything
docker-compose up -d

# 2. Check logs
docker-compose logs -f

# 3. Test it
curl http://localhost:8000/health
```

### Option 3: Local Development

```bash
# 1. Install dependencies
pip install -e .

# 2. Run the server
python src/api.py

# 3. Test it
curl http://localhost:8000/health
```

## 📝 Submit a Document

```bash
curl -X POST http://localhost:8000/submit \
  -F "file=@/path/to/your/document.pdf"
```

## 📚 View API Documentation

Open in your browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🛑 Stop the Server

**Docker:**
```bash
docker stop raylm-api
```

**Docker Compose:**
```bash
docker-compose down
```

**Local:**
Press `Ctrl+C` in the terminal

## 📖 Full Documentation

- [API Usage Guide](API_USAGE.md)
- [Docker Usage Guide](DOCKER_USAGE.md)

## 🔍 Useful Commands

```bash
# View logs (Docker)
docker logs -f raylm-api

# View logs (Docker Compose)
docker-compose logs -f

# Check container status
docker ps

# Access container shell
docker exec -it raylm-api /bin/bash
```

