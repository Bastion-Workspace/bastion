#!/bin/bash

# Bastion Deployment Script
# Allows multiple deployments with different project names

set -e

# Default project name
DEFAULT_PROJECT_NAME="bastion"

# Get project name from argument or use default
PROJECT_NAME=${1:-$DEFAULT_PROJECT_NAME}

echo "🚀 Deploying Bastion with project name: $PROJECT_NAME"
echo "📁 Project directory: $(pwd)"
echo "🐳 Docker images will be tagged as: ${PROJECT_NAME}-backend and ${PROJECT_NAME}-frontend"
echo ""

# Set the COMPOSE_PROJECT_NAME environment variable
export COMPOSE_PROJECT_NAME=$PROJECT_NAME

# Show what will be created
echo "📋 Will create:"
echo "   - Images: ${PROJECT_NAME}-backend:latest, ${PROJECT_NAME}-frontend:latest"
echo "   - Containers: ${PROJECT_NAME}-backend, ${PROJECT_NAME}-frontend, etc."
echo "   - Volumes: ${PROJECT_NAME}_postgres_data, ${PROJECT_NAME}_redis_data, etc."
echo "   - Network: ${PROJECT_NAME}-network"
echo ""

# Confirm deployment
read -p "Continue with deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 1
fi

echo "🔨 Building and starting services..."
docker compose up --build -d

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Access your application:"
echo "   - Frontend: http://localhost:3051"
echo "   - Backend API: http://localhost:8081"
echo "   - API Docs: http://localhost:8081/docs"
echo ""
echo "📊 Check status:"
echo "   docker compose ps"
echo ""
echo "📝 View logs:"
echo "   docker compose logs -f"
echo ""
echo "🛑 Stop services:"
echo "   docker compose down"
echo ""
echo "🗑️  Remove everything (including volumes):"
echo "   docker compose down -v"
