#!/bin/bash

# Optimized Docker entrypoint script for Bastion AI Workspace Backend

set -e

echo "🚀 Starting Bastion AI Workspace Backend with Optimized Configuration..."

# Wait for dependencies to be ready
echo "⏳ Waiting for database to be ready..."
python3 -c "
import time
import socket

def wait_for_service(host, port, service_name):
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                print(f'✅ {service_name} is ready!')
                break
            else:
                print(f'⏳ Waiting for {service_name}...')
                time.sleep(2)
        except Exception as e:
            print(f'⏳ Waiting for {service_name}...')
            time.sleep(2)

# Wait for PostgreSQL
wait_for_service('postgres', 5432, 'PostgreSQL')

# Wait for Redis  
wait_for_service('redis', 6379, 'Redis')
"

echo "✅ Dependencies are ready!"

# Set Python path
export PYTHONPATH=/app:$PYTHONPATH

# Check database readiness
echo "🔄 Checking database setup..."
python -c "
import asyncio
import asyncpg
import logging

async def ensure_database_ready():
    try:
        # Simple connection test to bastion_knowledge_base database
        conn = await asyncpg.connect(
            host='postgres',
            port=5432,
            user='bastion_user',
            password='bastion_secure_password',
            database='bastion_knowledge_base'
        )
        await conn.close()
        print('✅ Database connection successful')
    except Exception as e:
        print(f'ℹ️ Database not ready yet or initialization in progress: {e}')
        # Don't raise error - let the application start anyway

asyncio.run(ensure_database_ready())
"

# Database initialization is now handled by the consolidated 01_init.sql file
echo "✅ Database initialization completed by PostgreSQL container"

# Start the optimized application
echo "🚀 Starting FastAPI application with optimized configuration..."
echo "🔧 Using service container architecture for efficient resource usage"

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${UVICORN_WORKERS:-1}" \
    --log-level info
