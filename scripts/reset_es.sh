#!/usr/bin/env bash
set -e

docker compose -f docker/docker-compose.yml down -v
docker compose -f docker/docker-compose.yml up -d

echo "Elasticsearch reset complete"