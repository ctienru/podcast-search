FROM docker.elastic.co/elasticsearch/elasticsearch:8.11.1

# Install IK Analyzer plugin for Chinese word segmentation
# https://github.com/infinilabs/analysis-ik
RUN success=false; \
  for i in 1 2 3 4 5; do \
  if elasticsearch-plugin install --batch https://get.infini.cloud/elasticsearch/analysis-ik/8.11.1; then \
  success=true; \
  break; \
  fi; \
  echo "Attempt $i failed, retrying in 30s..."; \
  sleep 30; \
  done; \
  if [ "$success" != "true" ]; then \
  echo "Failed to install IK Analyzer plugin after 5 attempts."; \
  exit 1; \
  fi
