#!/usr/bin/env bash
set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-/opt/wordtoken}"
CONTAINER_NAME="${CONTAINER_NAME:-wordtoken}"
IMAGE_NAME="${IMAGE_NAME:-wordtoken-wordtoken:latest}"
ENV_FILE="${ENV_FILE:-$DEPLOY_PATH/.env}"
PORT_BIND="${PORT_BIND:-127.0.0.1:8000:8000}"
HF_VOLUME="${HF_VOLUME:-wordtoken_huggingface_cache}"
HEALTHCHECK_URL="${HEALTHCHECK_URL:-http://127.0.0.1:8000/health}"
HEALTHCHECK_TIMEOUT_SECONDS="${HEALTHCHECK_TIMEOUT_SECONDS:-900}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

cd "$DEPLOY_PATH"

DOCKER_BUILDKIT=1 docker build --pull \
  --build-arg "TORCH_VERSION=$TORCH_VERSION" \
  --build-arg "TORCH_INDEX_URL=$TORCH_INDEX_URL" \
  -t "$IMAGE_NAME" .
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --env-file "$ENV_FILE" \
  -p "$PORT_BIND" \
  -v "$HF_VOLUME:/root/.cache/huggingface" \
  "$IMAGE_NAME" >/dev/null

deadline=$((SECONDS + HEALTHCHECK_TIMEOUT_SECONDS))
until curl -fsS "$HEALTHCHECK_URL" >/tmp/wordtoken-health.json 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "Deployment timed out waiting for $HEALTHCHECK_URL" >&2
    docker logs --tail=200 "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  sleep 5
done

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files caddy.service >/dev/null 2>&1; then
  systemctl reload caddy || true
fi

docker image prune -f --filter dangling=true >/dev/null 2>&1 || true
cat /tmp/wordtoken-health.json
