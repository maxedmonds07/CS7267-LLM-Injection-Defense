#!/usr/bin/env bash

set -e

cat > /tmp/basic_auth.ini <<EOF
[mlflow]
default_permission = MANAGE
database_uri = ${DATABASE_URL}
admin_username = admin
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF

export MLFLOW_AUTH_CONFIG_PATH=/tmp/basic_auth.ini

exec mlflow server \
    --backend-store-uri "$DATABASE_URL" \
    --artifacts-destination "s3://$MLFLOW_ARTIFACT_BUCKET" \
    --host 0.0.0.0 \
    --port "${PORT:-10000}" \
    --workers 1 \
    --allowed-hosts "$RENDER_EXTERNAL_HOSTNAME"