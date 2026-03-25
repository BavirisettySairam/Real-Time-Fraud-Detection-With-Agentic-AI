#!/usr/bin/env bash
# Generate a self-signed TLS certificate for local/demo HTTPS.
# Output: nginx/certs/selfsigned.crt + selfsigned.key

set -euo pipefail

CERT_DIR="$(dirname "$0")/../nginx/certs"
mkdir -p "$CERT_DIR"

openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout "$CERT_DIR/selfsigned.key" \
  -out "$CERT_DIR/selfsigned.crt" \
  -subj "/C=IN/ST=Local/L=Dev/O=FraudDetection/CN=localhost"

echo "TLS certificate generated in $CERT_DIR"
