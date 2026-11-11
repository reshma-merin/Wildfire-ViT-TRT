#!/bin/sh
# Ensure the engine exists (no-op when the Kubernetes init container already built it), then run the service.
set -e
/app/build_engine.sh
exec "$@"
