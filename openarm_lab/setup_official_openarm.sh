#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT="cb7ff7547bdb81f290591edb9302b4845b1d49a7"
DEST="$LAB_DIR/vendor/openarm_isaac_lab"

if [[ -f "$DEST/.source-commit" ]]; then
    if [[ "$(cat "$DEST/.source-commit")" == "$COMMIT" ]]; then
        echo "OpenArm source is already present at $DEST ($COMMIT)"
        exit 0
    fi
    echo "Existing OpenArm source has a different commit: $DEST" >&2
    exit 1
fi
if [[ -e "$DEST" ]]; then
    echo "Destination already exists without a commit marker: $DEST" >&2
    exit 1
fi

mkdir -p "$LAB_DIR/vendor"
ARCHIVE="$(mktemp --suffix=.tar.gz)"
STAGING="$(mktemp -d "$LAB_DIR/vendor/.openarm-staging.XXXXXX")"
cleanup() { rm -f "$ARCHIVE"; rm -rf "$STAGING"; }
trap cleanup EXIT

curl --fail --location --silent --show-error \
    "https://github.com/enactic/openarm_isaac_lab/archive/$COMMIT.tar.gz" \
    --output "$ARCHIVE"
tar -xzf "$ARCHIVE" -C "$STAGING" --strip-components=1 --exclude='*/video/*'
test -f "$STAGING/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/openarm_bimanual/openarm_bimanual.usd"
test -f "$STAGING/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/config/agents/rl_games_ppo_cfg.yaml"
printf '%s\n' "$COMMIT" > "$STAGING/.source-commit"
mv "$STAGING" "$DEST"
trap - EXIT
rm -f "$ARCHIVE"
echo "OpenArm source ready at $DEST ($COMMIT)"
