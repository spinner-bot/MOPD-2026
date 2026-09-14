#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <full-40-char-commit> <new-worktree-path>" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
[[ $# -eq 2 ]] || { usage; exit 2; }

COMMIT="$1"
DESTINATION_INPUT="$2"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "commit must be the full 40-character lowercase hash" >&2
  exit 2
}
[[ ! -e "${DESTINATION_INPUT}" ]] || {
  echo "destination already exists: ${DESTINATION_INPUT}" >&2
  exit 2
}
DESTINATION_PARENT="$(cd -- "$(dirname -- "${DESTINATION_INPUT}")" && pwd -P)"
DESTINATION="${DESTINATION_PARENT}/$(basename -- "${DESTINATION_INPUT}")"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
git -C "${REPO_ROOT}" cat-file -e "${COMMIT}^{commit}"
git -C "${REPO_ROOT}" worktree add --detach "${DESTINATION}" "${COMMIT}" >&2

ACTUAL="$(git -C "${DESTINATION}" rev-parse HEAD)"
[[ "${ACTUAL}" == "${COMMIT}" ]] || {
  echo "worktree commit mismatch: expected=${COMMIT} actual=${ACTUAL}" >&2
  exit 2
}

echo "pinned worktree ready: ${DESTINATION}" >&2
printf 'export MOPD_CHECKOUT_ROOT=%q\n' "${DESTINATION}"
printf 'export MOPD_EXPECTED_CODE_COMMIT=%q\n' "${COMMIT}"
printf 'export MOPD_RUNS_ROOT=%q\n' "${REPO_ROOT}/research/main/outputs/runs"
