#!/usr/bin/env bash
# A separate shell preserves the user's interactive terminal on any failure.
set -u
set -o pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 9>"$PACKAGE_DIR/.server_acceptance.lock"
if ! flock -n 9; then
    printf 'Another acceptance is already running for this candidate.\n' >&2
    exit 1
fi
RUN_DIR="$(mktemp -d /root/autodl-tmp/dasdg_release_acceptance_20261012_XXXXXXXX)" || exit 1
PYTHON_BIN="${DASDG_PYTHON:-python}"
RESULT_DIR="$RUN_DIR/evidence"
mkdir -p "$RESULT_DIR"
printf '%s\n' "$RUN_DIR" > "$PACKAGE_DIR/LATEST_RUN.txt"
printf 'RUN_DIR=%s\n' "$RUN_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

main() {
    cd "$PACKAGE_DIR" || return 1
    sha256sum -c SHA256SUMS.txt > "$RESULT_DIR/package_hash_check.txt" 2>&1 || return 1
    "$PYTHON_BIN" -c 'import sys,numpy,torch; print(sys.version); print(numpy.__version__); print(torch.__version__); assert (3,10) <= sys.version_info[:2] < (3,13); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' > "$RESULT_DIR/environment_preflight.txt" 2>&1 || return 1
    "$PYTHON_BIN" -m venv --system-site-packages "$RUN_DIR/venv" || return 1
    "$RUN_DIR/venv/bin/python" -m pip install --no-index --no-deps "$PACKAGE_DIR/dist/dasdg-1.1.0-py3-none-any.whl" > "$RESULT_DIR/install.txt" 2>&1 || return 1
    # Record inherited environment conflicts without mistaking unrelated packages
    # for failures of this candidate. Its own requirements are checked below.
    "$RUN_DIR/venv/bin/python" -m pip check > "$RESULT_DIR/pip_check.txt" 2>&1
    "$RUN_DIR/venv/bin/python" "$PACKAGE_DIR/tools/check_runtime.py" > "$RESULT_DIR/candidate_runtime_check.txt" 2>&1 || return 1
    cd "$RUN_DIR" || return 1
    # Bound the short acceptance, not an existing training process. Timeout and
    # nonzero exits preserve evidence and do not retry or start formal training.
    printf 'STAGE_CPU_ACCEPTANCE\n'
    PYTHONPATH='' timeout --signal=TERM --kill-after=30s 45m "$RUN_DIR/venv/bin/python" -u "$PACKAGE_DIR/run_cpu_acceptance.py" --output "$RESULT_DIR/cpu_tests" || return 1
    printf 'STAGE_UNIFIED_GPU_ACCEPTANCE\n'
    PYTHONPATH='' timeout --signal=TERM --kill-after=30s 45m "$RUN_DIR/venv/bin/python" -u "$PACKAGE_DIR/run_release_acceptance.py" --output "$RESULT_DIR/release_tests" --cuda || return 1
}

main 2>&1 | tee "$RESULT_DIR/console.log"
TASK_RC=${PIPESTATUS[0]}
printf '%s\n' "$TASK_RC" > "$RESULT_DIR/exit_code.txt"
ARCHIVE="${RUN_DIR}_results.tar.gz"
tar -czf "$ARCHIVE" -C "$RUN_DIR" evidence || exit 1
printf '%s\n' "$ARCHIVE" > "$PACKAGE_DIR/LATEST_RESULT.txt"
printf '\nRESULT_ARCHIVE=%s\n' "$ARCHIVE"
if [ "$TASK_RC" -eq 0 ]; then
    printf 'SERVER_RELEASE_ACCEPTANCE_PASS\n'
else
    printf 'SERVER_RELEASE_ACCEPTANCE_FAILED (evidence preserved; do not retry blindly)\n'
fi
exit "$TASK_RC"
