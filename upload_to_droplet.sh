#!/usr/bin/env bash
# ============================================================================
# upload_to_droplet.sh — LOCAL control script (lives on the RPI5, NOT in the
# droplet bundle). Stages the graftingmodular deploy bundle + binary + model +
# eval data + graft onto a paid MI300X droplet, triggers run_all.sh, and pulls
# results back.
#
# It supports two ways to get the HIP binary onto the droplet:
#   * BUILD_ON_DROPLET=1 (default): upload grafting.hip + src/ + safetensors.h
#     and compile on the droplet with hipcc (the droplet has ROCm).
#   * BINARY_LOCAL=<path>: upload a pre-built grafting.shadow instead.
#
# Nothing here is committed to the repo; it is operator glue. Fill in the
# values below (or export them) before running.
# ============================================================================
set -u

# ---- REQUIRED: droplet connection ----
DROPLET_SSH="${DROPLET_SSH:-}"          # e.g. root@203.0.113.10
SSH_KEY="${SSH_KEY:-}"                  # optional: -i /path/to/key

# ---- REQUIRED: local asset locations ----
LOCAL_REPO="${LOCAL_REPO:-/home/hinedes/grafting}"
MODEL_LOCAL="${MODEL_LOCAL:-}"          # dir: real_SmolLM3-3B (safetensors + index)
EVAL_LOCAL="${EVAL_LOCAL:-}"            # file: real_medical_domain.bin (or a dir)
GRAFT_LOCAL="${GRAFT_LOCAL:-${LOCAL_REPO}/runs/medical_all3.graft}"

# ---- binary strategy ----
BUILD_ON_DROPLET="${BUILD_ON_DROPLET:-1}"   # 1 = compile on droplet, 0 = use BINARY_LOCAL
BINARY_LOCAL="${BINARY_LOCAL:-}"            # pre-built grafting.shadow (if BUILD_ON_DROPLET=0)

# ---- droplet layout (must match deploy/config.sh defaults) ----
REMOTE_ROOT="${REMOTE_ROOT:-/workspace}"
MODEL_NAME="${MODEL_NAME:-real_SmolLM3-3B}"

# ---- optional pass-throughs to run_all.sh ----
TPHS_CMD="${TPHS_CMD:-}"                # e.g. "bash /workspace/tphs_run.sh"
TPHS_ENTRY="${TPHS_ENTRY:-}"            # original TPHS cmd TEMPLATE (see tphs_run.sh); required if TPHS_CMD set
EXFIL="${EXFIL:-}"                      # user@host:/path to exfil after run (optional)
RUN_TIMEOUT="${RUN_TIMEOUT:-21600}"     # 6h wall-clock cap for run_all.sh

# ---- training data (domain + OOD .bin files) so HIP and TPHS train identically ----
DATA_LOCAL="${DATA_LOCAL:-}"            # local dir of *.bin -> ${REMOTE_ROOT}/data/

# ============================================================================
usage() { echo "usage: DROPLET_SSH=user@host MODEL_LOCAL=/path/to/model EVAL_LOCAL=/path/to/bin [BINARY_LOCAL=/path] $0"; exit 2; }
[ -z "${DROPLET_SSH}" ] && usage
[ -z "${MODEL_LOCAL}" ] && usage
[ -z "${EVAL_LOCAL}" ] && usage
if [ "${BUILD_ON_DROPLET}" = "1" ]; then
  [ -f "${LOCAL_REPO}/grafting.hip" ] || { echo "FATAL: grafting.hip missing in ${LOCAL_REPO}"; exit 1; }
else
  [ -f "${BINARY_LOCAL}" ] || { echo "FATAL: BINARY_LOCAL missing: ${BINARY_LOCAL}"; exit 1; }
fi
[ -d "${MODEL_LOCAL}" ] || { echo "FATAL: MODEL_LOCAL dir missing: ${MODEL_LOCAL}"; exit 1; }
[ -f "${GRAFT_LOCAL}" ] || { echo "FATAL: GRAFT_LOCAL missing: ${GRAFT_LOCAL}"; exit 1; }

ssh_opts=()
[ -n "${SSH_KEY}" ] && ssh_opts+=(-i "${SSH_KEY}")

remote() { ssh -o StrictHostKeyChecking=no "${ssh_opts[@]}" "${DROPLET_SSH}" "$@"; }
rsync_up() { rsync -az --progress -e "ssh -o StrictHostKeyChecking=no ${SSH_KEY:+-i ${SSH_KEY}}" "$1" "${DROPLET_SSH}:$2"; }
scp_up()   { scp -o StrictHostKeyChecking=no ${SSH_KEY:+-i "${SSH_KEY}"} "$1" "${DROPLET_SSH}:$2"; }

echo "==> [1/6] preparing remote layout"
remote "mkdir -p ${REMOTE_ROOT}/model/${MODEL_NAME} ${REMOTE_ROOT}/results ${REMOTE_ROOT}/out"

echo "==> [2/6] uploading deploy bundle (flat -> ${REMOTE_ROOT}/)"
remote "mkdir -p ${REMOTE_ROOT}"
for f in "${LOCAL_REPO}"/deploy/*.sh "${LOCAL_REPO}"/deploy/*.py "${LOCAL_REPO}"/deploy/manifest.sha256; do
  scp_up "$f" "${REMOTE_ROOT}/$(basename "$f")"
done

echo "==> [3/6] uploading model + eval + graft"
rsync_up "${MODEL_LOCAL}/" "${REMOTE_ROOT}/model/${MODEL_NAME}/"
if [ -d "${EVAL_LOCAL}" ]; then
  rsync_up "${EVAL_LOCAL}/" "${REMOTE_ROOT}/"
else
  scp_up "${EVAL_LOCAL}" "${REMOTE_ROOT}/$(basename "${EVAL_LOCAL}")"
fi
scp_up "${GRAFT_LOCAL}" "${REMOTE_ROOT}/medical_all3.graft"

if [ -n "${DATA_LOCAL:-}" ]; then
  echo "==> [3b] uploading training data (domain + OOD) -> ${REMOTE_ROOT}/data/"
  rsync_up "${DATA_LOCAL}/" "${REMOTE_ROOT}/data/"
fi

echo "==> [4/6] providing the HIP binary"
if [ "${BUILD_ON_DROPLET}" = "1" ]; then
  echo "    building on droplet from source (hipcc)"
  scp_up "${LOCAL_REPO}/grafting.hip"        "${REMOTE_ROOT}/grafting.hip"
  scp_up "${LOCAL_REPO}/safetensors.h"       "${REMOTE_ROOT}/safetensors.h"
  rsync_up "${LOCAL_REPO}/src/"              "${REMOTE_ROOT}/src/"
  # Mirror the known ROCm 7.2.1 + torch venv build flags. Adjust torch/rocm
  # paths on the droplet if they differ.
  remote "cd ${REMOTE_ROOT} && hipcc -std=c++17 -O3 \
    -I${REMOTE_ROOT} -I/opt/venv/lib/python3.10/site-packages/torch/include \
    -I/opt/venv/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
    -L/opt/venv/lib/python3.10/site-packages/torch/lib -L/opt/rocm-7.2.1/lib \
    -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -lrocblas -lhipblaslt \
    -Wl,-rpath,/opt/venv/lib/python3.10/site-packages/torch/lib \
    -Wl,-rpath,/opt/rocm-7.2.1/lib --hip-link \
    -o ${REMOTE_ROOT}/grafting.shadow ${REMOTE_ROOT}/grafting.hip && chmod +x ${REMOTE_ROOT}/grafting.shadow"
else
  echo "    uploading pre-built binary"
  scp_up "${BINARY_LOCAL}" "${REMOTE_ROOT}/grafting.shadow"
  remote "chmod +x ${REMOTE_ROOT}/grafting.shadow"
fi

echo "==> [5/6] triggering run_all.sh (timeout ${RUN_TIMEOUT}s)"
remote "cd ${REMOTE_ROOT} && nohup env TPHS_CMD='${TPHS_CMD}' TPHS_ENTRY='${TPHS_ENTRY}' EXFIL='${EXFIL}' \
  bash run_all.sh > ${REMOTE_ROOT}/run_all.log 2>&1 &"
echo "    launched in background; watching log (tail) ..."
sleep 15
remote "tail -n 40 ${REMOTE_ROOT}/run_all.log" || true

echo "==> [6/6] pulling results"
RESULTS="${LOCAL_REPO}/droplet_results.tar.gz"
scp_up_reverse() { scp -o StrictHostKeyChecking=no ${SSH_KEY:+-i "${SSH_KEY}"} "${DROPLET_SSH}:$1" "$2"; }
# Poll until the result artifact appears (or timeout).
elapsed=0
while ! remote "[ -f ${REMOTE_ROOT}/results/droplet_results.tar.gz ]"; do
  sleep 30; elapsed=$((elapsed+30))
  if [ "$elapsed" -ge "$RUN_TIMEOUT" ]; then echo "TIMEOUT waiting for results"; exit 3; fi
done
scp_up_reverse "${REMOTE_ROOT}/results/droplet_results.tar.gz" "${RESULTS}"
echo "==> DONE. results at ${RESULTS}"
