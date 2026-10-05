#!/usr/bin/env bash
# =============================================================================
# slurm_run_all.sbatch -- nop TOAN BO du an (RQ1-RQ6) len AAU AI Cloud (Slurm).
#
# Repo tren front-end: ~/aegis_agency/AegisAgency   (user: tungkvt@cs.aau.dk@ai-fe02)
#
#   ssh aicloud                       # ai-fe02.srv.aau.dk (VPN ssl-vpn1.aau.dk)
#   cd ~/aegis_agency/AegisAgency
#   mkdir -p logs                     # logs/ PHAI ton tai truoc khi sbatch (SLURM mo file output)
#   sbatch scripts/slurm_run_all.sbatch
#   squeue --me ; tail -f logs/aegis_<jobid>.out
#
# Ket qua ghi NGAY TRONG REPO (binh thuong): ./outputs/, ./outputs/cache/, ./logs/
# (muon doi cho ghi thi export AEGIS_OUTPUT / AEGIS_CACHE truoc khi sbatch).
#
# Bi preempt / het time-limit / node loi: CHI CAN SUBMIT LAI dung lenh cu. Driver bo qua buoc da
# xong (sentinel) va verdict cache giu moi call da chay -> chay lai chi lam not phan thieu.
#
# Truyen tham so cho driver qua dong lenh sbatch:
#   sbatch scripts/slurm_run_all.sbatch --phases measure,rq1,rq3     # job ngan hon
#   sbatch scripts/slurm_run_all.sbatch --quick                      # rehearsal (synthetic)
#
# Bien moi truong tuy chon (export truoc khi sbatch):
#   AEGIS_SIF=/home/container/vllm-openai_latest.sif   container vLLM co san cua AAU (mac dinh)
#   AEGIS_VENV=~/aegis_agency/venv                     dung venv thay container (uu tien neu co)
#   AEGIS_OUTPUT=outputs / AEGIS_CACHE=outputs/cache   doi cho ghi ket qua
#   AEGIS_GRES=gpu:l40s:1 / AEGIS_TIME=48:00:00 / AEGIS_PARTITION=batch / AEGIS_NODELIST=...
# =============================================================================
#SBATCH --job-name=aegis-rq16
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/aegis_%j.out
#SBATCH --error=logs/aegis_%j.err
#
# --- Ghi chu chon tai nguyen tren AAU AI Cloud (xem hpc.aau.dk/ai-cloud) ----------------------
# * KHONG dat --partition=gpu: AI Cloud dung partition mac dinh; cac node thuoc research group
#   (vi du nv-ai-04, i256-a40-[01-02]) chi dung duoc qua --partition=batch va job CO THE bi
#   interrupt -> chi them khi that su can:   #SBATCH --partition=batch
# * Chon GPU theo VRAM: 8B fp16 can ~16GB weights + KV cache => T4 16GB KHONG du.
#   Nen dung: gpu:l40s:1 (48GB) | gpu:a40:1 (48GB) | gpu:a100:1 (40GB) | gpu:v100:1 (32GB).
# * Ghim node cu the (giong cach ban van dung voi srun -w):
#       #SBATCH --nodelist=a768-l40s-03
# * Quota mac dinh: 12 job dong thoi / 12 GPU, hang doi khong gioi han.
# * Model weights: HuggingFace mac dinh cache o ~/.cache/huggingface (tren $HOME). Neu $HOME
#   khong duoc mount giong nhau giua front-end va compute node, dat HF_HOME vao thu muc repo:
#       export HF_HOME=$PWD/.hf_cache
# =============================================================================
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

AEGIS_SIF="${AEGIS_SIF:-/home/container/vllm-openai_latest.sif}"
AEGIS_VENV="${AEGIS_VENV:-}"
AEGIS_OUTPUT="${AEGIS_OUTPUT:-}"
AEGIS_CACHE="${AEGIS_CACHE:-}"

# HF_TOKEN: uu tien bien moi truong, sau do ~/.bashrc (theo huong dan AI Cloud dat token o day).
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" || true
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# HF_HOME: chi dat khi nguoi dung export; mac dinh de HuggingFace dung ~/.cache/huggingface.
if [[ -n "${HF_HOME:-}" ]]; then export HF_HOME; fi

# Tham so cho driver: chi truyen khi nguoi dung doi cho ghi (mac dinh = repo-relative).
DRIVER_ARGS=()
if [[ -n "$AEGIS_OUTPUT" ]]; then DRIVER_ARGS+=(--output-root "$AEGIS_OUTPUT"); fi
if [[ -n "$AEGIS_CACHE" ]]; then DRIVER_ARGS+=(--cache-dir "$AEGIS_CACHE"); fi

echo "job=${SLURM_JOB_ID:-local} node=$(hostname) start=$(date -Is)"
echo "repo=$PWD outputs=${AEGIS_OUTPUT:-outputs} cache=${AEGIS_CACHE:-outputs/cache}"
echo "hf_home=${HF_HOME:-<mac dinh ~/.cache/huggingface>} hf_token=$([[ -n "$HF_TOKEN" ]] && echo set || echo EMPTY)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || nvidia-smi || true

if [[ -n "$AEGIS_VENV" ]]; then
  # ---- Route B: venv rieng (co vllm) --------------------------------------------------------
  export PYTHON="$AEGIS_VENV/bin/python"
  echo "env: venv $AEGIS_VENV"
  "$PYTHON" -c "import vllm, torch; print('vllm', vllm.__version__, '| cuda', torch.cuda.is_available())"
  bash scripts/run_all.sh ${DRIVER_ARGS[@]+"${DRIVER_ARGS[@]}"} "$@"

elif [[ -f "$AEGIS_SIF" ]]; then
  # ---- Route A: container vLLM co san cua AAU (khong phai cai vLLM) -------------------------
  # Repo duoc dua vao container qua PYTHONPATH; thu vien phu (pyyaml/matplotlib) cai --user mot
  # lan va nam lai o $HOME/.local nen cac job sau khong phai cai lai.
  WRAPPER="$PWD/.aegis-python.sh"
  cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
# Wrapper do slurm_run_all.sbatch sinh ra: moi lenh python cua driver chay trong container vLLM.
export HF_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN"
export PYTHONPATH="$PWD/src:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [[ -n "${HF_HOME:-}" ]]; then export HF_HOME="$HF_HOME"; fi
# Bind chinh thu muc repo (chua ca outputs/ va cache) de container thay duoc file.
exec singularity exec --nv --bind "$PWD" "$AEGIS_SIF" python3 "\$@"
EOF
  chmod +x "$WRAPPER"
  export PYTHON="$WRAPPER"
  echo "env: singularity $AEGIS_SIF (PYTHON=$WRAPPER)"

  # Bootstrap mot lan: pyyaml (doc config) + matplotlib (ve hinh). vLLM/numpy/torch co trong image.
  "$PYTHON" -c "import vllm, torch; print('vllm', vllm.__version__, '| cuda', torch.cuda.is_available())"
  if ! "$PYTHON" -c "import yaml, matplotlib" 2>/dev/null; then
    echo "cai them pyyaml + matplotlib vao ~/.local trong container (mot lan) ..."
    "$PYTHON" -m pip install --user --quiet pyyaml matplotlib
  fi
  "$PYTHON" -c "import yaml, matplotlib; print('helpers ok')"

  bash scripts/run_all.sh ${DRIVER_ARGS[@]+"${DRIVER_ARGS[@]}"} "$@"

else
  echo "CANH BAO: khong thay container ($AEGIS_SIF) va khong co AEGIS_VENV -> dung python he thong." >&2
  export PYTHON="${PYTHON:-python3}"
  bash scripts/run_all.sh ${DRIVER_ARGS[@]+"${DRIVER_ARGS[@]}"} "$@"
fi

echo "ket thuc: $(date -Is)"
