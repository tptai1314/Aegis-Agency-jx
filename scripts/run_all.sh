#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- chay TRON BO giao thuc Aegis-Agency (RQ1 -> RQ6) trong MOT lan submit.
#
# Dung cho hang doi (SLURM/PBS/k8s): script la mot tien trinh bash duy nhat, khong can tmux,
# tu tao config, tu bo qua buoc da xong (sentinel) nen co the bi cat job roi submit lai ma
# khong mat ket qua (verdict cache + sentinel lo viec resume).
#
#   bash scripts/run_all.sh --dry-run                 # in ra moi lenh se chay, khong chay
#   bash scripts/run_all.sh --quick --backend synthetic   # rehearsal toan bo tren laptop, khong GPU
#   bash scripts/run_all.sh                           # chay that tren GPU server (backend vllm)
#   bash scripts/run_all.sh --phases measure,rq1,rq3  # chi chay mot vai pha
#   bash scripts/run_all.sh --max-tokens 512          # model "suy nghi" (Qwen3...) can nhieu token hon
#   bash scripts/run_all.sh --baselines-config configs/real_llama_baselines.yaml  # them pha baseline
#   bash scripts/run_all.sh --force                   # chay lai ca nhung buoc da co sentinel
#
# Anh xa RQ -> pha:
#   RQ1 integrity vs f (compromise + collusion voi r do duoc) ....... rq1, rq1_collusion
#   RQ2 second-order injection (epsilon do duoc, isolation on/off) ... measure, rq2
#   RQ3 adaptivity (adaptive_search toi uu theo rule) .............. rq3
#   RQ4 correlated failure (homogeneous vs diverse, rho) .......... measure, rq4
#   RQ5 cost frontier (cold cache, n = 1..7) ...................... rq5
#   RQ6 utility / over-refusal (benign sets) ...................... rq6
# =============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------- defaults
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python}"

BACKEND="vllm"
OUT_ROOT="outputs"
CACHE_DIR="outputs/cache"
SEEDS="0,1,2,3,4"
N_PAYLOADS="1000"
BENCHMARK="formal"
PAIRS_BENCHMARK="second_order"
BENIGN_BENCHMARKS="benign,benign_xstest"
VARIANTS="judgedeceiver,naive_override,roleplay_takeover,delimiter_escape"
MAX_PAIRS=""
COST_N="1,3,5,7"
PHASES="all"
BASELINES_CONFIG=""
MAX_TOKENS="256"           # Qwen3/Gemma-3 co the sinh "suy nghi" truoc JSON -> 64 token la qua it
TEMPERATURE="0.0"
MARK_REAL=0
DRY_RUN=0
FORCE=0
QUICK=0

# backbone mac dinh (ten:HF id). Sua bang --backbones "ten:id ..." khi server dung model khac.
BACKBONES_DEFAULT="llama31:meta-llama/Llama-3.1-8B-Instruct qwen3:Qwen/Qwen3-8B gemma3:google/gemma-3-12b-it"
BACKBONES="$BACKBONES_DEFAULT"

MAX_RESIDENT_ENGINES=0     # 0 = khong gioi han (homogeneous); diverse luon dung 1

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_DIR="$2"; shift 2;;
    --backend) BACKEND="$2"; shift 2;;
    --output-root) OUT_ROOT="$2"; shift 2;;
    --cache-dir) CACHE_DIR="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --n-payloads) N_PAYLOADS="$2"; shift 2;;
    --benchmark) BENCHMARK="$2"; shift 2;;
    --pairs-benchmark) PAIRS_BENCHMARK="$2"; shift 2;;
    --benign-benchmarks) BENIGN_BENCHMARKS="$2"; shift 2;;
    --variants) VARIANTS="$2"; shift 2;;
    --max-pairs) MAX_PAIRS="$2"; shift 2;;
    --cost-n) COST_N="$2"; shift 2;;
    --phases) PHASES="$2"; shift 2;;
    --backbones) BACKBONES="$2"; shift 2;;
    --max-tokens) MAX_TOKENS="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --baselines-config) BASELINES_CONFIG="$2"; shift 2;;
    --mark-real) MARK_REAL=1; shift;;
    --quick) QUICK=1; shift;;
    --force) FORCE=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage 0;;
    *) echo "Tham so khong hop le: $1" >&2; usage 2;;
  esac
done

if [[ "$QUICK" == "1" ]]; then
  BACKEND="synthetic"
  N_PAYLOADS="120"
  SEEDS="0"
  MAX_PAIRS="20"
  VARIANTS="judgedeceiver,delimiter_escape"
  COST_N="1,3"
  OUT_ROOT="${OUT_ROOT}/quick"
  CACHE_DIR="${CACHE_DIR}/quick"
fi

cd "$REPO_DIR"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$CACHE_DIR" "$LOG_DIR"

# --------------------------------------------------------------------------------- helpers
say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

run() {                      # run <mo ta> <lenh...>
  local desc="$1"; shift
  if [[ "$DRY_RUN" == "1" ]]; then printf '   [dry-run] %s\n' "$*"; return 0; fi
  printf '   -> %s\n' "$desc"
  "$@"
}

# Bo qua buoc da hoan thanh (sentinel) tru khi --force. Day la co che resume cho hang doi.
step_done() { [[ "$FORCE" == "0" && -e "$1" ]]; }

run_step() {                 # run_step <sentinel> <mo ta> <lenh...>
  local sentinel="$1"; shift
  local desc="$1"; shift
  if step_done "$sentinel"; then note "SKIP (da co $sentinel) - $desc"; return 0; fi
  run "$desc" "$@"
  if [[ "$DRY_RUN" != "1" && ! -e "$sentinel" ]]; then
    echo "LOI: buoc '$desc' khong tao ra $sentinel" >&2; exit 1
  fi
}

# Chay mot stage cua run_experiment.py, ghi log rieng.
stage() {                    # stage <config> <stage> <outdir> <logname> [them tham so...]
  local cfg="$1" st="$2" out="$3" logname="$4"; shift 4
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '   [dry-run] %s scripts/run_experiment.py --config %s --stage %s --output %s %s\n' \
      "$PYTHON" "$cfg" "$st" "$out" "$*"
    return 0
  fi
  mkdir -p "$out"
  "$PYTHON" scripts/run_experiment.py --config "$cfg" --stage "$st" --output "$out" "$@" \
    2>&1 | tee "$LOG_DIR/${logname}.log"
}

# Sinh config bang YAML (khong dung sed: it loi hon va giu nguyen comment cua file goc).
write_config() {             # write_config <out.yaml> <backbone|diverse|synthetic> [benchmark]
  local out="$1" kind="$2" bench="${3:-$BENCHMARK}"
  if [[ "$DRY_RUN" == "1" ]]; then printf '   [dry-run] sinh config %s (%s, benchmark=%s)\n' "$out" "$kind" "$bench"; return 0; fi
  "$PYTHON" - "$out" "$kind" "$bench" "$BACKEND" "$N_PAYLOADS" "$CACHE_DIR" "$MAX_RESIDENT_ENGINES" \
             "$BACKBONES" "$SEEDS" "$MAX_TOKENS" "$TEMPERATURE" <<'PY'
import pathlib, sys, yaml

(out, kind, bench, backend, n_payloads, cache_dir, max_resident, backbones, seeds,
 max_tokens, temperature) = sys.argv[1:12]
ids = {name: mid for name, mid in (item.split(":", 1) for item in backbones.split())}

cfg = yaml.safe_load(pathlib.Path("configs/real_experiment.yaml").read_text())
cfg.setdefault("experiment", {})
cfg["experiment"]["n_payloads"] = int(n_payloads)
cfg["experiment"]["seed"] = int(str(seeds).split(",")[0])
cfg.setdefault("data", {})
cfg["data"]["root"] = "data/benchmarks"
cfg["data"]["benchmark"] = bench
cfg["data"]["split"] = "test"

if backend == "synthetic":
    cfg.pop("judges", None)          # verdict synthetic: khong can LLM (moi kind)
    cfg["experiment"]["n_payloads"] = int(n_payloads)   # giu nho cho rehearsal
else:
    models = list(ids.values()) if kind == "diverse" else [ids[kind]]
    cfg["judges"] = {
        "backend": "vllm",
        "backbones": models,
        "isolation": True,
        "max_resident_engines": 1 if kind == "diverse" else int(max_resident),
        # Nhieu model (Qwen3...) sinh phan "suy nghi" truoc JSON -> can du token, neu khong
        # verdict se khong parse duoc va run se dung ngay (khong bao gio doan verdict).
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "verdict_cache": f"{cache_dir}/real_{kind}.jsonl",
    }

pathlib.Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"   config {out}: kind={kind} benchmark={bench} backend={backend} "
      f"n_payloads={n_payloads} max_tokens={max_tokens}")
PY
}

# Doc r_hat / epsilon_hat tu ket qua do luong de nuoi vao attack (RQ1 collusion, RQ2 injection).
read_measurement() {         # read_measurement <measure_dir> <key> ; in ra gia tri hoac rong
  local dir="$1" key="$2"
  [[ -e "$dir/empirical_committee.csv" ]] || { echo ""; return 0; }
  "$PYTHON" - "$dir" "$key" <<'PY'
import csv, pathlib, sys
d = pathlib.Path(sys.argv[1]); key = sys.argv[2]
value = ""
if key == "radius":
    row = next(csv.DictReader((d / "empirical_committee.csv").open()), {})
    value = row.get("r_p95", "")
elif key == "epsilon":
    path = d / "empirical_isolation_summary.csv"
    if path.exists():
        rows = [r for r in csv.DictReader(path.open()) if r.get("isolation") in ("False", "false")]
        if rows:
            value = max(rows, key=lambda r: float(r["epsilon_decision_flip_max"] or 0))["epsilon_decision_flip_max"]
print(value)
PY
}

phase_enabled() {            # phase_enabled <ten>
  [[ "$PHASES" == "all" || ",${PHASES}," == *",$1,"* ]]
}

# ------------------------------------------------------------------------------ 0. kiem tra
say "PHA 0 - kiem tra moi truong (backend=$BACKEND, quick=$QUICK)"
note "repo: $REPO_DIR"
note "output: $OUT_ROOT | cache: $CACHE_DIR | seeds: $SEEDS | n_payloads: $N_PAYLOADS"
[[ -d data/benchmarks ]] || { echo "LOI: thieu data/benchmarks (sync data truoc). Xem docs/run_all_commands.md Pha 1-2." >&2; exit 1; }
[[ -f "data/benchmarks/$BENCHMARK/test.csv" ]] || { echo "LOI: thieu data/benchmarks/$BENCHMARK/test.csv" >&2; exit 1; }
if [[ "$BACKEND" == "vllm" && "$DRY_RUN" == "0" ]]; then
  "$PYTHON" -c "import vllm, torch; assert torch.cuda.is_available(), 'CUDA khong san sang'; print('   vLLM', vllm.__version__, '| GPU', torch.cuda.device_count())" \
    || { echo "LOI: vLLM/CUDA chua san sang -> pip install -e '.[dev,vllm]' + kiem tra GPU." >&2; exit 1; }
fi
note "uoc luong: calibrate+evaluate 1 backbone ~9 800 call; epsilon day du ~112 000 call; frontier ~4x7 000 call"
note "nho dat time-limit cua hang doi du lon (xem docs/run_all_commands.md, bang chi phi)"

# ------------------------------------------------------------------- 1. do r/gamma/rho + epsilon
if phase_enabled measure; then
  say "PHA 1 - do r/gamma/rho (RQ4) va epsilon (RQ2)"
  first_backbone="$(echo "$BACKBONES" | awk '{print $1}' | cut -d: -f1)"
  write_config "configs/generated_${first_backbone}.yaml" "$first_backbone"
  write_config "configs/generated_diverse.yaml" diverse
  extra_measure=()
  if [[ -n "$MAX_PAIRS" ]]; then extra_measure+=(--max-pairs "$MAX_PAIRS"); fi
  run_step "$OUT_ROOT/measure_${first_backbone}/empirical_committee.csv" \
    "measure r/gamma/rho/epsilon - $first_backbone" \
    "$PYTHON" scripts/measure_empirical.py --config "configs/generated_${first_backbone}.yaml" \
      --output "$OUT_ROOT/measure_${first_backbone}" --seeds "$SEEDS" --variants "$VARIANTS" \
      ${extra_measure[@]+"${extra_measure[@]}"}
  if [[ "$BACKEND" == "vllm" ]]; then
    run_step "$OUT_ROOT/measure_diverse/empirical_committee.csv" \
      "measure r/gamma/rho/epsilon - diverse (RQ4)" \
      "$PYTHON" scripts/measure_empirical.py --config configs/generated_diverse.yaml \
        --output "$OUT_ROOT/measure_diverse" --seeds "$SEEDS" --variants "$VARIANTS" \
        ${extra_measure[@]+"${extra_measure[@]}"}
  fi
fi

RADIUS_HAT="$(read_measurement "$OUT_ROOT/measure_$(echo "$BACKBONES" | awk '{print $1}' | cut -d: -f1)" radius)"
EPSILON_HAT="$(read_measurement "$OUT_ROOT/measure_$(echo "$BACKBONES" | awk '{print $1}' | cut -d: -f1)" epsilon)"
note "r_hat (collusion radius) = ${RADIUS_HAT:-<chua co - se co sau khi do luong xong>}"
note "epsilon_hat (un-isolated worst case) = ${EPSILON_HAT:-<chua co - se co sau khi do luong xong>}"
if [[ "$DRY_RUN" == "1" ]]; then
  note "dry-run: pha rq1_collusion / rq2 can r_hat / epsilon_hat nen chi hien khi chay that."
fi

# Ghi config attack co tham so do duoc (chi khi co so).
write_attack_config() {      # write_attack_config <out.yaml> <config goc> <attack> <key=value...>
  local out="$1" base="$2" attack="$3"; shift 3
  [[ "$DRY_RUN" == "1" ]] && { printf '   [dry-run] attack config %s <- %s (%s %s)\n' "$out" "$base" "$attack" "$*"; return 0; }
  "$PYTHON" - "$out" "$base" "$attack" "$@" <<'PY'
import pathlib, sys, yaml
out, base, attack = sys.argv[1:4]
kwargs = dict(item.split("=", 1) for item in sys.argv[4:])
cfg = yaml.safe_load(pathlib.Path(base).read_text())
cfg.setdefault("experiment", {})
cfg["experiment"]["attack"] = attack
if kwargs:
    cfg["experiment"]["attack_kwargs"] = {k: (float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v)
                                          for k, v in kwargs.items()}
pathlib.Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"   {out}: attack={attack} kwargs={kwargs}")
PY
}

# ------------------------------------------------------- 2. RQ1 (compromise + collusion) / RQ3
for spec in $BACKBONES; do
  tag="${spec%%:*}"; model="${spec#*:}"
  if phase_enabled rq1; then
    say "PHA 2 - RQ1: compromise sweep tren $tag"
    write_config "configs/generated_${tag}.yaml" "$tag"
    stage "configs/generated_${tag}.yaml" calibrate "$OUT_ROOT/${tag}/calibrate" "${tag}_calibrate"
    stage "configs/generated_${tag}.yaml" evaluate  "$OUT_ROOT/${tag}/evaluate"  "${tag}_evaluate" --seeds "$SEEDS"
    stage "configs/generated_${tag}.yaml" ablate    "$OUT_ROOT/${tag}/ablate"    "${tag}_ablate"
  fi
  if phase_enabled rq1_collusion && [[ -n "$RADIUS_HAT" ]]; then
    say "PHA 2b - RQ1: collusion voi r_hat=$RADIUS_HAT tren $tag"
    write_config "configs/generated_${tag}.yaml" "$tag"
    write_attack_config "configs/generated_${tag}_collusion.yaml" "configs/generated_${tag}.yaml" collusion "radius=$RADIUS_HAT"
    stage "configs/generated_${tag}_collusion.yaml" evaluate "$OUT_ROOT/${tag}_collusion/evaluate" "${tag}_collusion" --seeds "$SEEDS"
  fi
  if phase_enabled rq2 && [[ -n "$EPSILON_HAT" ]]; then
    say "PHA 3 - RQ2: injection voi epsilon_hat=$EPSILON_HAT tren $tag"
    write_config "configs/generated_${tag}.yaml" "$tag"
    write_attack_config "configs/generated_${tag}_injection.yaml" "configs/generated_${tag}.yaml" injection "epsilon=$EPSILON_HAT"
    stage "configs/generated_${tag}_injection.yaml" evaluate "$OUT_ROOT/${tag}_injection/evaluate" "${tag}_injection" --seeds "$SEEDS"
  fi
  if phase_enabled rq3; then
    say "PHA 4 - RQ3: adaptive_search (toi uu theo tung rule) tren $tag"
    write_config "configs/generated_${tag}.yaml" "$tag"
    write_attack_config "configs/generated_${tag}_adaptive_search.yaml" "configs/generated_${tag}.yaml" adaptive_search
    stage "configs/generated_${tag}_adaptive_search.yaml" evaluate "$OUT_ROOT/${tag}_adaptive_search/evaluate" "${tag}_adaptive_search" --seeds "$SEEDS"
  fi
done

# ------------------------------------------------------------------- 3. RQ4 diverse committee
if phase_enabled rq4; then
  say "PHA 5 - RQ4: committee diverse (round-robin, max_resident_engines=1)"
  write_config configs/generated_diverse.yaml diverse
  stage configs/generated_diverse.yaml calibrate "$OUT_ROOT/diverse/calibrate" diverse_calibrate
  stage configs/generated_diverse.yaml evaluate  "$OUT_ROOT/diverse/evaluate"  diverse_evaluate --seeds "$SEEDS"
  stage configs/generated_diverse.yaml ablate    "$OUT_ROOT/diverse/ablate"    diverse_ablate
fi

# ------------------------------------------------------------------- 4. RQ5 cost frontier
if phase_enabled rq5; then
  say "PHA 6 - RQ5: so do that (cold cache) + frontier n=$COST_N"
  first_backbone="$(echo "$BACKBONES" | awk '{print $1}' | cut -d: -f1)"
  write_config "configs/generated_${first_backbone}.yaml" "$first_backbone"
  COLD_CACHE="$CACHE_DIR/cold_$(date +%Y%m%d_%H%M%S).jsonl"
  stage "configs/generated_${first_backbone}.yaml" evaluate "$OUT_ROOT/${first_backbone}_cold/evaluate" \
        "${first_backbone}_cold" --seeds "${SEEDS%%,*}" --cache "$COLD_CACHE"
  for n in ${COST_N//,/ }; do
    write_config "configs/generated_${first_backbone}_n${n}.yaml" "$first_backbone"
    if [[ "$DRY_RUN" == "0" ]]; then
      "$PYTHON" - "configs/generated_${first_backbone}_n${n}.yaml" "$n" <<'PY'
import pathlib, sys, yaml
path, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
cfg = yaml.safe_load(path.read_text()); cfg["experiment"]["n_judges"] = n
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
    fi
    stage "configs/generated_${first_backbone}_n${n}.yaml" evaluate \
          "$OUT_ROOT/${first_backbone}_n${n}/evaluate" "${first_backbone}_n${n}" \
          --seeds "${SEEDS%%,*}" --cache "$CACHE_DIR/cold_n${n}.jsonl"
  done
  stage "configs/generated_${first_backbone}.yaml" ablate "$OUT_ROOT/${first_backbone}_ablate_cost/ablate" \
        "${first_backbone}_ablate_cost"
fi

# ------------------------------------------------------------------- 5. RQ6 utility / ORR
if phase_enabled rq6; then
  say "PHA 7 - RQ6: utility / over-refusal tren $BENIGN_BENCHMARKS"
  for bench in ${BENIGN_BENCHMARKS//,/ }; do
    for spec in $BACKBONES; do
      tag="${spec%%:*}"
      write_config "configs/generated_${tag}_${bench}.yaml" "$tag" "$bench"
      stage "configs/generated_${tag}_${bench}.yaml" evaluate "$OUT_ROOT/${tag}_${bench}/evaluate" "${tag}_${bench}"
    done
    write_config "configs/generated_diverse_${bench}.yaml" diverse "$bench"
    stage "configs/generated_diverse_${bench}.yaml" evaluate "$OUT_ROOT/diverse_${bench}/evaluate" "diverse_${bench}"
  done
fi

# ------------------------------------------------------------------- 6. baseline ngoai (tuy chon)
if phase_enabled baselines && [[ -n "$BASELINES_CONFIG" ]]; then
  say "PHA 8 - baseline ngoai head-to-head ($BASELINES_CONFIG)"
  stage "$BASELINES_CONFIG" evaluate "$OUT_ROOT/baselines/evaluate" baselines --seeds "${SEEDS%%,*}" \
        --cache "$CACHE_DIR/baselines_$(date +%Y%m%d_%H%M%S).jsonl"
elif phase_enabled baselines; then
  note "bo qua pha baseline: chua truyen --baselines-config (xem runbook 8.8)"
fi

# ------------------------------------------------------------------- 7. hinh + manifest
if phase_enabled plots; then
  say "PHA 9 - ve hinh"
  plot_args=()
  if [[ "$MARK_REAL" == "1" ]]; then plot_args+=(--real); fi
  while IFS= read -r sweep; do
    if [[ -z "$sweep" ]]; then continue; fi
    out="${sweep%/evaluation_sweep.csv}/asr_vs_f.png"
    run "plot $out" "$PYTHON" scripts/make_plots.py --input "$sweep" --output "$out" ${plot_args[@]+"${plot_args[@]}"}
  done < <(find "$OUT_ROOT" -name evaluation_sweep.csv | sort)
  if [[ "$MARK_REAL" == "0" ]]; then
    note "hinh dang o che do synthetic (co watermark). Chi dung --mark-real sau khi xac minh."
  fi
fi

say "PHA 10 - manifest ket qua"
if [[ "$DRY_RUN" == "1" ]]; then
  note "[dry-run] se ghi $OUT_ROOT/RUN_MANIFEST.json"
else
  "$PYTHON" - "$OUT_ROOT" <<'PY'
import csv, hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
entries = []
for prov in sorted(root.rglob("*provenance*.json")):
    data = json.loads(prov.read_text())
    record = {
        "dir": str(prov.parent),
        "stage": data.get("stage"),
        "data_source": data.get("data_source"),
        "is_paper_result": data.get("is_paper_result"),
        "platform": data.get("platform"),
        "config_seeds": (data.get("config") or {}).get("seeds"),
        "config_models": (data.get("config") or {}).get("models") or (data.get("config") or {}).get("backbones"),
    }
    for name in ("evaluation_sweep.csv", "evaluation_summary.csv", "ablation.csv",
                 "empirical_committee.csv", "empirical_isolation_summary.csv",
                 "evaluation_cost.csv", "evaluation_cost_payloads.csv"):
        path = prov.parent / name
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            rows = sum(1 for _ in path.open()) - 1
            record[name] = {"rows": rows, "sha256_16": digest}
    entries.append(record)
manifest = {"output_root": str(root), "n_provenance_records": len(entries), "entries": entries}
(pathlib.Path(root) / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(f"   RUN_MANIFEST.json: {len(entries)} provenance record(s)")
for record in entries:
    print(f"   - {record['dir']:48s} stage={record['stage']:<10s} data={record['data_source']}")
PY
  find "$OUT_ROOT" -name "*.csv" -o -name "*.png" | sort | head -40
  note "Buoc tiep theo (BAT BUOC): cap nhat audits/result_integrity_audit.md bang so trong RUN_MANIFEST.json"
  note "Chi dat is_paper_result=true sau khi ghi provenance; cac file da ship trong repo phai giu false."
fi

say "XONG"
note "log: $LOG_DIR | ket qua: $OUT_ROOT"
