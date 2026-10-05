# Lệnh chạy hoàn thành dự án (end-to-end)

Danh sách lệnh **đầy đủ, theo đúng thứ tự**, từ laptop → GPU server → kết quả cuối. Không rút gọn
stage nào: đúng 3 stage `calibrate → evaluate → ablate`, cộng các phép đo RQ2/RQ4 và vòng cold-cache
cho RQ5.

## Cách 1 (khuyến nghị khi nộp hàng đợi): một script chạy trọn RQ1→RQ6

```bash
bash scripts/run_all.sh --dry-run                 # in ra moi lenh se chay, khong chay gi
bash scripts/run_all.sh --quick                   # rehearsal toan bo tren laptop (synthetic, khong GPU)
bash scripts/run_all.sh                           # chay that tren GPU server (backend vllm)
sbatch scripts/slurm_run_all.sbatch               # nop vao hang doi SLURM (job dai)
```

### AAU AI Cloud (front-end `ai-fe02`, Slurm + Singularity)

Repo trên server: `~/aegis_agency/AegisAgency` (user `tungkvt@cs.aau.dk@ai-fe02`).

```bash
ssh aicloud                             # host ai-fe02.srv.aau.dk, VPN ssl-vpn1.aau.dk
cd ~/aegis_agency/AegisAgency
mkdir -p logs                           # logs/ PHAI co truoc khi sbatch (SLURM mo file output)
sbatch scripts/slurm_run_all.sbatch
squeue --me
tail -f logs/aegis_<jobid>.out
```

Kết quả ghi **ngay trong repo** như bình thường: `outputs/`, `outputs/cache/`, `logs/`.
(Chỉ đổi chỗ ghi khi thật sự cần: `export AEGIS_OUTPUT=... AEGIS_CACHE=...` trước khi `sbatch`.)

`slurm_run_all.sbatch` đã được chỉnh đúng cho AI Cloud:

| Điểm | Trước | Sau (AI Cloud) |
|---|---|---|
| Partition | `--partition=gpu` (không tồn tại) | bỏ trống = partition mặc định; chỉ thêm `--partition=batch` khi cần node của research group (job **có thể bị interrupt**) |
| GPU | `--gres=gpu:1` (có thể rơi vào T4 16GB → OOM với 8B) | `--gres=gpu:l40s:1` (48GB); đổi được sang `gpu:a40:1` / `gpu:a100:1` / `gpu:v100:1` |
| Môi trường | `source .venv/bin/activate` | **Route A** (mặc định): container có sẵn của AAU `/home/container/vllm-openai_latest.sif` + `PYTHONPATH=<repo>/src`, không phải cài vLLM; **Route B**: `AEGIS_VENV=~/aegis_agency/venv` |
| Thư viện phụ | giả định có trong venv | Route A tự `pip install --user pyyaml matplotlib` một lần (nằm lại ở `$HOME/.local`) |
| Container thấy file repo | không bind → không thấy `outputs/` | wrapper sinh ra có `--bind "$PWD"` (bind chính thư mục repo) |
| HF token | tự export | tự đọc `HF_TOKEN` từ môi trường rồi từ `~/.bashrc` (đúng cách AAU hướng dẫn) |
| Model cache | — | mặc định `~/.cache/huggingface`; nếu `$HOME` không giống nhau giữa front-end và compute node thì `export HF_HOME=$PWD/.hf_cache` |
| Ghim node | không có | thêm `#SBATCH --nodelist=a768-l40s-03` nếu muốn (như `srun -w`) |

Biến môi trường chỉnh nhanh: `AEGIS_SIF`, `AEGIS_VENV`, `AEGIS_OUTPUT`, `AEGIS_CACHE`, `AEGIS_GRES`,
`AEGIS_MEM`, `AEGIS_TIME`, `AEGIS_PARTITION`, `AEGIS_NODELIST`, `HF_HOME`.

**Smoke test 5–10 phút trên node GPU** (nên chạy trước khi nộp job dài):

```bash
srun -w a768-l40s-03 --gres=gpu:1 --time=01:00:00 --pty bash -c \
  'bash scripts/slurm_run_all.sbatch --n-payloads 8 --seeds 7 --max-pairs 4 \
   --variants judgedeceiver --phases measure,rq1'
```

Chia job cho vừa time-limit / giảm rủi ro preempt (quota mặc định AAU: 12 job & 12 GPU đồng thời):

```bash
sbatch scripts/slurm_run_all.sbatch --phases measure            # do r/gamma/rho + epsilon truoc
sbatch scripts/slurm_run_all.sbatch --phases rq1,rq1_collusion,rq2,rq3
sbatch scripts/slurm_run_all.sbatch --phases rq4,rq5,rq6,plots
```

Bị preempt/cắt job: **submit lại đúng lệnh cũ** — sentinel + verdict cache tự tiếp tục.

Script tự: kiểm tra môi trường (vLLM/CUDA/data) → sinh config → đo r/γ/ρ + ε (RQ2/RQ4) → **lấy
`r_hat` nuôi vào attack collusion** và **`epsilon_hat` nuôi vào attack injection** → chạy 3 stage cho
từng backbone (RQ1) → `adaptive_search` (RQ3) → committee diverse (RQ4) → cold-cache + frontier
(RQ5) → các bộ benign cho ORR/utility (RQ6) → vẽ hình → ghi `RUN_MANIFEST.json`.

| Cờ | Ý nghĩa |
|---|---|
| `--backend vllm\|synthetic` | backend judge (mặc định `vllm`) |
| `--phases measure,rq1,rq1_collusion,rq2,rq3,rq4,rq5,rq6,baselines,plots` | chạy một phần (job ngắn hơn) |
| `--seeds 0,1,2,3,4` / `--n-payloads 1000` | quy mô chạy |
| `--backbones "llama:<HF id> qwen:<HF id> mistral:<HF id>"` | đổi model trên server khác |
| `--max-pairs 200` / `--variants A,B` | tiết chế phép đo ε |
| `--cost-n 1,3,5,7` | các `n` cho frontier RQ5 |
| `--baselines-config <yaml>` | bật pha head-to-head baseline ngoài |
| `--force` / `--dry-run` / `--mark-real` | chạy lại tất cả / xem trước / bỏ watermark hình |

**Resume sau khi hàng đợi cắt job**: mỗi bước có *sentinel* (file kết quả). Chỉ cần **submit lại đúng
lệnh cũ** — bước nào đã xong sẽ in `SKIP`, verdict cache giữ nguyên mọi call đã chạy, chỉ phần thiếu
được làm tiếp. Log nằm ở `<output-root>/logs/`.

Lệnh đã được kiểm chứng chạy thật (chế độ `--quick`, dữ liệu thật, verdict synthetic): 10/10 pha,
26 provenance record, 21 hình, resume trả về exit 0, và fail-fast khi thiếu data/CUDA.

## Cách 2: chạy tay từng pha (nếu muốn kiểm soát từng bước)


Quy ước: `<repo>` = thư mục repo trên laptop, `$GPU_HOST` = `user@host`, `$GPU_DIR` = thư mục repo
trên server. Các khối **PowerShell** chạy ở laptop, các khối **bash** chạy trên GPU server.

> Thời gian/chi phí call thật đã đo (n_judges = 7): calibrate + evaluate 1 backbone trên `formal`
> với 5 seed ≈ **9 800** call; `ablate` sau đó ≈ **0** (toàn cache hit); đo r/γ/ρ 1 seed ≈ **7 000**,
> 5 seed ≈ **9 800**; ε đầy đủ (1000 cặp × 4 biến thể × 2 mode) = **112 000**, pilot 200 cặp ×
> 2 biến thể × 1 mode = **5 600**. Nhân với `latency` đo ở Pha 4.

---

## Pha 0 — Trên laptop: rehearsal (rẻ, không tốn GPU)

```powershell
cd <repo>
python -m pytest -q                       # 163 test, phải pass
python -m ruff check .                    # phải sạch
python -m mypy src                        # phải sạch
```

Rehearsal toàn bộ đường ống trên **payload thật + verdict synthetic** (không cần GPU):

```powershell
@"
experiment:
  n_judges: 7
  rules: [cmed, gmed, krum]
  attack: compromise
  f: 2
  n_payloads: 200
  seed: 7
data:
  root: "data/benchmarks"
  benchmark: "formal"
  split: test
"@ | Set-Content configs/rehearsal_synthetic.yaml -Encoding utf8

python scripts/run_experiment.py --config configs/rehearsal_synthetic.yaml --stage calibrate --output outputs/rehearsal
python scripts/run_experiment.py --config configs/rehearsal_synthetic.yaml --stage evaluate  --output outputs/rehearsal --seeds 0,1
python scripts/run_experiment.py --config configs/rehearsal_synthetic.yaml --stage ablate    --output outputs/rehearsal
python scripts/measure_empirical.py --config configs/synthetic_demo.yaml --output outputs/measure_rehearsal --max-pairs 20
python scripts/make_plots.py --input outputs/rehearsal/evaluation_sweep.csv --output outputs/rehearsal/asr.png
```

**Cổng kiểm chứng**: `outputs/rehearsal/` có `calibration.json`, `evaluation_sweep.csv`,
`evaluation_summary.csv`, `evaluation_provenance.json`, `ablation.csv`; `outputs/measure_rehearsal/`
có `empirical_committee.csv`, `empirical_isolation.csv`, `empirical_isolation_summary.csv`.
Số của rehearsal là **kiểm tra đường ống**, tuyệt đối không báo cáo.

## Pha 1 — Đóng gói và đẩy lên server

```powershell
$GPU_HOST = "user@gpu-host.example.com"
$GPU_DIR  = "/home/user/aegis"
cd <repo>

# data/ và outputs/ nằm trong .gitignore -> phải sync thủ công.
tar -czf aegis_code.tgz `
  --exclude=".git" --exclude="outputs" --exclude="data" `
  --exclude="__pycache__" --exclude=".mypy_cache" --exclude=".pytest_cache" `
  --exclude=".ruff_cache" --exclude="*.egg-info" --exclude=".venv" .
tar -czf aegis_data.tgz --exclude="data/benchmarks/_raw" data\benchmarks

tar -tzf aegis_data.tgz | Select-String "test.csv$"      # phải đủ 8 dòng
ssh $GPU_HOST "mkdir -p $GPU_DIR"
scp aegis_code.tgz aegis_data.tgz "${GPU_HOST}:${GPU_DIR}/"
```

## Pha 2 — Trên server: giải nén, xác minh data, cài môi trường

```bash
cd "$GPU_DIR"
tar -xzf aegis_code.tgz && tar -xzf aegis_data.tgz && rm -f aegis_code.tgz aegis_data.tgz
mkdir -p outputs/cache logs

# Xác minh toàn vẹn (so với data/benchmarks/PROVENANCE.json)
sha256sum data/benchmarks/formal/test.csv data/benchmarks/second_order/test.csv data/benchmarks/benign_xstest/test.csv
#   formal        fb3101e3549a5c50eb9c8419da9301121c6ab5162c5e9c05220fc8d00a676cb8
#   second_order  f5c0fef406e021dac0c9a8dfca59d0d2d8bed26ee0e5e9d2e621802cdcde510d
#   benign_xstest f9c58dc0596910ff8919d9ddd793ad1d2fb3eaface3b63c5c8063fd4c5432566

python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,vllm]"
python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), vllm.__version__)"
export HF_TOKEN="<token>"        # không commit; chỉ cần cho backbone gated (meta-llama/*)
huggingface-cli whoami
pytest -q                        # 163 test
nvidia-smi
```

## Pha 3 — Sinh config cho 3 backbone + preflight

```bash
cd "$GPU_DIR" && source .venv/bin/activate
for tag in llama qwen mistral; do cp configs/real_experiment.yaml "configs/real_${tag}.yaml"; done

sed -i 's#^  backbones: .*#  backbones: [meta-llama/Meta-Llama-3-8B-Instruct]#; s#^  verdict_cache: .*#  verdict_cache: "outputs/cache/real_llama.jsonl"#'    configs/real_llama.yaml
sed -i 's#^  backbones: .*#  backbones: [Qwen/Qwen2.5-7B-Instruct]#;          s#^  verdict_cache: .*#  verdict_cache: "outputs/cache/real_qwen.jsonl"#'     configs/real_qwen.yaml
sed -i 's#^  backbones: .*#  backbones: [mistralai/Mistral-7B-Instruct-v0.3]#; s#^  verdict_cache: .*#  verdict_cache: "outputs/cache/real_mistral.jsonl"#' configs/real_mistral.yaml
grep -n "backbones\|verdict_cache\|isolation\|max_resident" configs/real_llama.yaml configs/real_qwen.yaml configs/real_mistral.yaml

tmux new -s aegis          # job dài: chạy trong tmux (Ctrl-b d để detach)
```

Preflight **một** payload (lấy `score` + `latency` để ước lượng thời gian):

```bash
python - <<'PY'
import numpy as np, time
from pathlib import Path
from aegis_agency.data.adapters import CsvBenchmarkAdapter
from aegis_agency.judges.llm_judge import VllmLLMJudge

p = next(CsvBenchmarkAdapter(Path("data/benchmarks/formal"), "test").iter_payloads())
j = VllmLLMJudge(backbone="meta-llama/Meta-Llama-3-8B-Instruct", threshold=0.5,
                 cache_path="outputs/cache/_preflight.jsonl")
t0 = time.time(); v = j.judge(p, 0, np.random.default_rng(0)); dt = time.time() - t0
print(f"score={v.score:.4f} decision={v.decision} latency={dt:.2f}s")
print(f"uoc luong 1 backbone: {dt*7*1400/3600:.2f} gio (1400 payload x 7 judge)")
PY
```

## Pha 4 — Đo r, γ, ρ (RQ4) và ε (RQ2)

```bash
# 4a. Pilot rẻ trước (2 biến thể, 1 mode, 200 cặp)
python scripts/measure_empirical.py --config configs/real_llama.yaml \
  --output outputs/measure_llama_pilot --seeds 7 \
  --max-pairs 200 --variants judgedeceiver,delimiter_escape --skip-isolation-ablation

cat outputs/measure_llama_pilot/empirical_committee.csv
cat outputs/measure_llama_pilot/empirical_gamma.csv
cat outputs/measure_llama_pilot/empirical_isolation_summary.csv

# 4b. Đo đầy đủ: 4 biến thể x isolation ON/OFF, 5 seed cho r/gamma/rho
python scripts/measure_empirical.py --config configs/real_llama.yaml \
  --output outputs/measure_llama --seeds 0,1,2,3,4 2>&1 | tee logs/measure_llama.log

# 4c. Cùng phép đo cho committee diverse (so rho giữa hai cấu hình - RQ4)
python scripts/measure_empirical.py --config configs/real_diverse.yaml \
  --output outputs/measure_diverse --seeds 0,1,2,3,4 2>&1 | tee logs/measure_diverse.log
```

**Cổng kiểm chứng**: `empirical_committee.csv` có `r_p95`, `gamma_min`, `margin_condition_ok`;
`empirical_isolation_summary.csv` có `epsilon_decision_flip_max` + biến thể đạt max;
`empirical_provenance.json` có `is_paper_result: false` và danh sách caveat.

## Pha 5 — Ba backbone homogeneous, đủ 3 stage

```bash
for tag in llama qwen mistral; do
  python scripts/run_experiment.py --config "configs/real_${tag}.yaml" --stage calibrate \
    --output "outputs/real_vllm_${tag}" 2>&1 | tee "logs/${tag}_calibrate.log"
  python scripts/run_experiment.py --config "configs/real_${tag}.yaml" --stage evaluate \
    --output "outputs/real_vllm_${tag}" --seeds 0,1,2,3,4 2>&1 | tee "logs/${tag}_evaluate.log"
  python scripts/run_experiment.py --config "configs/real_${tag}.yaml" --stage ablate \
    --output "outputs/real_vllm_${tag}" 2>&1 | tee "logs/${tag}_ablate.log"
done
```

**Cổng kiểm chứng** (chạy sau mỗi backbone):

```bash
for d in outputs/real_vllm_llama outputs/real_vllm_qwen outputs/real_vllm_mistral; do
  echo "--- $d"; ls -1 "$d"
  python - "$d" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
for name in ("calibration.json","evaluation_sweep.csv","evaluation_summary.csv",
             "evaluation_provenance.json","ablation.csv","ablation_provenance.json"):
    p = d / name
    print(f"  {'OK ' if p.exists() else 'MISSING'} {name} ({p.stat().st_size if p.exists() else 0} B)")
prov = json.loads((d / "evaluation_provenance.json").read_text())
print("  data_source:", prov["data_source"], "| is_paper_result:", prov["is_paper_result"])
print("  models:", prov["config"]["models"], "| seeds:", prov["config"]["seeds"])
PY
done
```

## Pha 6 — RQ4: committee diverse trong một lần chạy

```bash
python scripts/run_experiment.py --config configs/real_diverse.yaml --stage calibrate --output outputs/real_vllm_diverse
python scripts/run_experiment.py --config configs/real_diverse.yaml --stage evaluate  --output outputs/real_vllm_diverse --seeds 0,1,2,3,4
python scripts/run_experiment.py --config configs/real_diverse.yaml --stage ablate    --output outputs/real_vllm_diverse
```

Lưu ý: `n_judges: 7` chia 3 backbone thành 3+2+2. Muốn cân bằng tuyệt đối thì
`sed -i 's/^  n_judges: .*/  n_judges: 6/' configs/real_diverse.yaml` (2+2+2) và ghi rõ trong bài;
giữ cùng `n_judges` với các lần homogeneous nếu muốn so trực tiếp.

## Pha 7 — RQ5: số đo thật (cold cache) + frontier theo n

```bash
# 7a. Cold cache: bắt buộc để có evaluation_cost*.csv (cache hit không phải là call)
python scripts/run_experiment.py --config configs/real_llama.yaml --stage evaluate \
  --output outputs/real_vllm_llama_cold --seeds 7 \
  --cache "outputs/cache/cold_$(date +%Y%m%d_%H%M).jsonl"

# 7b. Frontier n = 1,3,5,7: mỗi n một cache TRẮNG riêng (không lấy từ ablation_cost.csv)
for n in 1 3 5 7; do
  sed "s/^  n_judges: .*/  n_judges: $n/" configs/real_llama.yaml > "configs/real_llama_n${n}.yaml"
  python scripts/run_experiment.py --config "configs/real_llama_n${n}.yaml" --stage evaluate \
    --output "outputs/real_vllm_llama_n${n}" --seeds 7 \
    --cache "outputs/cache/cold_n${n}.jsonl"
done
```

**Cổng kiểm chứng**: các thư mục `*_cold`, `*_n1/_n3/_n5/_n7` phải có `evaluation_cost.csv` và
`evaluation_cost_payloads.csv` (nếu thiếu ⇒ run đó toàn cache hit, chưa có số RQ5).

## Pha 8 — Head-to-head baseline ngoài (chỉ sau khi xác minh checkpoint/prompt)

> Đây là bước **có điều kiện**: phải ghim checkpoint đã kiểm chứng với repo tác giả trước, và nhớ
> rằng rows của hệ ngoài là **hệ chưa bị compromise** — không so trực tiếp với `asr_uc(f)`.

```bash
python - <<'PY'
import pathlib, yaml
cfg = yaml.safe_load(pathlib.Path("configs/real_llama.yaml").read_text())
cfg.setdefault("experiment", {})
cfg["experiment"]["external_baselines"] = ["autodefense", "secalign", "struq"]
cfg["experiment"]["external_baseline_kwargs"] = {
    "autodefense": {"n_judges": 3},
    "secalign": {"checkpoint": "<HF id da kiem chung>"},
    "struq":    {"checkpoint": "<HF id da kiem chung>"},
}
pathlib.Path("configs/real_llama_baselines.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote configs/real_llama_baselines.yaml")
PY

python scripts/run_experiment.py --config configs/real_llama_baselines.yaml --stage evaluate \
  --output outputs/real_vllm_baselines --seeds 7 \
  --cache "outputs/cache/baselines_$(date +%Y%m%d_%H%M).jsonl" 2>&1 | tee logs/baselines.log
```

Nếu checkpoint không có chat template, vLLM sẽ báo lỗi ngay ở call đầu — lúc đó xem
`docs/baseline_adapters.md` và báo lại để bổ sung chế độ completion thô.

## Pha 9 — Vẽ hình (chỉ sau khi xác minh)

```bash
for d in outputs/real_vllm_llama outputs/real_vllm_qwen outputs/real_vllm_mistral outputs/real_vllm_diverse; do
  python scripts/make_plots.py --input "$d/evaluation_sweep.csv" --output "$d/asr_vs_f.png" --real
done
```

`--real` chỉ dùng khi inputs đã xác minh (`platform` = Linux GPU box, `data_source` = `benchmark:*`).

## Pha 10 — Kéo kết quả về và cập nhật audit

```bash
# Trên server
tar -czf results.tgz outputs/real_vllm_* outputs/measure_* outputs/cache logs
```

```powershell
# Trên laptop
scp "${GPU_HOST}:${GPU_DIR}/results.tgz" .
tar -xzf results.tgz -C .
Get-ChildItem outputs\real_vllm_*\evaluation_provenance.json | ForEach-Object { $_.FullName; Get-Content $_ }
```

Sau đó (bắt buộc, không bỏ):
1. Cập nhật `audits/result_integrity_audit.md`: lệnh đã chạy, cache path, seed, `platform`, số nào
   vào bảng/hình nào của bài.
2. Chỉ khi đó mới set `is_paper_result: true` cho artefact tương ứng (các file đã ship trong repo
   phải giữ `false`).
3. Ghi rõ mọi caveat: ε là **max trên họ biến thể** (không phải supremum); `rho` không xác định với
   committee homogeneous temperature-0; judge thật hiện **score-only** (`d = 1`); ablation
   `diverse/homogeneous` + `hardened_*` trên backend thật chỉ là proxy synthetic; baseline ngoài là
   **bản cài lại**, hệ chưa bị compromise.

---

## Phụ lục — Bảng chi phí và các lệnh kiểm tra nhanh

| Việc | Lệnh | Call (n=7) |
|---|---|---|
| calibrate + evaluate 1 backbone, 5 seed (`formal`) | `--stage calibrate` + `--stage evaluate --seeds 0,1,2,3,4` | ≈ 9 800 |
| ablate 1 backbone (sau đó, cùng seed) | `--stage ablate` | ≈ 0 (cache hit) |
| r/γ/ρ 1 seed / 5 seed | `measure_empirical.py --seeds ...` | 7 000 / 9 800 |
| ε pilot | `--max-pairs 200 --variants A,B --skip-isolation-ablation` | 5 600 |
| ε đầy đủ | `measure_empirical.py` (mặc định) | 112 000 |
| RQ5 frontier | 4 × `--stage evaluate --seeds 7 --cache cold_n<N>.jsonl` | 4 × 7 000 |

Kiểm tra nhanh trong lúc chạy:

```bash
watch -n 60 'wc -l outputs/cache/*.jsonl | tail -3'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
grep -c "" logs/*.log
```

Sự cố thường gặp: xem `docs/gpu_server_runbook.md` §11 (vLLM chưa cài, CUDA OOM, parse score lỗi,
401/403 HF, thiếu cột CSV, cache cũ, SSH rớt).
