# GPU-Server Runbook (SSH-only)

> **Danh sách lệnh copy-paste cho toàn bộ dự án (đủ 10 pha, kèm cổng kiểm chứng):**
> `docs/run_all_commands.md`. Tài liệu này giải thích *vì sao* từng bước.

Runbook này thay thế giả định "máy chạy harness chính là EC2" bằng: **harness + data + weights
nằm trên một GPU server Linux, điều khiển qua SSH**. Không có HTTP endpoint, không có API cloud.

> **Ràng buộc bất di bất dịch (không được thay đổi / rút gọn):**
> 1. Đúng **3 stage** theo thứ tự `calibrate → evaluate → ablate`, đúng
>    `scripts/run_experiment.py`, đúng config trong `configs/`.
> 2. Không đổi cơ chế, math, metrics, hay tập baseline. Nhánh `synthetic` giữ nguyên.
> 3. Mọi artifact phải kèm `*_provenance.json`; `is_paper_result` chỉ được set `true` sau khi
>    có provenance thật và đã cập nhật `audits/result_integrity_audit.md`.
> 4. Runbook này **không sửa** `src/`, `scripts/`, `configs/` — chỉ mô tả thao tác vận hành.

---

## 0. Vì sao phương án này (không code mạng)

Đường chạy "thật" duy nhất của repo là **vLLM in-process**:

- `src/aegis_agency/judges/llm_judge.py:104-125` — `_get_llm_engine()` gọi
  `from vllm import LLM; LLM(model=backbone, trust_remote_code=True)` **trong chính process
  đang chạy harness**, cache engine theo process.
- `src/aegis_agency/judges/llm_judge.py:146-151` — `_score()` gọi `engine.chat(...)` trực tiếp,
  từng prompt một, không batch.
- `configs/real_experiment.yaml:37-40` — chỉ có `backend / backbones / verdict_cache`; không có
  chỗ khai báo host/port/token. `LLMJudgeAdapter.endpoint_or_path`
  (`src/aegis_agency/data/adapters.py:94`) tồn tại nhưng `VllmLLMJudge` không dùng.
- Toàn bộ I/O là filesystem cục bộ: data ở `data/benchmarks/`, cache ở
  `outputs/cache/*.jsonl`, kết quả ở `outputs/`.

⇒ Máy chạy harness **phải là** GPU server. Vì vậy runbook chỉ gồm: đồng bộ lên, chạy 3 stage,
kéo kết quả về.

---

## 1. Điều kiện tiên quyết trên GPU server

| Hạng mục | Yêu cầu | Cách kiểm tra |
|---|---|---|
| OS | Linux x86_64 (vLLM không hỗ trợ Windows native) | `uname -a` |
| GPU | NVIDIA, driver + CUDA đủ cho vLLM | `nvidia-smi` |
| VRAM | ≥ 24 GB cho **một** backbone 8B (fp16) + KV cache | `nvidia-smi --query-gpu=memory.total --format=csv` |
| Python | ≥ 3.11 | `python3 --version` |
| Đĩa | ≥ 10 GB (venv + vLLM + HF cache) + data 35 MB | `df -h ~` |
| Mạng | Ra được HuggingFace Hub **hoặc** đã có weights local | `curl -sI https://huggingface.co` |
| HF token | `meta-llama/*` là model gated → phải accept license + `HF_TOKEN` | `huggingface-cli whoami` |
| Quyền | Ghi được vào thư mục project + `outputs/` | `touch outputs/.w && rm outputs/.w` |

Nếu muốn dùng nhiều backbone (RQ4), xem §8 — có ràng buộc VRAM và một chặn ở tầng code.

---

## 2. Biến môi trường (điền trước khi chạy)

```bash
# ---- chạy trên MÁY LOCAL (Windows PowerShell) ----
$GPU_HOST = "user@gpu-host.example.com"     # SSH target
$GPU_DIR  = "/home/user/aegis"              # thư mục project trên server
$SSH_KEY  = "$env:USERPROFILE\.ssh\id_ed25519"   # bỏ qua nếu dùng ssh-agent
```

```bash
# ---- chạy trên GPU SERVER ----
export AEGIS_DIR="$HOME/aegis"
export AEGIS_CFG="configs/real_experiment.yaml"
export AEGIS_OUT="outputs/real_vllm"
export HF_HOME="$HOME/.cache/huggingface"
```

---

## 3. Bước 1 — Đóng gói ở máy local

`data/` và `outputs/**` **nằm trong `.gitignore`** (`.gitignore:19,26`), nên `git clone` trên
server sẽ **không** có data. Phải sync thủ công.

Đóng gói **hai** gói riêng (code ~0.5 MB, data ~35 MB):

```powershell
# 1) Code + tests + configs + docs (KHÔNG gồm data/outputs/.git)
tar -czf aegis_code.tgz `
  --exclude=".git" --exclude="outputs" --exclude="data" `
  --exclude="__pycache__" --exclude=".mypy_cache" --exclude=".pytest_cache" `
  --exclude=".ruff_cache" --exclude="*.egg-info" --exclude=".venv" `
  .

# 2) Data thật + provenance (bắt buộc, không có trong git)
#    _raw/ là 27.7 MB trong tổng 34.8 MB và CHỈ cần nếu muốn chạy lại tiền xử lý
#    (script tiền xử lý không nằm trong repo này) -> mặc định bỏ qua cho nhẹ.
tar -czf aegis_data.tgz --exclude="data/benchmarks/_raw" data/benchmarks

# Kiểm tra nội dung trước khi gửi
tar -tzf aegis_code.tgz | Select-Object -First 15
tar -tzf aegis_data.tgz | Select-Object -First 15
```

Kích thước đã đo trên bản hiện tại: code ~2.0 MB, data (không `_raw`) ~1.6 MB, data đầy đủ
~10.4 MB. Gói data **phải** chứa đủ 8 file `test.csv` + `PROVENANCE.json`:

```powershell
tar -tzf aegis_data.tgz | Select-String "test.csv$"
```

Ghi lại checksum của gói data để đối chiếu sau khi copy:

```powershell
Get-FileHash aegis_code.tgz,aegis_data.tgz -Algorithm SHA256 | Format-Table Hash,Path
```

---

## 4. Bước 2 — Copy lên server và kiểm tra toàn vẹn

Local có `ssh/scp/tar` sẵn (không có `rsync`). Dùng `scp` hai bước:

```powershell
ssh $GPU_HOST "mkdir -p $GPU_DIR"
scp aegis_code.tgz aegis_data.tgz "${GPU_HOST}:${GPU_DIR}/"
```

```bash
# ---- trên GPU SERVER ----
cd "$AEGIS_DIR"
tar -xzf aegis_code.tgz && tar -xzf aegis_data.tgz
rm -f aegis_code.tgz aegis_data.tgz
mkdir -p outputs/cache

# Đối chiếu checksum gói (so với giá trị in ở Bước 1)
sha256sum  # (nếu đã xoá gói thì bỏ qua; đối chiếu bằng hash file CSV bên dưới là đủ)
```

Kiểm tra **toàn vẹn data** bằng hash đã ghi trong `data/benchmarks/PROVENANCE.json`:

```bash
sha256sum data/benchmarks/formal/test.csv data/benchmarks/second_order/test.csv data/benchmarks/benign_xstest/test.csv
# Kỳ vọng (PROVENANCE.json):
#   formal/test.csv        fb3101e3549a5c50eb9c8419da9301121c6ab5162c5e9c05220fc8d00a676cb8
#   second_order/test.csv  f5c0fef406e021dac0c9a8dfca59d0d2d8bed26ee0e5e9d2e621802cdcde510d
#   benign_xstest/test.csv f9c58dc0596910ff8919d9ddd793ad1d2fb3eaface3b63c5c8063fd4c5432566

# Đếm payload từng bộ (khớp bảng dưới)
python - <<'PY'
from pathlib import Path
from aegis_agency.data.adapters import CsvBenchmarkAdapter
for b in ["harmbench","advbench","dan","injecagent","benign","benign_xstest","formal","second_order"]:
    p = list(CsvBenchmarkAdapter(Path("data/benchmarks")/b, "test").iter_payloads())
    pos = sum(1 for x in p if int(x.true_label) == 1)
    print(f"{b:14s} rows={len(p):5d} label1={pos:5d} label0={len(p)-pos:5d}")
PY
```

Kỳ vọng (đã kiểm chứng trên bản data hiện có):

| benchmark | rows | label=1 | label=0 |
|---|---|---|---|
| harmbench | 320 | 320 | 0 |
| advbench | 520 | 520 | 0 |
| dan | 1405 | 1405 | 0 |
| injecagent | 510 | 510 | 0 |
| benign | 299 | 0 | 299 |
| benign_xstest | 250 | 0 | 250 |
| formal | 1400 | 700 | 700 |
| second_order | 2000 | 1000 | 1000 |

---

## 5. Bước 3 — Cài môi trường (chỉ trên GPU server)

```bash
cd "$AEGIS_DIR"
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,vllm]"          # extras: pyproject.toml:20-29

# Xác nhận vLLM + CUDA nhìn thấy GPU
python - <<'PY'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "n", torch.cuda.device_count())
print("vllm", vllm.__version__)
PY
```

Nếu server **không có internet**: build wheelhouse ở máy có mạng rồi copy sang
(`pip download -d wheelhouse -e ".[dev,vllm]"` → `pip install --no-index --find-links wheelhouse ...`).

Đăng nhập HuggingFace nếu dùng backbone gated (`meta-llama/*`):

```bash
export HF_TOKEN="<token>"        # KHÔNG commit, không ghi vào config
huggingface-cli whoami
```

**Không commit secrets** — đúng quy định đã ghi ở `docs/ec2_experiment_guide.md` §3.

---

## 6. Bước 4 — Preflight (kiểm tra đường ống, KHÔNG thay thế stage nào)

```bash
cd "$AEGIS_DIR" && source .venv/bin/activate
python -m aegis_agency.cli info
pytest -q                       # 47+ test synthetic, xác nhận môi trường lành
```

Smoke-test **một** payload qua judge thật để bắt lỗi prompt/parse/VRAM trước khi chạy dài.
Đây chỉ là kiểm tra đường ống — output của nó **không** đi vào `outputs/` của paper:

```bash
python - <<'PY'
import numpy as np, time
from pathlib import Path
from aegis_agency.data.adapters import CsvBenchmarkAdapter
from aegis_agency.judges.llm_judge import VllmLLMJudge

p = next(CsvBenchmarkAdapter(Path("data/benchmarks/formal"), "test").iter_payloads())
j = VllmLLMJudge(backbone="meta-llama/Meta-Llama-3-8B-Instruct",
                 threshold=0.5, cache_path="outputs/cache/_preflight.jsonl")
t0 = time.time()
v = j.judge(p, 0, np.random.default_rng(0))
dt = time.time() - t0
print(f"score={v.score:.4f} decision={v.decision} latency={dt:.2f}s")
print(f"uoc luong cho 1 backbone: {dt*7*1000/3600:.2f} gio cho 7 judge x 1000 payload")
PY
```

> Con số `latency` ở trên là **công cụ lập kế hoạch**, không phải metric RQ5. RQ5 trong code chỉ
> có mô hình công thức (`src/aegis_agency/metrics/cost.py`), chưa có runner nào đo latency/token
> thật — xem §10 mục 5 và `docs/runner_metrics_plan.md` Mức 2.

---

## 7. Bước 5 — Chạy 3 stage (đúng thứ tự, không rút gọn)

Dùng `tmux` để không mất job khi SSH rớt (`tmux new -s aegis`, `Ctrl-b d` để detach,
`tmux attach -t aegis` để quay lại). Nếu không có `tmux`, dùng `nohup ... &` + `tee`.

```bash
cd "$AEGIS_DIR" && source .venv/bin/activate
mkdir -p "$AEGIS_OUT" logs

# Stage 1/3
python scripts/run_experiment.py --config "$AEGIS_CFG" --stage calibrate --output "$AEGIS_OUT" \
  2>&1 | tee logs/calibrate.log

# Stage 2/3 -- seed sweep gives evaluation_summary.csv (mean +/- std) theo checklist của paper
python scripts/run_experiment.py --config "$AEGIS_CFG" --stage evaluate --output "$AEGIS_OUT" \
  --seeds 0,1,2,3,4 2>&1 | tee logs/evaluate.log

# Stage 3/3
python scripts/run_experiment.py --config "$AEGIS_CFG" --stage ablate --output "$AEGIS_OUT" \
  2>&1 | tee logs/ablate.log
```

> `--seeds` chỉ có tác dụng ở stage `evaluate` (các stage khác sẽ cảnh báo và bỏ qua). Bỏ cờ này
> thì hành vi y như trước (đúng 1 seed của config).

Điểm vận hành cần biết (đọc từ code, không phải giả định):

- **Cache dùng chung**: `judges.verdict_cache: outputs/cache/real_vllm.jsonl`
  (`configs/real_experiment.yaml:40`) key theo `(model, payload_id, judge_id)`
  (`src/aegis_agency/utils/cache.py`). Chạy lại / resume **không** prompt lại payload đã có.
  ⇒ Nếu đổi backbone/threshold/prompt, **phải** đổi `verdict_cache` sang file mới, nếu không
  verdict cũ sẽ được tái sử dụng âm thầm.
- **Seed sweep không nhân 5 lần chi phí GPU**: cache key **không** chứa seed, nên verdict của
  payload đã gặp ở seed trước được tái sử dụng; tổng số call bị chặn trên bởi số payload của
  benchmark (ví dụ `formal` 1400 dòng × n judge), không phải 5 × 1000 × n.
- **Nguồn payload**: `n_payloads: 1000` trên `formal` (1400 dòng) → subsample seeded
  (`harness.py`), **250 dòng đầu làm calibration, 750 dòng còn lại để evaluate**.
- **Evaluate** sweep `f = 0..(n-1)//2 = 0..3` với `n_judges: 7`
  (`experiments/run_evaluation.py`), mỗi `f` là một trial đầy đủ; calibration được lặp lại
  mỗi trial nhưng đã có cache nên không phát sinh prompt mới.
- **Đầu ra của evaluate**: `evaluation_sweep.csv` (một dòng / `(seed, f, method)`),
  `evaluation_summary.csv` (mean ± std theo seed), `evaluation_cost.csv` (**chỉ** khi judge thật
  có đo latency/token — cache hit không tính là call nên chạy fully-cached sẽ không có file này),
  và `evaluation_provenance.json`. Chi tiết cột: `docs/reproducibility.md`.
- **Ablate** chạy thêm ~12 trial với `calibrate=False` (`experiments/run_ablation.py`).
- Ước lượng tổng prompt nếu cache trắng: ~7000 call cho evaluate + ~12×750 cho ablate, **tuần tự**
  (`llm_judge.py:150`, không batch). Dùng số đo ở §6 để tính thời gian, rồi chạy `tmux`.

Theo dõi tiến độ mà không làm hỏng job:

```bash
watch -n 60 'wc -l outputs/cache/real_vllm.jsonl'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

### 7.1 RQ5 — chạy cold-cache để có số đo THẬT

Số đo latency/token chỉ có khi judge **thực sự được gọi**: cache hit không phải là một call, nên
chạy lại trên cache cũ sẽ **không** sinh `evaluation_cost.csv`. Muốn số cho RQ5 thì chỉ định một
cache mới bằng `--cache` (hoặc sửa `judges.verdict_cache`):

```bash
# evaluate trên cache trắng (cold cache) -> evaluation_cost.csv + evaluation_cost_payloads.csv
python scripts/run_experiment.py --config configs/real_llama.yaml --stage evaluate \
  --output outputs/real_vllm_llama_cold --cache outputs/cache/cold_$(date +%Y%m%d_%H%M).jsonl
```

**Frontier RQ5 theo `n` — cách ĐÚNG** (mỗi `n` một cache trắng riêng, chạy `evaluate` với
`n_judges` cố định):

```bash
for n in 1 3 5 7; do
  sed "s/^  n_judges: .*/  n_judges: $n/" configs/real_llama.yaml > "configs/real_llama_n${n}.yaml"
  python scripts/run_experiment.py --config "configs/real_llama_n${n}.yaml" --stage evaluate \
    --output "outputs/real_vllm_llama_n${n}" --seeds 7 \
    --cache "outputs/cache/cold_n${n}.jsonl"
done
```

> **Đừng lấy frontier từ `ablation_cost.csv`.** Sweep `committee_n1..n7` trong stage `ablate`
> chạy **sau** các hàng ablation ở `n = 7` trên cùng một cache, nên khi tới `committee_n1` thì
> slot judge 0..6 đã có verdict ⇒ `n_calls = 0` (và các `n` sau chỉ thêm slot mới). Đó là số đo
> của *các trial ablation như đã cấu hình*, không phải frontier sạch. Frontier sạch = một
> `evaluate` cho mỗi `n` với cache trắng như trên (mỗi lần chỉ 1 seed để tiết kiệm GPU).

Đọc số thế nào:

| File | Nội dung |
|---|---|
| `evaluation_cost.csv` | mỗi `(seed, f)`: `n_calls`, latency mean/p50/p95, prompt/completion tokens |
| `evaluation_cost_payloads.csv` | mỗi `(seed, f, payload_id)`: `max_latency_s` = Eq. (lat-par) `max_k lat(J_k)` **đo thật**, `sum_latency_s` = Eq. (lat-seq) `sum_k lat(J_k)`; kèm token |
| `ablation_cost.csv` / `ablation_cost_payloads.csv` | số đo của các trial ablation (theo `ablation` + `n_judges`); **không** dùng làm frontier, xem cảnh báo trên |

### 7.2 Ngân sách GPU ước lượng (số call thật, cache trắng)

Đo bằng RNG subsampling thật, không phải phỏng đoán (`n_judges = 7`):

| Việc | Payload | Judge call |
|---|---|---|
| calibrate + evaluate 1 backbone trên `formal` (seed 7 + seeds 0..4) | union ≈ **1400**/1400 dòng | ≈ **9 800** |
| `ablate` sau đó (cùng seed, cùng payload) | đã có trong union | ≈ **0** (toàn cache hit) |
| đo ε trên `second_order`, **đầy đủ** 1000 cặp × 2 mode × **4 biến thể** | 2000 | **112 000** |
| đo ε, pilot `--max-pairs 200 --skip-isolation-ablation --variants judgedeceiver,delimiter_escape` | 400 | **5 600** |
| đo r/γ/ρ (1 seed, `formal`) | 1000 (union 1400 nếu nhiều seed) | 7 000 (≈9 800 cho 5 seed) |

Với ε: **tăng số biến thể là tăng chi phí tuyến tính**, nên chạy pilot rồi mở rộng dần. Bộ đầy đủ
(1000 cặp × 4 biến thể × 2 mode) là việc nhiều giờ; pilot 200 cặp × 2 biến thể × 1 mode là vài giờ
đổ lại. Payload "clean" dùng chung cho mọi biến thể nên chỉ tính một lần.

Nhân với `latency` đo ở §6 để ra thời gian. Với committee diverse (`max_resident_engines: 1`),
cộng thêm chi phí **nạp lại engine**: mỗi seed mới phát sinh payload mới ⇒ ~3 lần nạp model cho
mỗi seed (~1–3 phút/lần với 8B) ⇒ khoảng 15–45 phút cho 5 seed. Verdict cache làm cho các lần
chạy lại gần như miễn phí.

---

## 8. Judge backbones: homogeneous, diverse, và đo RQ2/RQ4

Máy GPU: **1 GPU ≥ 40 GB**. Ba backbone 8B (fp16) **không** nằm cùng lúc được trong 40 GB, nên
committee diverse chạy theo **session từng backbone** (`judges.max_resident_engines: 1`): nạp A →
chạy hết payload cho các judge dùng A → giải phóng VRAM → nạp B → ... Các engine mà verdict đã có
trong cache thì **không nạp gì cả**.

### 8.1 Sinh config homogeneous cho từng backbone

```bash
cd "$AEGIS_DIR"
for tag in llama qwen mistral; do cp configs/real_experiment.yaml "configs/real_${tag}.yaml"; done

# Chỉ đổi ĐÚNG 2 dòng: backbones + verdict_cache. Mọi tham số khác giữ nguyên.
sed -i 's#^  backbones: .*#  backbones: [meta-llama/Meta-Llama-3-8B-Instruct]#; s#^  verdict_cache: .*#  verdict_cache: "outputs/cache/real_llama.jsonl"#'    configs/real_llama.yaml
sed -i 's#^  backbones: .*#  backbones: [Qwen/Qwen2.5-7B-Instruct]#;          s#^  verdict_cache: .*#  verdict_cache: "outputs/cache/real_qwen.jsonl"#'     configs/real_qwen.yaml
sed -i 's#^  backbones: .*#  backbones: [mistralai/Mistral-7B-Instruct-v0.3]#; s#^  verdict_cache: .*#  verdict_cache: "outputs/cache/real_mistral.jsonl"#' configs/real_mistral.yaml

grep -n "backbones\|verdict_cache\|max_resident\|isolation" configs/real_llama.yaml configs/real_qwen.yaml configs/real_mistral.yaml
```

> Cache key là `(model, payload_id, judge_id)` **cộng thêm hậu tố chế độ isolation**
> (`<backbone>+noisolation` khi `isolation: false`), nên nhánh RQ2 un-isolated không bao giờ đọc
> nhầm verdict đã cách ly. Tách file cache theo backbone chỉ để theo dõi tiến độ (`wc -l`) và cô
> lập lỗi khi resume.

### 8.2 Chạy tuần tự (1 GPU nên không song song)

```bash
cd "$AEGIS_DIR" && source .venv/bin/activate
mkdir -p logs
tmux new -s aegis

for tag in llama qwen mistral; do
  python scripts/run_experiment.py --config "configs/real_${tag}.yaml" --stage calibrate \
    --output "outputs/real_vllm_${tag}" 2>&1 | tee "logs/${tag}_calibrate.log"
  python scripts/run_experiment.py --config "configs/real_${tag}.yaml" --stage evaluate \
    --output "outputs/real_vllm_${tag}" --seeds 0,1,2,3,4 2>&1 | tee "logs/${tag}_evaluate.log"
  python scripts/run_experiment.py --config "configs/real_${tag}.yaml" --stage ablate \
    --output "outputs/real_vllm_${tag}" 2>&1 | tee "logs/${tag}_ablate.log"
done
```

Thời gian ≈ 3 × (thời gian một backbone, tính từ số `latency` đo ở §6).

### 8.3 RQ4 — committee diverse trong MỘT lần chạy (đã cài)

`configs/real_diverse.yaml` gán ba backbone **round-robin** vào 7 slot judge
(`llama, qwen, mistral, llama, qwen, mistral, llama`) và đặt `max_resident_engines: 1`:

```bash
python scripts/run_experiment.py --config configs/real_diverse.yaml --stage calibrate --output outputs/real_vllm_diverse
python scripts/run_experiment.py --config configs/real_diverse.yaml --stage evaluate  --output outputs/real_vllm_diverse --seeds 0,1,2,3,4
python scripts/run_experiment.py --config configs/real_diverse.yaml --stage ablate    --output outputs/real_vllm_diverse
```

Đúng ba stage, cùng thứ tự, không rút gọn. Lưu ý cân bằng: `n_judges: 7` chia 3 backbone thành
3+2+2; nếu muốn cân bằng tuyệt đối thì dùng `n_judges: 6` (2+2+2) **và ghi rõ** trong bài. Muốn
so homogeneous-vs-diverse thì giữ cùng `n_judges` giữa hai cấu hình.

> ⚠️ **Đừng lấy RQ4 từ `ablation.csv` khi chạy backend thật.** Hai hàng `diverse_backbones` /
> `homogeneous_backbones` trong stage `ablate` được điều khiển bằng tham số **mô hình synthetic**
> `correlation` (`rho`), và `hardened_on/off` bằng `radius` (`r`) — với judge thật, các tham số đó
> **không tác động gì**, nên những hàng ấy chỉ là cùng một cấu hình chạy lại (số giống hệt nhau,
> dễ bị đọc thành "diverse không khác homogeneous"). Bằng chứng RQ4 thật lấy từ việc **so các
> config**: 3 lần chạy homogeneous (`real_llama`/`real_qwen`/`real_mistral`) vs 1 lần diverse
> (`real_diverse`) ở cùng `n_judges`, cộng `empirical_rho_pairs.csv` từ §8.5 (ρ theo từng cặp
> backbone). Stage `ablate` trên backend thật vẫn dùng được cho các hàng *isolation on/off* và
> *robust aggregation on/off*, vì hai hàng đó điều khiển **attack/threshold**, không phải judge.

### 8.4 Chọn benchmark — cần bạn xác nhận (protocol)

Harness chạy **một** benchmark mỗi config (`data.benchmark`). Bảng ánh xạ gợi ý, suy ra từ
`family` trong `data/benchmarks/PROVENANCE.json` + `docs/data_format.md` — **tôi không tự chốt**:

| benchmark | rows (label 1/0) | family | RQ gợi ý |
|---|---|---|---|
| formal | 1400 (700/700) | injection (class-balanced) | RQ1, RQ3 — cấu hình mặc định hiện tại |
| injecagent | 510 (510/0) | injection | RQ1, RQ2 |
| second_order | 2000 (1000/1000) | second-order (JudgeDeceiver) | RQ2 — đo ε bằng §8.5 |
| harmbench | 320 (320/0) | jailbreak | RQ1 |
| advbench | 520 (520/0) | jailbreak | RQ1 |
| dan | 1405 (1405/0) | jailbreak in-the-wild | RQ1 |
| benign | 299 (0/299) | benign | RQ6 (ORR / utility) |
| benign_xstest | 250 (0/250) | benign (XSTest safe) | RQ6 (over-refusal, prompt hợp lệ) |

Nếu chạy nhiều benchmark, mỗi (backbone × benchmark) là một output dir riêng:
`--output outputs/real_vllm_<tag>_<benchmark>` và `verdict_cache: "outputs/cache/real_<tag>_<benchmark>.jsonl"`.
Đã kiểm chứng `payload_id` được namespace theo benchmark (`formal_*`, `second_order_*`,
`injecagent_*`, `harmbench_*`, `advbench_*`, `dan_*`, `benign_*`, và **`xstest_*`** cho
`benign_xstest`) nên id giữa các bộ **không trùng**.

**Khoảng trống coverage (chốt — không thêm data, không claim đủ bộ):** paper chỉ báo cáo
**8 bộ đang có** ở bảng trên. Các nguồn còn thiếu — **PAIR, TAP, GPTFuzz, universal-injection** —
**không được thêm vào repo và không được claim** khi viết bài, vì:
PAIR/TAP/GPTFuzz là bộ *sinh* tấn công cần victim model để tối ưu (nếu chạy phải coi là hệ
sinh tấn công, không phải benchmark tĩnh); universal-injection là suffix phải huấn luyện riêng.
Báo cáo đúng coverage 8/12 nguồn trong danh sách `docs/data_format.md`, kèm dòng này trong
`docs/ec2_experiment_guide.md`/manuscript limitation (quyết định ghi ở `docs/runner_metrics_plan.md` §8.4).

### 8.5 Đo r, γ, ρ (RQ4) và ε (RQ2) — `scripts/measure_empirical.py`

Đây là entry point **bổ sung**: nó không thay thế/đổi thứ tự 3 stage, chỉ ghi thêm artifact.
`second_order` đã có sẵn 1000 cặp `_clean`/`_attack` khớp id, nên **ε đo được ngay, không cần repo
JudgeDeceiver bên ngoài**.

```bash
# r, gamma, rho cho committee homogeneous (một backbone) + epsilon isolation ON và OFF
python scripts/measure_empirical.py --config configs/real_llama.yaml \
    --output outputs/measure_llama --seeds 0,1,2,3,4

# RQ4: cùng phép đo cho committee diverse (so rho giữa hai cấu hình)
python scripts/measure_empirical.py --config configs/real_diverse.yaml \
    --output outputs/measure_diverse --seeds 0,1,2,3,4

# chạy thử trước cho rẻ (200 cặp, chỉ isolation ON)
python scripts/measure_empirical.py --config configs/real_diverse.yaml \
    --output outputs/measure_diverse_pilot --max-pairs 200 --skip-isolation-ablation
```

Định nghĩa đã cài (khớp paper, xem `src/aegis_agency/experiments/measure_empirical.py`):

| Đại lượng | Paper | Cách đo trong code |
|---|---|---|
| `r` | Assumption 1: `‖v_k − u*‖ ≤ r` | `u*` = trung bình verdict honest theo lớp (ước lượng consensus); `r` báo **profile** p50/p90/p95/max/rms, thêm `r_within` = khoảng cách tới consensus từng payload |
| `γ` | Assumption 2: `|π₁(u*) − τ| ≥ γ`, `D(u*) = y*` | theo từng lớp + `gamma_min` + cờ `condition_ok` kiểm luôn chiều đúng |
| `ρ` | Prop 3: tương quan cặp của `Z_k = 1{J_k đúng}` | mức decision + mức residual + mức score, kèm số cặp xác định được |
| `ε` | Def 1: `sup_c TV(P(·|c,x), P(·|c,x'))` | `epsilon_decision_flip` (đại lượng Prop 2 dùng) + `epsilon_tv_score` (TV chia bin) + `mean_abs_score_shift`, đo cho **cả họ biến thể injection** và cả isolation ON/OFF; `empirical_isolation_summary.csv` ghi **max theo biến thể** (đại diện gần nhất cho `sup`) |

**Bốn biến thể injection** (`INJECTION_VARIANTS`, mỗi biến thể có id payload riêng nên không đọc lẫn
verdict của nhau):

| variant | nội dung |
|---|---|
| `judgedeceiver` | payload tấn công có sẵn, **không** sửa gì |
| `naive_override` | ghi đè chỉ thị đơn giản ở cuối payload |
| `roleplay_takeover` | đổi vai/persona ở cuối payload |
| `delimiter_escape` | đóng sớm kênh `<|data|>` rồi giả làm operator — **tấn công trực tiếp vào giả định isolation của Def 1** |

Chạy: `--variants judgedeceiver,delimiter_escape` để giới hạn; mặc định là cả bốn.

Cảnh báo phải ghi vào bài khi báo số:
- `u*` được ước lượng **từ chính các judge**, không biết trước.
- ε dùng **một họ biến thể injection** (mặc định 4, trong đó `delimiter_escape` nhắm thẳng vào kênh
  cách ly); Def 1 là `sup` trên **mọi** injection nên **không đạt được bằng thực nghiệm** — báo
  per-variant + max (cột `epsilon_decision_flip_max` trong `empirical_isolation_summary.csv`), và
  **không được gọi đó là supremum**.
- Committee homogeneous temperature-0 **không có phương sai giữa các judge** ⇒ `rho` không xác
  định (ghi `NaN`, 0 cặp defined), không được bịa số. Muốn có ρ có nghĩa thì phải diverse.
- Judge thật hiện trả **score-only** (`embedding = zeros(0)`) ⇒ verdict vector `d = 1`, không phải
  `d = 1 + m` với rationale embedding như paper (`experiments.tex:54`); aggregation/attack/metric
  không phụ thuộc `d` nên kết quả vẫn đúng, nhưng phải ghi rõ cấu hình này khi viết bài.
- Chỉ số đo trên backend `synthetic` là **kiểm tra đường ống**, tuyệt đối không báo như kết quả.
- `is_paper_result` vẫn là `false`; chỉ lật sau khi cập nhật `audits/result_integrity_audit.md`.

### 8.6 Ghim revision + decoding params (checklist paper)

`judges.model_revision`, `judges.temperature`, `judges.max_tokens` được truyền thẳng xuống vLLM và
**ghi tự động vào mọi `*_provenance.json`** (`model_revision: "<unpinned>"` khi không ghim):

```yaml
judges:
  backend: vllm
  backbones: [meta-llama/Meta-Llama-3-8B-Instruct]
  model_revision: "main"      # commit/tag cụ thể; nên ghim đúng commit đã dùng
  temperature: 0.0            # đổi khác 0 sẽ phá giả định verdict tất định + cache
  max_tokens: 64
```

Engine được đăng ký theo khoá `backbone@revision`, nên hai revision khác nhau không dùng lẫn
engine. Giữ `temperature: 0.0`: verdict cache giả định judge tất định.

### 8.7 Nối số đo vào attack (RQ1/RQ2/RQ3 trên verdict thật)

Các attack trong `attacks/` thao tác trên **verdict vector**, nên chúng chạy được với verdict thật
(không gọi LLM). Nhưng tham số của chúng là hằng số mô hình synthetic, phải nối số đo thật vào
nếu không sẽ báo cáo một kịch bản không khớp judge:

| `experiment.attack` | Trên backend thật | Việc phải làm |
|---|---|---|
| `compromise` (RQ1) | dùng được ngay — thay `f` verdict bằng giá trị cực đoan | không cần gì thêm |
| `collusion` (RQ1, "a-little-is-enough") | dùng được, **nhưng** colluder ẩn trong bán kính `radius` — mà `radius` trong config là `r` **tổng hợp**, không phải `r` đo được | đặt `experiment.attack_kwargs: {radius: <r_p95 đo được>}` từ `empirical_committee.csv` |
| `adaptive` (RQ3) | dùng được; đây là **surrogate định hướng** (bước nhảy cố định), không tối ưu | ghi rõ là heuristic khi báo cáo |
| `adaptive_search` (RQ3) | **tối ưu theo đúng rule của từng pipeline**: tìm trong tập candidate verdict + mọi cách đặt `f` slot, chấm bằng chính aggregator, chọn cái đẩy aggregate xa ngưỡng nhất. Harness tự truyền `rule` + `threshold` đã calibrate cho từng method | dùng cái này cho RQ3; vẫn **không** phải tối ưu toàn cục (tìm trên tập hữu hạn) — nói rõ khi báo cáo |
| `injection` (RQ2) | dùng được, **nhưng** `epsilon` phải là ε **đo được** | đặt `attack_kwargs: {epsilon: <epsilon_decision_flip đo được>}` từ `empirical_isolation.csv`; đây chính là cách dựng hàng "isolation on/off" cho thật |

Ví dụ config cho RQ2/RQ1 trên judge thật (sau khi đã đo ε và r):

```yaml
experiment:
  attack: injection
  attack_kwargs: {epsilon: 0.07}     # <- epsilon_decision_flip do measure_empirical đo, isolation OFF
# hoặc
experiment:
  attack: collusion
  attack_kwargs: {radius: 0.11}      # <- r_p95 do measure_empirical đo
```

Nếu để nguyên mặc định, các hàng đó vẫn chạy nhưng **không** phải kịch bản của judge thật — phải
ghi rõ là mô hình tham số.

### 8.8 Baseline ngoài chạy head-to-head (AutoDefense / SecAlign / StruQ)

Ba hệ ngoài đã được **cài lại** trên chính engine local (`baselines/llm_baselines.py`) và được nối
vào harness như pipeline hạng nhất. Bật bằng config (mặc định **tắt**, nên tập method lịch sử không
đổi):

```yaml
experiment:
  external_baselines: [autodefense, secalign, struq]
  external_baseline_kwargs:
    autodefense:
      n_judges: 3                      # analyzer -> judge(s) -> coordinator; mặc định 1
    secalign:
      checkpoint: "<HF id đã kiểm chứng>"   # BẮT BUỘC, xem cảnh báo bên dưới
    struq:
      checkpoint: "<HF id đã kiểm chứng>"
judges:
  backbones: [meta-llama/Meta-Llama-3-8B-Instruct]   # analyzer=first, coordinator=last, judges=middle
  max_resident_engines: 1
  verdict_cache: "outputs/cache/real_llama.jsonl"    # mỗi role một namespace riêng trong cùng file
```

Khoá `extra` mà từng class chấp nhận:

| Class | Khoá |
|---|---|
| `AutoDefenseStyleBaseline` | `backbones` (list: analyzer=đầu, coordinator=cuối, judges=giữa; hoặc mapping role→backbone) · `analyzer`/`judges`/`coordinator`: `{backbone: str}` (`analyzer.enabled: false` để tắt) · `judges` có thể là list `[{backbone: ...}]` · `n_judges` (mặc định 1) · `judge_backbone` · `backbone` · `prompt_tag` (mặc định `autodefense`; namespace `<tag>:analyzer`, `<tag>:judge:<slot>`, `<tag>:coordinator`) · `cache_path` · `temperature` · `max_tokens` (mặc định 256) · `model_revision` · `max_resident_engines` (mặc định 1) · `threshold` · `isolation` · `prompts` (`analyzer_system`, `judge_system`, `coordinator_system`, `coordinator_user_template`) |
| `HardenedSingleModelAdapter` / `SecAlignHardenedAdapter` / `StruQHardenedAdapter` | checkpoint qua `model_path_or_endpoint` **hoặc** `extra['checkpoint']` **hoặc** `experiment.external_baseline_kwargs.<name>.checkpoint` (bắt buộc; `extra['backbone']` chỉ dùng khi `extra['allow_unhardened_backbone']: true`) · `system_prompt` · `user_template` (có `{payload}`, tuỳ chọn `{instruction}`) · `prompt_tag` · `cache_path` · `threshold` · `temperature` · `max_tokens` (mặc định 64) · `model_revision` · `max_resident_engines` · `isolation` |

Cảnh báo **phải** dẫn khi báo cáo:

- Đây là **bản cài lại**, không phải code của tác giả (`describe()["kind"] == "reimplementation"`).
  Prompt AutoDefense được **trích nguyên văn** từ `XHMY/AutoDefense` (`data/prompt/defense_prompts.json`,
  block `explicit_3_agent`); hai chỗ là của mình: (a) nối thêm contract JSON điểm vào prompt đã công
  bố, (b) tổng quát hoá "một Judge" thành *n* judge + một coordinator (điểm tin cậy đơn).
- **SecAlign/StruQ: checkpoint id và template prompt là mặc định cài lại, phải kiểm chứng lại với
  repo của tác giả trước khi báo bất kỳ số nào** — chúng là mirror HF bên thứ ba.
- Khi model trả lời theo **contract đã công bố** (`Judgment: VALID/INVALID`) mà không có điểm,
  verdict của judge được lấy ở mức decision (score 0/1) và ghi `score_source: "validity_text"` trong
  `last_trace`; chỉ khi **không** có cả hai contract mới raise (không bao giờ đoán).
- Rows của hệ ngoài là **hệ chưa bị compromise**: không thể inject verdict Byzantine vào một hệ
  không có interface verdict, nên ASR-UC của chúng là mức né sạch, **không** so trực tiếp với
  `asr_uc(f)` của các committee. Phải ghi rõ điều này trong bảng.
- Chi phí: mỗi payload tốn `1 analyzer + n_judges + 1 coordinator` call thật (đã cache); cộng thêm
  VRAM cho checkpoint riêng của SecAlign/StruQ.
- Sidecar `*.roles.jsonl` được ghi cạnh file cache (đã thêm vào `.gitignore`) để lần chạy lại từ
  cache vẫn dựng đúng prompt coordinator.

---

## 9. Bước 6 — Kéo kết quả về máy local

```bash
# ---- trên GPU SERVER ----
tar -czf real_vllm_out.tgz outputs/real_vllm_* outputs/measure_* outputs/cache logs
ls -la outputs/real_vllm_llama
```

```powershell
# ---- trên MÁY LOCAL ----
scp "${GPU_HOST}:${GPU_DIR}/real_vllm_out.tgz" .
tar -xzf real_vllm_out.tgz -C .
```

Kiểm tra provenance trước khi tin bất kỳ con số nào:

```powershell
Get-ChildItem outputs\real_vllm_*\evaluation_provenance.json | ForEach-Object { $_.FullName; Get-Content $_ }
```

Phải thấy tối thiểu:

- `"data_source": "benchmark:<tên bộ>"` (không phải `"synthetic"`)
- `"platform"` = Linux của GPU server (bằng chứng run diễn ra trên máy GPU, không phải laptop)
- `"is_paper_result": false` — chỉ set `true` sau khi cập nhật `audits/result_integrity_audit.md`

---

## 10. Bước 7 — Sau khi có kết quả thật (bắt buộc, không được bỏ)

1. Cập nhật `audits/result_integrity_audit.md` (hiện vẫn ghi "No real experiments were run" —
   sẽ sai sau run này).
2. Vẽ hình với `--real` **chỉ khi** inputs đã xác minh:
   `python scripts/make_plots.py --input outputs/real_vllm_llama/evaluation_sweep.csv --output outputs/real_vllm_llama/asr_vs_f.png --real`
   (lặp cho `<tag>` = `llama`, `qwen`, `mistral`)
3. Ghi rõ backbone + revision + decoding params (paper checklist, `experiments.tex:100-106`):
   `model_revision`, `temperature`, `max_tokens` đã được ghi **tự động** vào mọi
   `*_provenance.json` (§8.6) — chỉ cần đảm bảo bạn đã ghim revision trong config và dẫn lại giá
   trị đó khi viết bài.
4. Cấu hình verdict space phải nói rõ: judge thật hiện trả **score-only** (`d = 1`, m = 0), không
   kèm rationale embedding (§8.5).
4. RQ2/RQ4: chạy `scripts/measure_empirical.py` (§8.5) cho **cả** committee homogeneous
   (`configs/real_llama.yaml`) và diverse (`configs/real_diverse.yaml`), đọc
   `empirical_committee.csv` (r, γ, ρ), `empirical_gamma.csv`, `empirical_rho_pairs.csv` (ρ theo
   từng cặp backbone) và `empirical_isolation.csv` (ε với isolation ON vs OFF). Nhớ dẫn lại các
   caveat trong `empirical_provenance.json` khi viết bài.
5. RQ5: đọc `evaluation_cost.csv`, `evaluation_cost_payloads.csv` và `ablation_cost*.csv` từ lần
   chạy **cold-cache** ở §7.1 (`max_latency_s`/`sum_latency_s` là Eq. (lat-par)/(lat-seq) đo thật).
   Nếu các file này trống thì run đó chỉ toàn cache hit — chưa có số RQ5.
6. Đầu ra đã được mở rộng (đã triển khai, xem `docs/runner_metrics_plan.md` §9–§10 và
   `docs/reproducibility.md`): `evaluation_sweep.csv` có thêm `utility_retention`,
   `escalation_rate`, `byzantine_tolerance_fraction`, Wilson CI cho ASR-UC/ORR,
   `detection_auroc`/`detection_f1`, cost **mô hình** (`*_model`) và `group_asr_json`;
   `evaluation_summary.csv` cho mean ± std theo seed. Nếu muốn cost mô hình có số, thêm section
   `cost:` vào config (mô tả trong `docs/reproducibility.md`); không có `cost:` thì các cột
   `*_model` là NaN chứ không phải 0.

---

## 11. Troubleshooting

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `RuntimeError: vllm is not installed` | Chưa `pip install -e ".[vllm]"` hoặc đang chạy ngoài venv | Kích hoạt venv, cài lại; kiểm tra `python -c "import vllm"` |
| `CUDA out of memory` | Backbone/committee lớn hơn VRAM | Đặt `judges.max_resident_engines: 1` (chạy từng backbone session); xem §8 |
| `Could not parse a score from judge output` | Model trả lời không đúng format JSON | Xem `build_judge_messages` (`llm_judge.py`); kiểm tra chat template của backbone |
| `401/403` khi load `meta-llama/*` | Chưa accept license / thiếu `HF_TOKEN` | Accept trên HF Hub, export `HF_TOKEN` |
| `Expected .../test.csv` / thiếu cột | Data chưa sync hoặc sai schema | Làm lại §4; xem `docs/data_format.md` |
| `n base ids lack a clean/attack counterpart` | Bộ pairs không khớp `_clean`/`_attack` | Kiểm `data/benchmarks/second_order/test.csv` (phải có đủ cặp); dùng `--pairs-benchmark` khác nếu cần |
| `evaluation_cost.csv` không xuất hiện | Run toàn cache hit (không có call thật) | Chạy lại với `--cache <path mới>` (§7.1) |
| `rho_decision = nan`, `pairs_defined = 0` | Committee homogeneous temperature-0: không có phương sai giữa judge | Đúng về mặt toán; đo ρ có nghĩa bằng committee diverse (`configs/real_diverse.yaml`) |
| Kết quả trùng lặp bất thường / không đổi sau khi sửa prompt | Đang đọc `verdict_cache` cũ | Dùng `verdict_cache` mới cho mỗi cấu hình/prompt mới |
| Job chết khi SSH rớt | Chạy foreground | Dùng `tmux` hoặc `nohup` (§7) |

---

## 12. Checklist ngắn

- [ ] `nvidia-smi` OK, VRAM đủ cho backbone đã chọn
- [ ] Data đã sync; `sha256sum` khớp `PROVENANCE.json`; đếm rows khớp bảng §8.4
- [ ] `pip install -e ".[dev,vllm]"` + `pytest -q` pass trên server
- [ ] Preflight 1 payload trả `score` hợp lệ; có số `latency` để ước lượng
- [ ] `tmux` đã bật; log ghi vào `logs/`
- [ ] Chạy đủ **3** stage theo đúng thứ tự, cùng một `--output`, cho **cả 3** backbone (`llama`, `qwen`, `mistral`)
- [ ] Chạy thêm **một** vòng diverse (`configs/real_diverse.yaml`, `max_resident_engines: 1`) cho RQ4
- [ ] `scripts/measure_empirical.py` trên homogeneous **và** diverse → `empirical_*.csv` (r, γ, ρ, ε)
- [ ] Cold-cache run cho RQ5 → `evaluation_cost*.csv` / `ablation_cost*.csv` có số thật
- [ ] Nối số đo vào attack khi báo RQ1/RQ2/RQ3 (§8.7: `attack_kwargs.radius` = `r` đo được, `attack_kwargs.epsilon` = ε đo được)
- [ ] Nếu báo head-to-head hệ ngoài (§8.8): ghim `checkpoint` cho SecAlign/StruQ, kiểm chứng prompt với repo tác giả, và ghi rõ rows đó là hệ **chưa bị compromise**
- [ ] `evaluation_provenance.json`: `data_source` = `benchmark:*`, `platform` = Linux GPU box
- [ ] Kéo `outputs/real_vllm_*` + `outputs/measure_*` + `outputs/cache` + `logs` về local
- [ ] Cập nhật `audits/result_integrity_audit.md`; chỉ dùng `--real` sau khi xác minh
