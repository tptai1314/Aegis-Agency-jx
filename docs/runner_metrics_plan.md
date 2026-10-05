# Kế hoạch mở rộng runner metrics (CHỜ DUYỆT)

> **TRẠNG THÁI: ĐÃ TRIỂN KHAI Mức 1 + Mức 2** (được duyệt cùng lựa chọn `seeds 0..4`, mean ± std
> + Wilson CI, không làm significance test). Phần §9 "Cần bạn duyệt" bên dưới được giữ lại làm
> hồ sơ quyết định; kết quả thực tế xem §10.

## 1. Vấn đề: metric có trong code nhưng không bao giờ tới file kết quả

`outputs/*/evaluation_sweep.csv` hiện chỉ có 6 cột: `f, n_judges, method, asr_uc, orr,
defense_success_rate`. Trong khi đó:

| Metric (paper) | Hàm đã có | Trạng thái thực tế | Bằng chứng |
|---|---|---|---|
| ASR-UC (Eq. 4) | `metrics.asr_under_compromise` | **có ghi ra** | `harness.py:273,309` |
| ORR (Eq. 5) | `metrics.over_refusal_rate` | **có ghi ra** | `harness.py:274,310` |
| Defense success rate | `metrics.defense_success_rate` | **có ghi ra** | `harness.py:275` |
| Detection F1 / AUROC | `metrics.malicious_verdict_detection` | **tính rồi bị bỏ** | tính ở `harness.py:278-281`, `sweep_colluding_fraction:301-311` chỉ copy 3 key |
| Utility retention (RQ6) | `metrics.utility_retention` | **không ai gọi** | chỉ export ở `metrics/__init__.py:29` |
| Group-conditional ASR | `metrics.group_conditional_asr` | **không ai gọi** | chỉ export ở `metrics/__init__.py:30` |
| Byzantine tolerance fraction | `metrics.byzantine_tolerance_fraction` | **không ai gọi** | `metrics.py:102` |
| Wilson / bootstrap CI | `metrics.confidence_intervals` | **không ai gọi** | `confidence_intervals.py:14,25` |
| Token cost, latency (RQ5) | `metrics.cost` | **không ai gọi** | `cost.py:12,22,29` |

⇒ RQ2 (detection), RQ5 (cost), RQ6 (utility) và yêu cầu "mean ± std over seeds"
(`AegisAgency_manuscript/sections/experiments.tex:100-106`) **không sinh ra artifact nào**, dù
toàn bộ công thức đã được cài và test. Đây là khoảng trống của tầng chạy, không phải thiếu math.

## 2. Nguyên tắc bất di bất dịch khi sửa

1. **Chỉ thêm, không đổi/không xoá.** Schema và ngữ nghĩa các cột cũ của `evaluation_sweep.csv`
   phải giữ nguyên từng bit.
2. **Không thêm pipeline/baseline mới** — `tests/test_real_data.py:72` assert *đúng*
   `set(out["methods"])`.
3. **Không đổi** `metrics/metrics.py`, `metrics/theory.py`, `methods/aggregators.py`,
   `methods/gate.py`, `attacks/*`, `baselines/*`. Chỉ *gọi thêm* các hàm đã có.
4. **Không đổi** thứ tự/ý nghĩa 3 stage `calibrate → evaluate → ablate`, không đổi config hiện có.
5. Nhánh `synthetic` giữ nguyên hành vi; mọi cột mới mặc định `None`/`nan` khi không tính được
   (không bao giờ bịa số).

## 3. Mức 1 — chỉ thêm cột + file summary (không đụng judge, chạy được cả trên local)

1. `experiments/harness.py` — `run_trial()` bổ sung vào `metrics`:
   - `utility_retention` (gọi hàm đã có, `escalate_as="block"` như mặc định hiện tại);
   - `escalation_rate` = tỉ lệ `DecisionResult.mode is GateMode.ESCALATE` (đếm từ `results`);
   - `byzantine_tolerance_fraction(cfg.n_judges, rule)` cho các method `aegis_*`;
   - `asr_uc_ci_low/high`, `orr_ci_low/high` bằng `wilson_interval` (đếm `(k, n)` cục bộ trong
     harness, không sửa `metrics.py`);
   - `group_asr_json` = `json.dumps(group_conditional_asr(...))` khi có >1 group.
2. `experiments/harness.py` — `sweep_colluding_fraction()` **passthrough** mọi key của `metrics`
   vào row, gồm `detection_auroc`, `detection_f1` (hiện đang bị bỏ ở `harness.py:301-311`).
3. `experiments/run_evaluation.py` — thêm `seeds: Sequence[int] | None = None`:
   - mặc định `None` ⇒ hành vi hiện tại y như cũ (1 seed);
   - `--seeds 0,1,2,3,4` ⇒ CSV có thêm cột `seed`, mỗi (seed, f, method) một dòng;
   - ghi thêm **file mới** `evaluation_summary.csv`: mean ± std (và CI) theo `(f, method)`;
   - `evaluation_sweep.csv` **không** đổi tên/schema.
4. `experiments/run_ablation.py` — passthrough các key mới (giữ nguyên 3 cột hiện có).
5. Cost **mô hình** (nhãn rõ là model-based, không phải đo): thêm mục config
   `cost: {l_in, l_out, analyze_tokens, judge_latency_model, agg_latency_model}` → cột
   `tokens_model`, `latency_parallel_model`, `latency_sequential_model` cho mỗi row.
6. `cli.py` — map section `cost:` và cờ `--seeds` (chỉ thêm key, không đổi key cũ).
7. Test mới `tests/test_metrics_export.py`: khẳng định (a) CSV sau khi sửa vẫn chứa đủ 6 cột cũ,
   (b) các cột mới có mặt, (c) chạy 1 seed cho ra `asr_uc` **giống hệt** giá trị tính tay từ
   `run_trial`.

## 4. Mức 2 — đo latency/token THẬT cho RQ5 (đụng judge, cần duyệt riêng)

1. `judges/llm_judge.py` — `_score()` bọc `time.perf_counter()` và lấy token usage từ output của
   vLLM; trả kèm metadata (latency, prompt_tokens, completion_tokens).
2. `utils/cache.py` — JSONL thêm field `latency_s`, `prompt_tokens`, `completion_tokens`.
   **Tương thích ngược**: `_load()` (`cache.py:34-44`) chỉ đọc `model/payload_id/judge_id/score/decision`,
   nên cache cũ vẫn đọc được; verdict cũ thiếu field ⇒ cột `nan` + cảnh báo, **không suy diễn**.
3. `harness.run_trial()` — cộng dồn thành `latency_p50_measured`, `latency_p95_measured`,
   `tokens_measured` theo từng judge (committee dùng chung payload, nên đo theo judge chứ không
   theo pipeline; báo cáo rõ trong provenance).
4. Cảnh báo vận hành: cache cũ (`outputs/cache/*.jsonl`) **không có** latency ⇒ muốn có RQ5 thật
   phải chạy lại với cache mới, hoặc chấp nhận cột `nan`.

## 5. File dự kiến chạm (khi được duyệt)

| File | Mức | Loại sửa |
|---|---|---|
| `src/aegis_agency/experiments/harness.py` | 1 (+2) | thêm key metric, passthrough |
| `src/aegis_agency/experiments/run_evaluation.py` | 1 | `seeds` + `evaluation_summary.csv` |
| `src/aegis_agency/experiments/run_ablation.py` | 1 | passthrough |
| `src/aegis_agency/cli.py` | 1 | map `cost:` + `--seeds` |
| `src/aegis_agency/judges/llm_judge.py` | 2 | đo latency/token |
| `src/aegis_agency/utils/cache.py` | 2 | thêm field JSONL |
| `tests/test_metrics_export.py` | 1 | test mới (không sửa test cũ) |
| `docs/reproducibility.md`, `audits/result_integrity_audit.md` | 1–2 | mô tả cột mới / provenance |

**Không chạm:** `metrics/*`, `methods/*`, `attacks/*`, `baselines/*`, `data/*`, `judges/synthetic_judges.py`.

## 6. Tiêu chí không hồi quy (bắt buộc trước khi merge)

1. `pytest -q` xanh toàn bộ.
2. Chạy `evaluate` trên **cùng seed, cùng config** trước và sau khi sửa → 6 cột cũ của
   `evaluation_sweep.csv` phải **giống hệt** (so sánh từng ô).
3. `plot_evaluation_sweep` vẫn chạy với CSV mới (nó chỉ đọc `method/f/asr_uc`,
   `plot_results.py:37`) và ảnh không đổi.

## 7. Còn nợ sau kế hoạch này (không thuộc phạm vi)

- Đo **ε** (isolation leakage), **r** (honest radius), **γ** (margin), **ρ** (correlation) từ judge
  thật cho RQ2/RQ4 — hiện chỉ có mô hình tham số (`judges/isolation.py`,
  `judges/synthetic_judges.py`), **chưa có script đo**. Cần một kế hoạch riêng.
- **RQ4 diverse committee** trong một lần chạy: xem `docs/gpu_server_runbook.md` §8.5.
- Revision/checkpoint của backbone chưa được capture vào provenance (paper checklist
  `experiments.tex:100-106` yêu cầu) — cần quyết ghi tay vào `audits/` hay thêm field.
- Real `JudgeDeceiverAdapter.inject()` cho RQ2 vẫn là stub (`baselines/external_wrappers.py:86-98`).
- Significance test (bootstrap paired) giữa các method — chưa có; cần bạn chốt có làm hay không.

## 8. Quyết định đã chốt

- [x] **Mức 1** — duyệt
- [x] **Mức 2** — duyệt
- [x] Seed: **0, 1, 2, 3, 4** (CLI nhận danh sách tuỳ ý)
- [x] Thống kê: **mean ± std + Wilson CI**, **không** làm significance test

## 9. Kết quả triển khai

### Đã làm

| Hạng mục | File |
|---|---|
| Cột metric mới + helper tính CI/count, cost mô hình, builder row, `summarize_sweep_rows` | `src/aegis_agency/experiments/harness.py` |
| `seeds`, `evaluation_summary.csv`, `evaluation_cost.csv`, provenance ghi `seeds` + `cost_model` | `src/aegis_agency/experiments/run_evaluation.py` |
| Passthrough metric mới cho ablation | `src/aegis_agency/experiments/run_ablation.py` |
| Map section `cost:`, cờ `--seeds` | `src/aegis_agency/cli.py` |
| `--seeds` cho stage evaluate | `scripts/run_experiment.py` |
| Đo `latency_s` / `prompt_tokens` / `completion_tokens` mỗi call + `collect_call_stats()` | `src/aegis_agency/judges/llm_judge.py` |
| JSONL cache thêm 3 field đo (tuỳ chọn, tương thích ngược) + `get_stats()` | `src/aegis_agency/utils/cache.py` |
| 15 test mới (schema, CI khớp metric công khai, seed sweep, cache tương thích ngược, strict JSON) | `tests/test_metrics_export.py` |
| Mô tả cột/file mới | `docs/reproducibility.md` |

### Khác so với kế hoạch ban đầu (có chủ ý, không rút gọn)

1. Cột `seed` được **thêm ở cuối** `evaluation_sweep.csv` để 6 cột lịch sử giữ nguyên đúng vị
   trí; mọi cột mới khác cũng chỉ được **append**.
2. `evaluation_summary.csv` được ghi **luôn** (kể cả 1 seed, khi đó `*_std` là NaN) thay vì chỉ
   ghi khi sweep nhiều seed — để schema file ổn định.
3. Cost **đo được** đi vào file riêng `evaluation_cost.csv` theo `(seed, f)` thay vì thành cột
   per-method: một lần chạy committee dùng chung payload cho mọi method, nên gán latency đo được
   cho từng method sẽ là bịa. Cost **mô hình** thì vẫn là cột per-method (`*_model`).
4. `group_asr_json` được chuẩn hoá non-finite → `null` để JSON hợp lệ nghiêm ngặt (test có kiểm
   bằng `parse_constant`).
5. `cost:` chưa được thêm vào file config nào (không nằm trong phạm vi đã duyệt); muốn dùng thì
   thêm section `cost:` như mô tả trong `docs/reproducibility.md`.

### Bằng chứng kiểm chứng

- `pytest -q` → **91 passed** (trước: 76); `ruff check .` và `mypy src` đều sạch.
- **Không hồi quy:** tái tạo đúng lần chạy đã có trong `outputs/real_formal/` (payload
  `data/benchmarks/formal`, seed 7, n=7, rules cmed/gmed/krum, attack compromise, f=0..3) và so
  từng ô: **28/28 dòng, cả 6 cột lịch sử giống hệt từng ký tự**; 14 cột mới được thêm.
- Tích hợp CLI: `scripts/run_experiment.py --stage evaluate --seeds 0,1,2` sinh
  `evaluation_sweep.csv` (3 seed × 3 f × 6 method = 54 dòng), `evaluation_summary.csv` (18 dòng),
  provenance có `seeds` + `cost_model`, và **không** sinh `evaluation_cost.csv` (backend synthetic
  không đo gì → không bịa file).

### Vẫn còn nợ (không thuộc phạm vi đã duyệt)

- Đo ε / r / γ / ρ từ judge thật cho RQ2/RQ4 (chưa có script).
- RQ4 diverse committee trong một lần chạy (xem `docs/gpu_server_runbook.md` §8.5).
- Significance test giữa các method (đã chốt là chưa làm).
- RQ5 chỉ có số đo sau khi chạy thật với cache trắng; cache cũ không chứa latency nên cột sẽ trống.

## 10. Giai đoạn 2 — RQ2/RQ4/RQ5 (đã triển khai, chờ chạy GPU)

Bốn quyết định đã được duyệt: (A) RQ4 diverse bằng round-robin + session từng backbone có giải
phóng VRAM; (B) ε báo cả hai estimator + ablation isolation on/off; (C) r/γ/ρ báo cả profile;
(D) RQ5 đo theo sweep n = 1..7 với latency trên từng payload và cờ `--cache`.

| Hạng mục | File |
|---|---|
| Isolation ON/OFF (`NAIVE_SYSTEM_PROMPT`, `build_*` nhận `isolation`), cache namespace `<backbone>+noisolation` | `src/aegis_agency/judges/llm_judge.py` |
| Committee diverse round-robin, `backbone_groups()`, `is_diverse`, engine registry + `release_llm_engine`, cap `max_resident_engines` áp trên **mọi** đường gọi, `prefetch_honest` theo session backbone (bỏ qua engine nếu đã cache) | `src/aegis_agency/judges/llm_judge.py` |
| Stat mỗi call có `payload_id`/`judge_id`/`backbone`; `per_payload_latency()` (max = Eq. lat-par, sum = Eq. lat-seq) | `src/aegis_agency/judges/llm_judge.py` |
| `has()` cho cache | `src/aegis_agency/utils/cache.py` |
| Hook prefetch (chỉ áp cho backend thật, synthetic không đổi), `isolation`/`max_resident_engines`/`prefetch` trong `TrialConfig`, cost theo payload | `src/aegis_agency/experiments/harness.py` |
| `evaluation_cost*.csv` | `src/aegis_agency/experiments/run_evaluation.py` |
| `ablation_cost*.csv` (số đo của các trial ablation; **không** dùng làm frontier RQ5 — frontier = một `evaluate` cold-cache cho mỗi `n`, xem runbook §7.1) | `src/aegis_agency/experiments/run_ablation.py` |
| Prefetch ở stage calibrate | `src/aegis_agency/experiments/run_calibration.py` |
| Đo r/γ/ρ (RQ4) và ε với ablation isolation (RQ2) | `src/aegis_agency/experiments/measure_empirical.py` |
| Entry point đo | `scripts/measure_empirical.py` |
| Cờ `--cache` (cold cache cho RQ5) | `src/aegis_agency/cli.py`, `scripts/run_experiment.py` |
| Config diverse | `configs/real_diverse.yaml` |
| 28 test offline (round-robin, session/residency, isolation prompt + namespace, estimator, e2e) | `tests/test_empirical_measurement.py` |

### Sửa lỗi phát hiện khi kiểm tra sẵn sàng (cùng lượt)

| Lỗi | Ảnh hưởng | Sửa |
|---|---|---|
| `CsvBenchmarkAdapter` mở CSV thiếu `newline=""` | `\r\n` trong field nhúng bị đổi thành `\n`: `second_order` **2000/2000** dòng (chính là bộ đo ε), `benign` 194/299 | `data/adapters.py` + test hồi quy; đã chứng minh **8/8** bộ khớp byte, `formal` không đổi nên baseline không hồi quy |
| ε nhân bản theo seed | `--seeds 0..4` ghi 5 hàng ε giống hệt (tập cặp không phụ thuộc seed) | đo 1 lần/mode, provenance ghi `isolation_seed` + note |
| Frontier RQ5 lấy sai nguồn | `ablation_cost.csv` bị nhiễu cache (hàng `committee_n*` chạy sau các hàng `n=7`) | docs: frontier = một `evaluate` cold-cache cho mỗi `n` (runbook §7.1) |
| Provenance thiếu decoding params | checklist paper yêu cầu | `temperature`/`max_tokens`/`model_revision` ghi tự động; engine key `backbone@revision` |
| Ablation dùng proxy synthetic trên backend thật | `diverse/homogeneous_backbones` (rho) và `hardened_on/off` (r) vô tác dụng với judge thật ⇒ các hàng đó là cùng cấu hình chạy lại, dễ bị đọc thành kết luận RQ4 sai | cảnh báo trong runbook §8.3 + `docs/reproducibility.md`; RQ4 thật = so config homogeneous vs diverse + `empirical_rho_pairs.csv`. **Patch backend-aware cho `run_ablation` chưa làm** (cần duyệt) |

**Bằng chứng:** `pytest -q` → **120 passed**; `ruff`/`mypy` sạch; kiểm chứng không hồi quy trên
`outputs/real_formal` vẫn **28/28 dòng, 6 cột lịch sử giống hệt**; test dùng engine vLLM giả chứng
minh đúng thứ tự session (a=9, b=6, c=6 call cho 7 judge × 3 payload), cap `max_resident_engines`
giữ ≤ 1 engine, lần chạy thứ hai **không nạp engine nào**, và decoding params/revision tới đúng
engine + `SamplingParams`.

**Còn lại (thuộc vận hành GPU, không phải code):** chạy `scripts/measure_empirical.py` và vòng
cold-cache trên GPU server (xem `docs/gpu_server_runbook.md` §7.1, §8.3, §8.5) rồi cập nhật
`audits/result_integrity_audit.md`.

## 11. Giai đoạn 3 — đóng ba khoảng trống còn lại

| Khoảng trống | Cách xử lý | File |
|---|---|---|
| ε chỉ là **một** hiện thực tấn công | ε đo trên **họ 4 biến thể** injection (`judgedeceiver`, `naive_override`, `roleplay_takeover`, `delimiter_escape`), mỗi biến thể một id payload ⇒ một cache entry riêng; thêm `empirical_isolation_summary.csv` ghi **max theo biến thể** (đại diện gần nhất cho `sup` của Def 1) + cờ `--variants` | `src/aegis_agency/experiments/measure_empirical.py`, `scripts/measure_empirical.py` |
| `adaptive` chỉ là surrogate heuristic | Thêm `adaptive_search` (`per_rule = True`): tìm trong tập candidate verdict × mọi cách đặt `f` slot, chấm bằng **chính aggregator** của pipeline và ngưỡng đã calibrate, chọn cái đẩy aggregate xa ngưỡng nhất. Harness tự truyền `rule`/`threshold` cho từng method; `adaptive` cũ **không đổi** | `src/aegis_agency/attacks/adaptive.py`, `attacks/base.py`, `attacks/__init__.py`, `src/aegis_agency/experiments/harness.py` |
| Baseline ngoài còn là stub | Re-implementation `AutoDefenseStyleBaseline` (analyzer → judge → coordinator trên engine local, prompt trích nguyên văn từ `XHMY/AutoDefense`) + `HardenedSingleModelAdapter` cho SecAlign/StruQ (checkpoint + prompt riêng, cache namespace riêng), ghi rõ là **bản cài lại**, không phải code tác giả; **nối vào harness** qua `experiment.external_baselines` + `external_baseline_kwargs` (mặc định tắt) | `src/aegis_agency/baselines/llm_baselines.py`, `baselines/external_wrappers.py`, `src/aegis_agency/experiments/harness.py`, `tests/test_llm_baselines.py`, `tests/test_external_baseline_wiring.py` |
| Data PAIR/TAP/GPTFuzzer/universal-injection | **Không thêm**: báo cáo đúng coverage 8 bộ đang có, không claim đủ bộ (PAIR/TAP/GPTFuzz là bộ *sinh* tấn công phụ thuộc victim; universal injection là suffix phải huấn luyện) | `docs/gpu_server_runbook.md` §8.4 |

**Bằng chứng giai đoạn 3:** `adaptive_search` được kiểm chứng hành vi đúng theo Theorem 1: với
`f = 2 < n/2` và committee honest chặt, attacker **không lật được** `cmed`/`gmed`/`krum` nhưng lật
được `mean` (AutoDefense coordinator) và `single_model`; 129+ test xanh, `ruff`/`mypy` sạch, và
kiểm chứng không hồi quy vẫn giữ 6 cột lịch sử (đã chạy lại cho cả 4 attack cũ).


