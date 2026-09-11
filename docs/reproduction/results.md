# MambaPose ICME 2025 復現報告

## 結論

本次受控實驗 campaign 已完成 **9 個 300-epoch 訓練與 2 個 COCO test-dev 匯出（11/11）**。所有 completion artifacts、config、data、environment 與 Git provenance 均通過 hash 驗證；9 個推論 checkpoints 可載入，9 個 `epoch_300.pth` 也都含完整 optimizer training state、可供續訓。

科學結果必須分成兩層解讀：

- **主結果重現：通過。** COCO 與 CrowdPose 五個主模型的 AP 均落在論文值 ±0.18 AP 內。
- **消融結論：部分重現。** Prior 與 cycling 的增益方向重現；COCO 與 CrowdPose 的 PIF 增益方向沒有重現。本地結果中 no-PIF 反而分別高出 full model 0.244 與 0.068 AP。
- **COCO test-dev：artifact 完成、線上分數未驗證。** 兩份 submission JSON 已完整匯出並驗證，但未提交到需登入的 CodaLab；因此不宣稱本地得到論文的 72.4/73.5 test-dev AP。

所以，本報告證明實驗矩陣與主模型數值已復現，但不宣稱論文的所有消融主張都被重現。`final-verification.json` 的總體 `valid=false` 正是由兩個 PIF 方向偏差造成；其中 11 個 individual run 均為 `valid`，不是 checkpoint 或資料損毀。

## 取得 checkpoints

9 個 best inference checkpoints 與對應的 9 個 `epoch_300.pth` 續訓 checkpoints 已發布為 [`mambapose-icme2025-reproduction-v1`](https://github.com/vic9112/MambaPose/releases/tag/mambapose-icme2025-reproduction-v1) GitHub Release assets。完整 asset 名稱、下載 URL、檔案大小、epoch/iteration、AP、resolved-config SHA256 與 checkpoint SHA256 記錄在 [`reproduction/checkpoints.json`](../../reproduction/checkpoints.json)。best checkpoints 是刻意移除 optimizer state 的評估／推論格式；檔名含 `resume` 的資產則包含 `state_dict`、optimizer、兩組 schedulers、message hub 與訓練進度。

例如下載並驗證 COCO S-V1 續訓檔：

```bash
mkdir -p work_dirs/continuations/coco-s-v1
gh release download mambapose-icme2025-reproduction-v1 \
  --pattern mambapose-coco-s-v1-resume-epoch300.pth \
  --dir work_dirs/continuations/coco-s-v1
echo '61d543df733ec08c74bd295b6f5d0b8b6da88db9587411cdc2ab31f41de472e1  work_dirs/continuations/coco-s-v1/mambapose-coco-s-v1-resume-epoch300.pth' \
  | sha256sum --check
```

原 config 的 `train_cfg.max_epochs=300`，因此 epoch 300 已是論文 schedule 終點。要再訓練，必須明確開一個新的 continuation（例如加上 `--cfg-options train_cfg.max_epochs=330`）；這種延長訓練不可回報成原論文復現結果。若只需要評估或部署，應下載較小的 `best` asset。

## 主模型結果

AP 數值以百分點表示；delta = measured − paper。

| Dataset / split | Variant | Paper AP | Measured AP | Delta AP | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| COCO val2017 | S-V1 | 72.800 | 72.832 | +0.032 | 重現 |
| COCO val2017 | S-V2 | 74.200 | 74.212 | +0.012 | 重現 |
| COCO val2017 | B | 75.000 | 74.895 | -0.105 | 重現 |
| CrowdPose test | S-V1 | 65.600 | 65.422 | -0.178 | 重現 |
| CrowdPose test | S-V2 | 67.000 | 67.060 | +0.060 | 重現 |

其他論文指標也使用相同 evaluator 與 person-detection inputs 重算：

| Variant | Metric | Paper | Measured | Delta |
| --- | --- | ---: | ---: | ---: |
| COCO S-V1 | AP50 / AP75 / APM / APL / AR | 89.7 / 80.5 / 69.4 / 79.2 / 78.2 | 89.758 / 80.699 / 69.346 / 79.555 / 78.380 | +0.058 / +0.199 / -0.054 / +0.355 / +0.180 |
| COCO S-V2 | AP50 / AP75 / APM / APL / AR | 90.5 / 82.0 / 70.9 / 80.6 / 79.6 | 90.324 / 81.632 / 70.753 / 80.791 / 79.602 | -0.176 / -0.368 / -0.147 / +0.191 / +0.002 |
| COCO B | AP50 / AP75 / APM / APL / AR | 90.5 / 82.7 / 71.3 / 81.5 / 80.1 | 90.588 / 82.668 / 71.562 / 81.243 / 80.113 | +0.088 / -0.032 / +0.262 / -0.257 / +0.013 |
| CrowdPose S-V1 | AR | 75.2 | 75.135 | -0.065 |
| CrowdPose S-V2 | AR | 77.0 | 76.681 | -0.319 |

計算量使用同一份 config、batch 1、輸入 256×192，在 RTX 5090 上以 `get_flops.py --device cuda:0` 量測：

| Variant | Paper GFLOPs | Local traced GFLOPs | Params |
| --- | ---: | ---: | ---: |
| S-V1 | 2.8 | 2.738 | 22.029 M |
| S-V2 | 4.0 | 3.856 | 32.189 M |
| B | 5.2 | 4.984 | 36.204 M |

本地 counter 明確列出未支援的 `CrossScanTriton`、`SelectiveScanOflex`、`CrossMergeTriton`、`MambaInnerFn` 與部分 elementwise ops，因此 local traced GFLOPs 是不含這些 custom ops 的下界，不應視為比論文值更精確的總 FLOPs。MambaPose 的 Triton selective-scan 必須用 CUDA tensor，不能以 CPU-only trace 取代。

## 消融結果

下表的 effect = matched full − ablation；正值表示該元件提升 AP。

| Comparison | Paper ablation AP | Measured ablation AP | Paper effect | Measured effect | 方向 |
| --- | ---: | ---: | ---: | ---: | --- |
| COCO PIF | 72.600 | 73.076 | +0.200 | **-0.244** | 未重現 |
| CrowdPose PIF | 65.300 | 65.490 | +0.300 | **-0.068** | 未重現 |
| CrowdPose prior | 65.350 | 65.197 | +0.290 | **+0.225** | 重現 |
| CrowdPose cycling | 65.490 | 65.374 | +0.150 | **+0.048** | 重現 |

每個 ablation checkpoint、metrics 與 provenance 本身均有效，且個別 AP 與論文值差距不超過 0.477 AP。方向偏差源自 matched full/ablation 的相對排序，不應用挑 seed 或改寫結果掩飾。論文未提供多 seed 的平均值與變異，因此本次單次正式 run 不能判定差異是隨機變異或方法效應。

## COCO test-dev submissions

| Variant | Paper AP | Local status | Prediction rows | Image IDs | Submission SHA256 |
| --- | ---: | --- | ---: | ---: | --- |
| S-V1 | 72.4 | submission-only | 269,644 | 15,988 | `82fe1d983873839c4e490f287f6879ef2931793adf8c9f369559044ef561b413` |
| S-V2 | 73.5 | submission-only | 268,627 | 15,988 | `109854fbda21ee1e2abdeb8da276233a7e56bfa9a5a85027a33b1d42d9242238` |

兩個 JSON 的 ID set 精確覆蓋 detector input 的 15,988 個 image IDs；每列均有合法的 `image_id`、`category_id=1`、51 個 finite keypoint values、finite score 與 bbox。COCO test-dev annotations 有 20,288 個 images，其中 4,300 個沒有 detector bbox，故 top-down pipeline 不會產生 prediction。`submission-metrics.json` 保持空物件，沒有虛構本地 AP。

## Checkpoints 與 artifacts

表中 hash 為 SHA256；best checkpoint 是評估／推論格式，`epoch_300` 是含 optimizer state 的續訓格式。

| Run | Resolved config | Best checkpoint / epoch | Metrics | `epoch_300` resumable |
| --- | --- | --- | --- | --- |
| coco-s-v1 | `56d8fc12…1797` | `a6f76dae…7cdd2` / 300 | `2ef148c3…509b` | `61d543df…72e1` |
| coco-s-v2 | `80e64513…015c` | `c93e4362…18179` / 300 | `11faf100…a3d2` | `01a167d6…3bc7` |
| coco-b | `9bf26e44…c3ba` | `38b5e5b1…b9b2` / 290 | `67efe641…fe63` | `79b3b607…246c` |
| crowdpose-s-v1 | `b2ce742e…824b` | `9a71cd2e…e6f6` / 290 | `39e9c81c…8d74` | `c8fefb7f…c707` |
| crowdpose-s-v2 | `a541f3ae…e635` | `8be5badc…2322` / 260 | `8c7c037d…7588` | `04bc3cf2…657f` |
| coco-s-v1-no-pif | `b0d34ed1…0508` | `28cd0240…a5fb` / 300 | `95663c07…b806` | `6447eb62…4abc` |
| crowdpose-s-v1-no-pif | `2bfa6cf1…9062` | `8408255c…a298` / 290 | `5c39d20c…3d7e` | `8522406b…f01c` |
| crowdpose-s-v1-no-prior | `c73c37eb…621` | `da728b7b…d8c3` / 300 | `e21c85b8…242b` | `23f622d1…4161` |
| crowdpose-s-v1-no-cycling | `83694a28…f48b` | `e9d6eb2c…a9fe` / 290 | `0413d002…3fe` | `6a2ac535…52b1` |

完整 64-hex hashes、每個 artifact 的相對路徑與所有 evaluator metrics 位於 `work_dirs/reproduction/evidence/final-verification.json` 及各 run 的 `completion.json`。9/9 best checkpoints 已用 `require_training_state=False` 驗證；9/9 `epoch_300.pth` 已用 `require_training_state=True` 驗證。best checkpoints 不含 optimizer state 是預期格式差異。

## Reproduction provenance

- Training source commit: `1f4364d62279bf9fe3ac55e0e2d57339036aeb81`。
- Publication lineage: 本文件位於 training source 的 docs-only descendant commit；所有 completion 仍刻意綁定上述 training commit，沒有把 publication commit 偽裝成訓練來源。
- Paper PDF SHA256: `5bb5baba66a21e7c80aa7f99ddfa18bc4b10ff816f6ed637c8b996a27efe86b5`。
- Data inventory SHA256: `2a82ab3cfe05a514d141c17921463e72f22741d67eaf978d93317155bbeaf0ed`。
- Environment bundle SHA256: `691be7548cb7b2591f3c534bdef04ce235b7e8f9fefe1f00eb053f869810f8a0`。
- Conda explicit lock SHA256: `025219d3d85cc793c04165b966e5d4ce412eeffe0d9d3cd8d0525fe8a030c7d3`。
- Resolved-config inventory SHA256: `1c3c54bbd8ae4bb2633dec7aee04e16be109cf90efcdf5a7df9681b4ef58f381`。
- Final preflight SHA256: `6112b407d72117d61a30259d40ed5a2ba320259229c4ae9f597a776216ab472e`。
- Runtime: Python 3.11.15, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, CUDA 12.8, RTX 5090 (SM 12.0), driver 595.71.05。
- Training schedule: Adam, base LR 1e-3, 300 epochs, milestones 200/260；所有正式 batch、depth、PIF 與 evaluator settings 均由 resolved configs 鎖定。

11/11 `provenance.json` 與 `completion.json.provenance` 的 repo/data/environment/config hashes 均一致；completion 宣告的 artifact SHA 也與實檔 fresh hash 相同。campaign 從 2026-08-18 15:44:27 到 2026-08-26 03:00:09（Asia/Taipei），state generation 29，events generation 1–29 連續且 append-only。

## 背景執行與恢復紀錄

正式 campaign 由 user-level systemd service 執行，observer timer 持續提供 heartbeat、GPU telemetry 與磁碟監控；`Linger=yes` 使其不依賴 SSH/Codex session。獨立監控程序另外核對 GPU compute owners 與 `/dev/shm`。終局 service 為 `inactive/dead`、`Result=success`、`ExecMainStatus=0`、`NRestarts=0`，observer 顯示 `health=complete`，獨立終局檢查則確認 GPU 無 compute owner 且 `/dev/shm` 無 `torch_*` 殘留。

S-V2 曾遇到兩次 DataLoader AF_UNIX file-descriptor transport 的 `received 0 items of ancdata`。經 Torch 官方原始碼建議切換到 `file_system` sharing strategy，從有效 epoch-5 checkpoint 恢復。第一次正式 attempt-3 啟動又遭另一個不屬於本 campaign 的 GPU process 占用 17.43 GiB，第一個 forward 即 OOM；證據顯示尚未發生 optimizer step。一次性、hash-pinned operational replay 保留原 events、attempt lineage 與 checkpoint hash，再由 epoch 5 完整跑到 epoch 300。事故 manifest 與 replay authorization 保存在 `work_dirs/reproduction/evidence/incidents/`，沒有改寫舊 state/events 或訓練成果。campaign terminal 且 artifacts 獨立驗證後，已依 manifest 只移除臨時 `ancdata-file-system.conf` 並執行 daemon-reload；incident evidence 保留，installed base unit 與 repo source byte-identical。

## 驗證與限制

- `run_campaign.py --status --require-complete`：exit 0，state 為 11/11 complete。
- `run_campaign.py --run` terminal no-op revalidation：exit 0；在 training HEAD 上 fresh 重算 repo/data/environment/config provenance 與 artifact hashes，11/11 `CampaignExecutor.completion_valid` 通過，沒有啟動任何新 run。
- `pytest tests/test_reproduction -q`：122 passed。
- Result validator：11/11 individual artifacts valid；aggregate exit 1，唯一原因是上述兩個 PIF direction checks。
- Main AP reproduction tolerance：本報告沿用 validator 的 ±0.5 AP direct-reproduction threshold；五個主結果全部通過。
- FLOPs：三個 CUDA traces 均 exit 0；結果如主表所列，必須連同 unsupported-op 警告解讀為下界。
- Test-dev：沒有 CodaLab credentials，因此只能交付可提交 artifacts，不能本地驗證 hidden-label AP。
- Paper/config metadata：manifest 的 paper title 字串與 PDF 真正標題不同；部分 ablation/test-dev config 的巢狀 `paper_target.metrics` 因 base-config merge 保留主模型欄位。本報告的比較數值直接以 SHA-pinned PDF tables 與正確的 `paper_target.value` 重算，沒有採信殘留欄位。
- 論文沒有提供作者 pose checkpoints、訓練 seed 的多次統計或變異區間；因此本次是由上游 VMamba pretrained backbone 開始的獨立單次訓練復現，不能用來主張統計等價。
