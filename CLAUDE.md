# Wildfire Prediction Project — 上下文

## 项目目标
预测加拿大野火起火位置，提前 **14–46 天**，基于 ERA5/S2S 气象数据 + 历史起火频率。
输出：每日全加拿大 2km 网格的起火概率图。

## 当前模型：s2s_hotspot_cwfis_v2

### 架构
- **类型**：Patch-based S2S Transformer（编码器-解码器）
- **参数量**：8,488,704（~8.5M）
- **输入通道**：8 channels（FWI/FFMC/DMC/DC/ISI/BUI + ERA5 气象）
- **预测窗口**：lead_start=14d, lead_end=46d（33天预测窗口）
- **编码器历史**：in_days=7（前7天气象）
- **Patch size**：16×16 像素

### 训练配置
- neg_ratio=20（每个正样本配 20 个背景 patch）
- Train: 2018-05-01 → 2022-04-30 | Val: 2022-05-01 → 2024-10-31
- batch_size=256, lr=0.0001, optimizer=Adam

### 当前最优结果（Windows, Epoch 1）
- **Lift@5000 = 19.09x**（top-5000 像素火点密度是随机的 19 倍）
- **prec@5000 = 88%**（top-5000 预测中 88% 是真实火点）
- val_loss=0.1222, train_loss=0.1272
- Checkpoint: `checkpoints/s2s_hotspot_cwfis_v2/best_model.pt`

### 评估指标说明
```
Lift@K = Precision@K / baseline_fire_rate
baseline = total_fire_pixels / total_valid_pixels
```
Val 评估：随机抽样 20 个 val window（seed=0），快速估算（10-30秒）。

---

## 数据

### 空间参考
- **CRS：EPSG:3978**（加拿大 Lambert 等面积投影）
- **Grid：2709×2281 像素**，分辨率约 2km
- 所有 raster 数据必须对齐到此网格

### 数据源一览
| 数据 | 来源 | 格式 | 时间范围 | 存储路径 |
|------|------|------|---------|---------|
| FWI/FFMC/DMC/DC/ISI/BUI | Copernicus CDS (cems-fire-historical-v1) | GeoTIFF | 2018-2024 | `data/{fwi,ffmc,dmc,dc,isi,bui}_data/` |
| ERA5 气象观测 | Copernicus CDS (reanalysis-era5-single-levels) | GRIB | 2019-2024 | `data/era5_on_fwi_grid/` |
| 热点 CSV | CWFIS | CSV | 2018-2025 | `data/hotspot/hotspot_2018_2025.csv` |
| 火灾气候学 | 由热点+FWI生成 | GeoTIFF | 静态 | `data/fire_climatology.tif` |
| CWFIS FWI 参考文件 | CWFIS WCS | GeoTIFF | 单日 | `data/fwi_data/fwi_20250615.tif` |

### 下载脚本
```bash
# FWI 6分量（CDS下载 + EPSG:3978 reproject）
python -m src.data_ops.download.fwi_historical \
  --start 2018 --end 2024 --months 4 5 6 7 8 9 10 \
  --reference data/fwi_data/fwi_20250615.tif
  # --convert-only  # 在计算节点用（无外网），NC文件已下好时加此flag

# ERA5 观测（GRIB，nohup后台运行）
cd $SCRATCH/wildfire-refactored
nohup python -m src.data_ops.download.download_ecmwf_reanalysis_observations \
  2019-04-01 2024-10-31 --workers 2 > /scratch/jiaqi217/ecmwf_download.log 2>&1 &

# 热点 CSV
python -m src.data_ops.download.download_hotspot_csv

# fire_climatology（需要先有 EPSG:3978 的 FWI 参考文件）
python -m src.data_ops.processing.make_fire_climatology \
  --reference data/fwi_data/fwi_20250615.tif
```

---

## 基础设施

### Trillium HPC（主训练集群）
- **GPU 训练节点**：从 `trig-login01` 提交 SLURM job
- **CPU/下载节点**：`tri-login03`（有外网，可下载 CDS 数据）
- **计算节点**：无外网访问权限（不能连 CDS API）
- **存储**：`/scratch/jiaqi217/wildfire-refactored/`
- **Conda 环境**：`$SCRATCH/miniforge3/envs/wildfore-r/`（不能用 conda activate，用绝对路径）
- **Config**：`configs/paths_trillium.yaml`

### SLURM 关键语法（Trillium）
```bash
#SBATCH --gpus-per-node=1    # 不是 --gres=gpu:1
# 无 --mem 标志（自动分配 ~745 GB）
```

### 提交训练
```bash
# 必须从 trig-login01 提交 GPU job
sbatch slurm/train_v2.sh
squeue -u jiaqi217
```

---

## 已知问题 / TODO

### 模型
- **空间自相关未解决**：相邻 patch 正负样本高度相关，无 fire cluster 去重
- **时间 CV 是单一切割点**（2022-05-01），非 Leave-One-Year-Out（LOYO），评估偏乐观
- **Epoch 2 过拟合**：模型在 epoch 1 后开始记忆训练集，需要正则化（dropout / weight decay）
- **GPU 利用率低**（仅 9% VRAM）：训练是 IO-bound，数据加载是瓶颈

### 数据
- FWI 历史数据需要 reproject（CDS 下载为 WGS84，训练需要 EPSG:3978）
- ERA5 grib 文件需后续转换为训练用格式
- fire_climatology.tif 需在 EPSG:3978 参考文件存在后重新生成

---

## 关键文件路径
```
src/training/train_s2s_hotspot_cwfis_v2.py   # 主训练脚本
src/data_ops/download/fwi_historical.py       # FWI 下载+reproject（--convert-only 用于计算节点）
src/data_ops/download/download_ecmwf_reanalysis_observations.py  # ERA5 下载 → data/era5_on_fwi_grid/
src/data_ops/processing/make_fire_climatology.py  # 生成 fire_climatology.tif
configs/paths_trillium.yaml                   # Trillium 路径配置
slurm/train_v2.sh                            # SLURM 训练提交脚本
```
