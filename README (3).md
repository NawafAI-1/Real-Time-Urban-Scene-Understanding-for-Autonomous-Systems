# 🚗 Cityscapes Semantic Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/YOUR_USERNAME/cityscapes-segmentation/blob/main/cityscapes_segmentation.ipynb](https://colab.research.google.com/drive/1RqrHIwtXjJY3o97_GHCidB2UPi-SWCGd))
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end semantic segmentation of urban street scenes using a custom **DeepLabV3+** architecture with a **ResNet-50** backbone, trained on the [Cityscapes dataset](https://www.cityscapes-dataset.com/).

---

## ✨ Highlights

- **DeepLabV3+ from scratch** — Encoder (ResNet-50 with dilated convolutions) → ASPP → Decoder, all hand-coded with detailed explanations
- **Combined loss** — Cross-Entropy + Focal + Dice, weighted to handle rare classes like `bicycle` and `person`
- **Smart training** — Mixed-precision (AMP), differential learning rates, and warmup + cosine LR scheduling
- **Scene analysis engine** — Counts pedestrians, vehicles, and cyclists from the predicted mask and generates plain-language driving advice
- **Interactive Gradio demo** — Image and video inference with a live congestion HUD
- **Baseline comparison** — Automatically benchmarks your model against FCN, PSPNet, DeepLabV3, and FPN on the same val split

---

## 📊 Architecture

```
Input Image  (B, 3, 512, 512)
      │
      ▼
┌─────────────────────────────────────────────┐
│  ResNet-50 Encoder (ImageNet pretrained)    │
│  Dilated convolutions in layer3 & layer4    │
│  low_level  → (B,  256, H/4, W/4)           │
│  high_level → (B, 2048, H/8, W/8)           │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  ASPP — 5 parallel branches                 │
│  1×1 conv │ d=6 │ d=12 │ d=18 │ GAP         │
│  Output: (B, 256, H/8, W/8)                 │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  Decoder — fuses ASPP + low_level           │
│  Upsamples 4× → 19 class scores per pixel   │
└──────────────┬──────────────────────────────┘
               ▼
Output Mask  (B, 512, 512)
```

---

## 🗂️ Dataset

The notebook uses the [Cityscapes dataset](https://www.cityscapes-dataset.com/) downloaded via [KaggleHub](https://www.kaggle.com/datasets/electraawais/cityscape-dataset).

**19 semantic classes:** road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle.

You need a free Kaggle account and an API token to download the data. See the [Kaggle API docs](https://www.kaggle.com/docs/api) for setup.

---

## 🚀 Quick Start

### Google Colab (recommended)

1. Click the **Open in Colab** badge above.
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Run all cells top to bottom.

### Local

```bash
git clone https://github.com/YOUR_USERNAME/cityscapes-segmentation.git
cd cityscapes-segmentation
pip install kagglehub segmentation-models-pytorch albumentations gradio scipy torch torchvision
jupyter notebook cityscapes_segmentation.ipynb
```

> **Note:** The notebook calls `google.colab.drive` in Cell 3. Comment that block out if running locally and set `DRIVE_DIR` to a local path.

---

## ⚙️ Configuration (Cell 1)

| Variable | Default | Description |
|---|---|---|
| `SKIP_TRAINING` | `False` | Set `True` to skip training and load a saved checkpoint |
| `RESUME` | `False` | Set `True` to continue training from the last checkpoint |
| `NUM_EPOCHS` | `10` | Number of training epochs |
| `BATCH_SIZE` | `4` | Reduce to `2` if you get CUDA OOM errors |
| `IMG_H / IMG_W` | `512` | Input resolution |
| `BACKBONE` | `'ResNet-50'` | For reference only; model is fixed to ResNet-50 |

---

## 📈 Expected Results

Approximate benchmarks on the Cityscapes validation split after 10 epochs on a T4 GPU:

| Model | mIoU | Pixel Acc |
|---|---|---|
| FCN (ResNet-50) | ~35–40 % | ~88 % |
| DeepLabV3 (ResNet-50) | ~42–48 % | ~91 % |
| **This model (10 epochs)** | **~45–52 %** | **~91–93 %** |
| DeepLabV3+ paper (full training) | 78.8 % | 95.4 % |

> Full convergence requires ~50–100 epochs. Use `RESUME = True` to keep adding epochs across sessions.

---

## 📦 Dependencies

```
torch >= 2.0
torchvision >= 0.15
segmentation-models-pytorch
albumentations
gradio
kagglehub
scipy
opencv-python
matplotlib
numpy
```

---

## 📁 Repository Structure

```
cityscapes-segmentation/
├── cityscapes_segmentation.ipynb   # Main notebook (all 17 cells)
├── README.md
└── LICENSE
```

Checkpoints and training history are saved to Google Drive at the path set in `DRIVE_DIR`.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/) — Cordts et al., 2016
- [DeepLabV3+](https://arxiv.org/abs/1802.02611) — Chen et al., 2018
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://albumentations.ai/)
