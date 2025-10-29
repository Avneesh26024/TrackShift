# 🔍 TrackShift: Intelligent Visual Difference Engine

**By Team DeltaMind**

A production-ready, domain-agnostic visual anomaly detection system that works across industries without retraining.


##  The Problem

Traditional anomaly detection systems have a critical flaw: they require extensive training data for each new domain. Detect cracks in concrete? Train a model. Switch to PCB inspection? Train another model. Move to brand compliance? Train yet another model.

**This doesn't scale.**

##  Our Solution: The DeltaMind Approach

TrackShift uses a **three-stage modular pipeline** that works like a funnel, filtering out environmental noise (lighting, angles) to isolate true anomalies:

###  Stage 1: Align (The Foundation)
Makes the comparison fair by eliminating perspective and environmental differences:
- **SIFT/ORB keypoint detection** - Identifies matching features between images
- **Homography transformation** - Warps images into perfect alignment
- **Photometric normalization** - Adjusts for lighting variations

**Output:** Two perfectly aligned images, as if captured from the exact same position

---

###  Stage 2: Detect (The Dual-Check Brain)

This is where the magic happens. Instead of relying on a single method, we use **two parallel checks** that work together:

#### Fast Check (Pixel-Level)
- Uses **SSIM + Absolute Difference**
- Lightning-fast detection of all pixel-level changes
- Catches everything but will flag shadows, reflections, and minor lighting changes

#### Deep Check (Feature-Level)
- Uses **PatchCore** or similar "normal-only" models
- Trained only on images of "normal" states (perfect products, healthy infrastructure)
- Learns what "normal" looks like and flags anything abnormal
- Distinguishes between real anomalies (cracks, defects) and environmental noise (shadows, lighting)

#### The Fusion
The system combines both checks. **True anomalies** are regions that:
1. Show pixel-level differences (Fast Check)
2. Show abnormal features (Deep Check)

This dual-validation dramatically reduces false positives.

---

###  Stage 3: Segment (The Precise Cut)
Generates pixel-perfect boundaries around detected anomalies:
- High-confidence regions from Stage 2 become **point/box prompts**
- **SAM 2 (Segment Anything Model)** produces precise segmentation masks
- Extracts bounding boxes and contours for reporting

**Output:** JSON report with coordinates, masks, and anomaly scores

---

##  Why This Approach Works

| Challenge | Our Solution |
|-----------|--------------|
| **Different camera angles** | Stage 1 (Align) handles rotations, zooms, perspective shifts |
| **Lighting variations** | Photometric normalization + semantic feature comparison |
| **Domain switching** | "Normal-only" training - just show examples of "good" states |
| **False positives** | Dual-check system filters environmental noise |
| **Precision requirements** | SAM 2 generates pixel-perfect masks |

##  Real-World Proof: Satellite Imagery

We tested our generalized pipeline on satellite imagery to demonstrate cross-domain capability.

### Reference Image (Before)
![Reference Satellite Image](https://github.com/Avneesh26024/TrackShift/blob/main/Satellite_Image_Test/Golden_Image.jpg?raw=true)

### Current Image (After)
![Current Satellite Image](https://github.com/Avneesh26024/TrackShift/blob/main/Satellite_Image_Test/Current_Image.jpg?raw=true)

### Detection Results
![Visual Changes Detected](https://github.com/Avneesh26024/TrackShift/blob/main/Satellite_Image_Test/Visual_Changes.png?raw=true)

**Result:** The system successfully identified changes in regions including land use modifications, infrastructure development, and environmental alterations—**without any training on satellite imagery.**

---

##  Target Applications

###  Infrastructure Monitoring
**Example:** Bridge crack detection
- **Reference:** Drone photo from 2024
- **Current:** Drone photo from 2025 (different angle, sunny day)
- **Pipeline:**
  1. Align corrects perspective and normalizes lighting
  2. Fast Check flags crack + bird shadow
  3. Deep Check (trained on "normal concrete") flags only the crack
  4. SAM 2 generates precise crack mask
- **Output:** JSON report with crack coordinates, mask, and anomaly score (0.97)

###  Manufacturing Quality Control
- Detect missing components, assembly errors, surface defects
- Compare production units against golden reference
- Real-time inspection on assembly lines

###  Brand Compliance
- Verify product placement on retail shelves
- Detect packaging inconsistencies
- Ensure logo and design accuracy

###  Satellite & Aerial Analysis
- Track urban development and land use changes
- Monitor environmental degradation
- Disaster damage assessment

---

##  Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/Avneesh26024/TrackShift.git
cd TrackShift

# Install dependencies
pip install opencv-python-headless numpy torch torchvision transformers pillow scipy
```

---

## 📖 Usage

### Jupyter Notebook (Recommended)
The **[PatchCore.ipynb](PatchCore%20(1).ipynb)** notebook contains our complete generalized pipeline with:
- Stage-by-stage execution
- Debug visualizations (heatmaps, alignment checks)
- Configurable parameters
- Example outputs

### Quick Configuration
Key parameters you can adjust:
```python
# Stage 2: Detection sensitivity
heatmap_threshold = 200        # Higher = fewer false positives (range: 80-250)
blur_kernel_size = 21          # Larger = smoother heatmaps (range: 11-31)

# Stage 3: Segmentation
min_peak_distance = 20         # Minimum spacing between detections (pixels)
```

---

##  Project Structure

```
TrackShift/
├── PatchCore (1).ipynb              # 🎯 Main pipeline (generalized prototype)
├── Satellite_Image_Test/            # Proof-of-concept results
│   ├── Golden_Image.jpg             # Reference image
│   ├── Current_Image.jpg            # Test image
│   └── Visual_Changes.png           # Detection output
├── old_experiments/                 # Archived: early pixel-diff approaches
└── README.md                        # This file
```

---

##  Technical Deep Dive

### The "Normal-Only" Training Philosophy
Unlike supervised learning (which requires labeled defects), our approach only needs examples of **good/normal states**:

1. Collect 500-1000 images of "perfect" products/infrastructure
2. Train PatchCore to learn the feature distribution of "normal"
3. Deploy: anything outside this distribution = anomaly

**Advantage:** No need to collect rare defect examples or label data.

### Why PatchCore?
PatchCore is a memory-bank approach that:
- Extracts features from pre-trained networks (ResNet, WideResNet)
- Stores "normal" feature patches in memory
- Detects anomalies via nearest-neighbor distance
- Requires no gradient-based training

### The Dual-Check Fusion Strategy
```
If (Fast Check = DIFFERENT) AND (Deep Check = ABNORMAL):
    → True Anomaly
Else If (Fast Check = DIFFERENT) AND (Deep Check = NORMAL):
    → Environmental Noise (shadow, lighting)
Else:
    → No Change
```

This logic drastically reduces false positives in real-world deployments.

---

##  Current Status & Roadmap

### ✅ Completed
- [x] Three-stage pipeline architecture
- [x] SIFT-based alignment module
- [x] Dual-check detection (SSIM + semantic features)
- [x] SAM 2 integration for segmentation
- [x] Proof-of-concept on satellite imagery

###  In Progress
- [ ] PatchCore integration (replacing DINOv2 prototype)
- [ ] Automated "normal-only" training scripts
- [ ] Benchmark validation on multiple domains

###  Future Work
- [ ] Video stream processing for temporal tracking
- [ ] REST API for production deployment
- [ ] Interactive web UI for result visualization
- [ ] Edge device optimization (Jetson, Raspberry Pi)

---


##  Team DeltaMind

Built with precision for real-world visual intelligence.

