Huawei Cloud - Arabic Sign Language Recognition Project




https://github.com/user-attachments/assets/79354a4f-3351-4c50-8fbe-4ca4cc373cf4




🎯 Goal**: Real-time Arabic Sign Language Recognition with Enterprise Cloud Deployment
<img width="1338" height="371" alt="roadmap" src="https://github.com/user-attachments/assets/16f045d0-aba6-4b28-9cbd-34f4c7cc50c2" />

## 🚀 **Complete Cloud Integration Available!**

This project now includes **full Huawei Cloud integration** for enterprise-scale deployment:

- ☁️ **Object Storage Service (OBS)** - Dataset and model storage
- 🤖 **ModelArts** - GPU training with auto-scaling  
- 🌐 **API Gateway** - Real-time inference deployment
- 📊 **Cloud Eye** - Monitoring and alerting
- ⚡ **ECS** - Scalable compute resources

### 📋 **Quick Start with Cloud**
```bash
# Clone repository
git clone https://github.com/Youssef123ya/Bridge-Talk.git
cd Bridge-Talk

# Follow cloud deployment guides
see guides/00_master_checklist.md
```

A deep learning project for recognizing Arabic Sign Language (ArSL) gestures using computer vision techniques with complete cloud infrastructure.

## 🎯 Project Overview

# 2. Install cloud dependencies
pip install -r requirements.txt

# 3. Run complete setup
python scripts\setup_huawei_cloud.py --upload-data --start-training
This project implements a CNN-based approach to classify Arabic sign language gestures representing the 32 letters of the Arabic alphabet. The dataset contains over 108,000 images across 32 classes.

## 📊 Dataset Statistics

- **Total Images**: 108,098
- **Classes**: 32 (Arabic alphabet letters)
- **Image Format**: 64x64 grayscale images
- **Average per class**: ~3,378 images

## 🚀 Phase 1: Environment Setup ✅

### Completed Tasks:
- ✅ Python 3.13+ environment with virtual environment
- ✅ All required packages installed (PyTorch, OpenCV, etc.)
- ✅ Dataset validation and analysis
- ✅ Data visualization and reporting
- ✅ Basic data pipeline testing
- ✅ Project structure setup

### Quick Start

1. **Activate the virtual environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Check Phase 1 status:**
   ```powershell
   python scripts\check_phase1_status.py
   ```

3. **Validate dataset:**
   ```powershell
   python scripts\validate_data.py
   ```

4. **Test data pipeline:**
   ```powershell
   python scripts\test_data_pipeline.py
   ```

## 📁 Project Structure

```
├── data/
│   ├── raw/                 # Original image data (32 class folders)
│   ├── processed/           # Processed data
│   ├── analysis/            # Analysis results and visualizations
│   └── ArSL_Data_Labels.csv # Dataset labels
├── src/
│   ├── data/               # Data processing modules
│   ├── models/             # Model definitions
│   └── utils/              # Utility functions
├── scripts/                # Executable scripts
├── config/                 # Configuration files
├── models/                 # Saved model checkpoints
├── logs/                   # Training logs
├── notebooks/              # Jupyter notebooks
└── requirements.txt        # Python dependencies
```

## 🔧 Requirements

### System Requirements:
- Windows 10/11
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space

### Installed Packages:
- PyTorch 2.8.0
- OpenCV 4.12.0
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn, tqdm, Pillow

## 📝 Usage Examples

### Check Environment Status
```python
python scripts\check_phase1_status.py
```

### Validate Dataset
```python
python scripts\validate_data.py
```

### Test Data Loading
```python
python scripts\test_data_pipeline.py
```

## 🔍 Analysis Results

View the generated analysis files:
- `data/analysis/class_distribution.png` - Class distribution visualization
- `data/analysis/validation_report.txt` - Detailed dataset report�
