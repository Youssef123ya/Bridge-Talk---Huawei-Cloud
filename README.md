# Arabic Sign Language Recognition Project

A deep learning project for recognizing Arabic Sign Language (ArSL) gestures using computer vision techniques.

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

## 📈 Results (Phase 1)

- **Environment Setup**: ✅ Complete
- **Dataset Validation**: ✅ 100% file integrity
- **Data Pipeline**: ✅ All tests passed
- **Project Structure**: ✅ Ready for development

## 🔄 Next Steps (Phase 2)

1. Implement advanced data loaders with augmentation
2. Design and implement CNN architectures
3. Set up training pipeline with validation
4. Implement model evaluation and metrics
5. Create inference pipeline

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
- `data/analysis/validation_report.txt` - Detailed dataset report

## 👥 Contributing

This is a learning project for Arabic Sign Language recognition. Future improvements could include:
- Data augmentation techniques
- Advanced CNN architectures
- Transfer learning approaches
- Real-time inference capabilities

## 📄 License

Educational project - please respect the original dataset licensing terms.

---

**Status**: Phase 1 Complete ✅ | Ready for Phase 2 🚀
