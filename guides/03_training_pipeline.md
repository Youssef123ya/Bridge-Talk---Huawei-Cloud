# 🚀 Step 3: Setup ModelArts Training Pipeline

## 📋 **Prerequisites**
- ✅ OBS bucket created with your dataset
- ✅ Dataset uploaded to `obs://arsl-youssef-af-cairo-2025/datasets/`
- ✅ Environment configured

## 🎯 **Training Pipeline Overview**

Your cloud training pipeline will:
- 🔥 **Train on GPU instances** (faster than local training)
- 📊 **Auto-track experiments** with metrics and logs
- 💾 **Save models to OBS** automatically
- 📈 **Monitor in real-time** through ModelArts console
- 🔄 **Auto-scale resources** based on workload

## 🛠️ **Option A: Automated Training Setup (Recommended)**

### **1. Upload Training Code to OBS**
```powershell
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/setup_training_code.py
```

### **2. Start Training Job**
```powershell
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/start_training.py
```

## 🖥️ **Option B: Manual Setup in ModelArts Console**

### **1. Access ModelArts Console**
1. Go to [Huawei Cloud Console](https://console.huaweicloud.com)
2. Navigate to **ModelArts** service
3. Select **Training Management** → **Training Jobs**

### **2. Create Training Job**
Click **"Create Training Job"** and configure:

```yaml
Job Name: arsl-training-{timestamp}
Description: Arabic Sign Language CNN Training

Algorithm Settings:
  Code Directory: obs://arsl-youssef-af-cairo-2025/code/
  Boot File: train_arsl.py
  AI Engine: PyTorch 1.8.0 Python 3.7

Data Settings:
  Dataset Path: obs://arsl-youssef-af-cairo-2025/datasets/
  Output Path: obs://arsl-youssef-af-cairo-2025/output/

Resource Settings:
  Resource Pool: Public resource pool
  Compute Type: GPU (modelarts.vm.gpu.p4)
  Instance Count: 1

Hyperparameters:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  num_classes: 32
  image_size: 64
  dropout: 0.5
```

### **3. Monitor Training**
- 📊 **Real-time metrics**: Loss, accuracy, learning rate
- 📋 **Logs**: View training progress and debug issues
- 📈 **Resource usage**: GPU utilization, memory usage
- ⏱️ **Time estimates**: Training completion time

## 🔧 **Training Configuration Options**

### **GPU Instance Types Available:**
| Instance Type | GPU | Memory | Use Case |
|---------------|-----|--------|----------|
| `modelarts.vm.gpu.p4` | V100 | 32GB | Recommended for ARSL |
| `modelarts.vm.gpu.p2` | K80 | 12GB | Budget option |
| `modelarts.vm.gpu.t4` | T4 | 16GB | Good performance/cost |

### **Hyperparameter Tuning:**
Enable automatic hyperparameter optimization:
```yaml
hyperparameter_tuning:
  enabled: true
  tuning_strategy: "bayesian"
  max_trials: 10
  parameters:
    learning_rate: [0.0001, 0.001, 0.01]
    batch_size: [32, 64, 128]
    dropout: [0.3, 0.5, 0.7]
```

## 📊 **Expected Training Performance**

### **Dataset**: 108,000 images, 32 classes
### **Hardware**: GPU P4 instance
### **Training Time**: 
- **50 epochs**: ~4-6 hours
- **100 epochs**: ~8-12 hours  
- **200 epochs**: ~16-24 hours

### **Expected Results**:
- **Training Accuracy**: 95-98%
- **Validation Accuracy**: 85-92%
- **Model Size**: ~50-100 MB

## 🔍 **Monitoring Training Progress**

### **Real-time Dashboards**:
1. **ModelArts Console**: Training metrics and logs
2. **Cloud Eye**: Resource utilization
3. **OBS**: Model checkpoints and outputs

### **Key Metrics to Watch**:
- 📈 **Training Loss**: Should decrease steadily
- 📊 **Validation Accuracy**: Should increase then plateau
- 🔥 **GPU Utilization**: Should be >80% for efficiency
- 💾 **Memory Usage**: Monitor for OOM errors

## 🚨 **Troubleshooting Training Issues**

### **Issue**: Training job fails to start
**Solutions**:
- Verify dataset path in OBS
- Check training code uploaded correctly
- Ensure sufficient quota for GPU instances

### **Issue**: Low accuracy or slow convergence
**Solutions**:
- Reduce learning rate (try 0.0001)
- Increase batch size if GPU memory allows
- Add data augmentation
- Try different model architecture

### **Issue**: Out of memory errors
**Solutions**:
- Reduce batch size (try 32 or 16)
- Use smaller image size (try 32x32)
- Enable gradient accumulation

## 📁 **Training Outputs**

After training completes, you'll find in OBS:
```
obs://arsl-youssef-af-cairo-2025/output/
├── best_model.pth          # Best performing model
├── final_model.pth         # Final epoch model
├── training_history.json   # Metrics history
├── config.json            # Training configuration
├── logs/                  # Detailed training logs
└── checkpoints/           # Epoch checkpoints
```

## ⏭️ **Next Steps After Training**

1. **Download Best Model**: Get the trained model for inference
2. **Evaluate Performance**: Test on validation data
3. **Deploy to Inference**: Setup real-time API service
4. **Monitor Production**: Track model performance

---

## 🎯 **Quick Start Commands**

Once you've created your bucket and uploaded data:

```powershell
# 1. Setup training code
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/setup_training_code.py

# 2. Start training job
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/start_training.py

# 3. Monitor training
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/monitor_training.py
```

**💡 Pro Tip**: Training runs in the cloud - you can close your laptop and check progress later!