# 🚀 MODELARTS TRAINING - STEP-BY-STEP EXECUTION GUIDE

## 📋 Current Status
- ✅ **Phase 1:** OBS Bucket Created
- ⏳ **Phase 2:** Dataset Upload In Progress (54K/108K images, 1.9%)
- 🎯 **Phase 3:** ModelArts Training - **READY TO START**

---

## 🎯 PHASE 3: MODELARTS TRAINING - COMPLETE GUIDE

### ⚠️ IMPORTANT: Before You Start

**Prerequisites Checklist:**
- ☐ Dataset upload to OBS completed (108,098 images)
- ☐ Training script `train_arsl.py` ready
- ☐ Huawei Cloud credentials configured
- ☐ GPU quota available in AF-Cairo region

**Estimated Time:** 5-7 hours total
**Estimated Cost:** $5-10 USD

---

## 📤 STEP 1: Upload Training Code to OBS (5 minutes)

### Option A: Manual Upload via OBS Console

1. **Navigate to OBS Console:**
   ```
   🌐 https://console.huaweicloud.com/obs
   ```

2. **Open your bucket:**
   - Bucket name: `arsl-youssef-af-cairo-2025`

3. **Create `code` folder:**
   - Click "Create Folder"
   - Name: `code`
   - Click "OK"

4. **Upload training script:**
   - Navigate into `code/` folder
   - Click "Upload Object"
   - Select: `scripts/train_arsl.py` from your local computer
   - Click "Upload"
   - Wait for upload to complete

5. **Verify upload:**
   - Path should be: `obs://arsl-youssef-af-cairo-2025/code/train_arsl.py`
   - File size: ~12 KB

### Option B: Upload via OBS Browser Tool (Recommended for large files)

```bash
# Install OBS Browser (if not installed)
# Download from: https://support.huaweicloud.com/intl/en-us/browsertg-obs/obs_03_1000.html

# Then drag and drop train_arsl.py to code/ folder
```

---

## 🏗️ STEP 2: Create ModelArts Training Job (15 minutes)

### 2.1 Navigate to ModelArts Console

1. **Open ModelArts:**
   ```
   🌐 https://console.huaweicloud.com/modelarts
   ```

2. **Select Region:**
   - Ensure you're in: **AF-Cairo (af-north-1)**

3. **Go to Training Management:**
   - Left sidebar → "Training Management"
   - Click "Training Jobs"
   - Click "Create Training Job" button

---

### 2.2 Configure Basic Information

**Job Settings:**
```
Job Name: arsl-recognition-training-v1
Description: Arabic Sign Language Recognition CNN Training for 32 Arabic classes
```

---

### 2.3 Configure Algorithm

**Algorithm Source:**
- Select: **"Custom Algorithm"**

**Code Configuration:**
```
Code Directory: obs://arsl-youssef-af-cairo-2025/code/
Boot File: train_arsl.py
```

**AI Engine:**
```
Framework: PyTorch
Version: PyTorch 1.8.0-python3.7
Engine ID: 122
```

💡 **Tip:** If you don't see the exact version, choose the closest PyTorch 1.x version with Python 3.7

---

### 2.4 Configure Data Paths

**Input and Output:**
```
Training Input (data_url):
obs://arsl-youssef-af-cairo-2025/datasets/raw/

Training Output (train_url):
obs://arsl-youssef-af-cairo-2025/output/

Job Log Path (log_url):
obs://arsl-youssef-af-cairo-2025/logs/
```

💡 **Important:** Make sure these folders exist in your OBS bucket, or ModelArts will create them

---

### 2.5 Configure Resources

**Resource Pool:**
- Select: **"Public Resource Pools"**

**Compute Resources:**
```
Resource Type: GPU
Resource Flavor: GPU: 1 * V100-32GB (modelarts.vm.gpu.v100)
Compute Nodes: 1
```

**Alternative if V100 not available:**
```
GPU: 1 * P100-16GB (modelarts.vm.gpu.p100)
```

💰 **Cost Estimate:**
- V100: ~$1.8-2.0/hour
- P100: ~$1.2-1.5/hour
- Total: $7-14 for 4-7 hours training

---

### 2.6 Configure Hyperparameters

Click "Add Hyperparameter" for each of these:

| Parameter Name | Type | Value | Description |
|----------------|------|-------|-------------|
| `batch_size` | Integer | `64` | Training batch size |
| `learning_rate` | Float | `0.001` | Learning rate for optimizer |
| `epochs` | Integer | `100` | Maximum training epochs |
| `num_classes` | Integer | `32` | Number of Arabic sign classes |
| `image_size` | Integer | `64` | Input image size (64x64) |
| `validation_split` | Float | `0.2` | Validation set size (20%) |
| `patience` | Integer | `15` | Early stopping patience |
| `random_seed` | Integer | `42` | Random seed for reproducibility |

💡 **How to add each parameter:**
1. Click "Add Hyperparameter"
2. Enter parameter name (e.g., `batch_size`)
3. Enter value (e.g., `64`)
4. Repeat for all 8 parameters

---

### 2.7 Configure Training Constraints

**Running Configuration:**
```
Maximum Running Time: 8 hours (28800 seconds)
Auto Stop: ✅ Enabled
Automatic Restart: ❌ Disabled
```

**Checkpoints:**
```
☐ Enable Automatic Saving of Training Checkpoints (Optional)
Checkpoint Path: obs://arsl-youssef-af-cairo-2025/output/checkpoints/
```

---

### 2.8 Review and Submit

1. **Review all settings carefully**
2. **Estimated Cost Preview:** Check the cost estimate displayed
3. **Click "Submit"** or "Create Now"
4. **Job will appear in Training Jobs list with status: "Creating"**

---

## 📊 STEP 3: Monitor Training Progress (4-7 hours)

### 3.1 Real-Time Monitoring in Console

1. **View Job List:**
   - Go to: Training Management → Training Jobs
   - Find your job: `arsl-recognition-training-v1`

2. **Job Status Stages:**
   ```
   Creating (2-5 min) → Pending (5-10 min) → Running (4-6 hours) → Completed
   ```

3. **Click on Job Name** to view details:
   - **Overview Tab:** Job info, status, resource usage
   - **Logs Tab:** Real-time training logs
   - **Metrics Tab:** Training/validation metrics
   - **Configuration Tab:** Job parameters

---

### 3.2 Key Metrics to Watch

**Training Logs (Check every 30 mins):**
```
✅ Good Signs:
   • Training loss decreasing steadily
   • Validation accuracy increasing
   • No out-of-memory errors
   • GPU utilization >70%

⚠️ Warning Signs:
   • Loss stuck or increasing
   • Validation accuracy plateauing early
   • Frequent errors in logs
   • Low GPU utilization <30%
```

**Expected Progress:**
```
Epoch 1-10:    Train Acc ~40-60%, Val Acc ~35-55%
Epoch 11-30:   Train Acc ~70-80%, Val Acc ~65-75%
Epoch 31-50:   Train Acc ~85-90%, Val Acc ~80-85%
Epoch 51-100:  Train Acc ~92-95%, Val Acc ~88-92%
```

---

### 3.3 Training Timeline

| Time | Activity | Status |
|------|----------|--------|
| 0-5 min | Job creation and validation | Creating |
| 5-15 min | Resource allocation and environment setup | Pending |
| 15-30 min | Data loading and preprocessing | Running (Initializing) |
| 30 min - 4 hours | Model training (epochs 1-80) | Running (Training) |
| 4-5 hours | Final epochs and validation | Running (Finalizing) |
| 5-7 hours | Saving model and cleanup | Completing |

---

### 3.4 Download Logs for Local Monitoring

```bash
# Logs are automatically saved to:
obs://arsl-youssef-af-cairo-2025/logs/

# You can download and view them locally
# Or check them directly in the ModelArts console
```

---

## ✅ STEP 4: Verify Training Completion

### 4.1 Check Job Status

**Success Indicators:**
- ✅ Job Status: **"Completed"** (green)
- ✅ Duration: 4-7 hours
- ✅ Exit Code: 0
- ✅ Final validation accuracy: >85%

### 4.2 Verify Output Files in OBS

Navigate to `obs://arsl-youssef-af-cairo-2025/output/` and verify:

```
✅ Required Files:
   📁 best_model.pth (Model checkpoint)
   📁 final_model.pth (Final model)
   📄 training_history.json (Metrics history)
   📁 checkpoint_epoch_*.pth (Periodic checkpoints)
```

### 4.3 Download and Review Metrics

1. **Download `training_history.json`** from OBS
2. **Check final metrics:**
   ```json
   {
     "epoch": 85,
     "train_acc": 93.5,
     "val_acc": 89.2,
     "val_loss": 0.35,
     "f1_score": 0.88
   }
   ```

3. **Success Criteria:**
   - ✅ Validation Accuracy > 85%
   - ✅ F1 Score > 0.80
   - ✅ Training/Validation gap < 10%
   - ✅ Final loss < 0.5

---

## 🚨 TROUBLESHOOTING

### Problem: Job Fails Immediately

**Possible Causes:**
- ❌ Training script has syntax errors
- ❌ OBS paths are incorrect
- ❌ Required Python packages missing

**Solutions:**
1. Check job logs for error messages
2. Verify `train_arsl.py` was uploaded correctly
3. Ensure all OBS paths exist and are accessible
4. Review algorithm configuration settings

---

### Problem: Training Takes Too Long

**If training runs > 8 hours:**
- Check if early stopping is working
- Review logs for repeated errors
- Consider reducing epochs or batch size
- Check if dataset loaded correctly

---

### Problem: Low Accuracy (<70%)

**Possible Issues:**
- Dataset quality problems
- Hyperparameters need tuning
- Model architecture too simple
- Data augmentation needed

**Solutions:**
1. Verify dataset quality and balance
2. Try different learning rates (0.0001, 0.01)
3. Adjust batch size (32 or 128)
4. Add more training epochs

---

## 🎉 STEP 5: Next Steps After Training

### Once Training Completes Successfully:

1. **✅ Download Model Files:**
   - Download `best_model.pth` from OBS output folder
   - Save locally for testing

2. **📊 Review Training Metrics:**
   - Analyze accuracy trends
   - Check for overfitting
   - Document model performance

3. **🚀 Proceed to Phase 4: API Deployment**
   ```bash
   # Import model to VIAS
   python scripts/phase4_step1_import_model.py
   
   # Create inference service
   python scripts/phase4_step2_inference_service.py
   
   # Configure API Gateway
   python scripts/phase4_step3_api_gateway.py
   
   # Setup monitoring
   python scripts/phase4_step4_monitoring.py
   
   # Test API
   python scripts/phase4_step5_api_testing.py
   ```

---

## 📝 QUICK REFERENCE

### Important URLs
- **ModelArts Console:** https://console.huaweicloud.com/modelarts
- **OBS Console:** https://console.huaweicloud.com/obs
- **Training Jobs:** ModelArts → Training Management → Training Jobs

### Key OBS Paths
```
Code:   obs://arsl-youssef-af-cairo-2025/code/
Input:  obs://arsl-youssef-af-cairo-2025/datasets/raw/
Output: obs://arsl-youssef-af-cairo-2025/output/
Logs:   obs://arsl-youssef-af-cairo-2025/logs/
```

### Support
- **Documentation:** Check `docs/` folder
- **Logs:** ModelArts console or OBS logs folder
- **Status:** Run `python quick_status.py`

---

## 🎯 SUCCESS METRICS

**Target Performance:**
- ✅ Training Accuracy: >90%
- ✅ Validation Accuracy: >85%
- ✅ F1 Score: >0.80
- ✅ Training Time: 4-7 hours
- ✅ Cost: <$15

---

*Last Updated: October 1, 2025*
*Account: yyacoup | Region: AF-Cairo (af-north-1)*
*Project: Arabic Sign Language Recognition*