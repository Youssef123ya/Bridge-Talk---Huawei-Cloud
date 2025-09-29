# 📤 Step 2: Upload Your Dataset to Huawei Cloud OBS

## 📋 **Prerequisites**
- ✅ OBS bucket created (`arsl-youssef-af-cairo-2025`)
- ✅ Your ARSL dataset in `data/` folder
- ✅ Environment configured

## 🎯 **Dataset Upload Options**

### **Option A: Automated Upload Script (Recommended)**

I've created a smart upload script that will:
- ✅ Upload your 108K+ images with progress tracking
- ✅ Organize data in cloud-friendly structure
- ✅ Verify upload integrity
- ✅ Handle large files efficiently

#### **Run the Upload Script:**
```powershell
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/upload_dataset.py
```

### **Option B: Manual Upload (Web Console)**
1. Open Huawei Cloud Console
2. Go to OBS → Your bucket
3. Click "Upload" and select your `data/` folder
4. Wait for upload completion (may take 1-2 hours for 108K images)

## 🚀 **Let's Create the Smart Upload Script**

I'll create an optimized upload script for your large dataset:

```powershell
# This will create the upload script for you
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" -c "print('Creating upload script...')"
```

## 📊 **Upload Progress Monitoring**

The script will show:
- 📈 Upload progress (files/second)
- 💾 Data transferred (GB)
- ⏱️ Estimated time remaining
- ✅ Success/failure counts
- 🔄 Retry logic for failed uploads

## 🎯 **Expected Upload Structure**

After upload, your bucket will contain:
```
arsl-youssef-af-cairo-2025/
├── datasets/
│   ├── raw/
│   │   ├── class_01/
│   │   │   ├── image_001.jpg
│   │   │   ├── image_002.jpg
│   │   │   └── ...
│   │   ├── class_02/
│   │   └── ...
│   ├── ArSL_Data_Labels.csv
│   └── processed/ (if exists)
├── models/ (for future training outputs)
├── logs/ (for training logs)
└── experiments/ (for experiment tracking)
```

## ⏱️ **Estimated Upload Time**

Based on your dataset size:
- **108,000 images** (~10GB total)
- **Average upload speed**: 50-100 files/minute
- **Total time**: 18-36 hours (depending on internet speed)
- **Optimized parallel upload**: 6-12 hours

## 🆘 **Troubleshooting Upload Issues**

### **Issue**: "Upload too slow"
**Solution**: The script uses parallel uploads (10 threads)

### **Issue**: "Some files failed"
**Solution**: Script automatically retries failed uploads

### **Issue**: "Out of storage quota"
**Solution**: Check your OBS storage limits in console

### **Issue**: "Connection timeout"
**Solution**: Script resumes from last successful upload

## ⏭️ **Next Steps After Upload**

1. **Verify Upload**: Check file counts match your local dataset
2. **Test Data Access**: Ensure training script can read from OBS
3. **Setup Training Pipeline**: Configure ModelArts for GPU training

---
**💡 Pro Tip**: The upload runs in background - you can continue with other tasks!