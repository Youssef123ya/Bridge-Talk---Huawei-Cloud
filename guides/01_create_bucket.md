# 🪣 Step 1: Creating OBS Bucket in Huawei Cloud Console

## 📋 **Prerequisites**
- ✅ Huawei Cloud account (your account is ready)
- ✅ AF-Cairo region access
- ✅ Your credentials configured

## 🎯 **Step-by-Step Instructions**

### **1. Access Huawei Cloud Console**
1. Open your web browser
2. Go to: [https://console.huaweicloud.com](https://console.huaweicloud.com)
3. Sign in with your Huawei Cloud credentials
4. Ensure you're in the **AF-Cairo** region (top-right corner)

### **2. Navigate to Object Storage Service**
1. In the console dashboard, look for **Storage** in the left menu
2. Click on **Object Storage Service (OBS)**
3. You'll see the OBS dashboard

### **3. Create New Bucket**
1. Click the **"Create Bucket"** button (usually blue/orange button)
2. Fill in the bucket details:

```
Bucket Name: arsl-youssef-af-cairo-2025
Region: AF-Cairo (af-north-1)
Storage Class: Standard
Access Control: Private (recommended for your data)
Versioning: Disabled (you can enable later if needed)
Encryption: Server-side encryption (recommended)
```

### **4. Configure Bucket Settings**
1. **Access Permissions**: Keep as "Private"
2. **Bucket Policy**: Default (we'll configure later if needed)
3. **CORS**: Default (we'll add rules later for API access)
4. **Lifecycle**: Skip for now

### **5. Verify Bucket Creation**
After clicking "Create", you should see:
- ✅ Success message
- 📁 Your bucket `arsl-youssef-af-cairo-2025` in the bucket list
- 🌍 Region shows as "AF-Cairo"

### **6. Test Connection from Your Code**
Run this command to verify the bucket is accessible:

```powershell
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" -c "
import sys; sys.path.append('src')
from cloud.huawei_storage import HuaweiCloudStorage
storage = HuaweiCloudStorage()
bucket_info = storage.get_bucket_info()
print('Bucket Status:', bucket_info)
if bucket_info['exists']:
    print('✅ SUCCESS: Bucket is accessible from your code!')
else:
    print('❌ ISSUE: Bucket not found or not accessible')
storage.close()
"
```

## 🎉 **Expected Result**
You should see:
```
Bucket Status: {'name': 'arsl-youssef-af-cairo-2025', 'exists': True, 'region': 'af-north-1'}
✅ SUCCESS: Bucket is accessible from your code!
```

## 🆘 **Troubleshooting**

### **Issue**: "Bucket name already exists"
**Solution**: Try adding your username or current date:
- `arsl-youssef-af-cairo-20250929`
- `arsl-recognition-youssef-2025`

### **Issue**: "Permission denied"
**Solution**: 
1. Check if OBS service is enabled in your account
2. Verify your IAM permissions include OBS access
3. Contact Huawei Cloud support for service activation

### **Issue**: "Region not available"
**Solution**:
1. Try `cn-north-1` (Beijing) as alternative
2. Update your `config/huawei_cloud_config.yaml` with the working region

## ⏭️ **Next Step**
Once your bucket is created and verified, proceed to **Step 2: Upload Dataset** 🚀

---
**💡 Tip**: Keep the Huawei Cloud console open - you'll use it to monitor uploads and training jobs!