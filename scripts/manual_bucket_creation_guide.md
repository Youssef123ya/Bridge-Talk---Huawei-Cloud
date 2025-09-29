# 📋 **Manual Bucket Creation Guide for yyacoup Account**

## 🔐 **Your Account Details**
- **IAM Username**: `yyacoup`
- **Account Name**: `yyacoup`
- **IAM User ID**: `d39414080c594b3296c5490459fde0e0`
- **Account ID**: `7a6d7cdc0fec4b9cbfd3ae7f6fbea19e`
- **Region**: `AF-Cairo (af-north-1)`

## 🌐 **Step-by-Step Bucket Creation**

### **1. Login to Huawei Cloud Console**
1. Go to: https://console.huaweicloud.com
2. Login with:
   - **Username**: `yyacoup`
   - **Password**: [Your password]
3. Ensure you're in the **AF-Cairo** region (top-right corner)

### **2. Navigate to Object Storage Service**
1. From the console dashboard, click **"Storage"** → **"Object Storage Service"**
2. Or search for **"OBS"** in the search bar
3. Click on **"Object Storage Service"**

### **3. Create New Bucket**
1. Click **"Create Bucket"** button
2. Fill in the details:

```yaml
Basic Information:
  Bucket Name: arsl-youssef-af-cairo-2025
  Region: AF-Cairo
  Storage Class: Standard
  
Access Control:
  Bucket ACL: Private
  Bucket Policy: None (default)
  
Advanced Settings:
  Versioning: Disabled
  Logging: Disabled
  Event Notification: Disabled
  Cross-Region Replication: Disabled
  Tags: (Optional)
    - Key: Project
    - Value: Arabic-Sign-Language
```

### **4. Configure Bucket Policies (Important!)**
After bucket creation:
1. Go to the bucket → **"Permissions"** tab
2. Click **"Bucket Policies"**
3. Add this policy for your API access:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowAPIAccess",
      "Effect": "Allow",
      "Principal": {
        "ID": ["domain/7a6d7cdc0fec4b9cbfd3ae7f6fbea19e:user/yyacoup"]
      },
      "Action": [
        "obs:object:GetObject",
        "obs:object:PutObject",
        "obs:object:DeleteObject",
        "obs:bucket:ListBucket"
      ],
      "Resource": [
        "arsl-youssef-af-cairo-2025/*",
        "arsl-youssef-af-cairo-2025"
      ]
    }
  ]
}
```

### **5. Create Folder Structure**
Create these folders in your bucket:
1. Click **"Create Folder"** and create:
   - `data/`
   - `data/raw/`
   - `data/processed/`
   - `models/`
   - `output/`
   - `logs/`

## ✅ **Verification Checklist**
- [ ] Bucket `arsl-youssef-af-cairo-2025` created
- [ ] Region set to `AF-Cairo`
- [ ] Storage class is `Standard`
- [ ] Access control is `Private`
- [ ] Bucket policy configured
- [ ] Folder structure created
- [ ] You can see the bucket in OBS console

## 🚨 **Common Issues & Solutions**

### **Issue**: Can't see "Create Bucket" button
**Solution**: Ensure you have OBS permissions in IAM:
1. Go to **IAM** → **Users** → **yyacoup**
2. Check permissions include: `OBS Administrator` or `OBS OperateAccess`

### **Issue**: Bucket name already exists
**Solution**: Try these alternatives:
- `arsl-youssef-af-cairo-2025-v2`
- `arsl-yyacoup-af-cairo-2025`
- `arsl-recognition-yyacoup-2025`

### **Issue**: Region not showing AF-Cairo
**Solution**: 
1. Check top-right corner region selector
2. Switch to **AF-Cairo** region
3. Refresh the page

## 📞 **Need Help?**
If you encounter any issues:
1. Check IAM permissions for OBS access
2. Verify you're in the correct region (AF-Cairo)
3. Contact Huawei Cloud support if needed

---

**🎯 Once completed, return to VS Code and we'll verify the bucket connection!**