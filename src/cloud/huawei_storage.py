"""
Huawei Cloud Object Storage Service (OBS) Integration
For Arabic Sign Language Recognition Project
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import yaml
try:
    from obs import ObsClient
    OBS_AVAILABLE = True
except ImportError:
    print("OBS SDK not properly installed. Please install: pip install esdk-obs-python")
    ObsClient = None
    OBS_AVAILABLE = False
import mimetypes
from tqdm import tqdm

class HuaweiCloudStorage:
    """
    Huawei Cloud Object Storage Service (OBS) client for dataset and model management
    """
    
    def __init__(self, config_path: str = "config/huawei_cloud_config.yaml"):
        """
        Initialize OBS client with configuration
        
        Args:
            config_path: Path to Huawei Cloud configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.client = self._create_obs_client()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Huawei Cloud configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Replace environment variables
            auth_config = config['auth']
            for key, value in auth_config.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    auth_config[key] = os.getenv(env_var)
                    if not auth_config[key]:
                        raise ValueError(f"Environment variable {env_var} not set")
            
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for OBS operations"""
        logger = logging.getLogger('HuaweiCloudStorage')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_obs_client(self):
        """Create and configure OBS client"""
        try:
            if not OBS_AVAILABLE:
                raise RuntimeError("OBS SDK not available")
            
            # Configure OBS SDK logging (skip if not available)
            try:
                from obs.log.log_conf import LogConf, INFO
                LogConf.set_log_level(INFO)
            except ImportError:
                pass  # Skip logging configuration if not available
            
            auth = self.config['auth']
            obs_config = self.config['obs']
            
            # Get endpoint based on region
            endpoint = obs_config['endpoints'].get(
                auth['region'], 
                f"obs.{auth['region']}.myhuaweicloud.com"
            )
            
            client = ObsClient(
                access_key_id=auth['access_key_id'],
                secret_access_key=auth['secret_access_key'],
                server=f"https://{endpoint}"
            )
            
            self.logger.info(f"Connected to OBS in region: {auth['region']}")
            return client
            
        except Exception as e:
            raise RuntimeError(f"Failed to create OBS client: {e}")
    
    def create_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Create OBS bucket if it doesn't exist
        
        Args:
            bucket_name: Name of bucket to create (uses config default if None)
            
        Returns:
            True if bucket created or already exists, False otherwise
        """
        if not bucket_name:
            bucket_name = self.config['obs']['bucket_name']
        
        try:
            # Check if bucket exists
            resp = self.client.headBucket(bucket_name)
            if resp.status < 300:
                self.logger.info(f"Bucket '{bucket_name}' already exists")
                return True
        except:
            pass
        
        try:
            # Create bucket
            location = self.config['auth']['region']
            resp = self.client.createBucket(bucket_name, location=location)
            
            if resp.status < 300:
                self.logger.info(f"Successfully created bucket: {bucket_name}")
                return True
            else:
                self.logger.error(f"Failed to create bucket: {resp.reason}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating bucket: {e}")
            return False
    
    def upload_file(self, local_path: str, obs_path: str, 
                   bucket_name: Optional[str] = None) -> bool:
        """
        Upload a file to OBS
        
        Args:
            local_path: Local file path
            obs_path: OBS object key (path in bucket)
            bucket_name: Bucket name (uses config default if None)
            
        Returns:
            True if upload successful, False otherwise
        """
        if not bucket_name:
            bucket_name = self.config['obs']['bucket_name']
        
        try:
            # Ensure file exists
            if not os.path.exists(local_path):
                self.logger.error(f"Local file not found: {local_path}")
                return False
            
            # Get file content type
            content_type, _ = mimetypes.guess_type(local_path)
            if not content_type:
                content_type = 'application/octet-stream'
            
            # Upload file
            resp = self.client.putFile(
                bucketName=bucket_name,
                objectKey=obs_path,
                file_path=local_path,
                metadata={'Content-Type': content_type}
            )
            
            if resp.status < 300:
                self.logger.info(f"Successfully uploaded: {local_path} -> obs://{bucket_name}/{obs_path}")
                return True
            else:
                self.logger.error(f"Upload failed: {resp.reason}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error uploading file: {e}")
            return False
    
    def upload_directory(self, local_dir: str, obs_prefix: str,
                        bucket_name: Optional[str] = None,
                        file_extensions: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Upload entire directory to OBS with progress tracking
        
        Args:
            local_dir: Local directory path
            obs_prefix: OBS prefix (directory path in bucket)
            bucket_name: Bucket name (uses config default if None)
            file_extensions: List of file extensions to include (e.g., ['.jpg', '.png'])
            
        Returns:
            Dictionary with upload statistics
        """
        if not bucket_name:
            bucket_name = self.config['obs']['bucket_name']
        
        stats = {'uploaded': 0, 'failed': 0, 'skipped': 0}
        
        try:
            local_path = Path(local_dir)
            if not local_path.exists():
                self.logger.error(f"Local directory not found: {local_dir}")
                return stats
            
            # Get all files to upload
            all_files = []
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    if file_extensions:
                        if file_path.suffix.lower() in file_extensions:
                            all_files.append(file_path)
                        else:
                            stats['skipped'] += 1
                    else:
                        all_files.append(file_path)
            
            self.logger.info(f"Found {len(all_files)} files to upload")
            
            # Upload files with progress bar
            with tqdm(total=len(all_files), desc="Uploading files") as pbar:
                for file_path in all_files:
                    # Calculate relative path and OBS key
                    rel_path = file_path.relative_to(local_path)
                    obs_key = f"{obs_prefix.rstrip('/')}/{rel_path.as_posix()}"
                    
                    if self.upload_file(str(file_path), obs_key, bucket_name):
                        stats['uploaded'] += 1
                    else:
                        stats['failed'] += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'uploaded': stats['uploaded'],
                        'failed': stats['failed']
                    })
            
            self.logger.info(f"Upload completed: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error uploading directory: {e}")
            return stats
    
    def download_file(self, obs_path: str, local_path: str,
                     bucket_name: Optional[str] = None) -> bool:
        """
        Download a file from OBS
        
        Args:
            obs_path: OBS object key
            local_path: Local file path to save
            bucket_name: Bucket name (uses config default if None)
            
        Returns:
            True if download successful, False otherwise
        """
        if not bucket_name:
            bucket_name = self.config['obs']['bucket_name']
        
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            resp = self.client.getObject(
                bucketName=bucket_name,
                objectKey=obs_path,
                downloadPath=local_path
            )
            
            if resp.status < 300:
                self.logger.info(f"Successfully downloaded: obs://{bucket_name}/{obs_path} -> {local_path}")
                return True
            else:
                self.logger.error(f"Download failed: {resp.reason}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            return False
    
    def list_objects(self, prefix: str = "", bucket_name: Optional[str] = None,
                    max_keys: int = 1000) -> List[str]:
        """
        List objects in OBS bucket
        
        Args:
            prefix: Object key prefix to filter
            bucket_name: Bucket name (uses config default if None)
            max_keys: Maximum number of objects to return
            
        Returns:
            List of object keys
        """
        if not bucket_name:
            bucket_name = self.config['obs']['bucket_name']
        
        try:
            resp = self.client.listObjects(
                bucketName=bucket_name,
                prefix=prefix,
                max_keys=max_keys
            )
            
            if resp.status < 300:
                objects = [obj.key for obj in resp.body.contents] if resp.body.contents else []
                self.logger.info(f"Found {len(objects)} objects with prefix '{prefix}'")
                return objects
            else:
                self.logger.error(f"Failed to list objects: {resp.reason}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error listing objects: {e}")
            return []
    
    def delete_object(self, obs_path: str, bucket_name: Optional[str] = None) -> bool:
        """
        Delete an object from OBS
        
        Args:
            obs_path: OBS object key
            bucket_name: Bucket name (uses config default if None)
            
        Returns:
            True if deletion successful, False otherwise
        """
        if not bucket_name:
            bucket_name = self.config['obs']['bucket_name']
        
        try:
            resp = self.client.deleteObject(
                bucketName=bucket_name,
                objectKey=obs_path
            )
            
            if resp.status < 300:
                self.logger.info(f"Successfully deleted: obs://{bucket_name}/{obs_path}")
                return True
            else:
                self.logger.error(f"Delete failed: {resp.reason}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting object: {e}")
            return False
    
    def get_bucket_info(self, bucket_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get bucket information and statistics
        
        Args:
            bucket_name: Bucket name (uses config default if None)
            
        Returns:
            Dictionary with bucket information
        """
        if not bucket_name:
            bucket_name = self.config['obs']['bucket_name']
        
        try:
            # Get bucket metadata
            resp = self.client.getBucketMetadata(bucket_name)
            info = {
                'name': bucket_name,
                'exists': resp.status < 300,
                'region': self.config['auth']['region']
            }
            
            if info['exists']:
                # Count objects by category
                paths = self.config['obs']['paths']
                for category, prefix in paths.items():
                    objects = self.list_objects(prefix, bucket_name)
                    info[f'{category}_count'] = len(objects)
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting bucket info: {e}")
            return {'name': bucket_name, 'exists': False, 'error': str(e)}
    
    def close(self):
        """Close OBS client connection"""
        if self.client:
            self.client.close()
            self.logger.info("OBS client connection closed")


# Example usage and utility functions
def upload_arsl_dataset(storage_client: HuaweiCloudStorage, 
                       local_data_path: str = "data/") -> bool:
    """
    Upload the Arabic Sign Language dataset to OBS
    
    Args:
        storage_client: Initialized HuaweiCloudStorage client
        local_data_path: Local path to data directory
        
    Returns:
        True if upload successful, False otherwise
    """
    try:
        # Create bucket if needed
        if not storage_client.create_bucket():
            return False
        
        # Upload raw dataset
        raw_path = os.path.join(local_data_path, "raw")
        if os.path.exists(raw_path):
            print("Uploading raw dataset...")
            stats = storage_client.upload_directory(
                raw_path, 
                "datasets/raw/",
                file_extensions=['.jpg', '.jpeg', '.png', '.bmp']
            )
            print(f"Raw dataset upload: {stats}")
        
        # Upload labels file
        labels_file = os.path.join(local_data_path, "ArSL_Data_Labels.csv")
        if os.path.exists(labels_file):
            print("Uploading labels file...")
            storage_client.upload_file(labels_file, "datasets/ArSL_Data_Labels.csv")
        
        # Upload processed data if exists
        processed_path = os.path.join(local_data_path, "processed")
        if os.path.exists(processed_path):
            print("Uploading processed data...")
            stats = storage_client.upload_directory(processed_path, "datasets/processed/")
            print(f"Processed data upload: {stats}")
        
        return True
        
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    storage = HuaweiCloudStorage()
    
    # Get bucket info
    bucket_info = storage.get_bucket_info()
    print("Bucket Info:", bucket_info)
    
    # Upload dataset (uncomment to use)
    # upload_arsl_dataset(storage)
    
    storage.close()