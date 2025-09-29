"""
Smart Dataset Upload Script for Huawei Cloud OBS
Arabic Sign Language Recognition Project
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from cloud.huawei_storage import HuaweiCloudStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/dataset_upload_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class DatasetUploader:
    """
    Smart uploader for ARSL dataset to Huawei Cloud OBS
    """
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize uploader
        
        Args:
            data_path: Local path to dataset
        """
        self.data_path = Path(data_path)
        self.storage = HuaweiCloudStorage()
        self.stats = {
            'files_found': 0,
            'files_uploaded': 0,
            'files_failed': 0,
            'bytes_uploaded': 0,
            'start_time': None,
            'errors': []
        }
        
    def discover_files(self) -> List[Dict[str, str]]:
        """
        Discover all files to upload
        
        Returns:
            List of file information
        """
        logger.info("🔍 Discovering files to upload...")
        
        files_to_upload = []
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        
        # Scan data directory
        for file_path in self.data_path.rglob('*'):
            if file_path.is_file():
                # Check if it's an image or important file
                if (file_path.suffix.lower() in image_extensions or 
                    file_path.name.endswith('.csv') or
                    file_path.name.endswith('.txt')):
                    
                    # Calculate relative path from data root
                    relative_path = file_path.relative_to(self.data_path)
                    obs_path = f"datasets/{relative_path.as_posix()}"
                    
                    files_to_upload.append({
                        'local_path': str(file_path),
                        'obs_path': obs_path,
                        'size': file_path.stat().st_size,
                        'name': file_path.name
                    })
        
        self.stats['files_found'] = len(files_to_upload)
        
        # Sort by size (upload small files first for quick feedback)
        files_to_upload.sort(key=lambda x: x['size'])
        
        logger.info(f"📊 Found {len(files_to_upload)} files to upload")
        
        # Show size distribution
        total_size = sum(f['size'] for f in files_to_upload)
        logger.info(f"📦 Total size: {total_size / (1024*1024):.1f} MB")
        
        return files_to_upload
    
    def upload_file(self, file_info: Dict[str, str]) -> bool:
        """
        Upload a single file
        
        Args:
            file_info: File information dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.storage.upload_file(
                local_path=file_info['local_path'],
                obs_path=file_info['obs_path']
            )
            
            if success:
                self.stats['files_uploaded'] += 1
                self.stats['bytes_uploaded'] += file_info['size']
            else:
                self.stats['files_failed'] += 1
                self.stats['errors'].append(f"Failed to upload: {file_info['name']}")
            
            return success
            
        except Exception as e:
            self.stats['files_failed'] += 1
            error_msg = f"Error uploading {file_info['name']}: {str(e)}"
            self.stats['errors'].append(error_msg)
            logger.error(error_msg)
            return False
    
    def upload_with_progress(self, files_to_upload: List[Dict[str, str]], 
                           max_workers: int = 8) -> bool:
        """
        Upload files with progress tracking and parallel processing
        
        Args:
            files_to_upload: List of files to upload
            max_workers: Number of parallel upload threads
            
        Returns:
            True if majority of uploads successful
        """
        logger.info(f"🚀 Starting upload with {max_workers} parallel workers...")
        
        self.stats['start_time'] = time.time()
        
        # Create bucket if needed
        if not self.storage.create_bucket():
            logger.error("❌ Failed to create/access bucket")
            return False
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload jobs
            future_to_file = {
                executor.submit(self.upload_file, file_info): file_info 
                for file_info in files_to_upload
            }
            
            # Process completed uploads with progress reporting
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_info = future_to_file[future]
                
                try:
                    success = future.result()
                    
                    # Progress reporting every 100 files or at milestones
                    if i % 100 == 0 or i in [1, 10, 50, 100, 500, 1000]:
                        self._print_progress(i, len(files_to_upload))
                    
                except Exception as e:
                    logger.error(f"❌ Upload failed for {file_info['name']}: {e}")
                    self.stats['files_failed'] += 1
        
        # Final progress report
        self._print_progress(len(files_to_upload), len(files_to_upload))
        
        success_rate = self.stats['files_uploaded'] / self.stats['files_found'] if self.stats['files_found'] > 0 else 0
        return success_rate > 0.8  # Consider successful if >80% uploaded
    
    def _print_progress(self, completed: int, total: int):
        """Print upload progress"""
        if self.stats['start_time'] is None:
            return
            
        elapsed = time.time() - self.stats['start_time']
        progress = (completed / total) * 100
        
        if elapsed > 0:
            files_per_sec = completed / elapsed
            eta_seconds = (total - completed) / files_per_sec if files_per_sec > 0 else 0
            eta_mins = eta_seconds / 60
        else:
            files_per_sec = 0
            eta_mins = 0
        
        mb_uploaded = self.stats['bytes_uploaded'] / (1024 * 1024)
        
        logger.info(
            f"📈 Progress: {completed}/{total} ({progress:.1f}%) | "
            f"Speed: {files_per_sec:.1f} files/sec | "
            f"Data: {mb_uploaded:.1f} MB | "
            f"ETA: {eta_mins:.1f} min | "
            f"Failed: {self.stats['files_failed']}"
        )
    
    def verify_upload(self) -> Dict[str, any]:
        """
        Verify upload completion and integrity
        
        Returns:
            Verification report
        """
        logger.info("🔍 Verifying upload...")
        
        # List objects in bucket
        objects = self.storage.list_objects("datasets/")
        
        report = {
            'local_files': self.stats['files_found'],
            'uploaded_files': len(objects),
            'missing_files': [],
            'upload_success_rate': self.stats['files_uploaded'] / self.stats['files_found'] if self.stats['files_found'] > 0 else 0,
            'total_size_mb': self.stats['bytes_uploaded'] / (1024 * 1024),
            'errors': self.stats['errors'][:10]  # Show first 10 errors
        }
        
        logger.info(f"📊 Verification Report:")
        logger.info(f"   Local files: {report['local_files']}")
        logger.info(f"   Uploaded files: {report['uploaded_files']}")
        logger.info(f"   Success rate: {report['upload_success_rate']:.2%}")
        logger.info(f"   Total uploaded: {report['total_size_mb']:.1f} MB")
        
        return report
    
    def save_upload_report(self, report: Dict[str, any]):
        """Save upload report to file"""
        os.makedirs('logs', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"logs/upload_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Upload report saved: {report_path}")
    
    def close(self):
        """Close storage connection"""
        self.storage.close()


def main():
    """Main upload function"""
    print("🌟 Arabic Sign Language Dataset Upload to Huawei Cloud")
    print("=" * 60)
    
    # Check if data directory exists
    data_path = "data/"
    if not os.path.exists(data_path):
        print(f"❌ Data directory not found: {data_path}")
        print("Please ensure your dataset is in the 'data/' folder")
        return False
    
    try:
        # Initialize uploader
        uploader = DatasetUploader(data_path)
        
        # Discover files
        files_to_upload = uploader.discover_files()
        
        if not files_to_upload:
            print("❌ No files found to upload")
            return False
        
        # Confirm upload
        total_size_mb = sum(f['size'] for f in files_to_upload) / (1024 * 1024)
        print(f"\n📊 Upload Summary:")
        print(f"   Files to upload: {len(files_to_upload)}")
        print(f"   Total size: {total_size_mb:.1f} MB")
        print(f"   Estimated time: {len(files_to_upload) / 100:.0f} - {len(files_to_upload) / 50:.0f} minutes")
        
        # Ask for confirmation
        confirm = input("\n🤔 Proceed with upload? (y/N): ").lower().strip()
        if confirm != 'y':
            print("❌ Upload cancelled by user")
            return False
        
        # Start upload
        print("\n🚀 Starting upload...")
        success = uploader.upload_with_progress(files_to_upload)
        
        # Verify upload
        report = uploader.verify_upload()
        uploader.save_upload_report(report)
        
        # Final status
        if success:
            print("\n✅ Upload completed successfully!")
            print(f"📊 {report['uploaded_files']} files uploaded ({report['upload_success_rate']:.1%} success rate)")
            print(f"📦 {report['total_size_mb']:.1f} MB transferred")
        else:
            print("\n⚠️ Upload completed with issues")
            print(f"📊 {report['uploaded_files']} files uploaded ({report['upload_success_rate']:.1%} success rate)")
            if report['errors']:
                print("❌ Some errors occurred - check the log file")
        
        # Cleanup
        uploader.close()
        
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️ Upload interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)