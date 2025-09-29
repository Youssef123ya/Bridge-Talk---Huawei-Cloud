"""
Complete setup script for Huawei Cloud integration
Arabic Sign Language Recognition Project
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from cloud.huawei_storage import HuaweiCloudStorage, upload_arsl_dataset
from cloud.huawei_modelarts import ModelArtsManager, create_arsl_training_job
from cloud.api_deployment import HuaweiCloudDeployment, deploy_arsl_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/huawei_cloud_setup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class ARSLCloudSetup:
    """
    Complete setup manager for ARSL project on Huawei Cloud
    """
    
    def __init__(self, config_path: str = "config/huawei_cloud_config.yaml"):
        """
        Initialize cloud setup manager
        
        Args:
            config_path: Path to Huawei Cloud configuration
        """
        self.config_path = config_path
        self.setup_results = {
            'storage': {'status': 'not_started'},
            'model_training': {'status': 'not_started'},
            'deployment': {'status': 'not_started'},
            'monitoring': {'status': 'not_started'}
        }
        
        # Verify configuration exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info("ARSL Cloud Setup Manager initialized")
    
    def verify_environment(self) -> bool:
        """
        Verify that required environment variables are set
        
        Returns:
            True if environment is ready, False otherwise
        """
        required_env_vars = [
            'HUAWEI_ACCESS_KEY_ID',
            'HUAWEI_SECRET_ACCESS_KEY',
            'HUAWEI_PROJECT_ID'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            logger.error("Please set the following environment variables:")
            for var in missing_vars:
                logger.error(f"  set {var}=your_value")
            return False
        
        logger.info("Environment verification passed")
        return True
    
    def setup_storage(self, upload_data: bool = True) -> bool:
        """
        Setup Object Storage Service (OBS) and upload dataset
        
        Args:
            upload_data: Whether to upload the local dataset
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("=== Setting up Object Storage Service (OBS) ===")
        
        try:
            # Initialize storage client
            storage = HuaweiCloudStorage(self.config_path)
            
            # Create bucket
            if not storage.create_bucket():
                self.setup_results['storage'] = {
                    'status': 'failed',
                    'error': 'Failed to create bucket'
                }
                return False
            
            # Get bucket info
            bucket_info = storage.get_bucket_info()
            logger.info(f"Bucket info: {bucket_info}")
            
            # Upload dataset if requested
            if upload_data:
                logger.info("Uploading dataset to OBS...")
                if os.path.exists("data/"):
                    success = upload_arsl_dataset(storage, "data/")
                    if not success:
                        logger.warning("Dataset upload failed, but continuing...")
                else:
                    logger.warning("Local data directory not found, skipping upload")
            
            storage.close()
            
            self.setup_results['storage'] = {
                'status': 'completed',
                'bucket_info': bucket_info,
                'data_uploaded': upload_data
            }
            
            logger.info("✅ Storage setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Storage setup failed: {e}")
            self.setup_results['storage'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def setup_model_training(self, start_training: bool = False) -> Optional[str]:
        """
        Setup ModelArts for training
        
        Args:
            start_training: Whether to immediately start a training job
            
        Returns:
            Training job ID if started, None otherwise
        """
        logger.info("=== Setting up ModelArts for Model Training ===")
        
        try:
            # Initialize ModelArts manager
            modelarts = ModelArtsManager(self.config_path)
            
            # List existing training jobs
            existing_jobs = modelarts.list_training_jobs()
            logger.info(f"Found {len(existing_jobs)} existing training jobs")
            
            job_id = None
            if start_training:
                # Create and start training job
                logger.info("Starting new training job...")
                job_id = create_arsl_training_job(modelarts)
                
                if job_id:
                    logger.info(f"Training job started: {job_id}")
                    
                    # Monitor initial status
                    time.sleep(5)  # Wait a bit for job to initialize
                    status = modelarts.get_training_job_status(job_id)
                    if status:
                        logger.info(f"Job status: {status}")
                else:
                    logger.error("Failed to start training job")
            
            self.setup_results['model_training'] = {
                'status': 'completed',
                'existing_jobs': len(existing_jobs),
                'new_job_id': job_id,
                'training_started': start_training
            }
            
            logger.info("✅ ModelArts setup completed successfully")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ ModelArts setup failed: {e}")
            self.setup_results['model_training'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None
    
    def setup_deployment(self, modelarts_service_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Setup API Gateway deployment
        
        Args:
            modelarts_service_url: URL of deployed ModelArts inference service
            
        Returns:
            Deployment information
        """
        logger.info("=== Setting up API Gateway Deployment ===")
        
        try:
            # Initialize deployment manager
            deployment = HuaweiCloudDeployment(self.config_path)
            
            # If no service URL provided, create a placeholder
            if not modelarts_service_url:
                logger.warning("No ModelArts service URL provided, creating placeholder deployment")
                modelarts_service_url = "https://placeholder-service-url.com"
            
            # Deploy service
            results = deploy_arsl_service(deployment, modelarts_service_url)
            
            self.setup_results['deployment'] = {
                'status': 'completed' if results['success'] else 'failed',
                'results': results
            }
            
            if results['success']:
                logger.info("✅ API Gateway deployment completed successfully")
                logger.info(f"API Group ID: {results['api_group_id']}")
                logger.info(f"Endpoints created: {list(results['endpoints'].keys())}")
            else:
                logger.error(f"❌ API Gateway deployment failed: {results.get('error', 'Unknown error')}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Deployment setup failed: {e}")
            self.setup_results['deployment'] = {
                'status': 'failed',
                'error': str(e)
            }
            return {'success': False, 'error': str(e)}
    
    def setup_monitoring(self, api_group_id: Optional[str] = None) -> bool:
        """
        Setup monitoring and alerting
        
        Args:
            api_group_id: API Gateway group ID
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("=== Setting up Monitoring and Alerting ===")
        
        try:
            deployment = HuaweiCloudDeployment(self.config_path)
            
            # If no group ID provided, try to get from deployment results
            if not api_group_id:
                deployment_results = self.setup_results.get('deployment', {}).get('results', {})
                api_group_id = deployment_results.get('api_group_id')
            
            if not api_group_id:
                logger.error("No API group ID available for monitoring setup")
                self.setup_results['monitoring'] = {
                    'status': 'failed',
                    'error': 'No API group ID available'
                }
                return False
            
            # Setup monitoring
            success = deployment.setup_monitoring(api_group_id, "arsl-inference-service")
            
            self.setup_results['monitoring'] = {
                'status': 'completed' if success else 'failed',
                'api_group_id': api_group_id
            }
            
            if success:
                logger.info("✅ Monitoring setup completed successfully")
            else:
                logger.error("❌ Monitoring setup failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed: {e}")
            self.setup_results['monitoring'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def complete_setup(self, 
                      upload_data: bool = True,
                      start_training: bool = False,
                      modelarts_service_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete Huawei Cloud setup
        
        Args:
            upload_data: Whether to upload dataset to OBS
            start_training: Whether to start training job
            modelarts_service_url: ModelArts service URL for deployment
            
        Returns:
            Complete setup results
        """
        logger.info("🚀 Starting complete Huawei Cloud setup for ARSL project")
        
        start_time = time.time()
        
        # Verify environment
        if not self.verify_environment():
            return {'success': False, 'error': 'Environment verification failed'}
        
        # Setup storage
        storage_success = self.setup_storage(upload_data)
        
        # Setup model training
        training_job_id = self.setup_model_training(start_training)
        
        # Setup deployment
        deployment_results = self.setup_deployment(modelarts_service_url)
        
        # Setup monitoring
        monitoring_success = self.setup_monitoring()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            'success': all([
                self.setup_results['storage']['status'] == 'completed',
                self.setup_results['model_training']['status'] == 'completed',
                self.setup_results['deployment']['status'] == 'completed'
            ]),
            'setup_time_minutes': round(total_time / 60, 2),
            'components': self.setup_results,
            'next_steps': self._generate_next_steps()
        }
        
        # Save results
        self._save_setup_results(final_results)
        
        # Print summary
        self._print_setup_summary(final_results)
        
        return final_results
    
    def _generate_next_steps(self) -> list:
        """Generate next steps based on setup results"""
        next_steps = []
        
        # Check what was completed
        storage_completed = self.setup_results['storage']['status'] == 'completed'
        training_completed = self.setup_results['model_training']['status'] == 'completed'
        deployment_completed = self.setup_results['deployment']['status'] == 'completed'
        
        if storage_completed and not self.setup_results['storage'].get('data_uploaded', False):
            next_steps.append("Upload your dataset to OBS using the storage client")
        
        if training_completed and not self.setup_results['model_training'].get('training_started', False):
            next_steps.append("Start a training job using ModelArts manager")
        
        if training_completed:
            next_steps.append("Monitor training progress and wait for completion")
        
        if deployment_completed:
            next_steps.append("Deploy trained model to ModelArts inference service")
            next_steps.append("Update API Gateway with actual inference service URL")
        
        next_steps.append("Test the deployed API endpoints")
        next_steps.append("Monitor system performance using Cloud Eye dashboards")
        
        return next_steps
    
    def _save_setup_results(self, results: Dict[str, Any]):
        """Save setup results to file"""
        try:
            os.makedirs('logs', exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"logs/huawei_cloud_setup_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Setup results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save setup results: {e}")
    
    def _print_setup_summary(self, results: Dict[str, Any]):
        """Print setup summary"""
        print("\n" + "="*60)
        print("🏁 HUAWEI CLOUD SETUP SUMMARY")
        print("="*60)
        
        print(f"✅ Overall Success: {'YES' if results['success'] else 'NO'}")
        print(f"⏱️  Total Time: {results['setup_time_minutes']} minutes")
        print()
        
        print("📊 Component Status:")
        for component, info in results['components'].items():
            status = info['status']
            emoji = "✅" if status == 'completed' else "❌" if status == 'failed' else "⚠️"
            print(f"  {emoji} {component.replace('_', ' ').title()}: {status}")
        
        print()
        print("🔄 Next Steps:")
        for i, step in enumerate(results['next_steps'], 1):
            print(f"  {i}. {step}")
        
        print("\n" + "="*60)


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Huawei Cloud Setup for ARSL Project')
    parser.add_argument('--config', type=str, default='config/huawei_cloud_config.yaml',
                        help='Path to Huawei Cloud configuration file')
    parser.add_argument('--upload-data', action='store_true',
                        help='Upload dataset to OBS')
    parser.add_argument('--start-training', action='store_true',
                        help='Start training job immediately')
    parser.add_argument('--service-url', type=str,
                        help='ModelArts inference service URL for deployment')
    parser.add_argument('--component', type=str, choices=['storage', 'training', 'deployment', 'monitoring'],
                        help='Setup specific component only')
    
    args = parser.parse_args()
    
    try:
        # Initialize setup manager
        setup = ARSLCloudSetup(args.config)
        
        if args.component:
            # Setup specific component
            if args.component == 'storage':
                success = setup.setup_storage(args.upload_data)
            elif args.component == 'training':
                job_id = setup.setup_model_training(args.start_training)
                success = job_id is not None
            elif args.component == 'deployment':
                results = setup.setup_deployment(args.service_url)
                success = results['success']
            elif args.component == 'monitoring':
                success = setup.setup_monitoring()
            
            print(f"Component '{args.component}' setup: {'SUCCESS' if success else 'FAILED'}")
        else:
            # Complete setup
            results = setup.complete_setup(
                upload_data=args.upload_data,
                start_training=args.start_training,
                modelarts_service_url=args.service_url
            )
            
            if results['success']:
                print("🎉 Complete setup finished successfully!")
            else:
                print("❌ Setup encountered errors. Check logs for details.")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()