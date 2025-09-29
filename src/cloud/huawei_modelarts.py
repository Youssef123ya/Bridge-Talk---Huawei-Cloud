"""
Huawei Cloud ModelArts Integration
For Arabic Sign Language Recognition Project
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import time
from datetime import datetime

try:
    from huaweicloudsdkcore.auth.credentials import BasicCredentials
    from huaweicloudsdkcore.client import Client
    from huaweicloudsdkcore.http.http_config import HttpConfig
    from huaweicloudsdkmodelarts.v1.region.modelarts_region import ModelArtsRegion
    from huaweicloudsdkmodelarts.v1 import *
except ImportError:
    print("Huawei Cloud SDK not installed. Install with: pip install huaweicloudsdkmodelarts")


class ModelArtsManager:
    """
    Huawei Cloud ModelArts manager for training and deploying ML models
    """
    
    def __init__(self, config_path: str = "config/huawei_cloud_config.yaml"):
        """
        Initialize ModelArts client
        
        Args:
            config_path: Path to Huawei Cloud configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.client = self._create_modelarts_client()
        
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
        """Setup logging for ModelArts operations"""
        logger = logging.getLogger('ModelArtsManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_modelarts_client(self):
        """Create ModelArts client"""
        try:
            auth = self.config['auth']
            
            credentials = BasicCredentials(
                ak=auth['access_key_id'],
                sk=auth['secret_access_key'],
                project_id=auth['project_id']
            )
            
            config = HttpConfig.get_default_config()
            config.ignore_ssl_verification = True
            
            client = ModelArtsClient.new_builder() \
                .with_credentials(credentials) \
                .with_region(ModelArtsRegion.value_of(auth['region'])) \
                .with_http_config(config) \
                .build()
            
            self.logger.info(f"Connected to ModelArts in region: {auth['region']}")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to create ModelArts client: {e}")
            return None
    
    def create_training_job(self, 
                           job_name: str,
                           code_dir: str,
                           boot_file: str,
                           data_source: str,
                           output_path: str,
                           hyperparameters: Optional[Dict[str, Any]] = None,
                           compute_type: Optional[str] = None) -> Optional[str]:
        """
        Create a training job in ModelArts
        
        Args:
            job_name: Name of the training job
            code_dir: OBS path to training code
            boot_file: Main training script file
            data_source: OBS path to training data
            output_path: OBS path for outputs
            hyperparameters: Training hyperparameters
            compute_type: Compute instance type
            
        Returns:
            Job ID if successful, None otherwise
        """
        try:
            if not self.client:
                self.logger.error("ModelArts client not initialized")
                return None
            
            modelarts_config = self.config['modelarts']['training']
            
            # Use provided parameters or defaults from config
            compute_type = compute_type or modelarts_config['compute_type']
            
            # Default hyperparameters for ARSL project
            default_hyperparams = {
                'epochs': '50',
                'batch_size': '32',
                'learning_rate': '0.001',
                'num_classes': '32',
                'image_size': '64'
            }
            
            if hyperparameters:
                default_hyperparams.update(hyperparameters)
            
            # Convert hyperparameters to ModelArts format
            hyperparams_list = [
                TrainingJobParameter(label=k, value=str(v))
                for k, v in default_hyperparams.items()
            ]
            
            # Create training job specification
            spec = TrainingJobSpec(
                algorithm=TrainingJobAlgorithm(
                    code_dir=code_dir,
                    boot_file=boot_file,
                    parameters=hyperparams_list
                ),
                inputs=[
                    TrainingJobDataSource(
                        data_source=TrainingJobDataSourceDataSource(
                            type="obs",
                            data_url=data_source
                        ),
                        data_name="data_url"
                    )
                ],
                outputs=[
                    TrainingJobDataSource(
                        data_source=TrainingJobDataSourceDataSource(
                            type="obs",
                            data_url=output_path
                        ),
                        data_name="train_url"
                    )
                ],
                resource=TrainingJobResource(
                    flavor=compute_type,
                    node_count=1
                ),
                config=TrainingJobConfig(
                    framework_type=modelarts_config['framework'],
                    framework_version="1.8.0"
                )
            )
            
            # Create training job request
            request = CreateTrainingJobRequest(
                body=TrainingJob(
                    metadata=TrainingJobMetadata(name=job_name),
                    spec=spec
                )
            )
            
            # Submit training job
            response = self.client.create_training_job(request)
            
            if response.status_code == 200:
                job_id = response.metadata.name
                self.logger.info(f"Training job created successfully: {job_id}")
                return job_id
            else:
                self.logger.error(f"Failed to create training job: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating training job: {e}")
            return None
    
    def get_training_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get training job status and details
        
        Args:
            job_id: Training job ID
            
        Returns:
            Job status information
        """
        try:
            if not self.client:
                return None
            
            request = ShowTrainingJobRequest(job_id=job_id)
            response = self.client.show_training_job(request)
            
            if response.status_code == 200:
                job = response
                status_info = {
                    'job_id': job_id,
                    'status': job.status.phase,
                    'created_at': job.metadata.creation_timestamp,
                    'duration': job.status.duration,
                    'resource': {
                        'flavor': job.spec.resource.flavor,
                        'node_count': job.spec.resource.node_count
                    }
                }
                
                if hasattr(job.status, 'conditions') and job.status.conditions:
                    status_info['message'] = job.status.conditions[-1].message
                
                return status_info
            else:
                self.logger.error(f"Failed to get job status: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting job status: {e}")
            return None
    
    def list_training_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List training jobs
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of training job information
        """
        try:
            if not self.client:
                return []
            
            request = ListTrainingJobsRequest(limit=limit)
            response = self.client.list_training_jobs(request)
            
            if response.status_code == 200:
                jobs = []
                for job in response.items:
                    job_info = {
                        'job_id': job.metadata.name,
                        'status': job.status.phase,
                        'created_at': job.metadata.creation_timestamp,
                        'algorithm': job.spec.algorithm.code_dir if job.spec.algorithm else None
                    }
                    jobs.append(job_info)
                
                return jobs
            else:
                self.logger.error(f"Failed to list training jobs: {response}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error listing training jobs: {e}")
            return []
    
    def stop_training_job(self, job_id: str) -> bool:
        """
        Stop a running training job
        
        Args:
            job_id: Training job ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                return False
            
            request = DeleteTrainingJobRequest(job_id=job_id)
            response = self.client.delete_training_job(request)
            
            if response.status_code == 200:
                self.logger.info(f"Training job stopped: {job_id}")
                return True
            else:
                self.logger.error(f"Failed to stop training job: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error stopping training job: {e}")
            return False
    
    def create_model(self, 
                    model_name: str,
                    model_source: str,
                    description: str = "Arabic Sign Language Recognition Model") -> Optional[str]:
        """
        Create a model in ModelArts from training output
        
        Args:
            model_name: Name for the model
            model_source: OBS path to model artifacts
            description: Model description
            
        Returns:
            Model ID if successful, None otherwise
        """
        try:
            if not self.client:
                return None
            
            # Create model specification
            spec = ModelSpec(
                model_source=ModelSource(
                    source_type="obs",
                    source_location=model_source
                ),
                description=description,
                model_type="pytorch",
                runtime="python3.7"
            )
            
            request = CreateModelRequest(
                body=Model(
                    metadata=ModelMetadata(name=model_name),
                    spec=spec
                )
            )
            
            response = self.client.create_model(request)
            
            if response.status_code == 200:
                model_id = response.metadata.name
                self.logger.info(f"Model created successfully: {model_id}")
                return model_id
            else:
                self.logger.error(f"Failed to create model: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            return None
    
    def deploy_model(self, 
                    model_id: str,
                    service_name: str,
                    instance_count: int = 1,
                    compute_type: Optional[str] = None) -> Optional[str]:
        """
        Deploy a model as an inference service
        
        Args:
            model_id: Model ID to deploy
            service_name: Name for the inference service
            instance_count: Number of instances
            compute_type: Compute instance type
            
        Returns:
            Service ID if successful, None otherwise
        """
        try:
            if not self.client:
                return None
            
            modelarts_config = self.config['modelarts']['inference']
            compute_type = compute_type or modelarts_config['compute_type']
            
            # Create service specification
            spec = ServiceSpec(
                model_id=model_id,
                replicas=instance_count,
                resources=ServiceResources(
                    flavor=compute_type
                )
            )
            
            request = CreateServiceRequest(
                body=Service(
                    metadata=ServiceMetadata(name=service_name),
                    spec=spec
                )
            )
            
            response = self.client.create_service(request)
            
            if response.status_code == 200:
                service_id = response.metadata.name
                self.logger.info(f"Model deployed successfully: {service_id}")
                return service_id
            else:
                self.logger.error(f"Failed to deploy model: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error deploying model: {e}")
            return None
    
    def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """
        Get inference service status
        
        Args:
            service_id: Service ID
            
        Returns:
            Service status information
        """
        try:
            if not self.client:
                return None
            
            request = ShowServiceRequest(service_id=service_id)
            response = self.client.show_service(request)
            
            if response.status_code == 200:
                service = response
                status_info = {
                    'service_id': service_id,
                    'status': service.status.phase,
                    'replicas': service.spec.replicas,
                    'access_address': service.status.access_address if hasattr(service.status, 'access_address') else None,
                    'created_at': service.metadata.creation_timestamp
                }
                
                return status_info
            else:
                self.logger.error(f"Failed to get service status: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting service status: {e}")
            return None


def create_arsl_training_job(manager: ModelArtsManager,
                            job_name: Optional[str] = None) -> Optional[str]:
    """
    Create a training job specifically for Arabic Sign Language Recognition
    
    Args:
        manager: Initialized ModelArtsManager
        job_name: Custom job name (auto-generated if None)
        
    Returns:
        Job ID if successful, None otherwise
    """
    if not job_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"arsl-training-{timestamp}"
    
    # Hyperparameters optimized for ARSL dataset
    hyperparameters = {
        'epochs': '100',
        'batch_size': '64',
        'learning_rate': '0.001',
        'num_classes': '32',
        'image_size': '64',
        'dropout': '0.5',
        'weight_decay': '0.0001'
    }
    
    # OBS paths from config
    obs_config = manager.config['obs']
    bucket_name = obs_config['bucket_name']
    
    job_id = manager.create_training_job(
        job_name=job_name,
        code_dir=f"obs://{bucket_name}/code/",
        boot_file="train_arsl.py",
        data_source=f"obs://{bucket_name}/datasets/",
        output_path=f"obs://{bucket_name}/output/{job_name}/",
        hyperparameters=hyperparameters
    )
    
    return job_id


if __name__ == "__main__":
    # Example usage
    manager = ModelArtsManager()
    
    # List existing training jobs
    jobs = manager.list_training_jobs()
    print(f"Found {len(jobs)} training jobs:")
    for job in jobs:
        print(f"  - {job['job_id']}: {job['status']}")
    
    # Create new training job (uncomment to use)
    # job_id = create_arsl_training_job(manager)
    # if job_id:
    #     print(f"Created training job: {job_id}")