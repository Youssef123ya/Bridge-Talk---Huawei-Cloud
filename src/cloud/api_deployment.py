"""
API Gateway integration for Arabic Sign Language Recognition
Huawei Cloud deployment utilities
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import time
from datetime import datetime
import yaml

try:
    from huaweicloudsdkcore.auth.credentials import BasicCredentials
    from huaweicloudsdkcore.client import Client
    from huaweicloudsdkcore.http.http_config import HttpConfig
    from huaweicloudsdkapig.v2.region.apig_region import ApigRegion
    from huaweicloudsdkapig.v2 import *
    from huaweicloudsdkces.v1.region.ces_region import CesRegion
    from huaweicloudsdkces.v1 import *
except ImportError:
    print("Huawei Cloud SDK not installed. Install with: pip install huaweicloudsdkapig huaweicloudsdkces")

logger = logging.getLogger(__name__)


class HuaweiCloudDeployment:
    """
    Huawei Cloud deployment manager for ARSL API
    """
    
    def __init__(self, config_path: str = "config/huawei_cloud_config.yaml"):
        """
        Initialize deployment manager
        
        Args:
            config_path: Path to Huawei Cloud configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.apig_client = self._create_apig_client()
        self.ces_client = self._create_ces_client()
    
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
        """Setup logging"""
        logger = logging.getLogger('HuaweiCloudDeployment')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_apig_client(self):
        """Create API Gateway client"""
        try:
            auth = self.config['auth']
            
            credentials = BasicCredentials(
                ak=auth['access_key_id'],
                sk=auth['secret_access_key'],
                project_id=auth['project_id']
            )
            
            config = HttpConfig.get_default_config()
            config.ignore_ssl_verification = True
            
            client = ApigClient.new_builder() \
                .with_credentials(credentials) \
                .with_region(ApigRegion.value_of(auth['region'])) \
                .with_http_config(config) \
                .build()
            
            self.logger.info(f"Connected to API Gateway in region: {auth['region']}")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to create API Gateway client: {e}")
            return None
    
    def _create_ces_client(self):
        """Create Cloud Eye client for monitoring"""
        try:
            auth = self.config['auth']
            
            credentials = BasicCredentials(
                ak=auth['access_key_id'],
                sk=auth['secret_access_key'],
                project_id=auth['project_id']
            )
            
            config = HttpConfig.get_default_config()
            config.ignore_ssl_verification = True
            
            client = CesClient.new_builder() \
                .with_credentials(credentials) \
                .with_region(CesRegion.value_of(auth['region'])) \
                .with_http_config(config) \
                .build()
            
            self.logger.info(f"Connected to Cloud Eye in region: {auth['region']}")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to create Cloud Eye client: {e}")
            return None
    
    def create_api_group(self, group_name: Optional[str] = None) -> Optional[str]:
        """
        Create API group for ARSL endpoints
        
        Args:
            group_name: Name for API group
            
        Returns:
            Group ID if successful, None otherwise
        """
        try:
            if not self.apig_client:
                return None
            
            api_config = self.config['api_gateway']
            group_name = group_name or api_config['group_name']
            
            # Create API group
            request = CreateApiGroupV2Request(
                body=ApiGroupCreate(
                    name=group_name,
                    description="Arabic Sign Language Recognition APIs",
                    status=1  # Enable
                )
            )
            
            response = self.apig_client.create_api_group_v2(request)
            
            if response.status_code == 201:
                group_id = response.id
                self.logger.info(f"API group created: {group_id}")
                return group_id
            else:
                self.logger.error(f"Failed to create API group: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating API group: {e}")
            return None
    
    def create_api_endpoints(self, group_id: str, backend_url: str) -> Dict[str, str]:
        """
        Create API endpoints for ARSL service
        
        Args:
            group_id: API group ID
            backend_url: Backend service URL (ModelArts inference endpoint)
            
        Returns:
            Dictionary of endpoint names to API IDs
        """
        endpoints = {}
        
        try:
            if not self.apig_client:
                return endpoints
            
            # Predict endpoint
            predict_api = self._create_predict_api(group_id, backend_url)
            if predict_api:
                endpoints['predict'] = predict_api
            
            # Batch predict endpoint
            batch_predict_api = self._create_batch_predict_api(group_id, backend_url)
            if batch_predict_api:
                endpoints['batch_predict'] = batch_predict_api
            
            # Health check endpoint
            health_api = self._create_health_api(group_id, backend_url)
            if health_api:
                endpoints['health'] = health_api
            
            self.logger.info(f"Created {len(endpoints)} API endpoints")
            return endpoints
            
        except Exception as e:
            self.logger.error(f"Error creating API endpoints: {e}")
            return endpoints
    
    def _create_predict_api(self, group_id: str, backend_url: str) -> Optional[str]:
        """Create single prediction API endpoint"""
        try:
            request = CreateApiV2Request(
                group_id=group_id,
                body=ApiCreate(
                    name="arsl-predict",
                    type="HTTP",
                    req_method="POST",
                    req_uri="/predict",
                    auth_type="NONE",
                    backend_type="HTTP",
                    remark="Arabic Sign Language single image prediction",
                    backend_api=BackendApi(
                        url_domain=backend_url,
                        req_method="POST",
                        req_uri="/predict",
                        timeout=30000
                    )
                )
            )
            
            response = self.apig_client.create_api_v2(request)
            
            if response.status_code == 201:
                api_id = response.id
                self.logger.info(f"Predict API created: {api_id}")
                return api_id
            else:
                self.logger.error(f"Failed to create predict API: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating predict API: {e}")
            return None
    
    def _create_batch_predict_api(self, group_id: str, backend_url: str) -> Optional[str]:
        """Create batch prediction API endpoint"""
        try:
            request = CreateApiV2Request(
                group_id=group_id,
                body=ApiCreate(
                    name="arsl-batch-predict",
                    type="HTTP",
                    req_method="POST",
                    req_uri="/batch_predict",
                    auth_type="NONE",
                    backend_type="HTTP",
                    remark="Arabic Sign Language batch prediction",
                    backend_api=BackendApi(
                        url_domain=backend_url,
                        req_method="POST",
                        req_uri="/batch_predict",
                        timeout=60000  # Longer timeout for batch
                    )
                )
            )
            
            response = self.apig_client.create_api_v2(request)
            
            if response.status_code == 201:
                api_id = response.id
                self.logger.info(f"Batch predict API created: {api_id}")
                return api_id
            else:
                self.logger.error(f"Failed to create batch predict API: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating batch predict API: {e}")
            return None
    
    def _create_health_api(self, group_id: str, backend_url: str) -> Optional[str]:
        """Create health check API endpoint"""
        try:
            request = CreateApiV2Request(
                group_id=group_id,
                body=ApiCreate(
                    name="arsl-health",
                    type="HTTP",
                    req_method="GET",
                    req_uri="/health",
                    auth_type="NONE",
                    backend_type="HTTP",
                    remark="Arabic Sign Language service health check",
                    backend_api=BackendApi(
                        url_domain=backend_url,
                        req_method="GET",
                        req_uri="/health",
                        timeout=10000
                    )
                )
            )
            
            response = self.apig_client.create_api_v2(request)
            
            if response.status_code == 201:
                api_id = response.id
                self.logger.info(f"Health API created: {api_id}")
                return api_id
            else:
                self.logger.error(f"Failed to create health API: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating health API: {e}")
            return None
    
    def setup_monitoring(self, api_group_id: str, service_name: str) -> bool:
        """
        Setup Cloud Eye monitoring for API endpoints
        
        Args:
            api_group_id: API Gateway group ID
            service_name: ModelArts service name
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            if not self.ces_client:
                return False
            
            monitoring_config = self.config['monitoring']
            
            # Create alarm rules
            alarms_created = 0
            
            # High latency alarm
            if self._create_latency_alarm(api_group_id):
                alarms_created += 1
            
            # High error rate alarm
            if self._create_error_rate_alarm(api_group_id):
                alarms_created += 1
            
            # Low accuracy alarm (for ModelArts service)
            if self._create_accuracy_alarm(service_name):
                alarms_created += 1
            
            self.logger.info(f"Created {alarms_created} monitoring alarms")
            return alarms_created > 0
            
        except Exception as e:
            self.logger.error(f"Error setting up monitoring: {e}")
            return False
    
    def _create_latency_alarm(self, api_group_id: str) -> bool:
        """Create high latency alarm"""
        try:
            monitoring_config = self.config['monitoring']
            threshold = monitoring_config['alerts']['high_inference_latency']
            
            request = CreateAlarmRequest(
                body=CreateAlarmRequestBody(
                    alarm_name="ARSL-High-Latency",
                    alarm_description="High inference latency detected",
                    metric=MetricInfo(
                        namespace="SYS.APIG",
                        metric_name="response_time",
                        dimensions=[
                            MetricsDimension(
                                name="api_group_id",
                                value=api_group_id
                            )
                        ]
                    ),
                    condition=Condition(
                        period=300,  # 5 minutes
                        filter="average",
                        comparison_operator=">=",
                        value=threshold,
                        count=2
                    ),
                    alarm_enabled=True
                )
            )
            
            response = self.ces_client.create_alarm(request)
            
            if response.status_code == 201:
                self.logger.info("Latency alarm created successfully")
                return True
            else:
                self.logger.error(f"Failed to create latency alarm: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating latency alarm: {e}")
            return False
    
    def _create_error_rate_alarm(self, api_group_id: str) -> bool:
        """Create high error rate alarm"""
        try:
            monitoring_config = self.config['monitoring']
            threshold = monitoring_config['alerts']['high_error_rate']
            
            request = CreateAlarmRequest(
                body=CreateAlarmRequestBody(
                    alarm_name="ARSL-High-Error-Rate",
                    alarm_description="High error rate detected",
                    metric=MetricInfo(
                        namespace="SYS.APIG",
                        metric_name="error_rate",
                        dimensions=[
                            MetricsDimension(
                                name="api_group_id",
                                value=api_group_id
                            )
                        ]
                    ),
                    condition=Condition(
                        period=300,  # 5 minutes
                        filter="average",
                        comparison_operator=">=",
                        value=threshold,
                        count=2
                    ),
                    alarm_enabled=True
                )
            )
            
            response = self.ces_client.create_alarm(request)
            
            if response.status_code == 201:
                self.logger.info("Error rate alarm created successfully")
                return True
            else:
                self.logger.error(f"Failed to create error rate alarm: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating error rate alarm: {e}")
            return False
    
    def _create_accuracy_alarm(self, service_name: str) -> bool:
        """Create low accuracy alarm"""
        try:
            monitoring_config = self.config['monitoring']
            threshold = monitoring_config['alerts']['low_model_accuracy']
            
            request = CreateAlarmRequest(
                body=CreateAlarmRequestBody(
                    alarm_name="ARSL-Low-Accuracy",
                    alarm_description="Model accuracy below threshold",
                    metric=MetricInfo(
                        namespace="SYS.ModelArts",
                        metric_name="model_accuracy",
                        dimensions=[
                            MetricsDimension(
                                name="service_name",
                                value=service_name
                            )
                        ]
                    ),
                    condition=Condition(
                        period=300,  # 5 minutes
                        filter="average",
                        comparison_operator="<=",
                        value=threshold,
                        count=1
                    ),
                    alarm_enabled=True
                )
            )
            
            response = self.ces_client.create_alarm(request)
            
            if response.status_code == 201:
                self.logger.info("Accuracy alarm created successfully")
                return True
            else:
                self.logger.error(f"Failed to create accuracy alarm: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating accuracy alarm: {e}")
            return False
    
    def get_deployment_info(self, group_id: str) -> Dict[str, Any]:
        """
        Get deployment information and status
        
        Args:
            group_id: API Gateway group ID
            
        Returns:
            Deployment information
        """
        try:
            deployment_info = {
                'api_group_id': group_id,
                'region': self.config['auth']['region'],
                'endpoints': {},
                'monitoring': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get API endpoints
            if self.apig_client:
                try:
                    request = ListApisV2Request(group_id=group_id)
                    response = self.apig_client.list_apis_v2(request)
                    
                    if response.status_code == 200:
                        for api in response.apis:
                            deployment_info['endpoints'][api.name] = {
                                'id': api.id,
                                'uri': api.req_uri,
                                'method': api.req_method,
                                'status': api.status
                            }
                except Exception as e:
                    self.logger.error(f"Error getting API endpoints: {e}")
            
            # Check monitoring status
            if self.ces_client:
                try:
                    request = ListAlarmsRequest()
                    response = self.ces_client.list_alarms(request)
                    
                    if response.status_code == 200:
                        deployment_info['monitoring'] = len(response.metric_alarms) > 0
                except Exception as e:
                    self.logger.error(f"Error checking monitoring: {e}")
            
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"Error getting deployment info: {e}")
            return {'error': str(e)}


def deploy_arsl_service(deployment_manager: HuaweiCloudDeployment,
                       modelarts_service_url: str) -> Dict[str, Any]:
    """
    Complete deployment of ARSL service
    
    Args:
        deployment_manager: Initialized deployment manager
        modelarts_service_url: ModelArts inference service URL
        
    Returns:
        Deployment results
    """
    results = {
        'success': False,
        'api_group_id': None,
        'endpoints': {},
        'monitoring': False,
        'error': None
    }
    
    try:
        # Create API group
        group_id = deployment_manager.create_api_group()
        if not group_id:
            results['error'] = "Failed to create API group"
            return results
        
        results['api_group_id'] = group_id
        
        # Create API endpoints
        endpoints = deployment_manager.create_api_endpoints(group_id, modelarts_service_url)
        results['endpoints'] = endpoints
        
        if not endpoints:
            results['error'] = "Failed to create API endpoints"
            return results
        
        # Setup monitoring
        monitoring_success = deployment_manager.setup_monitoring(group_id, "arsl-inference-service")
        results['monitoring'] = monitoring_success
        
        results['success'] = True
        deployment_manager.logger.info("ARSL service deployment completed successfully")
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        deployment_manager.logger.error(f"Deployment failed: {e}")
        return results


if __name__ == "__main__":
    # Example usage
    deployment = HuaweiCloudDeployment()
    
    # Example ModelArts service URL (replace with actual URL)
    service_url = "https://modelarts-inference.ap-southeast-1.myhuaweicloud.com/v1/infers/your-service-id"
    
    # Deploy service (uncomment to use)
    # results = deploy_arsl_service(deployment, service_url)
    # print(f"Deployment results: {json.dumps(results, indent=2)}")