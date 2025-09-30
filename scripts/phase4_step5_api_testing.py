"""
Phase 4: Step 5 - Test API with Sample Images
Complete end-to-end testing of the deployed API system
"""

import base64
import json
import requests
import time
from pathlib import Path
import numpy as np
from PIL import Image
import io

def create_test_images():
    """Create sample test images for API testing"""
    
    print("🖼️ CREATING TEST IMAGES:")
    print("=" * 30)
    
    # Create test images directory
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Arabic letters for testing (subset of 32 classes)
    test_letters = [
        {"name": "alef", "index": 0, "arabic": "أ"},
        {"name": "baa", "index": 1, "arabic": "ب"},
        {"name": "taa", "index": 2, "arabic": "ت"},
        {"name": "thaa", "index": 3, "arabic": "ث"},
        {"name": "jeem", "index": 4, "arabic": "ج"},
        {"name": "haa", "index": 5, "arabic": "ح"},
        {"name": "khaa", "index": 6, "arabic": "خ"},
        {"name": "dal", "index": 7, "arabic": "د"}
    ]
    
    # Create simple test images (64x64 grayscale with patterns)
    for letter in test_letters:
        # Create a simple pattern image
        img_array = np.zeros((64, 64), dtype=np.uint8)
        
        # Add some pattern based on letter index
        for i in range(8):
            for j in range(8):
                if (i + j + letter["index"]) % 3 == 0:
                    img_array[i*8:(i+1)*8, j*8:(j+1)*8] = 128 + letter["index"] * 4
        
        # Convert to PIL Image and save
        img = Image.fromarray(img_array, mode='L')
        img_path = test_dir / f"test_{letter['name']}.jpg"
        img.save(img_path, "JPEG")
        
        print(f"   ✅ Created: {img_path} ({letter['arabic']} - {letter['name']})")
    
    print(f"\n📁 Test images created in: {test_dir}")
    return test_dir, test_letters

def create_comprehensive_test_suite():
    """Create comprehensive test suite for API testing"""
    
    test_suite_code = '''"""
Comprehensive Test Suite for Arabic Sign Language Recognition API
Tests all endpoints with various scenarios
"""

import base64
import json
import requests
import time
from pathlib import Path
import statistics
from datetime import datetime

class ARSLAPITestSuite:
    """Comprehensive test suite for ARSL API"""
    
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
        
        # Set up session headers
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
        
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "ARSL-API-TestSuite/1.0"
        })
    
    def log_test_result(self, test_name, success, duration_ms, details=None):
        """Log test result"""
        result = {
            "test_name": test_name,
            "success": success,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.test_results["tests"].append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name} ({duration_ms:.1f}ms)")
        
        if not success and details:
            print(f"      Error: {details}")
    
    def encode_image(self, image_path):
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def test_health_check(self):
        """Test health check endpoint"""
        print("\\n🏥 Testing Health Check Endpoint...")
        
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/v1/health", timeout=10)
            duration = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                health_data = response.json()
                success = health_data.get("status") == "healthy"
                self.log_test_result("health_check", success, duration, health_data)
                return health_data
            else:
                self.log_test_result("health_check", False, duration, 
                                   {"status_code": response.status_code, "response": response.text})
                return None
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test_result("health_check", False, duration, {"error": str(e)})
            return None
    
    def test_model_info(self):
        """Test model info endpoint"""
        print("\\n📋 Testing Model Info Endpoint...")
        
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/v1/model/info", timeout=10)
            duration = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                model_info = response.json()
                success = "model_name" in model_info and "classes" in model_info
                self.log_test_result("model_info", success, duration, model_info)
                return model_info
            else:
                self.log_test_result("model_info", False, duration,
                                   {"status_code": response.status_code, "response": response.text})
                return None
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test_result("model_info", False, duration, {"error": str(e)})
            return None
    
    def test_single_prediction(self, image_path, expected_class=None):
        """Test single image prediction"""
        test_name = f"single_prediction_{Path(image_path).stem}"
        
        start_time = time.time()
        try:
            image_data = self.encode_image(image_path)
            payload = {
                "image": image_data,
                "top_k": 3,
                "confidence_threshold": 0.1
            }
            
            response = self.session.post(f"{self.base_url}/v1/predict", 
                                       json=payload, timeout=30)
            duration = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False) and len(result.get("predictions", [])) > 0
                
                details = {
                    "predictions": result.get("predictions", []),
                    "processing_time": result.get("processing_time_ms", 0),
                    "model_version": result.get("model_version", "unknown")
                }
                
                if expected_class:
                    top_prediction = result.get("predictions", [{}])[0]
                    predicted_class = top_prediction.get("class", "")
                    details["expected"] = expected_class
                    details["predicted"] = predicted_class
                    details["correct"] = predicted_class == expected_class
                
                self.log_test_result(test_name, success, duration, details)
                return result
            else:
                self.log_test_result(test_name, False, duration,
                                   {"status_code": response.status_code, "response": response.text})
                return None
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test_result(test_name, False, duration, {"error": str(e)})
            return None
    
    def test_batch_prediction(self, image_paths):
        """Test batch prediction"""
        print("\\n📦 Testing Batch Prediction...")
        
        start_time = time.time()
        try:
            images = []
            for i, image_path in enumerate(image_paths):
                image_data = self.encode_image(image_path)
                images.append({
                    "id": f"test_image_{i}",
                    "image": image_data
                })
            
            payload = {
                "images": images,
                "top_k": 3
            }
            
            response = self.session.post(f"{self.base_url}/v1/predict/batch",
                                       json=payload, timeout=60)
            duration = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                batch_stats = result.get("batch_stats", {})
                
                details = {
                    "total_images": batch_stats.get("total_images", 0),
                    "successful": batch_stats.get("successful_predictions", 0),
                    "failed": batch_stats.get("failed_predictions", 0),
                    "total_time": batch_stats.get("total_processing_time_ms", 0)
                }
                
                self.log_test_result("batch_prediction", success, duration, details)
                return result
            else:
                self.log_test_result("batch_prediction", False, duration,
                                   {"status_code": response.status_code, "response": response.text})
                return None
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test_result("batch_prediction", False, duration, {"error": str(e)})
            return None
    
    def test_performance(self, image_path, num_requests=10):
        """Test API performance with multiple requests"""
        print(f"\\n⚡ Testing Performance ({num_requests} requests)...")
        
        image_data = self.encode_image(image_path)
        payload = {
            "image": image_data,
            "top_k": 1
        }
        
        response_times = []
        success_count = 0
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                response = self.session.post(f"{self.base_url}/v1/predict",
                                           json=payload, timeout=30)
                duration = (time.time() - start_time) * 1000
                response_times.append(duration)
                
                if response.status_code == 200:
                    success_count += 1
                
                print(f"      Request {i+1}/{num_requests}: {duration:.1f}ms")
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                response_times.append(duration)
                print(f"      Request {i+1}/{num_requests}: ERROR - {str(e)}")
        
        # Calculate performance metrics
        if response_times:
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            
            success_rate = (success_count / num_requests) * 100
            
            details = {
                "total_requests": num_requests,
                "successful_requests": success_count,
                "success_rate_percent": success_rate,
                "avg_response_time_ms": avg_time,
                "min_response_time_ms": min_time,
                "max_response_time_ms": max_time,
                "p95_response_time_ms": p95_time
            }
            
            success = success_rate >= 95 and avg_time <= 1000
            self.log_test_result("performance_test", success, avg_time, details)
    
    def test_error_conditions(self):
        """Test error handling"""
        print("\\n🚫 Testing Error Conditions...")
        
        # Test invalid image data
        start_time = time.time()
        try:
            payload = {"image": "invalid_base64_data", "top_k": 3}
            response = self.session.post(f"{self.base_url}/v1/predict", json=payload)
            duration = (time.time() - start_time) * 1000
            
            success = response.status_code in [400, 422]  # Bad request expected
            self.log_test_result("invalid_image_data", success, duration,
                               {"status_code": response.status_code})
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test_result("invalid_image_data", False, duration, {"error": str(e)})
        
        # Test missing required fields
        start_time = time.time()
        try:
            payload = {"top_k": 3}  # Missing image field
            response = self.session.post(f"{self.base_url}/v1/predict", json=payload)
            duration = (time.time() - start_time) * 1000
            
            success = response.status_code in [400, 422]  # Bad request expected
            self.log_test_result("missing_required_field", success, duration,
                               {"status_code": response.status_code})
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test_result("missing_required_field", False, duration, {"error": str(e)})
        
        # Test invalid endpoint
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/v1/invalid_endpoint")
            duration = (time.time() - start_time) * 1000
            
            success = response.status_code == 404  # Not found expected
            self.log_test_result("invalid_endpoint", success, duration,
                               {"status_code": response.status_code})
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test_result("invalid_endpoint", False, duration, {"error": str(e)})
    
    def generate_report(self):
        """Generate comprehensive test report"""
        self.test_results["end_time"] = datetime.now().isoformat()
        
        total_tests = len(self.test_results["tests"])
        passed_tests = sum(1 for test in self.test_results["tests"] if test["success"])
        failed_tests = total_tests - passed_tests
        
        if total_tests > 0:
            success_rate = (passed_tests / total_tests) * 100
            avg_duration = statistics.mean([test["duration_ms"] for test in self.test_results["tests"]])
        else:
            success_rate = 0
            avg_duration = 0
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate_percent": success_rate,
            "average_duration_ms": avg_duration
        }
        
        # Save detailed report
        report_file = Path("test_results") / f"api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Print summary
        print(f"\\n📊 TEST SUMMARY:")
        print(f"=" * 40)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({success_rate:.1f}%)")
        print(f"Failed: {failed_tests}")
        print(f"Average Duration: {avg_duration:.1f}ms")
        print(f"\\n📋 Detailed report saved: {report_file}")
        
        return self.test_results

def run_comprehensive_tests():
    """Run all API tests"""
    # Configuration
    API_BASE_URL = "https://arsl-api.apig.af-north-1.huaweicloudapis.com"
    API_KEY = None  # Set if using API key authentication
    
    # Initialize test suite
    test_suite = ARSLAPITestSuite(API_BASE_URL, API_KEY)
    
    print("🧪 STARTING COMPREHENSIVE API TESTS")
    print("=" * 50)
    print(f"Base URL: {API_BASE_URL}")
    print(f"API Key: {'Set' if API_KEY else 'Not Set'}")
    print()
    
    # Run tests
    test_suite.test_health_check()
    test_suite.test_model_info()
    
    # Test with sample images if they exist
    test_images_dir = Path("test_images")
    if test_images_dir.exists():
        image_files = list(test_images_dir.glob("*.jpg"))
        
        if image_files:
            print(f"\\n🖼️ Testing with {len(image_files)} sample images...")
            
            # Test single predictions
            for image_file in image_files[:5]:  # Test first 5 images
                expected_class = image_file.stem.replace("test_", "")
                test_suite.test_single_prediction(image_file, expected_class)
            
            # Test batch prediction
            if len(image_files) >= 3:
                test_suite.test_batch_prediction(image_files[:3])
            
            # Performance test
            test_suite.test_performance(image_files[0], num_requests=5)
    
    # Test error conditions
    test_suite.test_error_conditions()
    
    # Generate final report
    final_results = test_suite.generate_report()
    
    return final_results

if __name__ == "__main__":
    results = run_comprehensive_tests()
'''
    
    # Save test suite
    test_suite_file = Path("scripts/comprehensive_api_test.py")
    with open(test_suite_file, 'w', encoding='utf-8') as f:
        f.write(test_suite_code)
    
    print(f"🧪 Comprehensive test suite saved: {test_suite_file}")
    return test_suite_file

def create_curl_test_examples():
    """Create cURL command examples for manual testing"""
    
    curl_examples = '''# Arabic Sign Language Recognition API - cURL Test Examples

## 1. Health Check
curl -X GET \\
  "https://arsl-api.apig.af-north-1.huaweicloudapis.com/v1/health" \\
  -H "Content-Type: application/json"

## 2. Model Information
curl -X GET \\
  "https://arsl-api.apig.af-north-1.huaweicloudapis.com/v1/model/info" \\
  -H "Content-Type: application/json"

## 3. Single Image Prediction
# First, encode your image to base64:
# base64 -i test_image.jpg > image_base64.txt

curl -X POST \\
  "https://arsl-api.apig.af-north-1.huaweicloudapis.com/v1/predict" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: YOUR_API_KEY_HERE" \\
  -d '{
    "image": "$(cat image_base64.txt)",
    "top_k": 3,
    "confidence_threshold": 0.1
  }'

## 4. Batch Prediction
curl -X POST \\
  "https://arsl-api.apig.af-north-1.huaweicloudapis.com/v1/predict/batch" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: YOUR_API_KEY_HERE" \\
  -d '{
    "images": [
      {
        "id": "image_1",
        "image": "$(cat image1_base64.txt)"
      },
      {
        "id": "image_2", 
        "image": "$(cat image2_base64.txt)"
      }
    ],
    "top_k": 3
  }'

## 5. Performance Testing with ApacheBench
# Test 100 requests with concurrency of 10
ab -n 100 -c 10 -T "application/json" \\
   -H "X-API-Key: YOUR_API_KEY_HERE" \\
   -p post_data.json \\
   "https://arsl-api.apig.af-north-1.huaweicloudapis.com/v1/predict"

## 6. Load Testing with wrk
# Test for 30 seconds with 10 connections
wrk -t10 -c10 -d30s \\
    -H "Content-Type: application/json" \\
    -H "X-API-Key: YOUR_API_KEY_HERE" \\
    --script=post_image.lua \\
    "https://arsl-api.apig.af-north-1.huaweicloudapis.com/v1/predict"

## Expected Response Format
{
  "success": true,
  "predictions": [
    {
      "class": "alef",
      "class_index": 0,
      "confidence": 0.94,
      "arabic_letter": "أ"
    },
    {
      "class": "baa",
      "class_index": 1,
      "confidence": 0.04,
      "arabic_letter": "ب"
    }
  ],
  "model_version": "1.0.0",
  "processing_time_ms": 156.2,
  "timestamp": "2025-01-22T10:30:45Z"
}
'''
    
    # Save cURL examples
    curl_file = Path("examples/curl_api_tests.md")
    curl_file.parent.mkdir(exist_ok=True)
    
    with open(curl_file, 'w', encoding='utf-8') as f:
        f.write(curl_examples)
    
    print(f"📋 cURL examples saved: {curl_file}")
    return curl_file

def print_testing_guide():
    """Print comprehensive testing guide"""
    
    print("\n🧪 COMPREHENSIVE API TESTING GUIDE:")
    print("=" * 50)
    print()
    
    print("🎯 TESTING PHASES:")
    testing_phases = [
        {
            "phase": "1. Smoke Tests",
            "duration": "5 minutes",
            "tests": ["Health check", "Model info", "Basic prediction"]
        },
        {
            "phase": "2. Functional Tests", 
            "duration": "15 minutes",
            "tests": ["All endpoints", "Various image formats", "Parameter validation"]
        },
        {
            "phase": "3. Performance Tests",
            "duration": "20 minutes", 
            "tests": ["Response time", "Throughput", "Concurrent requests"]
        },
        {
            "phase": "4. Error Handling",
            "duration": "10 minutes",
            "tests": ["Invalid inputs", "Rate limiting", "Timeout scenarios"]
        },
        {
            "phase": "5. Security Tests",
            "duration": "10 minutes",
            "tests": ["Authentication", "Authorization", "Input validation"]
        }
    ]
    
    total_time = 0
    for phase in testing_phases:
        print(f"   {phase['phase']} ({phase['duration']}):")
        for test in phase['tests']:
            print(f"     • {test}")
        print()
        total_time += int(phase['duration'].split()[0])
    
    print(f"⏱️ Total testing time: ~{total_time} minutes")
    
    print(f"\n📊 SUCCESS CRITERIA:")
    criteria = [
        "Health check returns 'healthy' status",
        "Model info returns 32 Arabic classes",
        "Single prediction accuracy > 85%",
        "Response time < 500ms (95th percentile)",
        "Error rate < 2% under normal load",
        "API supports 100+ concurrent requests"
    ]
    
    for criterion in criteria:
        print(f"   ✅ {criterion}")

def main():
    """Main function"""
    print("🧪 PHASE 4: STEP 5 - API TESTING & VALIDATION")
    print("Account: yyacoup")
    print("Region: AF-Cairo")
    print("API: arsl-api-gateway")
    print("=" * 60)
    
    # Create test images
    test_dir, test_letters = create_test_images()
    
    # Create comprehensive test suite
    test_suite_file = create_comprehensive_test_suite()
    
    # Create cURL examples
    curl_file = create_curl_test_examples()
    
    # Print testing guide
    print_testing_guide()
    
    print(f"\n🎯 STEP 5 SUMMARY:")
    print(f"✅ Test images created ({len(test_letters)} samples)")
    print(f"✅ Comprehensive test suite ready ({test_suite_file})")
    print(f"✅ cURL examples prepared ({curl_file})")
    print(f"✅ Testing guide documented")
    print(f"📋 Ready for end-to-end API testing")
    print(f"🌐 Next: Execute test suite and validate deployment")

if __name__ == "__main__":
    main()