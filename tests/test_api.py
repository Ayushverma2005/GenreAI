"""
Test script for the ML Model Prediction API
This script demonstrates how to interact with the API using Python requests
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_health_check():
    """Test the health check endpoint"""
    print_section("Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_single_prediction():
    """Test single prediction endpoint"""
    print_section("Single Prediction")
    
    # Example input (modify based on your model's requirements)
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    
    print(f"Request Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nPredicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print_section("Batch Prediction")
    
    # Example batch input (modify based on your model's requirements)
    batch_payload = [
        {"features": [5.1, 3.5, 1.4, 0.2]},
        {"features": [6.2, 2.9, 4.3, 1.3]},
        {"features": [7.3, 2.9, 6.3, 1.8]}
    ]
    
    print(f"Request Payload: {json.dumps(batch_payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"\nTotal Predictions: {len(results)}")
            for i, result in enumerate(results, 1):
                print(f"Prediction {i}: Class={result['predicted_class']}, "
                      f"Confidence={result['confidence']:.4f}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_invalid_input():
    """Test error handling with invalid input"""
    print_section("Invalid Input Test")
    
    # Test with empty features
    payload = {
        "features": []
    }
    
    print(f"Request Payload (Invalid): {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # We expect this to fail with 422
        return response.status_code == 422
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_root_endpoint():
    """Test the root endpoint"""
    print_section("Root Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def measure_latency():
    """Measure prediction latency"""
    print_section("Latency Test")
    
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    
    num_requests = 10
    latencies = []
    
    print(f"Measuring latency over {num_requests} requests...")
    
    for i in range(num_requests):
        start_time = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"\nLatency Statistics:")
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  Min: {min_latency:.2f} ms")
        print(f"  Max: {max_latency:.2f} ms")
        print(f"  Successful Requests: {len(latencies)}/{num_requests}")
        
        return True
    
    return False


def run_all_tests():
    """Run all test cases"""
    print("\n" + "#"*70)
    print("  ML Model API Test Suite")
    print("#"*70)
    
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Invalid Input", test_invalid_input),
        ("Latency", measure_latency),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\nTest '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_section("Test Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to the API server")
        print(f"Please ensure the server is running at {BASE_URL}")
        print("\nStart the server with:")
        print("  uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
