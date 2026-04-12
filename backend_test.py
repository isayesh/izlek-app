#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime

class DashboardAPITester:
    def __init__(self, base_url="https://tablet-layout-fix.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_profile_id = "220abafe-963a-4074-af17-ac1d75047256"
        self.test_program_id = "6ae1baa0-36e9-462e-afc8-65da61e0b221"
        self.test_firebase_uid = "dev-dashboard-user"

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        default_headers = {'Content-Type': 'application/json'}
        if headers:
            default_headers.update(headers)

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=default_headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=default_headers, timeout=10)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=default_headers, timeout=10)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, list):
                        print(f"   Response: List with {len(response_data)} items")
                    elif isinstance(response_data, dict):
                        print(f"   Response keys: {list(response_data.keys())}")
                except:
                    print(f"   Response: {response.text[:100]}...")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")

            return success, response.json() if response.headers.get('content-type', '').startswith('application/json') else {}

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed - Network Error: {str(e)}")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_api_health(self):
        """Test API health endpoint"""
        return self.run_test("API Health Check", "GET", "", 200)

    def test_get_profile(self):
        """Test getting profile by firebase_uid"""
        return self.run_test(
            "Get Profile", 
            "GET", 
            f"profile?firebase_uid={self.test_firebase_uid}", 
            200
        )

    def test_get_programs(self):
        """Test getting programs for profile"""
        return self.run_test(
            "Get Programs", 
            "GET", 
            f"programs/{self.test_profile_id}", 
            200
        )

    def test_update_program_tasks(self):
        """Test updating program tasks (simulating task completion)"""
        # First get the current program
        success, program_data = self.run_test(
            "Get Program for Update", 
            "GET", 
            f"programs/{self.test_profile_id}", 
            200
        )
        
        if not success or not program_data:
            return False, {}
            
        programs = program_data if isinstance(program_data, list) else [program_data]
        if not programs:
            print("   No programs found to update")
            return False, {}
            
        program = programs[0]
        tasks = program.get('tasks', [])
        
        if not tasks:
            print("   No tasks found to update")
            return True, {}  # Not a failure if no tasks exist
            
        # Toggle completion status of first task
        tasks[0]['completed'] = not tasks[0].get('completed', False)
        
        return self.run_test(
            "Update Program Tasks", 
            "PUT", 
            f"programs/{program['id']}", 
            200,
            data={"tasks": tasks}
        )

    def test_create_task(self):
        """Test creating a new task"""
        # First get the current program
        success, program_data = self.run_test(
            "Get Program for Task Creation", 
            "GET", 
            f"programs/{self.test_profile_id}", 
            200
        )
        
        if not success or not program_data:
            return False, {}
            
        programs = program_data if isinstance(program_data, list) else [program_data]
        if not programs:
            print("   No programs found to add task to")
            return False, {}
            
        program = programs[0]
        current_tasks = program.get('tasks', [])
        
        # Add a new test task
        new_task = {
            "id": f"test-task-{datetime.now().strftime('%H%M%S')}",
            "lesson": "Test Ders",
            "topic": "Test Konu",
            "duration": 30,
            "day": "Pazartesi",
            "completed": False
        }
        
        updated_tasks = current_tasks + [new_task]
        
        return self.run_test(
            "Create New Task", 
            "PUT", 
            f"programs/{program['id']}", 
            200,
            data={"tasks": updated_tasks}
        )

    def test_cors_headers(self):
        """Test CORS headers are present"""
        url = f"{self.api_url}/"
        try:
            response = requests.options(url, timeout=10)
            cors_headers = {
                'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers'),
            }
            
            print(f"\n🔍 Testing CORS Headers...")
            print(f"   CORS Headers: {cors_headers}")
            
            if cors_headers['Access-Control-Allow-Origin']:
                self.tests_passed += 1
                print(f"✅ Passed - CORS headers present")
                return True, cors_headers
            else:
                print(f"❌ Failed - CORS headers missing")
                return False, {}
                
        except Exception as e:
            print(f"❌ Failed - CORS test error: {str(e)}")
            return False, {}
        finally:
            self.tests_run += 1

def main():
    print("🚀 Starting Dashboard API Tests...")
    print("=" * 50)
    
    tester = DashboardAPITester()
    
    # Run all tests
    tests = [
        tester.test_api_health,
        tester.test_cors_headers,
        tester.test_get_profile,
        tester.test_get_programs,
        tester.test_update_program_tasks,
        tester.test_create_task,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            tester.tests_run += 1
    
    # Print results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {tester.tests_run - tester.tests_passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())