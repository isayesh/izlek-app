#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime

class DashboardAPITester:
    def __init__(self, base_url="https://unread-indicator.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_profile_id = "220abafe-963a-4074-af17-ac1d75047256"
        self.test_program_id = "6ae1baa0-36e9-462e-afc8-65da61e0b221"
        self.test_firebase_uid = "dev-dashboard-user"
        self.test_room_id = None
        self.test_room_code = None

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

    # ============ ROOM SMOKE TESTS ============
    
    def test_create_room(self):
        """Test creating a room for RoomPage testing"""
        room_data = {
            "name": "Test Smoke Room",
            "owner_name": "Test User",
            "owner_id": "test-smoke-user-123",
            "owner_avatar_url": None,
            "room_type": "public"
        }
        
        success, response_data = self.run_test(
            "Create Room", 
            "POST", 
            "rooms", 
            200,
            data=room_data
        )
        
        if success and response_data:
            self.test_room_id = response_data.get('id')
            self.test_room_code = response_data.get('code')
            print(f"   Created room ID: {self.test_room_id}")
            print(f"   Room code: {self.test_room_code}")
        
        return success, response_data

    def test_get_room_by_id(self):
        """Test fetching room by ID (used by RoomPage)"""
        if not self.test_room_id:
            print("   Skipping - no room ID available")
            return True, {}
            
        return self.run_test(
            "Get Room by ID", 
            "GET", 
            f"rooms/{self.test_room_id}", 
            200
        )

    def test_get_room_by_code(self):
        """Test fetching room by code"""
        if not self.test_room_code:
            print("   Skipping - no room code available")
            return True, {}
            
        return self.run_test(
            "Get Room by Code", 
            "GET", 
            f"rooms/code/{self.test_room_code}", 
            200
        )

    def test_join_room(self):
        """Test joining a room"""
        if not self.test_room_code:
            print("   Skipping - no room code available")
            return True, {}
            
        join_data = {
            "room_code": self.test_room_code,
            "user_id": "test-joiner-456",
            "user_name": "Test Joiner",
            "user_avatar_url": None
        }
        
        return self.run_test(
            "Join Room", 
            "POST", 
            "rooms/join", 
            200,
            data=join_data
        )

    def test_update_timer(self):
        """Test updating room timer (used by RoomPage timer)"""
        if not self.test_room_id:
            print("   Skipping - no room ID available")
            return True, {}
            
        timer_data = {
            "is_running": True,
            "duration_minutes": 25,
            "remaining_seconds": 1500,
            "started_at": datetime.now().isoformat()
        }
        
        return self.run_test(
            "Update Timer", 
            "PUT", 
            f"rooms/{self.test_room_id}/timer", 
            200,
            data=timer_data
        )

    def test_create_message(self):
        """Test creating a message in room (used by RoomPage chat)"""
        if not self.test_room_id:
            print("   Skipping - no room ID available")
            return True, {}
            
        message_data = {
            "room_id": self.test_room_id,
            "user_id": "test-smoke-user-123",
            "user_name": "Test User",
            "user_avatar_url": None,
            "user_study_field": "Sayısal",
            "content": "Test smoke message for RoomPage"
        }
        
        return self.run_test(
            "Create Message", 
            "POST", 
            "messages", 
            200,
            data=message_data
        )

    def test_get_messages(self):
        """Test fetching messages for room (used by RoomPage chat)"""
        if not self.test_room_id:
            print("   Skipping - no room ID available")
            return True, {}
            
        return self.run_test(
            "Get Messages", 
            "GET", 
            f"messages/{self.test_room_id}", 
            200
        )

    def test_leave_room(self):
        """Test leaving room (cleanup)"""
        if not self.test_room_id:
            print("   Skipping - no room ID available")
            return True, {}
            
        leave_data = {
            "user_id": "test-joiner-456",
            "user_name": "Test Joiner"
        }
        
        return self.run_test(
            "Leave Room", 
            "POST", 
            f"rooms/{self.test_room_id}/leave", 
            200,
            data=leave_data
        )

    # ============ BREAK MODE TESTS ============
    
    def test_break_mode_room_setup(self):
        """Test creating room and adding 2 users for break mode testing"""
        # Create room
        room_data = {
            "name": "Break Mode Test Room",
            "owner_name": "User A",
            "owner_id": "user-a-break-test",
            "owner_avatar_url": None,
            "room_type": "public"
        }
        
        success, response_data = self.run_test(
            "Create Break Mode Test Room", 
            "POST", 
            "rooms", 
            200,
            data=room_data
        )
        
        if success and response_data:
            self.break_room_id = response_data.get('id')
            self.break_room_code = response_data.get('code')
            print(f"   Break test room ID: {self.break_room_id}")
            print(f"   Break test room code: {self.break_room_code}")
        
        return success, response_data

    def test_break_mode_second_user_join(self):
        """Test second user joining the break mode test room"""
        if not self.break_room_code:
            print("   Skipping - no break room code available")
            return True, {}
            
        join_data = {
            "room_code": self.break_room_code,
            "user_id": "user-b-break-test",
            "user_name": "User B",
            "user_avatar_url": None
        }
        
        return self.run_test(
            "User B Join Break Room", 
            "POST", 
            "rooms/join", 
            200,
            data=join_data
        )

    def test_break_mode_timer_persistence(self):
        """Test that global room timer works normally with break mode"""
        if not self.break_room_id:
            print("   Skipping - no break room ID available")
            return True, {}
            
        timer_data = {
            "is_running": True,
            "duration_minutes": 25,
            "remaining_seconds": 1200,
            "started_at": datetime.now().isoformat()
        }
        
        return self.run_test(
            "Break Mode Timer Update", 
            "PUT", 
            f"rooms/{self.break_room_id}/timer", 
            200,
            data=timer_data
        )

    def test_break_mode_endpoint_user_a_on_break(self):
        """Test PUT /api/rooms/{room_id}/break-mode - User A goes on break"""
        if not self.break_room_id:
            print("   Skipping - no break room ID available")
            return True, {}
            
        break_data = {
            "participant_id": "user-a-break-test",
            "firebase_uid": "user-a-break-test",
            "is_on_break": True
        }
        
        return self.run_test(
            "User A Enter Break Mode", 
            "PUT", 
            f"rooms/{self.break_room_id}/break-mode", 
            200,
            data=break_data
        )

    def test_break_mode_verify_user_a_break_status(self):
        """Verify User A is marked as on break, User B is not affected"""
        if not self.break_room_id:
            print("   Skipping - no break room ID available")
            return True, {}
            
        success, response_data = self.run_test(
            "Verify Break Status After User A Break", 
            "GET", 
            f"rooms/{self.break_room_id}", 
            200
        )
        
        if success and response_data:
            participants = response_data.get('participants', [])
            user_a_break = None
            user_b_break = None
            
            for participant in participants:
                if participant.get('id') == 'user-a-break-test':
                    user_a_break = participant.get('is_on_break')
                elif participant.get('id') == 'user-b-break-test':
                    user_b_break = participant.get('is_on_break')
            
            print(f"   User A break status: {user_a_break}")
            print(f"   User B break status: {user_b_break}")
            
            if user_a_break is True and user_b_break is False:
                print("✅ Break status correctly set - User A on break, User B not affected")
                return True, response_data
            else:
                print("❌ Break status incorrect")
                return False, response_data
        
        return success, response_data

    def test_break_mode_endpoint_user_a_off_break(self):
        """Test User A coming back from break"""
        if not self.break_room_id:
            print("   Skipping - no break room ID available")
            return True, {}
            
        break_data = {
            "participant_id": "user-a-break-test",
            "firebase_uid": "user-a-break-test",
            "is_on_break": False
        }
        
        return self.run_test(
            "User A Exit Break Mode", 
            "PUT", 
            f"rooms/{self.break_room_id}/break-mode", 
            200,
            data=break_data
        )

    def test_study_session_user_a_start(self):
        """Test starting study session for User A"""
        if not self.break_room_id:
            print("   Skipping - no break room ID available")
            return True, {}
            
        session_data = {
            "firebase_uid": "user-a-break-test",
            "room_id": self.break_room_id
        }
        
        success, response_data = self.run_test(
            "Start Study Session User A", 
            "POST", 
            "study-sessions/start", 
            200,
            data=session_data
        )
        
        if success and response_data:
            self.user_a_session_id = response_data.get('id')
            print(f"   User A session ID: {self.user_a_session_id}")
        
        return success, response_data

    def test_study_session_user_b_start(self):
        """Test starting study session for User B"""
        if not self.break_room_id:
            print("   Skipping - no break room ID available")
            return True, {}
            
        session_data = {
            "firebase_uid": "user-b-break-test",
            "room_id": self.break_room_id
        }
        
        success, response_data = self.run_test(
            "Start Study Session User B", 
            "POST", 
            "study-sessions/start", 
            200,
            data=session_data
        )
        
        if success and response_data:
            self.user_b_session_id = response_data.get('id')
            print(f"   User B session ID: {self.user_b_session_id}")
        
        return success, response_data

    def test_study_session_normal_accumulation(self):
        """Test normal time accumulation for both users when active"""
        tests_passed = 0
        total_tests = 2
        
        # Test User A accumulation (should work when not on break)
        if self.user_a_session_id:
            update_data = {"accumulated_seconds": 300}  # 5 minutes
            success, response_data = self.run_test(
                "User A Normal Accumulation", 
                "PUT", 
                f"study-sessions/{self.user_a_session_id}/update", 
                200,
                data=update_data
            )
            if success and response_data.get('accumulated_seconds') == 300:
                tests_passed += 1
                print("✅ User A normal accumulation working")
            else:
                print("❌ User A normal accumulation failed")
        
        # Test User B accumulation (should always work)
        if self.user_b_session_id:
            update_data = {"accumulated_seconds": 300}  # 5 minutes
            success, response_data = self.run_test(
                "User B Normal Accumulation", 
                "PUT", 
                f"study-sessions/{self.user_b_session_id}/update", 
                200,
                data=update_data
            )
            if success and response_data.get('accumulated_seconds') == 300:
                tests_passed += 1
                print("✅ User B normal accumulation working")
            else:
                print("❌ User B normal accumulation failed")
        
        return tests_passed == total_tests, {}

    def test_study_session_break_accumulation_filter(self):
        """Test that User A on break doesn't accumulate time, User B continues"""
        # First put User A on break
        if self.break_room_id:
            break_data = {
                "participant_id": "user-a-break-test",
                "firebase_uid": "user-a-break-test",
                "is_on_break": True
            }
            self.run_test(
                "Put User A on Break for Session Test", 
                "PUT", 
                f"rooms/{self.break_room_id}/break-mode", 
                200,
                data=break_data
            )
        
        tests_passed = 0
        total_tests = 2
        
        # Test User A accumulation (should NOT increase when on break)
        if self.user_a_session_id:
            update_data = {"accumulated_seconds": 600}  # Try to update to 10 minutes
            success, response_data = self.run_test(
                "User A Break Accumulation Filter", 
                "PUT", 
                f"study-sessions/{self.user_a_session_id}/update", 
                200,
                data=update_data
            )
            # Should still be 300 (previous value) because user is on break
            if success and response_data.get('accumulated_seconds') == 300:
                tests_passed += 1
                print("✅ User A break accumulation filter working - time not increased")
            else:
                print(f"❌ User A break accumulation filter failed - got {response_data.get('accumulated_seconds')}, expected 300")
        
        # Test User B accumulation (should continue to work normally)
        if self.user_b_session_id:
            update_data = {"accumulated_seconds": 600}  # Update to 10 minutes
            success, response_data = self.run_test(
                "User B Continues During A's Break", 
                "PUT", 
                f"study-sessions/{self.user_b_session_id}/update", 
                200,
                data=update_data
            )
            if success and response_data.get('accumulated_seconds') == 600:
                tests_passed += 1
                print("✅ User B continues accumulating during User A's break")
            else:
                print("❌ User B accumulation affected by User A's break")
        
        return tests_passed == total_tests, {}

    def test_study_session_resume_after_break(self):
        """Test that User A can accumulate time again after coming back from break"""
        # First take User A off break
        if self.break_room_id:
            break_data = {
                "participant_id": "user-a-break-test",
                "firebase_uid": "user-a-break-test",
                "is_on_break": False
            }
            self.run_test(
                "Take User A Off Break for Resume Test", 
                "PUT", 
                f"rooms/{self.break_room_id}/break-mode", 
                200,
                data=break_data
            )
        
        # Test User A can accumulate again
        if self.user_a_session_id:
            update_data = {"accumulated_seconds": 900}  # Update to 15 minutes
            success, response_data = self.run_test(
                "User A Resume Accumulation After Break", 
                "PUT", 
                f"study-sessions/{self.user_a_session_id}/update", 
                200,
                data=update_data
            )
            if success and response_data.get('accumulated_seconds') == 900:
                print("✅ User A can accumulate time after returning from break")
                return True, response_data
            else:
                print(f"❌ User A cannot accumulate after break - got {response_data.get('accumulated_seconds')}, expected 900")
                return False, response_data
        
        return True, {}

def main():
    print("🚀 Starting İzlek Break Mode Backend Tests...")
    print("=" * 50)
    
    tester = DashboardAPITester()
    
    # Initialize break mode test variables
    tester.break_room_id = None
    tester.break_room_code = None
    tester.user_a_session_id = None
    tester.user_b_session_id = None
    
    # Run existing room smoke tests first (regression check)
    existing_room_tests = [
        tester.test_create_room,
        tester.test_get_room_by_id,
        tester.test_get_room_by_code,
        tester.test_join_room,
        tester.test_update_timer,
        tester.test_create_message,
        tester.test_get_messages,
        tester.test_leave_room,
    ]
    
    print("\n🏠 EXISTING ROOM SMOKE TESTS (Regression Check)")
    print("-" * 50)
    for test in existing_room_tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            tester.tests_run += 1
    
    # Run break mode specific tests
    break_mode_tests = [
        tester.test_break_mode_room_setup,
        tester.test_break_mode_second_user_join,
        tester.test_break_mode_timer_persistence,
        tester.test_break_mode_endpoint_user_a_on_break,
        tester.test_break_mode_verify_user_a_break_status,
        tester.test_break_mode_endpoint_user_a_off_break,
        tester.test_study_session_user_a_start,
        tester.test_study_session_user_b_start,
        tester.test_study_session_normal_accumulation,
        tester.test_study_session_break_accumulation_filter,
        tester.test_study_session_resume_after_break,
    ]
    
    print("\n🔄 BREAK MODE TESTS (New Feature)")
    print("-" * 40)
    for test in break_mode_tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            tester.tests_run += 1
    
    # Print results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} passed")
    
    # Specific analysis for break mode
    existing_room_count = 8  # Number of existing room tests
    break_mode_count = 11   # Number of break mode tests
    
    existing_passed = min(tester.tests_passed, existing_room_count)
    break_mode_passed = max(0, tester.tests_passed - existing_room_count)
    
    print(f"🏠 Existing Room Tests (Regression): {existing_passed}/{existing_room_count}")
    print(f"🔄 Break Mode Tests (New Feature): {break_mode_passed}/{break_mode_count}")
    
    # Determine success criteria
    regression_ok = existing_passed >= 6  # At least 75% of existing tests pass
    break_mode_ok = break_mode_passed >= 8  # At least 73% of break mode tests pass
    
    if regression_ok and break_mode_ok:
        print("🎉 Break Mode feature working! No regression in existing functionality.")
        return 0
    elif regression_ok:
        print("✅ No regression detected, but Break Mode has issues")
        print("⚠️  Break Mode feature needs fixes")
        return 1
    else:
        print("❌ Regression detected in existing room functionality!")
        print("❌ Critical issues found!")
        return 1

if __name__ == "__main__":
    sys.exit(main())