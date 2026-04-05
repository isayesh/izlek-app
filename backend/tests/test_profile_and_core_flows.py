import os
import uuid

import pytest
import requests
from dotenv import dotenv_values


def _get_base_url() -> str:
    env_base = os.environ.get("REACT_APP_BACKEND_URL")
    if env_base:
        return env_base.rstrip("/")

    frontend_env = dotenv_values("/app/frontend/.env")
    file_base = frontend_env.get("REACT_APP_BACKEND_URL")
    if file_base:
        return str(file_base).rstrip("/")

    pytest.skip("REACT_APP_BACKEND_URL missing; cannot run API tests")


BASE_URL = _get_base_url()
API_BASE = f"{BASE_URL}/api"


@pytest.fixture
def api_client():
    """Shared HTTP client for API regression tests."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


class TestProfileAndCoreFlows:
    """Profile CRUD + core regression checks for program, rooms/timer, leaderboard."""

    def test_profile_post_get_put_persistence(self, api_client):
        firebase_uid = f"TEST_profile_uid_{uuid.uuid4().hex[:10]}"

        create_payload = {
            "firebase_uid": firebase_uid,
            "username": "TEST_profile_user",
            "email": "test.profile@example.com",
            "study_goal": "Günlük düzenli tekrar",
            "daily_study_hours": 2.5,
            "avatar_url": "https://example.com/avatar.png",
        }

        create_response = api_client.post(f"{API_BASE}/profile", json=create_payload, timeout=20)
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["firebase_uid"] == firebase_uid
        assert created["username"] == create_payload["username"]
        assert created["email"] == create_payload["email"]
        assert created["study_goal"] == create_payload["study_goal"]
        assert created["daily_study_hours"] == create_payload["daily_study_hours"]

        get_response = api_client.get(f"{API_BASE}/profile", params={"firebase_uid": firebase_uid}, timeout=20)
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["id"] == created["id"]
        assert fetched["username"] == create_payload["username"]
        assert fetched["email"] == create_payload["email"]

        update_payload = {
            "firebase_uid": firebase_uid,
            "username": "TEST_profile_user_updated",
            "email": "test.profile@example.com",
            "study_goal": "Sınav hedefi güncellendi",
            "daily_study_hours": 3,
            "avatar_url": "https://example.com/avatar-updated.png",
        }

        update_response = api_client.put(f"{API_BASE}/profile", json=update_payload, timeout=20)
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["id"] == created["id"]
        assert updated["username"] == update_payload["username"]
        assert updated["study_goal"] == update_payload["study_goal"]
        assert updated["daily_study_hours"] == update_payload["daily_study_hours"]

        get_after_update_response = api_client.get(
            f"{API_BASE}/profile", params={"firebase_uid": firebase_uid}, timeout=20
        )
        assert get_after_update_response.status_code == 200
        fetched_after_update = get_after_update_response.json()
        assert fetched_after_update["username"] == update_payload["username"]
        assert fetched_after_update["study_goal"] == update_payload["study_goal"]
        assert fetched_after_update["daily_study_hours"] == update_payload["daily_study_hours"]

    def test_program_creation_flow_still_works(self, api_client):
        firebase_uid = f"TEST_program_uid_{uuid.uuid4().hex[:10]}"
        profile_response = api_client.post(
            f"{API_BASE}/profiles",
            json={
                "firebase_uid": firebase_uid,
                "name": "TEST Program User",
                "study_field": "EA",
            },
            timeout=20,
        )
        assert profile_response.status_code == 200
        profile_data = profile_response.json()
        assert profile_data["firebase_uid"] == firebase_uid
        assert profile_data["name"] == "TEST Program User"
        assert isinstance(profile_data["id"], str)

        program_payload = {
            "profile_id": profile_data["id"],
            "exam_goal": "TYT",
            "daily_hours": "2-4",
            "study_days": 5,
        }
        create_program_response = api_client.post(f"{API_BASE}/programs", json=program_payload, timeout=20)
        assert create_program_response.status_code == 200
        created_program = create_program_response.json()
        assert created_program["profile_id"] == profile_data["id"]
        assert created_program["exam_goal"] == "TYT"
        assert isinstance(created_program["tasks"], list)

        get_programs_response = api_client.get(f"{API_BASE}/programs/{profile_data['id']}", timeout=20)
        assert get_programs_response.status_code == 200
        programs = get_programs_response.json()
        assert isinstance(programs, list)
        assert any(program["id"] == created_program["id"] for program in programs)

    def test_rooms_create_and_timer_update_still_work(self, api_client):
        room_payload = {
            "name": f"TEST Room {uuid.uuid4().hex[:6]}",
            "owner_name": "TEST Owner",
            "owner_study_field": "Sayısal",
        }
        create_room_response = api_client.post(f"{API_BASE}/rooms", json=room_payload, timeout=20)
        assert create_room_response.status_code == 200
        room = create_room_response.json()
        assert room["name"] == room_payload["name"]
        assert isinstance(room["id"], str)
        assert room["timer_state"]["is_running"] is False

        timer_payload = {
            "is_running": True,
            "duration_minutes": 30,
            "remaining_seconds": 1200,
            "started_at": "2026-02-01T10:00:00Z",
        }
        timer_update_response = api_client.put(
            f"{API_BASE}/rooms/{room['id']}/timer", json=timer_payload, timeout=20
        )
        assert timer_update_response.status_code == 200
        timer_update_data = timer_update_response.json()
        assert timer_update_data["success"] is True

        get_room_response = api_client.get(f"{API_BASE}/rooms/{room['id']}", timeout=20)
        assert get_room_response.status_code == 200
        updated_room = get_room_response.json()
        assert updated_room["timer_state"]["is_running"] is True
        assert updated_room["timer_state"]["duration_minutes"] == 30
        assert updated_room["timer_state"]["remaining_seconds"] == 1200

    def test_leaderboard_endpoint_responds(self, api_client):
        leaderboard_response = api_client.get(f"{API_BASE}/leaderboard", timeout=20)
        assert leaderboard_response.status_code == 200
        leaderboard = leaderboard_response.json()
        assert isinstance(leaderboard, list)
