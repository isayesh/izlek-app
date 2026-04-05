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
    """Shared HTTP client for study-session identity regression tests."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


def _leaderboard_totals_by_uid(api_client):
    response = api_client.get(f"{API_BASE}/leaderboard", timeout=20)
    assert response.status_code == 200
    entries = response.json()
    assert isinstance(entries, list)
    return {entry["user_id"]: int(entry["total_seconds"]) for entry in entries}


def _create_room(api_client, owner_name: str):
    response = api_client.post(
        f"{API_BASE}/rooms",
        json={
            "name": f"TEST UID Room {uuid.uuid4().hex[:6]}",
            "owner_name": owner_name,
            "owner_avatar_url": None,
            "owner_study_field": "EA",
        },
        timeout=20,
    )
    assert response.status_code == 200
    room = response.json()
    assert isinstance(room.get("id"), str)
    assert isinstance(room.get("participants"), list)
    assert len(room["participants"]) >= 1
    return room


def _start_session(api_client, firebase_uid: str, room_id: str):
    response = api_client.post(
        f"{API_BASE}/study-sessions/start",
        json={"firebase_uid": firebase_uid, "room_id": room_id},
        timeout=20,
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("id"), str)
    assert data["firebase_uid"] == firebase_uid
    assert data["room_id"] == room_id
    return data


def _complete_session(api_client, session_id: str, accumulated_seconds: int):
    response = api_client.put(
        f"{API_BASE}/study-sessions/{session_id}/complete",
        json={"accumulated_seconds": accumulated_seconds},
        timeout=20,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_completed"] is True
    assert int(data["accumulated_seconds"]) == accumulated_seconds
    return data


class TestStudySessionUidAttributionRegression:
    # Module: room timer -> study session ownership identity attribution by firebase_uid

    def test_room_timer_session_ownership_uses_exact_firebase_uid_not_room_participant_id(self, api_client):
        room = _create_room(api_client, owner_name="TEST Owner Identity")
        room_owner_participant_id = room["participants"][0]["id"]
        auth_uid = f"TEST_auth_uid_{uuid.uuid4().hex[:10]}"

        baseline = _leaderboard_totals_by_uid(api_client)

        started = _start_session(api_client, firebase_uid=auth_uid, room_id=room["id"])
        assert started["firebase_uid"] != room_owner_participant_id

        _complete_session(api_client, session_id=started["id"], accumulated_seconds=300)

        after = _leaderboard_totals_by_uid(api_client)

        assert after.get(auth_uid, 0) - baseline.get(auth_uid, 0) == 300
        assert after.get(room_owner_participant_id, 0) - baseline.get(room_owner_participant_id, 0) == 0

    def test_same_username_different_firebase_uids_do_not_share_study_minutes(self, api_client):
        shared_username = f"TEST_SAME_NAME_{uuid.uuid4().hex[:6]}"
        uid_a = f"TEST_uid_a_{uuid.uuid4().hex[:8]}"
        uid_b = f"TEST_uid_b_{uuid.uuid4().hex[:8]}"

        for firebase_uid in [uid_a, uid_b]:
            profile_res = api_client.post(
                f"{API_BASE}/profile",
                json={
                    "firebase_uid": firebase_uid,
                    "username": shared_username,
                    "email": f"{firebase_uid}@test.dev",
                    "avatar_url": None,
                },
                timeout=20,
            )
            assert profile_res.status_code == 200
            profile_data = profile_res.json()
            assert profile_data["firebase_uid"] == firebase_uid
            assert profile_data["username"] == shared_username

        room = _create_room(api_client, owner_name=shared_username)
        baseline = _leaderboard_totals_by_uid(api_client)

        started = _start_session(api_client, firebase_uid=uid_a, room_id=room["id"])
        _complete_session(api_client, session_id=started["id"], accumulated_seconds=420)

        after = _leaderboard_totals_by_uid(api_client)

        assert after.get(uid_a, 0) - baseline.get(uid_a, 0) == 420
        assert after.get(uid_b, 0) - baseline.get(uid_b, 0) == 0

    def test_room_timer_endpoint_still_updates_and_persists_after_identity_fix(self, api_client):
        room = _create_room(api_client, owner_name="TEST Timer Owner")

        timer_payload = {
            "is_running": True,
            "duration_minutes": 1,
            "remaining_seconds": 60,
            "started_at": "2026-02-01T12:00:00Z",
        }
        update_res = api_client.put(f"{API_BASE}/rooms/{room['id']}/timer", json=timer_payload, timeout=20)
        assert update_res.status_code == 200
        update_data = update_res.json()
        assert update_data["success"] is True

        room_res = api_client.get(f"{API_BASE}/rooms/{room['id']}", timeout=20)
        assert room_res.status_code == 200
        room_data = room_res.json()
        assert room_data["timer_state"]["is_running"] is True
        assert int(room_data["timer_state"]["remaining_seconds"]) == 60
        assert int(room_data["timer_state"]["duration_minutes"]) == 1