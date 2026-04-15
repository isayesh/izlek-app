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
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


def _create_room(api_client, owner_id: str, owner_name: str):
    response = api_client.post(
        f"{API_BASE}/rooms",
        json={
            "name": f"TEST Break Room {uuid.uuid4().hex[:6]}",
            "owner_id": owner_id,
            "owner_name": owner_name,
        },
        timeout=20,
    )
    assert response.status_code == 200
    return response.json()


def _join_room(api_client, room_code: str, user_id: str, user_name: str):
    response = api_client.post(
        f"{API_BASE}/rooms/join",
        json={
            "room_code": room_code,
            "user_id": user_id,
            "user_name": user_name,
        },
        timeout=20,
    )
    assert response.status_code == 200
    return response.json()


def _start_session(api_client, firebase_uid: str, room_id: str):
    response = api_client.post(
        f"{API_BASE}/study-sessions/start",
        json={"firebase_uid": firebase_uid, "room_id": room_id},
        timeout=20,
    )
    assert response.status_code == 200
    return response.json()


def _update_session(api_client, session_id: str, accumulated_seconds: int):
    response = api_client.put(
        f"{API_BASE}/study-sessions/{session_id}/update",
        json={"accumulated_seconds": accumulated_seconds},
        timeout=20,
    )
    assert response.status_code == 200
    return response.json()


def _set_break_mode(api_client, room_id: str, participant_id: str, firebase_uid: str, is_on_break: bool):
    response = api_client.put(
        f"{API_BASE}/rooms/{room_id}/break-mode",
        json={
            "participant_id": participant_id,
            "firebase_uid": firebase_uid,
            "is_on_break": is_on_break,
        },
        timeout=20,
    )
    assert response.status_code == 200
    return response.json()


def _participant_by_id(room_data, participant_id: str):
    participant = next((item for item in room_data["participants"] if item["id"] == participant_id), None)
    assert participant is not None
    return participant


class TestRoomBreakModeRegression:
    def test_break_mode_updates_only_target_participant_and_preserves_room_timer(self, api_client):
        owner_participant_id = f"TEST_break_owner_participant_{uuid.uuid4().hex[:8]}"
        owner_auth_uid = f"TEST_break_owner_auth_{uuid.uuid4().hex[:8]}"
        other_participant_id = f"TEST_break_other_participant_{uuid.uuid4().hex[:8]}"

        room = _create_room(api_client, owner_id=owner_participant_id, owner_name="TEST Break Owner")
        updated_room = _join_room(
            api_client,
            room_code=room["code"],
            user_id=other_participant_id,
            user_name="TEST Break Other",
        )
        assert len(updated_room["participants"]) == 2

        timer_payload = {
            "is_running": True,
            "duration_minutes": 25,
            "remaining_seconds": 1200,
            "started_at": "2026-03-01T12:00:00Z",
        }
        timer_res = api_client.put(f"{API_BASE}/rooms/{room['id']}/timer", json=timer_payload, timeout=20)
        assert timer_res.status_code == 200

        room_with_break = _set_break_mode(
            api_client,
            room_id=room["id"],
            participant_id=owner_participant_id,
            firebase_uid=owner_auth_uid,
            is_on_break=True,
        )

        owner = _participant_by_id(room_with_break, owner_participant_id)
        other = _participant_by_id(room_with_break, other_participant_id)

        assert owner["is_on_break"] is True
        assert other.get("is_on_break", False) is False
        assert room_with_break["timer_state"]["is_running"] is True
        assert int(room_with_break["timer_state"]["remaining_seconds"]) == 1200

    def test_break_mode_blocks_target_session_growth_until_user_resumes(self, api_client):
        owner_participant_id = f"TEST_break_session_participant_{uuid.uuid4().hex[:8]}"
        owner_auth_uid = f"TEST_break_session_auth_{uuid.uuid4().hex[:8]}"
        other_auth_uid = f"TEST_break_session_other_auth_{uuid.uuid4().hex[:8]}"
        other_participant_id = f"TEST_break_session_other_participant_{uuid.uuid4().hex[:8]}"

        room = _create_room(api_client, owner_id=owner_participant_id, owner_name="TEST Break Session Owner")
        _join_room(
            api_client,
            room_code=room["code"],
            user_id=other_participant_id,
            user_name="TEST Break Session Other",
        )

        owner_session = _start_session(api_client, firebase_uid=owner_auth_uid, room_id=room["id"])
        other_session = _start_session(api_client, firebase_uid=other_auth_uid, room_id=room["id"])

        owner_active = _update_session(api_client, owner_session["id"], accumulated_seconds=120)
        other_active = _update_session(api_client, other_session["id"], accumulated_seconds=200)

        assert int(owner_active["accumulated_seconds"]) == 120
        assert int(other_active["accumulated_seconds"]) == 200
        assert owner_active.get("is_on_break", False) is False

        _set_break_mode(
            api_client,
            room_id=room["id"],
            participant_id=owner_participant_id,
            firebase_uid=owner_auth_uid,
            is_on_break=True,
        )

        owner_blocked = _update_session(api_client, owner_session["id"], accumulated_seconds=360)
        other_still_active = _update_session(api_client, other_session["id"], accumulated_seconds=260)

        assert int(owner_blocked["accumulated_seconds"]) == 120
        assert owner_blocked["is_on_break"] is True
        assert int(other_still_active["accumulated_seconds"]) == 260
        assert other_still_active.get("is_on_break", False) is False

        room_after_resume = _set_break_mode(
            api_client,
            room_id=room["id"],
            participant_id=owner_participant_id,
            firebase_uid=owner_auth_uid,
            is_on_break=False,
        )
        owner_after_resume = _participant_by_id(room_after_resume, owner_participant_id)
        assert owner_after_resume["is_on_break"] is False

        owner_resumed = _update_session(api_client, owner_session["id"], accumulated_seconds=180)
        assert int(owner_resumed["accumulated_seconds"]) == 180
        assert owner_resumed.get("is_on_break", False) is False
