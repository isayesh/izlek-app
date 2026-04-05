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
    """HTTP client for leaderboard profile identity integration tests."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


def _create_completed_session(api_client, firebase_uid: str, room_id: str, seconds: int):
    start_res = api_client.post(
        f"{API_BASE}/study-sessions/start",
        json={"firebase_uid": firebase_uid, "room_id": room_id},
        timeout=20,
    )
    assert start_res.status_code == 200
    session = start_res.json()
    assert isinstance(session.get("id"), str)

    complete_res = api_client.put(
        f"{API_BASE}/study-sessions/{session['id']}/complete",
        json={"accumulated_seconds": seconds},
        timeout=20,
    )
    assert complete_res.status_code == 200
    completed = complete_res.json()
    assert completed["is_completed"] is True
    assert completed["accumulated_seconds"] == seconds


class TestLeaderboardProfileIdentityIntegration:
    """Leaderboard should follow central profile identity with safe fallback and keep ranking math."""

    # Module: leaderboard profile-avatar ownership must be exact by firebase_uid
    def test_leaderboard_duplicate_username_uses_exact_uid_profile_avatar(self, api_client):
        shared_name = f"Isa-{uuid.uuid4().hex[:4]}"
        uid_with_avatar = f"TEST_isa_a_{uuid.uuid4().hex[:8]}"
        uid_without_avatar = f"TEST_isa_b_{uuid.uuid4().hex[:8]}"
        strict_avatar_url = f"https://picsum.photos/seed/isa-a-{uuid.uuid4().hex[:6]}/80"

        profile_with_avatar_res = api_client.post(
            f"{API_BASE}/profile",
            json={
                "firebase_uid": uid_with_avatar,
                "username": shared_name,
                "email": f"{uid_with_avatar}@test.dev",
                "avatar_url": strict_avatar_url,
            },
            timeout=20,
        )
        assert profile_with_avatar_res.status_code == 200
        profile_with_avatar = profile_with_avatar_res.json()
        assert profile_with_avatar["firebase_uid"] == uid_with_avatar
        assert profile_with_avatar["avatar_url"] == strict_avatar_url

        profile_without_avatar_res = api_client.post(
            f"{API_BASE}/profile",
            json={
                "firebase_uid": uid_without_avatar,
                "username": shared_name,
                "email": f"{uid_without_avatar}@test.dev",
                "avatar_url": None,
            },
            timeout=20,
        )
        assert profile_without_avatar_res.status_code == 200
        profile_without_avatar = profile_without_avatar_res.json()
        assert profile_without_avatar["firebase_uid"] == uid_without_avatar
        assert profile_without_avatar["avatar_url"] is None

        room_res = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST Isa Room {uuid.uuid4().hex[:6]}",
                "owner_name": shared_name,
                "owner_avatar_url": f"https://example.com/room-avatar-{uuid.uuid4().hex[:6]}.png",
            },
            timeout=20,
        )
        assert room_res.status_code == 200
        room = room_res.json()

        _create_completed_session(api_client, uid_with_avatar, room["id"], 777)

        leaderboard_res = api_client.get(f"{API_BASE}/leaderboard", timeout=20)
        assert leaderboard_res.status_code == 200
        leaderboard = leaderboard_res.json()

        avatar_owner_entry = next((item for item in leaderboard if item["user_id"] == uid_with_avatar), None)
        same_name_other_uid_entry = next((item for item in leaderboard if item["user_id"] == uid_without_avatar), None)

        assert avatar_owner_entry is not None
        assert avatar_owner_entry["user_name"] == shared_name
        assert avatar_owner_entry["avatar_url"] == strict_avatar_url

        if same_name_other_uid_entry is not None:
            assert same_name_other_uid_entry["avatar_url"] != strict_avatar_url

    def test_leaderboard_prefers_profile_username_and_avatar(self, api_client):
        uid = f"TEST_lb_profile_{uuid.uuid4().hex[:10]}"
        username = f"TEST LB Profile {uuid.uuid4().hex[:4]}"
        avatar_url = f"https://picsum.photos/seed/lb-{uuid.uuid4().hex[:6]}/80"

        profile_res = api_client.post(
            f"{API_BASE}/profile",
            json={
                "firebase_uid": uid,
                "username": username,
                "email": f"{uid}@test.com",
                "avatar_url": avatar_url,
            },
            timeout=20,
        )
        assert profile_res.status_code == 200
        profile = profile_res.json()
        assert profile["username"] == username
        assert profile["avatar_url"] == avatar_url

        room_res = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST LB Room {uuid.uuid4().hex[:6]}",
                "owner_name": "Different Room Name",
                "owner_avatar_url": "https://example.com/room-only-avatar.png",
            },
            timeout=20,
        )
        assert room_res.status_code == 200
        room = room_res.json()

        _create_completed_session(api_client, uid, room["id"], 1111)

        leaderboard_res = api_client.get(f"{API_BASE}/leaderboard", timeout=20)
        assert leaderboard_res.status_code == 200
        leaderboard = leaderboard_res.json()
        entry = next((item for item in leaderboard if item["user_id"] == uid), None)
        assert entry is not None
        assert entry["user_name"] == username
        assert entry["avatar_url"] == avatar_url

    def test_leaderboard_prefers_handle_over_username(self, api_client):
        uid = f"TEST_lb_handle_{uuid.uuid4().hex[:10]}"
        handle = f"lb_{uuid.uuid4().hex[:8]}"

        profile_res = api_client.post(
            f"{API_BASE}/profile",
            json={
                "firebase_uid": uid,
                "username": "TEST Handle Username",
                "handle": handle,
                "email": f"{uid}@test.com",
                "avatar_url": None,
            },
            timeout=20,
        )
        assert profile_res.status_code == 200

        room_res = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST LB Handle Room {uuid.uuid4().hex[:6]}",
                "owner_name": "TEST Room Name",
                "owner_avatar_url": None,
            },
            timeout=20,
        )
        assert room_res.status_code == 200
        room = room_res.json()

        _create_completed_session(api_client, uid, room["id"], 944)

        leaderboard_res = api_client.get(f"{API_BASE}/leaderboard", timeout=20)
        assert leaderboard_res.status_code == 200
        leaderboard = leaderboard_res.json()
        entry = next((item for item in leaderboard if item["user_id"] == uid), None)
        assert entry is not None
        assert entry["user_name"] == f"@{handle}"

    def test_leaderboard_falls_back_to_room_identity_when_profile_missing(self, api_client):
        room_res = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST LB Room Fallback {uuid.uuid4().hex[:6]}",
                "owner_name": f"TEST Room Identity {uuid.uuid4().hex[:4]}",
                "owner_avatar_url": f"https://example.com/room-avatar-{uuid.uuid4().hex[:6]}.png",
            },
            timeout=20,
        )
        assert room_res.status_code == 200
        room = room_res.json()
        owner = room["participants"][0]

        _create_completed_session(api_client, owner["id"], room["id"], 1222)

        leaderboard_res = api_client.get(f"{API_BASE}/leaderboard", timeout=20)
        assert leaderboard_res.status_code == 200
        leaderboard = leaderboard_res.json()
        entry = next((item for item in leaderboard if item["user_id"] == owner["id"]), None)
        assert entry is not None
        assert entry["user_name"] == owner["name"]
        assert entry["avatar_url"] == owner["avatar_url"]

    def test_leaderboard_uses_safe_default_when_no_profile_and_no_room_identity(self, api_client):
        uid = f"TEST_lb_safe_{uuid.uuid4().hex[:10]}"
        orphan_room_id = f"room-{uuid.uuid4().hex[:8]}"

        _create_completed_session(api_client, uid, orphan_room_id, 1333)

        leaderboard_res = api_client.get(f"{API_BASE}/leaderboard", timeout=20)
        assert leaderboard_res.status_code == 200
        leaderboard = leaderboard_res.json()
        entry = next((item for item in leaderboard if item["user_id"] == uid), None)
        assert entry is not None
        assert entry["user_name"] == "Bilinmeyen Kullanıcı"
        assert entry["avatar_url"] is None

    def test_leaderboard_ranking_and_total_seconds_calculation_still_work(self, api_client):
        room_res = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST LB Ranking Room {uuid.uuid4().hex[:6]}",
                "owner_name": f"TEST Rank Owner {uuid.uuid4().hex[:4]}",
                "owner_avatar_url": None,
            },
            timeout=20,
        )
        assert room_res.status_code == 200
        room = room_res.json()

        uid_high = f"TEST_lb_rank_hi_{uuid.uuid4().hex[:8]}"
        uid_low = f"TEST_lb_rank_lo_{uuid.uuid4().hex[:8]}"

        _create_completed_session(api_client, uid_high, room["id"], 3000)
        _create_completed_session(api_client, uid_high, room["id"], 500)
        _create_completed_session(api_client, uid_low, room["id"], 1800)

        leaderboard_res = api_client.get(f"{API_BASE}/leaderboard", timeout=20)
        assert leaderboard_res.status_code == 200
        leaderboard = leaderboard_res.json()

        high_entry = next((item for item in leaderboard if item["user_id"] == uid_high), None)
        low_entry = next((item for item in leaderboard if item["user_id"] == uid_low), None)
        assert high_entry is not None
        assert low_entry is not None

        assert high_entry["total_seconds"] == 3500
        assert low_entry["total_seconds"] == 1800
        assert high_entry["rank"] < low_entry["rank"]
