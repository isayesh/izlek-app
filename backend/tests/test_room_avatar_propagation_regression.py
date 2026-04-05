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
    """Avatar propagation regression test client."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


class TestRoomAvatarPropagationRegression:
    """Room participant/chat avatar source must follow central profile identity payload."""

    def test_room_participant_uses_profile_avatar_when_present(self, api_client):
        user_uid = f"TEST_avatar_owner_{uuid.uuid4().hex[:10]}"
        username = f"TEST Avatar {uuid.uuid4().hex[:4]}"
        avatar_url = f"https://example.com/avatar-{uuid.uuid4().hex[:6]}.png"

        profile_res = api_client.post(
            f"{API_BASE}/profile",
            json={
                "firebase_uid": user_uid,
                "username": username,
                "email": f"{user_uid}@test.com",
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
                "name": f"TEST Avatar Room {uuid.uuid4().hex[:6]}",
                "owner_name": profile["username"],
                "owner_avatar_url": profile["avatar_url"],
                "owner_study_field": "EA",
            },
            timeout=20,
        )
        assert room_res.status_code == 200
        room = room_res.json()

        owner = room["participants"][0]
        assert owner["name"] == username
        assert owner["avatar_url"] == avatar_url

        persisted_room_res = api_client.get(f"{API_BASE}/rooms/{room['id']}", timeout=20)
        assert persisted_room_res.status_code == 200
        persisted_room = persisted_room_res.json()
        assert persisted_room["participants"][0]["name"] == username
        assert persisted_room["participants"][0]["avatar_url"] == avatar_url

    def test_message_uses_profile_avatar_when_present(self, api_client):
        username = f"TEST Chat Avatar {uuid.uuid4().hex[:4]}"
        avatar_url = f"https://example.com/chat-avatar-{uuid.uuid4().hex[:6]}.png"

        room_res = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST Chat Avatar Room {uuid.uuid4().hex[:6]}",
                "owner_name": username,
                "owner_avatar_url": avatar_url,
                "owner_study_field": "Sayısal",
            },
            timeout=20,
        )
        assert room_res.status_code == 200
        room = room_res.json()
        owner = room["participants"][0]

        msg_res = api_client.post(
            f"{API_BASE}/messages",
            json={
                "room_id": room["id"],
                "user_id": owner["id"],
                "user_name": owner["name"],
                "user_avatar_url": owner["avatar_url"],
                "content": f"TEST avatar msg {uuid.uuid4().hex[:6]}",
            },
            timeout=20,
        )
        assert msg_res.status_code == 200
        sent = msg_res.json()
        assert sent["user_name"] == username
        assert sent["user_avatar_url"] == avatar_url

        list_res = api_client.get(f"{API_BASE}/messages/{room['id']}", timeout=20)
        assert list_res.status_code == 200
        messages = list_res.json()
        assert any(
            m["id"] == sent["id"]
            and m["user_name"] == username
            and m["user_avatar_url"] == avatar_url
            for m in messages
        )

    def test_avatar_can_be_absent_without_backend_errors(self, api_client):
        room_res = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST No Avatar Room {uuid.uuid4().hex[:6]}",
                "owner_name": f"TEST NoAvatar {uuid.uuid4().hex[:4]}",
                "owner_avatar_url": None,
            },
            timeout=20,
        )
        assert room_res.status_code == 200
        room = room_res.json()

        owner = room["participants"][0]
        assert owner["avatar_url"] is None

        msg_res = api_client.post(
            f"{API_BASE}/messages",
            json={
                "room_id": room["id"],
                "user_id": owner["id"],
                "user_name": owner["name"],
                "user_avatar_url": None,
                "content": f"TEST no avatar msg {uuid.uuid4().hex[:6]}",
            },
            timeout=20,
        )
        assert msg_res.status_code == 200
        sent = msg_res.json()
        assert sent["user_avatar_url"] is None