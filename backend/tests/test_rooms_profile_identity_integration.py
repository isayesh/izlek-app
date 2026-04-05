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
    """Shared HTTP client for rooms/profile identity integration tests."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


class TestRoomsProfileIdentityIntegration:
    """Rooms create/join identity and avatar propagation from profile payloads."""

    def test_create_room_uses_profile_username_and_avatar_identity(self, api_client):
        owner_uid = f"TEST_rooms_owner_{uuid.uuid4().hex[:10]}"
        owner_username = f"TEST Owner {uuid.uuid4().hex[:4]}"
        owner_avatar = f"https://example.com/avatar-owner-{uuid.uuid4().hex[:6]}.png"
        owner_study_field = "EA"

        profile_response = api_client.post(
            f"{API_BASE}/profiles",
            json={
                "firebase_uid": owner_uid,
                "name": owner_username,
                "username": owner_username,
                "avatar_url": owner_avatar,
                "study_field": owner_study_field,
            },
            timeout=20,
        )
        assert profile_response.status_code == 200
        profile_data = profile_response.json()
        assert profile_data["username"] == owner_username
        assert profile_data["avatar_url"] == owner_avatar

        create_room_response = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST Identity Room {uuid.uuid4().hex[:6]}",
                "owner_name": profile_data["username"],
                "owner_id": owner_uid,
                "owner_avatar_url": profile_data["avatar_url"],
            },
            timeout=20,
        )
        assert create_room_response.status_code == 200
        room = create_room_response.json()

        assert room["participants"][0]["name"] == owner_username
        assert room["participants"][0]["avatar_url"] == owner_avatar
        assert room["participants"][0]["study_field"] == owner_study_field

        get_room_response = api_client.get(f"{API_BASE}/rooms/{room['id']}", timeout=20)
        assert get_room_response.status_code == 200
        persisted_room = get_room_response.json()
        assert persisted_room["participants"][0]["name"] == owner_username
        assert persisted_room["participants"][0]["avatar_url"] == owner_avatar
        assert persisted_room["participants"][0]["study_field"] == owner_study_field

    def test_join_room_uses_profile_username_and_avatar_identity(self, api_client):
        owner_uid = f"TEST_rooms_owner_{uuid.uuid4().hex[:10]}"
        owner_profile_response = api_client.post(
            f"{API_BASE}/profiles",
            json={
                "firebase_uid": owner_uid,
                "name": "TEST Room Owner",
                "username": "TEST Room Owner",
                "study_field": "Sayısal",
            },
            timeout=20,
        )
        assert owner_profile_response.status_code == 200

        owner_response = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST Join Room {uuid.uuid4().hex[:6]}",
                "owner_name": "TEST Room Owner",
                "owner_id": owner_uid,
                "owner_avatar_url": None,
            },
            timeout=20,
        )
        assert owner_response.status_code == 200
        room = owner_response.json()

        joiner_uid = f"TEST_rooms_joiner_{uuid.uuid4().hex[:10]}"
        joiner_username = f"TEST Joiner {uuid.uuid4().hex[:4]}"
        joiner_avatar = f"https://example.com/avatar-joiner-{uuid.uuid4().hex[:6]}.png"
        joiner_study_field = "Sözel"

        joiner_profile_response = api_client.post(
            f"{API_BASE}/profiles",
            json={
                "firebase_uid": joiner_uid,
                "name": joiner_username,
                "username": joiner_username,
                "avatar_url": joiner_avatar,
                "study_field": joiner_study_field,
            },
            timeout=20,
        )
        assert joiner_profile_response.status_code == 200
        joiner_profile = joiner_profile_response.json()
        assert joiner_profile["username"] == joiner_username

        join_response = api_client.post(
            f"{API_BASE}/rooms/join",
            json={
                "room_code": room["code"],
                "user_id": joiner_uid,
                "user_name": joiner_profile["username"],
                "user_avatar_url": joiner_profile["avatar_url"],
            },
            timeout=20,
        )
        assert join_response.status_code == 200
        joined_room = join_response.json()

        joined_participant = joined_room["participants"][-1]
        assert joined_participant["name"] == joiner_username
        assert joined_participant["avatar_url"] == joiner_avatar
        assert joined_participant["study_field"] == joiner_study_field

        get_room_response = api_client.get(f"{API_BASE}/rooms/{room['id']}", timeout=20)
        assert get_room_response.status_code == 200
        persisted_room = get_room_response.json()
        persisted_joiner = persisted_room["participants"][-1]
        assert persisted_joiner["name"] == joiner_username
        assert persisted_joiner["avatar_url"] == joiner_avatar
        assert persisted_joiner["study_field"] == joiner_study_field

    def test_chat_message_keeps_avatar_and_identity(self, api_client):
        owner_uid = f"TEST_chat_owner_{uuid.uuid4().hex[:10]}"
        profile_response = api_client.post(
            f"{API_BASE}/profiles",
            json={
                "firebase_uid": owner_uid,
                "name": "TEST Chat Owner",
                "username": "TEST Chat Owner",
                "avatar_url": "https://example.com/chat-owner.png",
                "study_field": "EA",
            },
            timeout=20,
        )
        assert profile_response.status_code == 200

        create_room_response = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST Chat Room {uuid.uuid4().hex[:6]}",
                "owner_name": "TEST Chat Owner",
                "owner_id": owner_uid,
                "owner_avatar_url": "https://example.com/chat-owner.png",
            },
            timeout=20,
        )
        assert create_room_response.status_code == 200
        room = create_room_response.json()
        owner = room["participants"][0]

        message_payload = {
            "room_id": room["id"],
            "user_id": owner_uid,
            "user_name": owner["name"],
            "user_avatar_url": owner["avatar_url"],
            "content": f"TEST message {uuid.uuid4().hex[:6]}",
        }
        send_message_response = api_client.post(f"{API_BASE}/messages", json=message_payload, timeout=20)
        assert send_message_response.status_code == 200
        sent_message = send_message_response.json()
        assert sent_message["user_name"] == owner["name"]
        assert sent_message["user_avatar_url"] == owner["avatar_url"]
        assert sent_message["user_study_field"] == "EA"

        get_messages_response = api_client.get(f"{API_BASE}/messages/{room['id']}", timeout=20)
        assert get_messages_response.status_code == 200
        messages = get_messages_response.json()
        assert isinstance(messages, list)
        assert any(
            m["content"] == message_payload["content"]
            and m["user_name"] == owner["name"]
            and m["user_avatar_url"] == owner["avatar_url"]
            and m.get("user_study_field") == "EA"
            for m in messages
        )
