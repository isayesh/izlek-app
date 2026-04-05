import os
import uuid

import pytest
import requests
from dotenv import dotenv_values
from pymongo import MongoClient


def _get_base_url() -> str:
    env_base = os.environ.get("REACT_APP_BACKEND_URL")
    if env_base:
        return env_base.rstrip("/")

    frontend_env = dotenv_values("/app/frontend/.env")
    file_base = frontend_env.get("REACT_APP_BACKEND_URL")
    if file_base:
        return str(file_base).rstrip("/")

    pytest.skip("REACT_APP_BACKEND_URL missing; cannot run room privacy tests")


def _get_rooms_collection():
    backend_env = dotenv_values("/app/backend/.env")
    mongo_url = os.environ.get("MONGO_URL") or backend_env.get("MONGO_URL")
    db_name = os.environ.get("DB_NAME") or backend_env.get("DB_NAME")
    if not mongo_url or not db_name:
        pytest.skip("MONGO_URL/DB_NAME missing; cannot inspect room documents")

    mongo_client = MongoClient(mongo_url)
    return mongo_client, mongo_client[db_name]["rooms"]


BASE_URL = _get_base_url()
API_BASE = f"{BASE_URL}/api"


@pytest.fixture
def api_client():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


class TestRoomPrivacyFlow:
    def test_public_room_join_flow_still_works_without_password(self, api_client):
        create_response = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST Public Room {uuid.uuid4().hex[:6]}",
                "owner_name": "TEST Public Owner",
                "room_type": "public",
            },
            timeout=20,
        )
        assert create_response.status_code == 200
        room = create_response.json()
        assert room["room_type"] == "public"
        assert room["is_private"] is False

        join_response = api_client.post(
            f"{API_BASE}/rooms/join",
            json={
                "room_code": room["code"],
                "user_name": "TEST Public Joiner",
            },
            timeout=20,
        )
        assert join_response.status_code == 200
        joined_room = join_response.json()
        assert any(participant["name"] == "TEST Public Joiner" for participant in joined_room["participants"])

    def test_private_room_password_is_hashed_and_not_exposed(self, api_client):
        plain_password = "roompass123"
        create_response = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST Private Room {uuid.uuid4().hex[:6]}",
                "owner_name": "TEST Private Owner",
                "room_type": "private",
                "room_password": plain_password,
            },
            timeout=20,
        )
        assert create_response.status_code == 200
        room = create_response.json()
        assert room["room_type"] == "private"
        assert room["is_private"] is True
        assert "room_password_hash" not in room

        mongo_client, rooms_collection = _get_rooms_collection()
        try:
            stored_room = rooms_collection.find_one({"id": room["id"]}, {"_id": 0})
            assert stored_room is not None
            assert stored_room["room_password_hash"] != plain_password
            assert stored_room["room_password_hash"]
        finally:
            mongo_client.close()

    def test_private_room_requires_correct_password_to_join(self, api_client):
        password = "secure123"
        create_response = api_client.post(
            f"{API_BASE}/rooms",
            json={
                "name": f"TEST Join Private {uuid.uuid4().hex[:6]}",
                "owner_name": "TEST Owner",
                "room_type": "private",
                "room_password": password,
            },
            timeout=20,
        )
        assert create_response.status_code == 200
        room = create_response.json()

        missing_password_response = api_client.post(
            f"{API_BASE}/rooms/join",
            json={
                "room_code": room["code"],
                "user_name": "TEST Missing Password",
            },
            timeout=20,
        )
        assert missing_password_response.status_code == 403

        wrong_password_response = api_client.post(
            f"{API_BASE}/rooms/join",
            json={
                "room_code": room["code"],
                "user_name": "TEST Wrong Password",
                "room_password": "wrongpass",
            },
            timeout=20,
        )
        assert wrong_password_response.status_code == 403

        correct_password_response = api_client.post(
            f"{API_BASE}/rooms/join",
            json={
                "room_code": room["code"],
                "user_name": "TEST Correct Password",
                "room_password": password,
            },
            timeout=20,
        )
        assert correct_password_response.status_code == 200
        joined_room = correct_password_response.json()
        assert any(participant["name"] == "TEST Correct Password" for participant in joined_room["participants"])