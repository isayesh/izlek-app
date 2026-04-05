import os
import uuid

import pytest
import requests
from dotenv import dotenv_values
from pymongo import MongoClient


def _get_backend_base_url() -> str:
    env_base = os.environ.get("REACT_APP_BACKEND_URL")
    if env_base:
        return env_base.rstrip("/")

    frontend_env = dotenv_values("/app/frontend/.env")
    file_base = frontend_env.get("REACT_APP_BACKEND_URL")
    if file_base:
        return str(file_base).rstrip("/")

    pytest.skip("REACT_APP_BACKEND_URL missing; cannot run API tests")


BASE_URL = _get_backend_base_url()
API_BASE = f"{BASE_URL}/api"


def _get_mongo_handles():
    backend_env = dotenv_values("/app/backend/.env")
    mongo_url = os.environ.get("MONGO_URL") or backend_env.get("MONGO_URL")
    db_name = os.environ.get("DB_NAME") or backend_env.get("DB_NAME")
    if not mongo_url or not db_name:
        pytest.skip("MONGO_URL/DB_NAME missing; cannot clean friendship test data")

    mongo_client = MongoClient(mongo_url)
    database = mongo_client[db_name]
    return mongo_client, database["profiles"], database["friend_requests"], database["friends"]


@pytest.fixture
def api_client():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


@pytest.fixture
def uid_tracker():
    tracked_uids = []
    yield tracked_uids

    mongo_client, profiles_collection, requests_collection, friends_collection = _get_mongo_handles()
    if tracked_uids:
        profiles_collection.delete_many({"firebase_uid": {"$in": tracked_uids}})
        requests_collection.delete_many({"$or": [{"from_uid": {"$in": tracked_uids}}, {"to_uid": {"$in": tracked_uids}}]})
        friends_collection.delete_many({"$or": [{"user_uid": {"$in": tracked_uids}}, {"friend_uid": {"$in": tracked_uids}}]})
    mongo_client.close()


def _create_profile(api_client, firebase_uid: str, username: str, handle: str | None = None):
    payload = {
        "firebase_uid": firebase_uid,
        "username": username,
        "handle": handle,
        "email": f"{firebase_uid}@test.dev",
        "study_goal": "Düzenli tekrar",
        "daily_study_hours": 3,
        "avatar_url": f"https://example.com/{firebase_uid}.png",
    }
    response = api_client.post(f"{API_BASE}/profile", json=payload, timeout=20)
    assert response.status_code == 200
    return response.json()


class TestFriendshipNotificationsFlow:
    def test_user_search_returns_public_profile_preview_without_private_fields(self, api_client, uid_tracker):
        current_uid = f"TEST_friend_search_self_{uuid.uuid4().hex[:8]}"
        other_uid = f"TEST_friend_search_other_{uuid.uuid4().hex[:8]}"
        uid_tracker.extend([current_uid, other_uid])

        _create_profile(api_client, current_uid, "Arayan Kullanıcı", "arayan_user")
        other_profile = _create_profile(api_client, other_uid, "Bulunan Kullanıcı", "bulunan_user")

        response = api_client.get(
            f"{API_BASE}/users/search",
            params={"q": "bulunan"},
            headers={"X-Firebase-UID": current_uid},
            timeout=20,
        )
        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)

        result = next((item for item in results if item["profile_id"] == other_profile["id"]), None)
        assert result is not None
        assert result["username"] == "Bulunan Kullanıcı"
        assert result["handle"] == "bulunan_user"
        assert result["handle_display"] == "@bulunan_user"
        assert result["relationship_status"] == "none"
        assert "email" not in result
        assert "firebase_uid" not in result
        assert "id" not in result

    def test_duplicate_pending_and_self_request_are_blocked(self, api_client, uid_tracker):
        sender_uid = f"TEST_friend_sender_{uuid.uuid4().hex[:8]}"
        receiver_uid = f"TEST_friend_receiver_{uuid.uuid4().hex[:8]}"
        uid_tracker.extend([sender_uid, receiver_uid])

        sender_profile = _create_profile(api_client, sender_uid, "Gönderen Kullanıcı", "gonderen_user")
        receiver_profile = _create_profile(api_client, receiver_uid, "Alan Kullanıcı", "alan_user")

        first_response = api_client.post(
            f"{API_BASE}/friends/requests",
            json={"to_profile_id": receiver_profile["id"]},
            headers={"X-Firebase-UID": sender_uid},
            timeout=20,
        )
        assert first_response.status_code == 200
        assert first_response.json()["status"] == "pending"

        duplicate_response = api_client.post(
            f"{API_BASE}/friends/requests",
            json={"to_profile_id": receiver_profile["id"]},
            headers={"X-Firebase-UID": sender_uid},
            timeout=20,
        )
        assert duplicate_response.status_code == 409

        self_response = api_client.post(
            f"{API_BASE}/friends/requests",
            json={"to_profile_id": sender_profile["id"]},
            headers={"X-Firebase-UID": sender_uid},
            timeout=20,
        )
        assert self_response.status_code == 400

    def test_incoming_requests_accept_creates_bidirectional_friend_lists(self, api_client, uid_tracker):
        sender_uid = f"TEST_friend_accept_a_{uuid.uuid4().hex[:8]}"
        receiver_uid = f"TEST_friend_accept_b_{uuid.uuid4().hex[:8]}"
        uid_tracker.extend([sender_uid, receiver_uid])

        _create_profile(api_client, sender_uid, "Kabul Gönderen", "kabul_gonderen")
        receiver_profile = _create_profile(api_client, receiver_uid, "Kabul Alan", "kabul_alan")

        create_request_response = api_client.post(
            f"{API_BASE}/friends/requests",
            json={"to_profile_id": receiver_profile["id"]},
            headers={"X-Firebase-UID": sender_uid},
            timeout=20,
        )
        assert create_request_response.status_code == 200

        incoming_response = api_client.get(
            f"{API_BASE}/friends/requests/incoming",
            headers={"X-Firebase-UID": receiver_uid},
            timeout=20,
        )
        assert incoming_response.status_code == 200
        incoming_requests = incoming_response.json()
        assert len(incoming_requests) >= 1
        friend_request = incoming_requests[0]
        assert friend_request["from_username"] == "Kabul Gönderen"
        assert friend_request["from_handle"] == "kabul_gonderen"
        assert "from_uid" not in friend_request
        assert "email" not in friend_request

        accept_response = api_client.post(
            f"{API_BASE}/friends/requests/{friend_request['id']}/accept",
            headers={"X-Firebase-UID": receiver_uid},
            timeout=20,
        )
        assert accept_response.status_code == 200
        assert accept_response.json()["status"] == "accepted"

        sender_friends_response = api_client.get(
            f"{API_BASE}/friends",
            headers={"X-Firebase-UID": sender_uid},
            timeout=20,
        )
        receiver_friends_response = api_client.get(
            f"{API_BASE}/friends",
            headers={"X-Firebase-UID": receiver_uid},
            timeout=20,
        )
        assert sender_friends_response.status_code == 200
        assert receiver_friends_response.status_code == 200

        sender_friends = sender_friends_response.json()
        receiver_friends = receiver_friends_response.json()
        assert any(friend["handle"] == "kabul_alan" for friend in sender_friends)
        assert any(friend["handle"] == "kabul_gonderen" for friend in receiver_friends)

    def test_reject_clears_pending_list_and_allows_resend(self, api_client, uid_tracker):
        sender_uid = f"TEST_friend_reject_a_{uuid.uuid4().hex[:8]}"
        receiver_uid = f"TEST_friend_reject_b_{uuid.uuid4().hex[:8]}"
        uid_tracker.extend([sender_uid, receiver_uid])

        _create_profile(api_client, sender_uid, "Red Gönderen", "red_gonderen")
        receiver_profile = _create_profile(api_client, receiver_uid, "Red Alan", "red_alan")

        first_request = api_client.post(
            f"{API_BASE}/friends/requests",
            json={"to_profile_id": receiver_profile["id"]},
            headers={"X-Firebase-UID": sender_uid},
            timeout=20,
        )
        assert first_request.status_code == 200

        incoming_response = api_client.get(
            f"{API_BASE}/friends/requests/incoming",
            headers={"X-Firebase-UID": receiver_uid},
            timeout=20,
        )
        assert incoming_response.status_code == 200
        pending_request = incoming_response.json()[0]

        reject_response = api_client.post(
            f"{API_BASE}/friends/requests/{pending_request['id']}/reject",
            headers={"X-Firebase-UID": receiver_uid},
            timeout=20,
        )
        assert reject_response.status_code == 200
        assert reject_response.json()["status"] == "rejected"

        incoming_after_reject = api_client.get(
            f"{API_BASE}/friends/requests/incoming",
            headers={"X-Firebase-UID": receiver_uid},
            timeout=20,
        )
        assert incoming_after_reject.status_code == 200
        assert incoming_after_reject.json() == []

        resend_response = api_client.post(
            f"{API_BASE}/friends/requests",
            json={"to_profile_id": receiver_profile["id"]},
            headers={"X-Firebase-UID": sender_uid},
            timeout=20,
        )
        assert resend_response.status_code == 200
        assert resend_response.json()["status"] == "pending"