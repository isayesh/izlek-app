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


def _get_mongo_collection():
    backend_env = dotenv_values("/app/backend/.env")
    mongo_url = os.environ.get("MONGO_URL") or backend_env.get("MONGO_URL")
    db_name = os.environ.get("DB_NAME") or backend_env.get("DB_NAME")
    if not mongo_url or not db_name:
        pytest.skip("MONGO_URL/DB_NAME missing; cannot verify duplicate profile persistence")

    mongo_client = MongoClient(mongo_url)
    return mongo_client, mongo_client[db_name]["profiles"]


@pytest.fixture
def api_client():
    """Shared HTTP session for profile API regression tests."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


@pytest.fixture
def uid_tracker():
    """Tracks created firebase_uids and cleans profile documents after tests."""
    tracked_uids = []
    yield tracked_uids

    mongo_client, profiles_collection = _get_mongo_collection()
    if tracked_uids:
        profiles_collection.delete_many({"firebase_uid": {"$in": tracked_uids}})
    mongo_client.close()


class TestProfileSystemRegression:
    """Profile save/update/persistence/duplicate/user-isolation and firebase_uid consistency checks."""

    def test_valid_handle_persists_on_create_and_update(self, api_client, uid_tracker):
        firebase_uid = f"TEST_profile_handle_{uuid.uuid4().hex[:10]}"
        uid_tracker.append(firebase_uid)

        create_payload = {
            "firebase_uid": firebase_uid,
            "username": "TEST Handle User",
            "handle": "test_handle_01",
            "email": "handle.user@test.com",
            "study_goal": "Handle test",
            "daily_study_hours": 2,
            "avatar_url": None,
        }

        create_response = api_client.post(f"{API_BASE}/profile", json=create_payload, timeout=20)
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["handle"] == "test_handle_01"

        get_response = api_client.get(f"{API_BASE}/profile", params={"firebase_uid": firebase_uid}, timeout=20)
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["handle"] == "test_handle_01"

        update_payload = {
            **create_payload,
            "handle": "test_handle_02",
        }
        update_response = api_client.put(f"{API_BASE}/profile", json=update_payload, timeout=20)
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["handle"] == "test_handle_02"

    def test_duplicate_handle_is_rejected_with_409(self, api_client, uid_tracker):
        first_uid = f"TEST_profile_handle_dup_a_{uuid.uuid4().hex[:8]}"
        second_uid = f"TEST_profile_handle_dup_b_{uuid.uuid4().hex[:8]}"
        uid_tracker.extend([first_uid, second_uid])

        shared_handle = f"dup_{uuid.uuid4().hex[:8]}"

        first_response = api_client.post(
            f"{API_BASE}/profile",
            json={
                "firebase_uid": first_uid,
                "username": "TEST Duplicate Handle One",
                "handle": shared_handle,
                "email": f"{first_uid}@test.com",
                "avatar_url": None,
            },
            timeout=20,
        )
        assert first_response.status_code == 200

        second_response = api_client.post(
            f"{API_BASE}/profile",
            json={
                "firebase_uid": second_uid,
                "username": "TEST Duplicate Handle Two",
                "handle": shared_handle,
                "email": f"{second_uid}@test.com",
                "avatar_url": None,
            },
            timeout=20,
        )
        assert second_response.status_code == 409
        assert "handle" in second_response.json()["detail"].lower()

    def test_invalid_handle_is_rejected_with_400(self, api_client, uid_tracker):
        firebase_uid = f"TEST_profile_handle_invalid_{uuid.uuid4().hex[:8]}"
        uid_tracker.append(firebase_uid)

        response = api_client.post(
            f"{API_BASE}/profile",
            json={
                "firebase_uid": firebase_uid,
                "username": "TEST Invalid Handle",
                "handle": "Invalid Handle",
                "email": "invalid.handle@test.com",
                "avatar_url": None,
            },
            timeout=20,
        )
        assert response.status_code == 400
        assert "handle" in response.json()["detail"].lower()

    def test_save_then_refresh_persists_profile(self, api_client, uid_tracker):
        firebase_uid = f"TEST_profile_refresh_{uuid.uuid4().hex[:10]}"
        uid_tracker.append(firebase_uid)

        create_payload = {
            "firebase_uid": firebase_uid,
            "username": "TEST Refresh User",
            "email": "refresh.user@test.com",
            "study_goal": "İlk hedef",
            "daily_study_hours": 2,
            "avatar_url": None,
        }

        create_response = api_client.post(f"{API_BASE}/profile", json=create_payload, timeout=20)
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["firebase_uid"] == firebase_uid
        assert created["username"] == "TEST Refresh User"

        # Refresh simulation: new GET call should return persisted data
        get_response = api_client.get(f"{API_BASE}/profile", params={"firebase_uid": firebase_uid}, timeout=20)
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["id"] == created["id"]
        assert fetched["username"] == create_payload["username"]
        assert fetched["study_goal"] == create_payload["study_goal"]

    def test_update_then_refresh_persists_changes(self, api_client, uid_tracker):
        firebase_uid = f"TEST_profile_update_{uuid.uuid4().hex[:10]}"
        uid_tracker.append(firebase_uid)

        create_payload = {
            "firebase_uid": firebase_uid,
            "username": "TEST Before Update",
            "email": "update.user@test.com",
            "study_goal": "Önce",
            "daily_study_hours": 1.5,
            "avatar_url": None,
        }
        create_response = api_client.post(f"{API_BASE}/profile", json=create_payload, timeout=20)
        assert create_response.status_code == 200
        created = create_response.json()

        update_payload = {
            "firebase_uid": firebase_uid,
            "username": "TEST After Update",
            "email": "update.user@test.com",
            "study_goal": "Sonra",
            "daily_study_hours": 3,
            "avatar_url": None,
        }
        update_response = api_client.put(f"{API_BASE}/profile", json=update_payload, timeout=20)
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["id"] == created["id"]
        assert updated["username"] == "TEST After Update"
        assert updated["study_goal"] == "Sonra"

        # Refresh simulation
        get_after_update = api_client.get(
            f"{API_BASE}/profile", params={"firebase_uid": firebase_uid}, timeout=20
        )
        assert get_after_update.status_code == 200
        persisted = get_after_update.json()
        assert persisted["id"] == created["id"]
        assert persisted["username"] == "TEST After Update"
        assert persisted["study_goal"] == "Sonra"

    def test_only_one_profile_exists_per_firebase_uid(self, api_client, uid_tracker):
        firebase_uid = f"TEST_profile_single_{uuid.uuid4().hex[:10]}"
        uid_tracker.append(firebase_uid)

        first_payload = {
            "firebase_uid": firebase_uid,
            "username": "TEST Single One",
            "email": "single.user@test.com",
            "study_goal": "Hedef 1",
            "daily_study_hours": 2,
            "avatar_url": None,
        }
        first_response = api_client.post(f"{API_BASE}/profile", json=first_payload, timeout=20)
        assert first_response.status_code == 200
        first_profile = first_response.json()

        # Repeated create should behave as upsert and keep same profile id
        second_payload = {
            "firebase_uid": firebase_uid,
            "username": "TEST Single Two",
            "email": "single.user@test.com",
            "study_goal": "Hedef 2",
            "daily_study_hours": 2.5,
            "avatar_url": None,
        }
        second_response = api_client.post(f"{API_BASE}/profile", json=second_payload, timeout=20)
        assert second_response.status_code == 200
        second_profile = second_response.json()
        assert second_profile["id"] == first_profile["id"]

        # Update should not create extra docs
        update_response = api_client.put(f"{API_BASE}/profile", json=second_payload, timeout=20)
        assert update_response.status_code == 200

        mongo_client, profiles_collection = _get_mongo_collection()
        count = profiles_collection.count_documents({"firebase_uid": firebase_uid})
        mongo_client.close()
        assert count == 1

    def test_new_user_profile_is_isolated(self, api_client, uid_tracker):
        firebase_uid_user_1 = f"TEST_profile_u1_{uuid.uuid4().hex[:8]}"
        firebase_uid_user_2 = f"TEST_profile_u2_{uuid.uuid4().hex[:8]}"
        uid_tracker.extend([firebase_uid_user_1, firebase_uid_user_2])

        user_1_payload = {
            "firebase_uid": firebase_uid_user_1,
            "username": "TEST User One",
            "email": "user.one@test.com",
            "study_goal": "User1 Goal",
            "daily_study_hours": 2,
            "avatar_url": None,
        }
        user_2_payload = {
            "firebase_uid": firebase_uid_user_2,
            "username": "TEST User Two",
            "email": "user.two@test.com",
            "study_goal": "User2 Goal",
            "daily_study_hours": 4,
            "avatar_url": None,
        }

        user_1_create = api_client.post(f"{API_BASE}/profile", json=user_1_payload, timeout=20)
        assert user_1_create.status_code == 200
        user_2_create = api_client.post(f"{API_BASE}/profile", json=user_2_payload, timeout=20)
        assert user_2_create.status_code == 200

        user_1_get = api_client.get(
            f"{API_BASE}/profile", params={"firebase_uid": firebase_uid_user_1}, timeout=20
        )
        user_2_get = api_client.get(
            f"{API_BASE}/profile", params={"firebase_uid": firebase_uid_user_2}, timeout=20
        )
        assert user_1_get.status_code == 200
        assert user_2_get.status_code == 200

        user_1_data = user_1_get.json()
        user_2_data = user_2_get.json()
        assert user_1_data["username"] == "TEST User One"
        assert user_2_data["username"] == "TEST User Two"
        assert user_1_data["id"] != user_2_data["id"]

    def test_firebase_uid_required_and_consistently_applied(self, api_client, uid_tracker):
        firebase_uid = f"TEST_profile_uid_check_{uuid.uuid4().hex[:8]}"
        uid_tracker.append(firebase_uid)

        create_payload = {
            "firebase_uid": firebase_uid,
            "username": "TEST UID Check",
            "email": "uid.check@test.com",
            "study_goal": "UID",
            "daily_study_hours": 2,
            "avatar_url": None,
        }

        create_response = api_client.post(f"{API_BASE}/profile", json=create_payload, timeout=20)
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["firebase_uid"] == firebase_uid

        # GET must require firebase_uid query
        missing_uid_get = api_client.get(f"{API_BASE}/profile", timeout=20)
        assert missing_uid_get.status_code == 422

        # PUT with unknown firebase_uid should not upsert and must return 404
        unknown_uid_payload = {
            "firebase_uid": f"TEST_unknown_uid_{uuid.uuid4().hex[:8]}",
            "username": "TEST Unknown",
            "email": "unknown@test.com",
            "study_goal": "none",
            "daily_study_hours": 1,
            "avatar_url": None,
        }
        update_unknown = api_client.put(f"{API_BASE}/profile", json=unknown_uid_payload, timeout=20)
        assert update_unknown.status_code == 404
