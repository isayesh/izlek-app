import os
import uuid
from datetime import datetime, timedelta

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
        pytest.skip("MONGO_URL/DB_NAME missing; cannot clean streak test data")

    mongo_client = MongoClient(mongo_url)
    database = mongo_client[db_name]
    return mongo_client, database["profiles"], database["study_sessions"]


@pytest.fixture
def api_client():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


@pytest.fixture
def uid_tracker():
    tracked_uids = []
    yield tracked_uids

    mongo_client, profiles_collection, study_sessions_collection = _get_mongo_handles()
    if tracked_uids:
        profiles_collection.delete_many({"firebase_uid": {"$in": tracked_uids}})
        study_sessions_collection.delete_many({"firebase_uid": {"$in": tracked_uids}})
    mongo_client.close()


def _create_profile(api_client, firebase_uid: str):
    response = api_client.post(
        f"{API_BASE}/profile",
        json={
            "firebase_uid": firebase_uid,
            "username": f"User {firebase_uid[-4:]}",
            "email": f"{firebase_uid}@test.dev",
            "study_goal": "Streak test",
            "daily_study_hours": 2,
            "avatar_url": None,
        },
        timeout=20,
    )
    assert response.status_code == 200
    return response.json()


def _complete_session(api_client, firebase_uid: str, seconds: int = 600):
    start_response = api_client.post(
        f"{API_BASE}/study-sessions/start",
        json={"firebase_uid": firebase_uid, "room_id": f"room-{uuid.uuid4().hex[:8]}"},
        timeout=20,
    )
    assert start_response.status_code == 200
    session = start_response.json()

    complete_response = api_client.put(
        f"{API_BASE}/study-sessions/{session['id']}/complete",
        json={"accumulated_seconds": seconds},
        timeout=20,
    )
    assert complete_response.status_code == 200


def _get_profile(api_client, firebase_uid: str):
    response = api_client.get(f"{API_BASE}/profile", params={"firebase_uid": firebase_uid}, timeout=20)
    assert response.status_code == 200
    return response.json()


def _set_profile_streak_state(firebase_uid: str, streak_count: int, last_active_date: str):
    mongo_client, profiles_collection, _ = _get_mongo_handles()
    profiles_collection.update_one(
        {"firebase_uid": firebase_uid},
        {"$set": {"streak_count": streak_count, "last_active_date": last_active_date}},
    )
    mongo_client.close()


class TestStreakSystem:
    def test_first_completed_session_sets_streak_to_one(self, api_client, uid_tracker):
        firebase_uid = f"TEST_streak_first_{uuid.uuid4().hex[:8]}"
        uid_tracker.append(firebase_uid)
        _create_profile(api_client, firebase_uid)

        _complete_session(api_client, firebase_uid)
        profile = _get_profile(api_client, firebase_uid)

        assert profile["streak_count"] == 1
        assert profile["last_active_date"] == datetime.now().date().isoformat()

    def test_same_day_completion_does_not_double_count(self, api_client, uid_tracker):
        firebase_uid = f"TEST_streak_same_day_{uuid.uuid4().hex[:8]}"
        uid_tracker.append(firebase_uid)
        _create_profile(api_client, firebase_uid)

        _complete_session(api_client, firebase_uid)
        _complete_session(api_client, firebase_uid)
        profile = _get_profile(api_client, firebase_uid)

        assert profile["streak_count"] == 1

    def test_next_day_completion_increments_streak(self, api_client, uid_tracker):
        firebase_uid = f"TEST_streak_next_day_{uuid.uuid4().hex[:8]}"
        uid_tracker.append(firebase_uid)
        _create_profile(api_client, firebase_uid)

        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
        _set_profile_streak_state(firebase_uid, streak_count=4, last_active_date=yesterday)

        _complete_session(api_client, firebase_uid)
        profile = _get_profile(api_client, firebase_uid)

        assert profile["streak_count"] == 5
        assert profile["last_active_date"] == datetime.now().date().isoformat()

    def test_gap_resets_streak_to_one(self, api_client, uid_tracker):
        firebase_uid = f"TEST_streak_reset_{uuid.uuid4().hex[:8]}"
        uid_tracker.append(firebase_uid)
        _create_profile(api_client, firebase_uid)

        old_date = (datetime.now().date() - timedelta(days=3)).isoformat()
        _set_profile_streak_state(firebase_uid, streak_count=7, last_active_date=old_date)

        _complete_session(api_client, firebase_uid)
        profile = _get_profile(api_client, firebase_uid)

        assert profile["streak_count"] == 1
        assert profile["last_active_date"] == datetime.now().date().isoformat()