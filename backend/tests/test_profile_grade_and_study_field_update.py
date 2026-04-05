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

    pytest.skip("REACT_APP_BACKEND_URL missing; cannot run profile update tests")


BASE_URL = _get_base_url()
API_BASE = f"{BASE_URL}/api"


@pytest.fixture
def api_client():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


class TestProfileGradeAndStudyFieldUpdate:
    def test_simple_profile_update_adds_grade_and_study_field_without_clearing_legacy_fields(self, api_client):
        user_uid = f"TEST_profile_grade_{uuid.uuid4().hex[:10]}"
        email = f"{user_uid}@test.com"

        create_response = api_client.post(
            f"{API_BASE}/profiles",
            json={
                "firebase_uid": user_uid,
                "name": "TEST Profile Grade",
                "username": "TEST Profile Grade",
                "email": email,
                "study_goal": "Matematik netlerini artırmak",
                "daily_study_hours": 3,
                "study_field": "EA",
            },
            timeout=20,
        )
        assert create_response.status_code == 200

        update_response = api_client.put(
            f"{API_BASE}/profile",
            json={
                "firebase_uid": user_uid,
                "username": "TEST Profile Grade Updated",
                "email": email,
                "grade_level": "mezun",
                "study_field": "Dil",
            },
            timeout=20,
        )
        assert update_response.status_code == 200
        updated_profile = update_response.json()

        assert updated_profile["username"] == "TEST Profile Grade Updated"
        assert updated_profile["grade_level"] == "mezun"
        assert updated_profile["study_field"] == "Dil"
        assert updated_profile["study_goal"] == "Matematik netlerini artırmak"
        assert updated_profile["daily_study_hours"] == 3

        get_response = api_client.get(
            f"{API_BASE}/profile",
            params={"firebase_uid": user_uid},
            timeout=20,
        )
        assert get_response.status_code == 200
        fetched_profile = get_response.json()

        assert fetched_profile["grade_level"] == "mezun"
        assert fetched_profile["study_field"] == "Dil"
        assert fetched_profile["study_goal"] == "Matematik netlerini artırmak"
        assert fetched_profile["daily_study_hours"] == 3