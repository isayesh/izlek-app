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

    pytest.skip("REACT_APP_BACKEND_URL missing; cannot run onboarding API tests")


BASE_URL = _get_base_url()
API_BASE = f"{BASE_URL}/api"


@pytest.fixture
def api_client():
    """Shared client for onboarding API regression checks."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


class TestOnboardingProgramCreateFlow:
    """Onboarding profile/program payload compatibility and persistence checks."""

    def test_profile_create_with_handle_and_dil_study_field_persists(self, api_client):
        firebase_uid = f"TEST_onboarding_uid_{uuid.uuid4().hex[:10]}"
        handle = f"test_handle_{uuid.uuid4().hex[:6]}"

        payload = {
            "firebase_uid": firebase_uid,
            "name": "TEST Onboarding User",
            "username": "TEST Onboarding User",
            "handle": handle,
            "study_field": "Dil",
        }

        create_response = api_client.post(f"{API_BASE}/profiles", json=payload, timeout=20)
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["firebase_uid"] == firebase_uid
        assert created["handle"] == handle
        assert created["study_field"] == "Dil"
        assert isinstance(created["id"], str)

        get_response = api_client.get(
            f"{API_BASE}/profiles/by-firebase-uid/{firebase_uid}",
            timeout=20,
        )
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["id"] == created["id"]
        assert fetched["handle"] == handle
        assert fetched["study_field"] == "Dil"

    def test_program_create_with_hidden_defaults_still_works(self, api_client):
        firebase_uid = f"TEST_onboarding_program_uid_{uuid.uuid4().hex[:10]}"
        profile_response = api_client.post(
            f"{API_BASE}/profiles",
            json={
                "firebase_uid": firebase_uid,
                "name": "TEST Program Defaults",
                "username": "TEST Program Defaults",
                "handle": f"test_defaults_{uuid.uuid4().hex[:6]}",
                "study_field": "EA",
            },
            timeout=20,
        )
        assert profile_response.status_code == 200
        profile = profile_response.json()

        program_payload = {
            "profile_id": profile["id"],
            "exam_goal": "TYT",
            "daily_hours": "2-4",
            "study_days": 7,
        }
        create_program_response = api_client.post(f"{API_BASE}/programs", json=program_payload, timeout=20)
        assert create_program_response.status_code == 200
        created_program = create_program_response.json()
        assert created_program["profile_id"] == profile["id"]
        assert created_program["exam_goal"] == "TYT"
        assert created_program["daily_hours"] == "2-4"
        assert created_program["study_days"] == 7
        assert isinstance(created_program["tasks"], list)
        assert len(created_program["tasks"]) > 0

        get_programs_response = api_client.get(f"{API_BASE}/programs/{profile['id']}", timeout=20)
        assert get_programs_response.status_code == 200
        programs = get_programs_response.json()
        matched = next((p for p in programs if p["id"] == created_program["id"]), None)
        assert matched is not None
        assert matched["exam_goal"] == "TYT"
        assert matched["daily_hours"] == "2-4"
        assert matched["study_days"] == 7

    def test_invalid_handle_returns_400(self, api_client):
        payload = {
            "firebase_uid": f"TEST_invalid_handle_{uuid.uuid4().hex[:10]}",
            "name": "TEST Invalid Handle",
            "username": "TEST Invalid Handle",
            "handle": "INVALID-HANDLE",
            "study_field": "Sayısal",
        }

        response = api_client.post(f"{API_BASE}/profiles", json=payload, timeout=20)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Handle" in data["detail"]

    def test_duplicate_handle_returns_409(self, api_client):
        shared_handle = f"test_dupe_{uuid.uuid4().hex[:7]}"

        first_payload = {
            "firebase_uid": f"TEST_dupe_first_{uuid.uuid4().hex[:10]}",
            "name": "TEST Dupe One",
            "username": "TEST Dupe One",
            "handle": shared_handle,
            "study_field": "Sözel",
        }
        second_payload = {
            "firebase_uid": f"TEST_dupe_second_{uuid.uuid4().hex[:10]}",
            "name": "TEST Dupe Two",
            "username": "TEST Dupe Two",
            "handle": shared_handle,
            "study_field": "Dil",
        }

        first_response = api_client.post(f"{API_BASE}/profiles", json=first_payload, timeout=20)
        assert first_response.status_code == 200

        second_response = api_client.post(f"{API_BASE}/profiles", json=second_payload, timeout=20)
        assert second_response.status_code == 409
        body = second_response.json()
        assert "detail" in body
