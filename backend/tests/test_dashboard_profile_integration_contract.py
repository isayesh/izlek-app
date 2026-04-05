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
    """Shared HTTP client for dashboard/profile integration contract tests."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


class TestDashboardProfileIntegrationContract:
    """Contract checks for profile data consumed by dashboard and fallback-ready profile endpoint behavior."""

    def test_profile_contract_returns_dashboard_identity_fields(self, api_client):
        firebase_uid = f"TEST_dash_contract_{uuid.uuid4().hex[:10]}"
        payload = {
            "firebase_uid": firebase_uid,
            "username": "TEST Dashboard Identity",
            "email": f"{firebase_uid}@test.com",
            "study_goal": "Günde soru çözmek",
            "daily_study_hours": 3,
            "avatar_url": "https://example.com/dash-contract-avatar.png",
        }

        create_response = api_client.post(f"{API_BASE}/profile", json=payload, timeout=20)
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["username"] == payload["username"]
        assert created["avatar_url"] == payload["avatar_url"]
        assert created["study_goal"] == payload["study_goal"]
        assert created["daily_study_hours"] == payload["daily_study_hours"]

        get_response = api_client.get(f"{API_BASE}/profile", params={"firebase_uid": firebase_uid}, timeout=20)
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["id"] == created["id"]
        assert fetched["username"] == payload["username"]
        assert fetched["avatar_url"] == payload["avatar_url"]
        assert fetched["study_goal"] == payload["study_goal"]
        assert fetched["daily_study_hours"] == payload["daily_study_hours"]

    def test_profile_endpoint_returns_null_for_missing_profile_uid(self, api_client):
        missing_uid = f"TEST_dash_missing_{uuid.uuid4().hex[:12]}"
        response = api_client.get(f"{API_BASE}/profile", params={"firebase_uid": missing_uid}, timeout=20)
        assert response.status_code == 200
        assert response.json() is None

    def test_program_and_progress_contract_unchanged_with_profile_flow(self, api_client):
        firebase_uid = f"TEST_dash_prog_{uuid.uuid4().hex[:10]}"
        profile_response = api_client.post(
            f"{API_BASE}/profiles",
            json={
                "firebase_uid": firebase_uid,
                "name": "TEST Dashboard Program User",
                "study_field": "EA",
            },
            timeout=20,
        )
        assert profile_response.status_code == 200
        profile = profile_response.json()
        assert isinstance(profile["id"], str)

        program_payload = {
            "profile_id": profile["id"],
            "exam_goal": "TYT",
            "daily_hours": "2-4",
            "study_days": 5,
        }
        create_program_response = api_client.post(f"{API_BASE}/programs", json=program_payload, timeout=20)
        assert create_program_response.status_code == 200
        created_program = create_program_response.json()
        assert created_program["profile_id"] == profile["id"]
        assert created_program["exam_goal"] == "TYT"
        assert isinstance(created_program["tasks"], list)
        assert len(created_program["tasks"]) > 0

        get_programs_response = api_client.get(f"{API_BASE}/programs/{profile['id']}", timeout=20)
        assert get_programs_response.status_code == 200
        programs = get_programs_response.json()
        assert any(p["id"] == created_program["id"] for p in programs)
