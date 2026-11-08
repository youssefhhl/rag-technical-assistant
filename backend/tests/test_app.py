"""Tests for application initialization (factory + configuration)."""

from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.main import app, create_app


def test_create_app_returns_fastapi_instance():
    instance = create_app()
    assert isinstance(instance, FastAPI)


def test_module_level_app_matches_construction():
    instance = create_app()
    assert instance.title == app.title
    assert instance.version == app.version


def test_app_metadata_comes_from_settings():
    settings = get_settings()
    assert app.title == settings.app_name
    assert app.version == settings.app_version


def test_health_route_is_registered():
    assert "/health" in app.openapi()["paths"]


def test_settings_respect_environment_fallback_to_defaults():
    settings = Settings(_env_file=None)  # isolate from any local .env
    assert settings.app_name == "Smart Knowledge Assistant"
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"


def test_settings_accepts_app_env_override(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    assert Settings(_env_file=None).app_env == "production"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
