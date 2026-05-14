"""Shared fixtures for all test modules."""
import pytest
import sys
from pathlib import Path

# Ensure backend root is on path when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.ingestion.indexer import get_pinecone_index

POLICY_ID = "care-insurance-sample"
TEST_QUERY = "What is the waiting period for maternity benefits?"
NON_WEB_QUERY = "Is maternity covered under this policy?"
WEB_QUERY = "Is Apollo Hospital Bannerghatta Road in Care's cashless network?"


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def pinecone_index(settings):
    return get_pinecone_index(settings.pinecone_api_key, settings.pinecone_index_name)


@pytest.fixture(scope="session")
def policy_id():
    return POLICY_ID
