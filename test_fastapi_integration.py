#!/usr/bin/env python3
"""
Test script for FastAPI + React integration
Verifies the API endpoints work correctly
"""

import sys
sys.path.insert(0, '.')

from react_api.server import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health():
    """Test health endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✅ Health check passed")

def test_pipeline_run():
    """Test pipeline execution endpoint"""
    spec = {
        "module_name": "test_counter",
        "description": "Simple 8-bit counter with enable and reset",
        "data_width": 8,
        "clock_freq": 100.0
    }

    response = client.post("/api/pipeline/run", json=spec)
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["status"] == "running"
    print(f"✅ Pipeline started: run_id={data['run_id']}")
    return data["run_id"]

def test_pipeline_status(run_id):
    """Test pipeline status endpoint"""
    response = client.get(f"/api/pipeline/status/{run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == run_id
    print(f"✅ Pipeline status: {data['status']}")

def test_list_runs():
    """Test list runs endpoint"""
    response = client.get("/api/pipeline/runs")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    print(f"✅ Listed runs: {data['total']} total")

if __name__ == "__main__":
    print("=" * 80)
    print("FastAPI Integration Test")
    print("=" * 80)
    print()

    try:
        test_health()
        run_id = test_pipeline_run()
        test_pipeline_status(run_id)
        test_list_runs()

        print()
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Start backend: cd ~/cda-agent-2C1 && ./launch_react_api.sh")
        print("  2. Start frontend: cd ~/ai-chip-design-ui && npm run dev")
        print("  3. Open browser: http://localhost:3000")

    except AssertionError as e:
        print()
        print("=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ TEST ERROR")
        print("=" * 80)
        print(f"Error: {e}")
        sys.exit(1)
