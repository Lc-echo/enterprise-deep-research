"""
API 安全测试：路径遍历防护、后缀白名单、CORS 白名单。
"""

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


# ------------------------- 路径遍历 / 后缀白名单 -------------------------

@pytest.mark.parametrize("malicious", [
    "..%2F..%2F.env",                    # 编码的正斜杠穿越
    "..%5C..%5C.env",                    # 编码的反斜杠穿越（Windows）
    "..%2F..%2F..%2F..%2FWindows%2Fwin.ini",
    "%2e%2e%2f%2e%2e%2f.env",            # 全编码点号
])
def test_path_traversal_blocked(malicious):
    r = client.get(f"/api/v1/reports/{malicious}")
    assert r.status_code in (400, 404), f"穿越路径 {malicious} 未被拦截，状态码 {r.status_code}"


def test_non_markdown_suffix_rejected():
    r = client.get("/api/v1/reports/foo.txt")
    assert r.status_code == 404


def test_nonexistent_markdown_returns_404():
    r = client.get("/api/v1/reports/__definitely_not_exist__.md")
    assert r.status_code == 404


# ------------------------- CORS 白名单 -------------------------

def test_cors_whitelisted_origin_is_reflected():
    r = client.get("/api/v1/health", headers={"Origin": "http://localhost:8000"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8000"


def test_cors_non_whitelisted_origin_has_no_acao_header():
    r = client.get("/api/v1/health", headers={"Origin": "https://evil.example.com"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_cors_preflight_rejects_non_whitelisted_origin():
    r = client.options(
        "/api/v1/research/start",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in r.headers
