"""信源权威度打分与重排算法测试。"""

from mcp_servers.search_tools import _score_domain_authority


def test_academic_domain_outranks_generic_blog():
    """Tier 1 学术域名（arxiv）权重必须显著高于普通未知站点。"""
    academic = _score_domain_authority("https://arxiv.org/abs/2401.00001", "某论文")
    generic = _score_domain_authority("https://random-personal-blog.net/post/1", "某帖子")
    assert academic > generic
    assert academic >= 4.0  # 基准 1.0 + 学术 Tier 3.5


def test_official_blog_domain_gets_tier_bonus():
    """大厂官方技术博客应获得 Tier 4 加权。"""
    aliyun = _score_domain_authority("https://developer.aliyun.com/article/1", "技术文章")
    generic = _score_domain_authority("https://unknown-blog.example/article", "技术文章")
    assert aliyun > generic


def test_professional_keyword_in_title_adds_bonus():
    """标题含白皮书/架构/报告等专业特征词应获得额外加分。"""
    base = _score_domain_authority("https://example.com/a", "一个普通的标题")
    boosted = _score_domain_authority("https://example.com/b", "2025 年技术架构白皮书与研究报告")
    assert boosted > base


def test_score_is_deterministic():
    """相同输入必须得到相同分数（重排稳定性）。"""
    url, title = "https://github.com/org/repo", "Agent 架构 benchmark"
    assert _score_domain_authority(url, title) == _score_domain_authority(url, title)
