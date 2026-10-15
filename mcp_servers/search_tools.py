"""
FastMCP 标准化工具服务模块 (Model Context Protocol Tool Service)
==================================================================
本模块基于 Anthropic 发起的 FastMCP (Model Context Protocol) 规范构建，
负责将外部互联网检索 (Web Search) 与网页深度去噪正文抓取 (Web Content Scraping) 封装为标准微服务。

【核心工程架构价值】：
1. 物理隔离与解耦：工具实现层与 Agent 核心编排拓扑解耦，便于独立部署为独立进程或 Docker 沙箱；
2. 权威信源分级引擎 (Domain Authority Tiering)：设计 Tier 1 ~ Tier 5 权威度打分与重排序算法，
   优先召回高校科研 (.edu/.ac.cn)、官方实验室、顶级智库与大厂官方架构，剔除营销号与灌水社区；
3. 多级检索容灾架构：
   - 优先层：Tavily API (search_depth="advanced") 高级语义深度搜索；
   - 备用层：DuckDuckGo 免费开源搜索引擎；
   - 兜底层：内置离线学术文献卡片高保真仿真库，确保国内弱网/断网环境下 0 秒崩溃；
4. 深度正文抽取与抗干扰：优先采用 Trafilatura 算法提取语义主体，降级使用 BeautifulSoup 去除 HTML 标签杂质。
"""

import time
import warnings
import logging
from typing import List, Dict, Any, Optional
import httpx

# 忽略第三方库 ddgs 历史命名兼容警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import trafilatura
from bs4 import BeautifulSoup
from fastmcp import FastMCP

# 配置模块日志输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCP-ResearchTools")
# 抑制第三方 trafilatura 对二进制/非标准 HTML 产生控制台噪音日志
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)

from core.config import settings
from core.observability import degradation_tracker, is_simulated_url

# 实例化 FastMCP 服务器单例：管理工具注册、JSON Schema 导出与协议通信
mcp = FastMCP("EnterpriseDeepResearchServer")


def _score_domain_authority(url: str, title: str) -> float:
    """
    权威信源量化打分算法 (Domain Authority Tiering Algorithm)。
    
    【设计考量】：
    通用搜索引擎返回的结果中往往充斥着营销号搬运、防爬阻断页面 (如 403 页面) 或论坛灌水帖。
    本算法通过分析目标 URL 域名层级与标题特征词，赋予不同的权威权重：
    - Tier 1: 顶级高校、学术科研机构、预印本平台 (+3.5 分)
    - Tier 2: 官方开源代码库与 AI 领军实验室 (+3.0 分)
    - Tier 3: 核心学术期刊、头部智库与主流科技媒体 (+2.5 分)
    - Tier 4: 国内外科技大厂官方技术团队博客 (+2.0 分)
    - Tier 5: 权威财经与全球主流智库 (+1.8 分)
    - 标题加权: 包含“白皮书”、“论文”、“架构”等专业特征词 (+1.2 分)
    
    Args:
        url: 候选网页的完整 URL 链接
        title: 候选网页的标题文本
        
    Returns:
        float: 权威度量化综合评分 (基准分 1.0)
    """
    score = 1.0
    url_lower = url.lower()
    title_lower = title.lower()
    
    # Tier 1：顶级学术科研机构与高校 (.edu, .gov, .ac.cn, arxiv, ieee)
    if any(d in url_lower for d in [".edu", ".gov", ".ac.cn", "arxiv.org", "acm.org", "ieee.org", "sciencenet.cn"]):
        score += 3.5
    # Tier 2：官方实验室与开源领军代码库 (github, huggingface, baai 智源, openai, anthropic)
    elif any(d in url_lower for d in ["github.com", "huggingface.co", "baai.ac.cn", "openai.com", "anthropic.com"]):
        score += 3.0
    # Tier 3：核心学术期刊、头部智库、主流科技媒体 (大数据研究期刊, infoq, 36kr, 机器之心, 信通院)
    elif any(d in url_lower for d in ["j-bigdataresearch.com.cn", "vzkoo.com", "infoq.cn", "36kr.com", "jiqizhixin.com", "qbitai.com", "caict.ac.cn", "geekpark.net", "ithome.com"]):
        score += 2.5
    # Tier 4：大厂官方架构团队与云原生技术底座 (aliyun, tencent, huawei, bytedance, meituan)
    elif any(d in url_lower for d in ["aliyun.com", "tencent.com", "huawei.com", "bytedance.com", "meituan.com"]):
        score += 2.0
    # Tier 5：权威财经与全球智库 (caixin, 21jingji, bloomberg, reuters, mckinsey)
    elif any(d in url_lower for d in ["caixin.com", "21jingji.com", "bloomberg.com", "reuters.com", "mckinsey.com", "investing.com"]):
        score += 1.8
        
    # 标题专业度特征词加权打分
    if any(k in title_lower for k in ["白皮书", "论文", "学术", "报告", "paper", "architecture", "benchmark", "架构", "专访", "综述"]):
        score += 1.2
        
    return score


@mcp.tool(
    name="web_search",
    description="在互联网上搜索指定关键词，返回高相关度网页的标题、链接和摘要片段。"
)
def web_search(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    通过 FastMCP 协议暴露的标准化联网搜索工具。
    
    【执行流水线】：
    1. 域名黑名单预过滤：屏蔽营销号、防爬拦截站与低信息密度论坛 (CSDN, 贴吧, 知乎 403 页面等)；
    2. 主通道 - Tavily Advanced Search：调用专为 AI Agent 设计的搜索引擎获取丰富候选集；
    3. 域名权威评分与重排：调用 `_score_domain_authority` 重新排序，筛选出 Top-N 高权威信源；
    4. 备用通道 - DuckDuckGo：若 Tavily 未配置或请求异常，无缝切换至 DDGS；
    5. 兜底通道 - 本地文献库仿真：在极端离线或断网情况下，返回合成的专业领域技术文献卡片。
    
    Args:
        query: 提纯后的精准检索关键词
        max_results: 最终返回的高权威结果数量上限
        
    Returns:
        包含 title, url, snippet, authority_score 的字典列表
    """
    logger.info(f"[FastMCP] 正在执行 Web 搜索: query='{query[:40]}...', max_results={max_results}")
    results: List[Dict[str, Any]] = []
    
    # 低质、营销、防爬阻断及失效域名黑名单列表
    blocked_domains = [
        "csdn.net", "faiusr.com", "zhihu.com", "zhidao.baidu.com", 
        "tieba.baidu.com", "cnblogs.com", "bilibili.com", "weibo.com",
        "xiaohongshu.com", "wenku.baidu.com", "baijiahao.baidu.com"
    ]
    
    # =========================================================================
    # 策略 1：优先采用 Tavily 官方深度搜索 API (search_depth=advanced)
    # =========================================================================
    if getattr(settings, "TAVILY_API_KEY", ""):
        for attempt in range(2):
            try:
                logger.info(f"[FastMCP] 正在通过 Tavily API (Advanced Deep Search) 执行高信噪比权威检索 (尝试 {attempt+1}/2)...")
                with httpx.Client(timeout=18.0) as client:
                    resp = client.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": settings.TAVILY_API_KEY,
                            "query": query,
                            "search_depth": "advanced",               # 开启高级深度模式，由模型对网页相关性做深度二次排序
                            "max_results": max(max_results * 2, 6),   # 扩大召回候选池，为后续权威重排提供充足样本
                            "exclude_domains": blocked_domains,       # 搜索引擎底层直拒黑名单域名
                            "include_answer": False
                        }
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = []
                        for item in data.get("results", []):
                            t = item.get("title", "")
                            u = item.get("url", "")
                            c = item.get("content", "")
                            
                            # 双重防线：代码层再次严格过滤黑名单域名与仿真占位域名
                            if any(b in u.lower() for b in blocked_domains) or is_simulated_url(u):
                                continue
                                
                            score = _score_domain_authority(u, t)
                            candidates.append({
                                "title": t,
                                "url": u,
                                "snippet": c,
                                "authority_score": score
                            })
                            
                        # 按计算得出的权威度评分从高到低重排序
                        candidates.sort(key=lambda x: x["authority_score"], reverse=True)
                        results = candidates[:max_results]
                        
                        if results:
                            logger.info(f"[FastMCP] Tavily 深度检索成功！命中并筛选出 {len(results)} 条权威信源")
                            return results
                    else:
                        logger.warning(f"[FastMCP] Tavily API 响应异常: HTTP {resp.status_code}")
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(1.0)  # 临时网络闪断时休眠 1 秒后重试
                    continue
                logger.warning(f"[FastMCP] Tavily 调用异常 ({e})，尝试备选搜索源")

    # =========================================================================
    # 策略 2：备选开源 DuckDuckGo 引擎检索
    # =========================================================================
    if DDGS:
        try:
            with DDGS(timeout=4) as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
                for item in raw_results:
                    u = item.get("href", "")
                    if any(b in u.lower() for b in blocked_domains):
                        continue
                    results.append({
                        "title": item.get("title", ""),
                        "url": u,
                        "snippet": item.get("body", ""),
                        "authority_score": _score_domain_authority(u, item.get("title", ""))
                    })
            if results:
                results.sort(key=lambda x: x["authority_score"], reverse=True)
                logger.info(f"[FastMCP] DuckDuckGo 搜索成功，命中 {len(results)} 条结果")
                return results[:max_results]
        except Exception:
            pass

    # =========================================================================
    # 策略 3：所有检索通道均失败 —— 显式降级，返回空结果，绝不伪造信源
    # =========================================================================
    # 历史实现会在此返回虚构域名（research.agent-systems.org 等）的合成卡片，
    # 这些"信源"会进入最终报告的参考文献，构成幻觉引用。现改为：
    # 返回空列表 + 登记降级事件，由上层 Worker/Synthesizer 决定如何标注。
    degradation_tracker.record(
        "联网搜索全部通道失败（Tavily 未配置或异常、DuckDuckGo 不可用），"
        "本次研究未获取到任何真实外部信源，报告相关内容仅代表模型先验知识"
    )
    logger.warning(f"[FastMCP] 所有搜索通道均失败，返回空结果（不伪造信源）: query='{query[:40]}'")
    return []


@mcp.tool(
    name="fetch_web_content",
    description="深入抓取并解析指定网页正文，自动剔除广告、导航栏与样式代码，提取高密度纯文本。"
)
def fetch_web_content(url: str, max_length: int = 4000) -> str:
    """
    网页深度抓取与语义正文清洗工具。
    
    【核心技术特点】：
    1. 抗反爬伪装：携带现代浏览器 User-Agent 头并设置 3 秒严格超时，防止单个挂起页面拖垮并发线程池；
    2. 双层抽取算法：
       - 第一层：采用 Python 工业界权威的 `trafilatura` 库，基于统计与启发式算法剥离页眉、页脚、侧边栏广告；
       - 第二层：若 trafilatura 抽取为空，自动降级为 BeautifulSoup 解析，并手动剔除 script/style/nav 等噪音标签；
    3. 长度剪裁与上下文保护：强行限制返回字符上限 (默认 4000 字符)，彻底解决单页海量文本撑爆上下文的问题。
    
    Args:
        url: 目标网页的真实 URL
        max_length: 提取文本的最大字符数上限
        
    Returns:
        清洗后的高信噪比纯文本正文内容；若抓取失败则返回空字符串
    """
    logger.info(f"[FastMCP] 正在抓取网页正文: url='{url}'")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    
    try:
        # 3.0 秒短超时设定，防止死链挂起 Worker 并发任务
        with httpx.Client(timeout=3.0, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                return ""
            content_type = resp.headers.get("content-type", "").lower()
            clean_url = url.lower().split("?")[0]
            is_pdf = clean_url.endswith(".pdf") or "application/pdf" in content_type

            extracted = ""
            if is_pdf:
                # 若为 PDF 文档，尝试以 pypdf 提取前若干页，或跳过 HTML 提取器避免报语法错误
                try:
                    import pypdf
                    import io
                    reader = pypdf.PdfReader(io.BytesIO(resp.content))
                    pages_text = []
                    for page in reader.pages[:10]:
                        t = page.extract_text()
                        if t:
                            pages_text.append(t)
                    extracted = "\n".join(pages_text)
                except Exception:
                    extracted = ""
            else:
                html_text = resp.text
                # 第一阶段抽取：trafilatura 深度语义提取
                try:
                    extracted = trafilatura.extract(
                        html_text,
                        include_comments=False,
                        include_tables=True,
                        no_fallback=False
                    )
                except Exception:
                    extracted = ""
                
                # 第二阶段抽取：BeautifulSoup 标签剥离降级保底
                if not extracted:
                    try:
                        soup = BeautifulSoup(html_text, "html.parser")
                        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                            tag.decompose()
                        extracted = soup.get_text(separator="\n", strip=True)
                    except Exception:
                        extracted = ""
            
            if not extracted:
                return ""
                
            cleaned_text = extracted.strip()
            # 截断过长正文，预留安全 Token 预算
            if len(cleaned_text) > max_length:
                cleaned_text = cleaned_text[:max_length] + "\n...(正文过长已截断)"
                
            return cleaned_text
        
    except Exception:
        # 网络异常、超时或 DNS 失败时静默降级，交由上层判定
        return ""


if __name__ == "__main__":
    # 作为独立 MCP 微服务启动时监听标准协议通信
    mcp.run()

