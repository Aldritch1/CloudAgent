# CloudAgent Phase2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Hybrid RAG (Milvus + Neo4j + PostgreSQL three-way retrieval with RRF fusion) and a RAG Agent, extending the Entry Agent to recognize `faq` and `workflow` intents.

**Architecture:** A new `cloudagent/retrieval/` package provides `VectorRetriever`, `GraphRetriever`, `KeywordRetriever`, and `HybridRetriever` (async concurrent search + RRF). The `RAGAgent` calls `HybridRetriever` to fetch context, injects it into a system prompt, and invokes the LLM. The existing `EntryAgent` prompt is expanded to classify `faq`/`workflow`/`chat`; `main.py` routes `faq` to `RAGAgent` and `workflow` to a placeholder message.

**Tech Stack:** Python 3.11+, FastAPI, LangChain, pymilvus, neo4j-python-driver, asyncpg, pytest, pytest-asyncio

---

## File Structure

```
cloudagent/
├── retrieval/               # NEW
│   ├── __init__.py
│   ├── base.py              # RetrievalResult dataclass + Retriever Protocol
│   ├── vector.py            # VectorRetriever (Milvus + OpenAI embeddings)
│   ├── graph.py             # GraphRetriever (Neo4j Cypher)
│   ├── keyword.py           # KeywordRetriever (PostgreSQL BM25)
│   └── hybrid.py            # HybridRetriever (async gather + RRF)
├── agent/
│   ├── __init__.py
│   ├── router.py            # MODIFIED: expand intents to faq/workflow/chat
│   ├── chat_agent.py        # unchanged
│   └── rag_agent.py         # NEW: RAG Agent
├── memory/
│   └── redis_store.py       # unchanged
├── main.py                  # MODIFIED: wire retrievers + RAGAgent + routing
├── models.py                # unchanged
└── config.py                # MODIFIED: add Milvus/Neo4j/PG settings

tests/
├── conftest.py              # MODIFIED: patch new env vars
├── test_config.py           # MODIFIED: assert new settings
├── test_main.py             # MODIFIED: cover faq routing path
├── test_router.py           # MODIFIED: cover faq/workflow intent
├── test_chat_agent.py       # unchanged
├── test_rag_agent.py        # NEW
└── retrieval/               # NEW
    ├── __init__.py
    ├── test_vector.py
    ├── test_graph.py
    ├── test_keyword.py
    └── test_hybrid.py

docker-compose.yml           # NEW: etcd + minio + milvus + neo4j + postgres
pyproject.toml               # MODIFIED: add pymilvus, neo4j, asyncpg
tests/fixtures/seed_faq.json # NEW: mock FAQ seed data
```

---

### Task 1: Docker Compose, Dependencies, and Config Extension

**Files:**
- Create: `docker-compose.yml`
- Modify: `pyproject.toml`
- Modify: `cloudagent/config.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Add three lines to the `[project] dependencies` list:

```toml
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.2.0",
    "langchain>=0.1.0",
    "langchain-openai>=0.0.8",
    "langgraph>=0.0.26",
    "redis>=5.0.0",
    "pymilvus>=2.4.0",
    "neo4j>=5.19.0",
    "asyncpg>=0.29.0",
]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd-data:/etcd
    command: etcd -advertise-client-urls http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio-data:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus-standalone:
    image: milvusdb/milvus:v2.4.1
    ports:
      - "19530:19530"
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio

  neo4j:
    image: neo4j:5.19-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password

  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: cloudagent
      POSTGRES_PASSWORD: cloudagent
      POSTGRES_DB: cloudagent

volumes:
  etcd-data:
  minio-data:
```

- [ ] **Step 3: Modify cloudagent/config.py**

Add new fields to `Settings` (all with defaults so existing tests don't break):

```python
from pydantic import RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: SecretStr
    redis_url: RedisDsn = "redis://localhost:6379/0"
    model_name: str = "gpt-3.5-turbo"
    milvus_uri: str = "http://localhost:19530"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("password")
    database_url: str = "postgresql://cloudagent:cloudagent@localhost:5432/cloudagent"


settings = Settings()
```

- [ ] **Step 4: Modify tests/conftest.py**

Add env var patches for the new settings:

```python
import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("MODEL_NAME", "gpt-test")
    monkeypatch.setenv("MILVUS_URI", "http://localhost:19530")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("DATABASE_URL", "postgresql://cloudagent:cloudagent@localhost:5432/cloudagent")
```

- [ ] **Step 5: Modify tests/test_config.py**

Add assertions for the new fields:

```python
import importlib

import pytest


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://test:6379/0")
    monkeypatch.setenv("MODEL_NAME", "gpt-4")
    monkeypatch.setenv("MILVUS_URI", "http://test:19530")
    monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@test:5432/db")


def test_settings_loads_from_env(patch_env):
    from cloudagent.config import settings

    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert str(settings.redis_url) == "redis://test:6379/0"
    assert settings.model_name == "gpt-4"
    assert settings.milvus_uri == "http://test:19530"
    assert settings.neo4j_uri == "bolt://test:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password.get_secret_value() == "secret"
    assert settings.database_url == "postgresql://u:p@test:5432/db"


def test_settings_class_instantiation(patch_env):
    from cloudagent.config import Settings

    s = Settings()
    assert s.openai_api_key.get_secret_value() == "test-key"
    assert str(s.redis_url) == "redis://test:6379/0"
    assert s.model_name == "gpt-4"
    assert s.milvus_uri == "http://test:19530"
```

- [ ] **Step 6: Run tests to verify config changes**

```bash
pytest tests/test_config.py -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml docker-compose.yml cloudagent/config.py tests/conftest.py tests/test_config.py
git commit -m "chore: add docker-compose and config for milvus/neo4j/postgres"
```

---

### Task 2: Retrieval Base Types

**Files:**
- Create: `cloudagent/retrieval/__init__.py`
- Create: `cloudagent/retrieval/base.py`
- Create: `tests/retrieval/__init__.py`

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/test_base.py`:

```python
from cloudagent.retrieval.base import RetrievalResult


def test_retrieval_result_defaults():
    r = RetrievalResult(content="hello", source="vector")
    assert r.content == "hello"
    assert r.source == "vector"
    assert r.score == 0.0
    assert r.metadata == {}
```

Run:
```bash
pytest tests/retrieval/test_base.py -v
```

Expected: `ImportError: cannot import name 'RetrievalResult' from 'cloudagent.retrieval.base'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/retrieval/__init__.py` (empty).

Create `cloudagent/retrieval/base.py`:

```python
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RetrievalResult:
    content: str
    source: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class Retriever(Protocol):
    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        ...
```

Create `tests/retrieval/__init__.py` (empty).

- [ ] **Step 3: Run test**

```bash
pytest tests/retrieval/test_base.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/retrieval/ tests/retrieval/
git commit -m "feat: add retrieval base types and protocol"
```

---

### Task 3: VectorRetriever (Milvus)

**Files:**
- Create: `cloudagent/retrieval/vector.py`
- Create: `tests/retrieval/test_vector.py`

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/test_vector.py`:

```python
from unittest.mock import MagicMock, patch

import pytest

from cloudagent.retrieval.vector import VectorRetriever


@pytest.fixture
def mock_milvus():
    with patch("cloudagent.retrieval.vector.MilvusClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_embeddings():
    with patch("cloudagent.retrieval.vector.OpenAIEmbeddings") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.aembed_query = MagicMock(return_value=[0.1] * 1536)
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
async def test_vector_search(mock_milvus, mock_embeddings):
    mock_milvus.search.return_value = [[
        {"entity": {"content": "退款政策", "category": "售后"}, "distance": 0.9},
    ]]

    retriever = VectorRetriever(uri="http://localhost:19530", api_key="test-key")
    results = await retriever.search("怎么退款", top_k=5)

    assert len(results) == 1
    assert results[0].content == "退款政策"
    assert results[0].source == "vector"
    assert results[0].score == 0.9
    assert results[0].metadata["category"] == "售后"
    mock_milvus.search.assert_called_once()


@pytest.mark.asyncio
async def test_vector_search_degrades_on_failure(mock_milvus, mock_embeddings):
    mock_milvus.search.side_effect = Exception("Milvus down")

    retriever = VectorRetriever(uri="http://localhost:19530", api_key="test-key")
    results = await retriever.search("怎么退款", top_k=5)

    assert results == []
```

Run:
```bash
pytest tests/retrieval/test_vector.py -v
```

Expected: `ImportError: cannot import name 'VectorRetriever'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/retrieval/vector.py`:

```python
import logging

from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

from cloudagent.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, uri: str, api_key: str, collection_name: str = "kb_documents"):
        self._collection = collection_name
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        try:
            self._client = MilvusClient(uri=uri)
            if not self._client.has_collection(collection_name):
                self._client.create_collection(
                    collection_name=collection_name,
                    dimension=1536,
                )
        except Exception as e:
            logger.warning(f"Milvus connection failed: {e}")
            self._client = None

    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self._client is None:
            return []
        try:
            embedding = await self._embeddings.aembed_query(query)
            results = self._client.search(
                collection_name=self._collection,
                data=[embedding],
                limit=top_k,
                output_fields=["content", "category"],
            )
            return [
                RetrievalResult(
                    content=hit["entity"]["content"],
                    source="vector",
                    score=hit.get("distance", 0.0),
                    metadata={"category": hit["entity"].get("category", "")},
                )
                for hit in results[0]
            ]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
```

- [ ] **Step 3: Run test**

```bash
pytest tests/retrieval/test_vector.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/retrieval/vector.py tests/retrieval/test_vector.py
git commit -m "feat: add VectorRetriever for Milvus semantic search"
```

---

### Task 4: GraphRetriever (Neo4j)

**Files:**
- Create: `cloudagent/retrieval/graph.py`
- Create: `tests/retrieval/test_graph.py`

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/test_graph.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloudagent.retrieval.graph import GraphRetriever


class AsyncIter:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


@pytest.fixture
def mock_neo4j():
    with patch("cloudagent.retrieval.graph.AsyncGraphDatabase.driver") as mock_driver:
        mock_session = AsyncMock()
        mock_driver.return_value.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.return_value.session.return_value.__aexit__ = AsyncMock(return_value=False)
        yield mock_session


@pytest.mark.asyncio
async def test_graph_search(mock_neo4j):
    mock_neo4j.run.return_value = AsyncIter([
        {"content": "退款流程说明", "metadata": "售后"},
    ])

    retriever = GraphRetriever(uri="bolt://localhost:7687", user="neo4j", password="pass")
    results = await retriever.search("怎么退款", top_k=5)

    assert len(results) == 1
    assert results[0].content == "退款流程说明"
    assert results[0].source == "graph"
    assert results[0].metadata["category"] == "售后"


@pytest.mark.asyncio
async def test_graph_search_degrades_on_failure(mock_neo4j):
    mock_neo4j.run.side_effect = Exception("Neo4j down")

    retriever = GraphRetriever(uri="bolt://localhost:7687", user="neo4j", password="pass")
    results = await retriever.search("怎么退款", top_k=5)

    assert results == []
```

Run:
```bash
pytest tests/retrieval/test_graph.py -v
```

Expected: `ImportError: cannot import name 'GraphRetriever'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/retrieval/graph.py`:

```python
import logging

from neo4j import AsyncGraphDatabase

from cloudagent.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class GraphRetriever:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = None
        try:
            self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")

    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self._driver is None:
            return []
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    MATCH (f:FAQ)
                    WHERE f.question CONTAINS $query OR f.answer CONTAINS $query
                    RETURN f.question AS content, f.category AS metadata
                    LIMIT $limit
                    """,
                    query=query,
                    limit=top_k,
                )
                records = []
                async for record in result:
                    records.append(
                        RetrievalResult(
                            content=record["content"],
                            source="graph",
                            metadata={"category": record.get("metadata", "")},
                        )
                    )
                return records
        except Exception as e:
            logger.warning(f"Graph search failed: {e}")
            return []
```

- [ ] **Step 3: Run test**

```bash
pytest tests/retrieval/test_graph.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/retrieval/graph.py tests/retrieval/test_graph.py
git commit -m "feat: add GraphRetriever for Neo4j FAQ search"
```

---

### Task 5: KeywordRetriever (PostgreSQL)

**Files:**
- Create: `cloudagent/retrieval/keyword.py`
- Create: `tests/retrieval/test_keyword.py`

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/test_keyword.py`:

```python
from unittest.mock import AsyncMock, patch

import pytest

from cloudagent.retrieval.keyword import KeywordRetriever


@pytest.fixture
def mock_asyncpg():
    with patch("cloudagent.retrieval.keyword.asyncpg.connect") as mock_connect:
        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn
        yield mock_conn


@pytest.mark.asyncio
async def test_keyword_search(mock_asyncpg):
    mock_asyncpg.fetch.return_value = [
        {"title": "退款政策", "content": "支持7天无理由退款", "category": "售后"},
    ]

    retriever = KeywordRetriever(dsn="postgresql://u:p@localhost/db")
    results = await retriever.search("怎么退款", top_k=5)

    assert len(results) == 1
    assert results[0].content == "支持7天无理由退款"
    assert results[0].source == "keyword"
    assert results[0].metadata["title"] == "退款政策"
    assert results[0].metadata["category"] == "售后"
    mock_asyncpg.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_keyword_search_degrades_on_failure(mock_asyncpg):
    mock_asyncpg.fetch.side_effect = Exception("PG down")

    retriever = KeywordRetriever(dsn="postgresql://u:p@localhost/db")
    results = await retriever.search("怎么退款", top_k=5)

    assert results == []
```

Run:
```bash
pytest tests/retrieval/test_keyword.py -v
```

Expected: `ImportError: cannot import name 'KeywordRetriever'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/retrieval/keyword.py`:

```python
import logging

import asyncpg

from cloudagent.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class KeywordRetriever:
    def __init__(self, dsn: str):
        self._dsn = dsn

    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        try:
            conn = await asyncpg.connect(self._dsn)
            try:
                rows = await conn.fetch(
                    """
                    SELECT title, content, category
                    FROM kb_documents
                    WHERE fts_vector @@ plainto_tsquery('chinese', $1)
                    ORDER BY ts_rank(fts_vector, plainto_tsquery('chinese', $1)) DESC
                    LIMIT $2
                    """,
                    query,
                    top_k,
                )
                return [
                    RetrievalResult(
                        content=row["content"],
                        source="keyword",
                        metadata={"title": row.get("title", ""), "category": row.get("category", "")},
                    )
                    for row in rows
                ]
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []
```

- [ ] **Step 3: Run test**

```bash
pytest tests/retrieval/test_keyword.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/retrieval/keyword.py tests/retrieval/test_keyword.py
git commit -m "feat: add KeywordRetriever for PostgreSQL BM25 search"
```

---

### Task 6: HybridRetriever + RRF Fusion

**Files:**
- Create: `cloudagent/retrieval/hybrid.py`
- Create: `tests/retrieval/test_hybrid.py`

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/test_hybrid.py`:

```python
from unittest.mock import AsyncMock

import pytest

from cloudagent.retrieval.hybrid import HybridRetriever, rrf_fuse
from cloudagent.retrieval.base import RetrievalResult


def test_rrf_fuse_ranking():
    vector_results = [
        RetrievalResult("doc-a", "vector"),
        RetrievalResult("doc-b", "vector"),
    ]
    graph_results = [
        RetrievalResult("doc-b", "graph"),
        RetrievalResult("doc-c", "graph"),
    ]
    keyword_results = [
        RetrievalResult("doc-c", "keyword"),
        RetrievalResult("doc-a", "keyword"),
    ]

    fused = rrf_fuse([vector_results, graph_results, keyword_results], k=60, final_top_k=5)
    contents = [r.content for r in fused]

    # doc-b: rank 2 (vector) + rank 1 (graph) -> highest RRF score
    assert contents[0] == "doc-b"
    assert len(contents) == 3


@pytest.mark.asyncio
async def test_hybrid_search_concurrent():
    v = AsyncMock()
    v.search.return_value = [RetrievalResult("a", "vector")]
    g = AsyncMock()
    g.search.return_value = [RetrievalResult("b", "graph")]
    k = AsyncMock()
    k.search.return_value = [RetrievalResult("c", "keyword")]

    hybrid = HybridRetriever(v, g, k)
    results = await hybrid.search("query", top_k=2)

    assert len(results) == 2
    v.search.assert_called_once_with("query", top_k=10)
    g.search.assert_called_once_with("query", top_k=10)
    k.search.assert_called_once_with("query", top_k=10)
```

Run:
```bash
pytest tests/retrieval/test_hybrid.py -v
```

Expected: `ImportError: cannot import name 'HybridRetriever'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/retrieval/hybrid.py`:

```python
import asyncio
import logging
from collections import defaultdict

from cloudagent.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


def rrf_fuse(result_lists: list[list[RetrievalResult]], k: int = 60, final_top_k: int = 5) -> list[RetrievalResult]:
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, RetrievalResult] = {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            scores[r.content] += 1.0 / (k + rank)
            if r.content not in items:
                items[r.content] = r
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [items[content] for content, _ in sorted_items[:final_top_k]]


class HybridRetriever:
    def __init__(self, vector, graph, keyword):
        self.vector = vector
        self.graph = graph
        self.keyword = keyword

    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        v_task = self.vector.search(query, top_k=10)
        g_task = self.graph.search(query, top_k=10)
        k_task = self.keyword.search(query, top_k=10)
        v_results, g_results, k_results = await asyncio.gather(v_task, g_task, k_task)
        return rrf_fuse([v_results, g_results, k_results], k=60, final_top_k=top_k)
```

- [ ] **Step 3: Run test**

```bash
pytest tests/retrieval/test_hybrid.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/retrieval/hybrid.py tests/retrieval/test_hybrid.py
git commit -m "feat: add HybridRetriever with RRF fusion"
```

---

### Task 7: RAG Agent

**Files:**
- Create: `cloudagent/agent/rag_agent.py`
- Create: `tests/test_rag_agent.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_rag_agent.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloudagent.agent.rag_agent import RAGAgent


@pytest.mark.asyncio
async def test_rag_agent_includes_context_in_prompt():
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = [
        MagicMock(content="支持7天无理由退款", source="vector", metadata={}),
    ]

    agent = RAGAgent(model_name="gpt-test", api_key="test-key", retriever=mock_retriever)

    with patch("cloudagent.agent.rag_agent.ChatOpenAI") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="我们支持7天无理由退款。"))
        mock_llm_cls.return_value = mock_llm

        result = await agent.run({
            "messages": [
                {"role": "user", "content": "怎么退款？"},
            ],
        })

        assert result == "我们支持7天无理由退款。"
        mock_retriever.search.assert_called_once_with("怎么退款？", top_k=5)

        # Verify the prompt contains retrieved context
        call_messages = mock_llm.ainvoke.call_args[0][0]
        assert "支持7天无理由退款" in call_messages[0].content
        assert call_messages[1].content == "怎么退款？"


@pytest.mark.asyncio
async def test_rag_agent_empty_context():
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = []

    agent = RAGAgent(model_name="gpt-test", api_key="test-key", retriever=mock_retriever)

    with patch("cloudagent.agent.rag_agent.ChatOpenAI") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="抱歉，我暂时无法回答。"))
        mock_llm_cls.return_value = mock_llm

        result = await agent.run({
            "messages": [{"role": "user", "content": "未知问题"}],
        })

        assert result == "抱歉，我暂时无法回答。"
```

Run:
```bash
pytest tests/test_rag_agent.py -v
```

Expected: `ImportError: cannot import name 'RAGAgent'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/agent/rag_agent.py`:

```python
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """你是 CloudAgent 智能客服助手。请根据以下参考资料回答用户问题。
如果参考资料不足以回答，请坦诚告知用户。

参考资料：
{context}
"""


class RAGAgent:
    def __init__(self, model_name: str, api_key: str, retriever):
        self._llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.3)
        self._retriever = retriever

    @staticmethod
    def _extract_last_user(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    @staticmethod
    def _convert_messages(messages: list[dict]):
        lc_messages = []
        for m in messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))
            elif m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
        return lc_messages

    async def run(self, state: dict) -> str:
        last_user_msg = self._extract_last_user(state["messages"])
        contexts = await self._retriever.search(last_user_msg, top_k=5)
        context_text = "\n\n".join([c.content for c in contexts])

        system_prompt = RAG_SYSTEM_PROMPT.format(context=context_text)
        lc_messages = [SystemMessage(content=system_prompt)]
        lc_messages.extend(self._convert_messages(state["messages"]))

        try:
            response = await self._llm.ainvoke(lc_messages)
            return response.content
        except Exception as e:
            logger.error(f"RAG agent failed: {e}")
            raise
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_rag_agent.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/agent/rag_agent.py tests/test_rag_agent.py
git commit -m "feat: add RAG Agent with retrieval-augmented generation"
```

---

### Task 8: Entry Agent Intent Expansion

**Files:**
- Modify: `cloudagent/agent/router.py`
- Modify: `tests/test_router.py`

- [ ] **Step 1: Write failing test additions**

Modify `tests/test_router.py` to add faq/workflow cases. If the file doesn't exist yet, create it. Assuming it exists from Phase 1, add these tests:

```python
from unittest.mock import MagicMock, patch

import pytest

from cloudagent.agent.router import EntryAgent


# ... existing tests ...

def test_entry_agent_recognizes_faq_intent():
    agent = EntryAgent(model_name="gpt-test", api_key="test-key")

    with patch.object(agent._llm, "invoke") as mock_invoke:
        mock_invoke.return_value = MagicMock(
            content='{"intent": "faq", "confidence": 0.92, "target_agent": "faq"}'
        )

        state = {
            "messages": [{"role": "user", "content": "怎么退款？"}],
            "intent": None,
            "confidence": 0.0,
            "target_agent": None,
            "context": {},
        }
        result = agent.run(state)

        assert result["intent"] == "faq"
        assert result["confidence"] == 0.92
        assert result["target_agent"] == "faq"


def test_entry_agent_recognizes_workflow_intent():
    agent = EntryAgent(model_name="gpt-test", api_key="test-key")

    with patch.object(agent._llm, "invoke") as mock_invoke:
        mock_invoke.return_value = MagicMock(
            content='{"intent": "workflow", "confidence": 0.88, "target_agent": "workflow"}'
        )

        state = {
            "messages": [{"role": "user", "content": "帮我查一下订单"}],
            "intent": None,
            "confidence": 0.0,
            "target_agent": None,
            "context": {},
        }
        result = agent.run(state)

        assert result["intent"] == "workflow"
        assert result["confidence"] == 0.88
        assert result["target_agent"] == "workflow"


def test_entry_agent_low_confidence_fallback_to_chat():
    agent = EntryAgent(model_name="gpt-test", api_key="test-key")

    with patch.object(agent._llm, "invoke") as mock_invoke:
        mock_invoke.return_value = MagicMock(
            content='{"intent": "faq", "confidence": 0.3, "target_agent": "faq"}'
        )

        state = {
            "messages": [{"role": "user", "content": "随便说点啥"}],
            "intent": None,
            "confidence": 0.0,
            "target_agent": None,
            "context": {},
        }
        result = agent.run(state)

        # Low confidence forces fallback to chat
        assert result["target_agent"] == "chat"
```

Run:
```bash
pytest tests/test_router.py -v
```

Expected: FAIL (new assertions about faq/workflow will fail if prompt doesn't mention them, or the test file might not exist)

- [ ] **Step 2: Modify router.py**

Replace `INTENT_PROMPT` in `cloudagent/agent/router.py`:

```python
INTENT_PROMPT = """You are an intent classifier for a customer service system.
Analyze the user's message and output ONLY a JSON object with this exact schema:
{{
  "intent": "chat|faq|workflow",
  "confidence": 0.0-1.0,
  "target_agent": "chat|faq|workflow"
}}

Intent definitions:
- "chat": casual conversation, greetings, small talk, general chitchat
- "faq": knowledge questions about policies, refunds, shipping, pricing, product info
- "workflow": business transactions like order queries, refunds, ticket creation

Rules:
- confidence > 0.8: user intent is clearly one of the above
- confidence <= 0.5: unclear or unrelated, fallback to chat agent
- Set target_agent to the most appropriate agent for the intent.

User message: {message}

Output JSON only, no markdown, no explanation."""
```

Also update the comment in `run()`:

```python
        # Routing logic: phase2 has chat, faq, and workflow intents
        if state["confidence"] <= 0.5:
            state["target_agent"] = "chat"
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_router.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/agent/router.py tests/test_router.py
git commit -m "feat: extend EntryAgent to recognize faq and workflow intents"
```

---

### Task 9: Main App Integration

**Files:**
- Modify: `cloudagent/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Modify main.py**

Add imports and initialization at the top of `cloudagent/main.py` (after existing imports):

```python
from cloudagent.retrieval.vector import VectorRetriever
from cloudagent.retrieval.graph import GraphRetriever
from cloudagent.retrieval.keyword import KeywordRetriever
from cloudagent.retrieval.hybrid import HybridRetriever
from cloudagent.agent.rag_agent import RAGAgent

# Initialize new retrievers (module-level, same pattern as Phase 1)
vector_retriever = VectorRetriever(
    uri=settings.milvus_uri,
    api_key=settings.openai_api_key.get_secret_value(),
)
graph_retriever = GraphRetriever(
    uri=settings.neo4j_uri,
    user=settings.neo4j_user,
    password=settings.neo4j_password.get_secret_value(),
)
keyword_retriever = KeywordRetriever(dsn=settings.database_url)
hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever, keyword_retriever)

rag_agent = RAGAgent(
    model_name=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    retriever=hybrid_retriever,
)
```

Modify the `chat()` endpoint routing logic:

```python
        # Phase2: route by target_agent
        target = state["target_agent"]
        try:
            if target == "faq":
                response_text = await rag_agent.run(state)
            elif target == "workflow":
                response_text = "业务办理功能正在开发中，请稍后再试。"
            else:
                response_text = chat_agent.run(state["messages"])
        except Exception as e:
            logger.error(f"Agent failed: {e}")
            raise HTTPException(status_code=500, detail="服务暂时繁忙，请稍后重试")
```

- [ ] **Step 2: Modify tests/test_main.py**

Add a test for the faq routing path. Patch retriever classes before importing main:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_faq_routing(mock_chat_cls, mock_entry_cls, mock_store_cls,
                                    mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_entry = MagicMock()
    mock_entry.run.return_value = {
        "messages": [{"role": "user", "content": "怎么退款？"}],
        "intent": "faq",
        "confidence": 0.94,
        "target_agent": "faq",
        "context": {},
    }
    mock_entry_cls.return_value = mock_entry

    mock_rag = MagicMock()
    mock_rag.run = AsyncMock(return_value="支持7天无理由退款。")
    mock_rag_cls.return_value = mock_rag

    mock_chat = MagicMock()
    mock_chat_cls.return_value = mock_chat

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "怎么退款？",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "支持7天无理由退款。"
    assert data["intent"] == "faq"
    mock_rag.run.assert_called_once()


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_workflow_placeholder(mock_chat_cls, mock_entry_cls, mock_store_cls,
                                             mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_entry = MagicMock()
    mock_entry.run.return_value = {
        "messages": [{"role": "user", "content": "查订单"}],
        "intent": "workflow",
        "confidence": 0.91,
        "target_agent": "workflow",
        "context": {},
    }
    mock_entry_cls.return_value = mock_entry

    mock_rag = MagicMock()
    mock_rag_cls.return_value = mock_rag
    mock_chat = MagicMock()
    mock_chat_cls.return_value = mock_chat

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "查订单",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "业务办理功能正在开发中，请稍后再试。"
    assert data["intent"] == "workflow"
    mock_rag.run.assert_not_called()
    mock_chat.run.assert_not_called()
```

Run:
```bash
pytest tests/test_main.py -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add cloudagent/main.py tests/test_main.py
git commit -m "feat: integrate RAG Agent and extend routing in main app"
```

---

### Task 10: Seed Data and Final Verification

**Files:**
- Create: `tests/fixtures/seed_faq.json`
- Run: full test suite

- [ ] **Step 1: Create seed FAQ fixture**

Create `tests/fixtures/seed_faq.json`:

```json
[
  {
    "question": "怎么退款？",
    "answer": "我们支持7天无理由退款，请在订单页面申请退款。",
    "category": "售后"
  },
  {
    "question": "运费是多少？",
    "answer": "满99元包邮，不满99元收取6元运费。",
    "category": "物流"
  },
  {
    "question": "发货需要多久？",
    "answer": "一般情况下，下单后24小时内发货。",
    "category": "物流"
  }
]
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass (including existing Phase 1 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/seed_faq.json
git commit -m "chore: add seed FAQ fixture for testing"
```

---

## Self-Review

**1. Spec coverage:**
- Docker Compose infrastructure → Task 1
- Config extension → Task 1
- VectorRetriever (Milvus) → Task 3
- GraphRetriever (Neo4j) → Task 4
- KeywordRetriever (PG) → Task 5
- HybridRetriever + RRF → Task 6
- RAG Agent → Task 7
- Entry Agent intent expansion → Task 8
- Main app routing → Task 9
- Error handling (degrade to empty list) → covered in each retriever test
- Testing strategy → covered in all tasks

**2. Placeholder scan:**
- No TBD/TODO/fill-in-details found.
- All test code is complete with assertions.
- All implementation code is complete.

**3. Type consistency:**
- `RetrievalResult` fields: `content`, `source`, `score`, `metadata` — consistent across all retrievers.
- `Retriever.search` signature: `async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]` — consistent.
- `RAGAgent.run` signature: `async def run(self, state: dict) -> str` — matches usage in main.py.
