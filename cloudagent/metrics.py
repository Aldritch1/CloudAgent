import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM calls",
    ["agent_type", "status"],
)

cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache lookups",
    ["tier", "hit"],
)

retrieval_results_total = Counter(
    "retrieval_results_total",
    "Total retrieval results",
    ["source"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        method = request.method
        endpoint = request.url.path
        status_code = str(response.status_code)

        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
        http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()

        return response


def record_llm_call(agent_type: str, status: str) -> None:
    llm_calls_total.labels(agent_type=agent_type, status=status).inc()


def record_cache_hit(tier: str, hit: bool) -> None:
    cache_hits_total.labels(tier=tier, hit=str(hit)).inc()


def record_retrieval_results(source: str, count: int) -> None:
    retrieval_results_total.labels(source=source).inc(count)


def get_metrics() -> bytes:
    return generate_latest()
