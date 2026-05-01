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
