"""Manual retrieval benchmark for the Day 7 vector-store lab.

Run from the project directory:

    python3 bench.py
    python3 bench.py --data-dir data/university --chunk-size 350

Edit BENCHMARKS below when the group finalizes its five questions, metadata
filters, and gold answers. This script measures retrieval only; it does not call
an LLM.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from src import Document, EmbeddingStore, RecursiveChunker


BENCHMARKS: list[dict[str, Any]] = [
    {
        "query": "Sinh viên đăng ký học phần ở đâu?",
        "filter": {"department": "academic-affairs"},
        "gold_answer": "Sinh viên đăng ký học phần trong cổng học vụ.",
    },
    {
        "query": "Trước khi xác nhận đăng ký học phần cần kiểm tra gì?",
        "filter": {"department": "academic-affairs"},
        "gold_answer": "Cần kiểm tra điều kiện học phần tiên quyết.",
    },
    {
        "query": "Sinh viên phải làm gì khi bị trùng lịch học?",
        "filter": {"department": "academic-affairs"},
        "gold_answer": "Điều chỉnh lớp học phần trước thời hạn được công bố.",
    },
    {
        "query": "Ai có thể sử dụng dịch vụ thư viện?",
        "filter": {"department": "library"},
        "gold_answer": "Sinh viên, giảng viên và nhân viên.",
    },
    {
        "query": "Cần mang gì khi mượn tài liệu tại thư viện?",
        "filter": {"department": "library"},
        "gold_answer": "Người dùng cần mang thẻ định danh hợp lệ.",
    },
]


def _parse_scalar(raw_value: str) -> Any:
    """Parse the small subset of YAML values used by this lab's metadata."""
    value = re.split(r"\s+#", raw_value, maxsplit=1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", "~"}:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", value):
        return float(value)
    return value


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(metadata, body)`` from a Markdown document."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()

    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("frontmatter starts with '---' but has no closing '---'") from exc

    metadata: dict[str, Any] = {}
    for line_number, line in enumerate(lines[1:closing_index], start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid frontmatter on line {line_number}: {line!r}")
        key, raw_value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(raw_value)

    body = "\n".join(lines[closing_index + 1 :]).strip()
    return metadata, body


def load_documents(data_dir: Path, chunk_size: int) -> list[Document]:
    """Load every Markdown file and convert its body into chunk Documents."""
    chunker = RecursiveChunker(chunk_size=chunk_size)
    documents: list[Document] = []

    paths = sorted(data_dir.rglob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no .md files found under {data_dir}")

    for path in paths:
        try:
            metadata, body = split_frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"Skipping {path}: {exc}")
            continue

        doc_id = str(metadata.get("doc_id") or path.stem)
        shared_metadata = {
            **metadata,
            "doc_id": doc_id,
            "source": str(path),
        }
        for index, chunk in enumerate(chunker.chunk(body)):
            documents.append(
                Document(
                    id=f"{path.stem}#{index}",
                    content=chunk,
                    metadata=shared_metadata,
                )
            )

    return documents


def run_benchmark(data_dir: Path, chunk_size: int, top_k: int) -> None:
    documents = load_documents(data_dir, chunk_size)
    store = EmbeddingStore(collection_name="retrieval_benchmark")
    store.add_documents(documents)

    print(f"Loaded {len(documents)} chunks from {data_dir}")
    print(f"Chunking strategy: RecursiveChunker(chunk_size={chunk_size})")

    for number, case in enumerate(BENCHMARKS, start=1):
        query = case["query"]
        metadata_filter = case.get("filter")
        results = store.search_with_filter(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        print("\n" + "=" * 80)
        print(f"QUERY {number}: {query}")
        print(f"FILTER: {metadata_filter or 'None'}")
        print(f"GOLD: {case['gold_answer']}")

        if not results:
            print("RESULTS: no matching chunks")
            continue

        print("RESULTS:")
        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            preview = " ".join(result["content"].split())
            print(
                f"  {rank}. score={result['score']:.4f} "
                f"doc_id={metadata.get('doc_id', 'unknown')} "
                f"chunk_id={result.get('id', 'unknown')}"
            )
            print(f"     {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retrieval benchmark")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/university"),
        help="directory containing Markdown documents (default: data/university)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="maximum recursive chunk size in characters (default: 500)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="number of results printed per query (default: 3)",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run_benchmark(arguments.data_dir, arguments.chunk_size, arguments.top_k)
