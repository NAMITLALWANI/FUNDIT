"""
Loader for unstructured fund documents stored as Markdown with a provenance front matter.

Expected file layout::

    ---
    document_id: sebi-scheme-categorisation-2017
    document_type: Regulatory
    title: SEBI Categorization and Rationalization of Mutual Fund Schemes
    source: SEBI
    source_url: https://www.sebi.gov.in/...
    publication_date: 2017-10-06
    fund_id:            # optional, blank for universe-wide documents
    ---
    ## Section heading
    Body text...
"""

import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import ValidationError

from app.core.exceptions import IngestionError
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.models import RawDocument

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_front_matter(text: str) -> tuple[Dict[str, str], str]:
    """Split a Markdown file into its key/value front matter and body."""
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    header, body = match.group(1), match.group(2)
    fields: Dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.split("#", 1)[0].strip() if not value.strip().startswith("http") else value.strip()
        fields[key.strip()] = value.strip()
    return fields, body


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


class MarkdownDocumentLoader(DocumentLoader):
    """Loads provenance-annotated Markdown documents from a file or directory tree."""

    def load(self, source_path: Union[str, Path]) -> List[RawDocument]:
        path = Path(source_path)
        if not path.exists():
            raise IngestionError(f"Document path does not exist: {path}")
        files = sorted(path.rglob("*.md")) if path.is_dir() else [path]
        documents: List[RawDocument] = []
        for file in files:
            documents.append(self._load_file(file))
        return documents

    def _load_file(self, file: Path) -> RawDocument:
        text = file.read_text(encoding="utf-8")
        fields, body = parse_front_matter(text)
        if "document_id" not in fields or "document_type" not in fields:
            raise IngestionError(f"Document {file} is missing required front matter (document_id, document_type)")
        try:
            return RawDocument(
                document_id=fields["document_id"],
                document_type=fields["document_type"],  # type: ignore[arg-type]
                title=fields.get("title") or file.stem,
                source=fields.get("source") or "Unknown",
                source_url=fields.get("source_url") or None,
                fund_id=fields.get("fund_id") or None,
                publication_date=_parse_date(fields.get("publication_date")),
                content=body.strip(),
                metadata={"path": str(file)},
            )
        except ValidationError as exc:
            raise IngestionError(f"Invalid document front matter in {file}: {exc}") from exc
