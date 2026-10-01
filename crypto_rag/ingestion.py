"""Document loaders and metadata classification.

A :class:`Document` is the canonical unit of ingested knowledge: raw text
plus the metadata attached to every chunk (``source``, ``title``, ``url``,
``type``).

Loaders provided:
* IETF RFC text by number (:meth:`DocumentLoader.fetch_rfc`),
* generic web page (:meth:`DocumentLoader.fetch_url`),
* local PDF (:meth:`DocumentLoader.read_pdf`),
* local plain text (:meth:`DocumentLoader.read_txt`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

DOC_TYPES = ("RFC Document", "NIST Standard", "Cryptanalysis Paper",
             "Cryptography Paper", "Cryptanalysis Article", "Local Document")


@dataclass
class Document:
    source: str
    title: str
    url: str
    type: str
    content: str
    extra: Dict[str, str] = field(default_factory=dict)

    def metadata(self) -> Dict[str, str]:
        md = {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "type": self.type,
        }
        md.update(self.extra)
        return md


def classify_doc_type(filename: str) -> str:
    """Heuristic doc-type classification based on the file name."""
    fn = filename.upper()
    # word-bounded so "NIST" does not match inside e.g. "DETERMINISTIC"
    if re.search(r"(?:^|[_\- ])(NIST|FIPS|SP[_\- ]?800)", fn):
        return "NIST Standard"
    if re.search(r"(?:^|[_\- ])RFC", fn):
        return "RFC Document"
    attack_terms = ("ATTACK", "COLLISION", "CRYPTANALYSIS", "IMPOSSIBLE",
                    "BOOMERANG", "TIMING", "CUBE", "RELATED_KEY", "ZERO")
    if any(t in fn for t in attack_terms):
        return "Cryptanalysis Paper"
    return "Cryptography Paper"


class DocumentLoader:
    """Fetch/read raw documents into the canonical :class:`Document` shape."""

    USER_AGENT = "CryptoRAG/1.0 (research RAG system)"

    def fetch_rfc(self, rfc_number: int) -> Document:
        url = f"https://www.rfc-editor.org/rfc/rfc{rfc_number}.txt"
        resp = requests.get(url, headers={"User-Agent": self.USER_AGENT},
                            timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch RFC {rfc_number} (HTTP {resp.status_code})"
            )
        text = resp.text
        title = f"RFC {rfc_number}"
        for line in (l.strip() for l in text.splitlines()[:15] if l.strip()):
            if (line.lower().startswith("network working group")
                    or "Request for Comments" in line
                    or "Category:" in line or len(line) <= 10):
                continue
            title = f"RFC {rfc_number}: {line}"
            break
        return Document(
            source=f"RFC {rfc_number}",
            title=title,
            url=url,
            type="RFC Document",
            content=text,
        )

    def fetch_url(self, url: str, doc_type: str = "Cryptanalysis Article"
                  ) -> Document:
        resp = requests.get(url, headers={"User-Agent": self.USER_AGENT},
                            timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch {url} (HTTP {resp.status_code})")
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        body = soup.find("article") or soup.find("main") or soup.body
        if body:
            parts = body.find_all(["p", "h1", "h2", "h3", "h4", "ul", "ol", "pre"])
            text = "\n\n".join(p.get_text().strip() for p in parts
                               if p.get_text().strip())
        else:
            text = soup.get_text(separator="\n\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return Document(source=url, title=title, url=url,
                        type=doc_type, content=text)

    def read_pdf(self, path: str, filename: Optional[str] = None) -> Document:
        name = filename or path.rsplit("/", 1)[-1]
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                pages.append(f"--- Page {i + 1} ---\n{extracted}")
        return Document(
            source=name,
            title=name[:-4].replace("_", " ") if name.lower().endswith(".pdf")
            else name,
            url=f"file://{path}",
            type=classify_doc_type(name),
            content="\n\n".join(pages),
        )

    def read_txt(self, path: str, filename: Optional[str] = None) -> Document:
        name = filename or path.rsplit("/", 1)[-1]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return Document(
            source=name,
            title=name.replace(".txt", "").replace("_", " "),
            url=f"file://{path}",
            type=classify_doc_type(name),
            content=content,
        )

    def load(self, path: str) -> Document:
        """Load a local file by extension."""
        if path.lower().endswith(".pdf"):
            return self.read_pdf(path)
        if path.lower().endswith((".txt", ".md")):
            return self.read_txt(path)
        raise ValueError(f"Unsupported file type: {path}")
