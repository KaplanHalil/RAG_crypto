import re
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from typing import List, Dict, Any

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into semantic chunks with overlapping boundaries and metadata."""
        # Clean text
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Split into paragraphs or major logical sections
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_word_count = 0
        chunk_idx = 0

        max_words = max(50, self.chunk_size // 5)
        overlap_word_count = max(10, self.chunk_overlap // 5)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_words = para.split()
            # If a single paragraph is longer than max_words, break it into word sub-chunks
            if len(para_words) > max_words:
                sub_chunks = []
                step = max_words - overlap_word_count
                for i in range(0, len(para_words), step):
                    sub_chunks.append(" ".join(para_words[i:i + max_words]))
                para_list = sub_chunks
            else:
                para_list = [para]

            for item in para_list:
                item_word_count = len(item.split())
                if current_word_count + item_word_count > max_words:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": {
                                **metadata,
                                "chunk_id": chunk_idx,
                                "char_count": len(current_chunk),
                            }
                        })
                        chunk_idx += 1
                        words = current_chunk.split()
                        overlap_words = words[-overlap_word_count:]
                        current_chunk = " ".join(overlap_words) + "\n\n" + item
                        current_word_count = len(current_chunk.split())
                    else:
                        current_chunk = item
                        current_word_count = item_word_count
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + item
                    else:
                        current_chunk = item
                    current_word_count += item_word_count

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {
                    **metadata,
                    "chunk_id": chunk_idx,
                    "char_count": len(current_chunk),
                }
            })

        return chunks

    def fetch_rfc(self, rfc_number: int) -> Dict[str, Any]:
        """Fetch IETF RFC text document by number."""
        url = f"https://www.ietf.org/rfc/rfc{rfc_number}.txt"
        headers = {'User-Agent': 'Mozilla/5.0 (Cryptanalysis-RAG-Tool/1.0)'}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                text = resp.text
                # Extract RFC Title from first few lines
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                title = f"RFC {rfc_number}"
                for line in lines[:15]:
                    if "Request for Comments:" in line or "Category:" in line:
                        continue
                    if len(line) > 10 and not line.startswith("Network Working Group"):
                        title = f"RFC {rfc_number}: {line}"
                        break
                return {
                    "source": f"RFC {rfc_number}",
                    "title": title,
                    "url": url,
                    "type": "RFC Document",
                    "content": text
                }
            else:
                return {"error": f"Failed to fetch RFC {rfc_number}. HTTP Status {resp.status_code}"}
        except Exception as e:
            return {"error": f"Exception while fetching RFC {rfc_number}: {str(e)}"}

    def fetch_web_article(self, url: str) -> Dict[str, Any]:
        """Fetch and extract readable text content from a cryptanalysis article URL."""
        headers = {'User-Agent': 'Mozilla/5.0 (Cryptanalysis-RAG-Tool/1.0)'}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                
                # Remove scripts, styles, nav, headers, footers
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.decompose()

                title = soup.title.string.strip() if soup.title else url
                # Main text content extraction
                article_body = soup.find('article') or soup.find('main') or soup.body
                if article_body:
                    paragraphs = article_body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'ul', 'ol', 'pre', 'code'])
                    text_parts = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
                    clean_text = "\n\n".join(text_parts)
                else:
                    clean_text = soup.get_text(separator="\n\n")
                    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

                return {
                    "source": url,
                    "title": title,
                    "url": url,
                    "type": "Cryptanalysis Article",
                    "content": clean_text
                }
            else:
                return {"error": f"Failed to fetch URL. HTTP Status {resp.status_code}"}
        except Exception as e:
            return {"error": f"Exception while fetching web article: {str(e)}"}

    def read_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Extract text from a local PDF document."""
        try:
            reader = PdfReader(file_path)
            extracted = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted.append(f"--- Page {i+1} ---\n{page_text}")
            full_text = "\n\n".join(extracted)
            return {
                "source": filename,
                "title": filename.replace(".pdf", ""),
                "url": f"file://{filename}",
                "type": "Cryptanalysis Paper / Local PDF",
                "content": full_text
            }
        except Exception as e:
            return {"error": f"Error reading PDF {filename}: {str(e)}"}

    def read_txt(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Read text from a local TXT/MD document."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {
                "source": filename,
                "title": filename,
                "url": f"file://{filename}",
                "type": "Local Document",
                "content": content
            }
        except Exception as e:
            return {"error": f"Error reading TXT {filename}: {str(e)}"}
