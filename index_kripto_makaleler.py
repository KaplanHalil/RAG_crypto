import os
import sys
import glob

# Ensure src module importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ollama_client import OllamaClient
from src.vector_store import VectorStore
from src.document_processor import DocumentProcessor

PAPERS_DIR = "/home/halil/Desktop/kripto_makaleler"

def determine_doc_type(filename: str) -> str:
    fn = filename.upper()
    if "NIST" in fn or "FIPS" in fn or "SP_800" in fn:
        return "NIST Standard"
    elif "RFC" in fn:
        return "RFC Document"
    elif any(term in fn for term in ["ATTACK", "COLLISION", "CRYPTANALYSIS", "IMPOSSIBLE", "BOOMERANG", "TIMING", "CUBE"]):
        return "Cryptanalysis Paper"
    else:
        return "Cryptography Paper"

def index_all_papers():
    if not os.path.exists(PAPERS_DIR):
        print(f"Klasör bulunamadı: {PAPERS_DIR}")
        return

    files = sorted(os.listdir(PAPERS_DIR))
    print(f"Bulunan toplam dosya sayısı: {len(files)} ({PAPERS_DIR})\n")

    ollama_client = OllamaClient()
    vector_store = VectorStore(ollama_client=ollama_client)
    processor = DocumentProcessor(chunk_size=1200, chunk_overlap=150)

    total_files_processed = 0
    total_chunks_added = 0

    for idx, filename in enumerate(files, 1):
        file_path = os.path.join(PAPERS_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        doc_type = determine_doc_type(filename)
        clean_title = filename.replace("_", " ").replace(".pdf", "").replace(".txt", "")

        print(f"[{idx}/{len(files)}] İndeksleniyor: {filename} ({doc_type})...")

        if filename.lower().endswith(".pdf"):
            doc_data = processor.read_pdf(file_path, filename)
        elif filename.lower().endswith(".txt"):
            doc_data = processor.read_txt(file_path, filename)
        else:
            print(f"  -> Desteklenmeyen dosya türü, atlanıyor: {filename}")
            continue

        if "error" in doc_data or not doc_data.get("content", "").strip():
            print(f"  -> UYARI: İçerik okunamadı veya boş: {doc_data.get('error', 'Boş metin')}")
            continue

        metadata = {
            "source": filename,
            "title": clean_title,
            "url": f"file://{file_path}",
            "type": doc_type
        }

        chunks = processor.chunk_text(doc_data["content"], metadata)
        if not chunks:
            print("  -> Metin parçalara ayrılamadı.")
            continue

        # Add chunks in batches to avoid Ollama timeout
        batch_size = 50
        added_for_file = 0
        for b_idx in range(0, len(chunks), batch_size):
            batch = chunks[b_idx:b_idx + batch_size]
            added_count = vector_store.add_chunks(batch)
            added_for_file += added_count

        total_files_processed += 1
        total_chunks_added += added_for_file
        print(f"  ✓ Tamamlandı: {added_for_file} vektör parçası eklendi.")

    stats = vector_store.get_stats()
    print("\n==========================================")
    print(f"İNDEKSLEME TAMAMLANDI!")
    print(f"İşlenen Dosya Sayısı: {total_files_processed}")
    print(f"Eklenen Yeni Vektör Parçası: {total_chunks_added}")
    print(f"Güncel Veri Tabanı İstatistikleri: {stats}")
    print("==========================================")

if __name__ == "__main__":
    index_all_papers()
