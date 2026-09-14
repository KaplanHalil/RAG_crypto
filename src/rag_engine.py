from typing import List, Dict, Any, Optional
from src.ollama_client import OllamaClient
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

class RAGEngine:
    def __init__(self, vector_store: VectorStore, ollama_client: OllamaClient):
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.processor = DocumentProcessor()

    def ingest_rfc(self, rfc_number: int) -> Dict[str, Any]:
        """Fetch and index an IETF RFC document into vector database."""
        rfc_data = self.processor.fetch_rfc(rfc_number)
        if "error" in rfc_data:
            return rfc_data

        chunks = self.processor.chunk_text(rfc_data["content"], {
            "source": rfc_data["source"],
            "title": rfc_data["title"],
            "url": rfc_data["url"],
            "type": rfc_data["type"]
        })

        added_count = self.vector_store.add_chunks(chunks)
        return {
            "success": True,
            "source": rfc_data["source"],
            "title": rfc_data["title"],
            "chunks_added": added_count
        }

    def ingest_url(self, url: str) -> Dict[str, Any]:
        """Fetch and index a web article (cryptanalysis technique, paper, blog) into vector DB."""
        article = self.processor.fetch_web_article(url)
        if "error" in article:
            return article

        chunks = self.processor.chunk_text(article["content"], {
            "source": article["source"],
            "title": article["title"],
            "url": article["url"],
            "type": article["type"]
        })

        added_count = self.vector_store.add_chunks(chunks)
        return {
            "success": True,
            "source": article["source"],
            "title": article["title"],
            "chunks_added": added_count
        }

    def ingest_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Index a local PDF or TXT file into vector store."""
        if filename.lower().endswith('.pdf'):
            doc = self.processor.read_pdf(file_path, filename)
        else:
            doc = self.processor.read_txt(file_path, filename)

        if "error" in doc:
            return doc

        chunks = self.processor.chunk_text(doc["content"], {
            "source": doc["source"],
            "title": doc["title"],
            "url": doc["url"],
            "type": doc["type"]
        })

        added_count = self.vector_store.add_chunks(chunks)
        return {
            "success": True,
            "source": doc["source"],
            "title": doc["title"],
            "chunks_added": added_count
        }

    def query(
        self,
        question: str,
        model: str = "llama3.1:8b",
        top_k: int = 5,
        doc_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute RAG query pipeline: search vector store -> construct prompt -> call local Ollama model."""
        # 1. Similarity search in vector store
        search_results = self.vector_store.search(question, top_k=top_k, doc_type_filter=doc_type_filter)

        if not search_results or (search_results and search_results[0]["similarity"] < 0.25):
            return {
                "answer": "Veritabanında sorunuzla doğrudan ilgili kriptografik standart (RFC) veya kriptoanaliz makalesi bulunamadı. Lütfen sorunuzu farklı anahtar kelimelerle tekrarlayın veya ilgili dokümanı sisteme yükleyin.",
                "sources": [],
                "retrieved_chunks": []
            }

        # 2. Build Context String with citations and consistent 1-based source indices
        context_parts = []
        sources = []
        seen_sources = {}

        for res in search_results:
            meta = res["metadata"]
            source_label = meta.get("source", "Bilinmeyen Kaynak")
            title_label = meta.get("title", source_label)

            if source_label not in seen_sources:
                source_id = len(seen_sources) + 1
                seen_sources[source_label] = source_id
                sources.append({
                    "id": source_id,
                    "source": source_label,
                    "title": title_label,
                    "url": meta.get("url", ""),
                    "type": meta.get("type", "Belge"),
                    "similarity": res["similarity"]
                })
            else:
                source_id = seen_sources[source_label]

            context_parts.append(
                f"--- [KAYNAK {source_id}]: {title_label} (Dosya/Kaynak: {source_label}) ---\n"
                f"{res['content']}\n"
            )

        context_str = "\n".join(context_parts)

        # 3. Construct System Prompt & User Prompt
        system_prompt = (
            "Sen uzman bir Kriptografi ve Kriptoanaliz Yapay Zeka Asistanısın. "
            "Kullanıcının sorularını SADECE aşağıda verilen 'RETRIEVED KNOWLEDGE CONTEXT' (Alınan Bilgi Bağlamı) "
            "bölümündeki kaynaklara dayanarak teknik, doğru ve detaylı şekilde yanıtla.\n\n"
            "ÖNEMLİ KURALLAR:\n"
            "1. YANITINI KESİNLİKLE VERİLEN KAYNAKLARA DAYANDIR. Verilen bağlamda yer almayan hiçbir makale, RFC veya sonucu uydurma (halüsinasyon görme).\n"
            "2. ATIF VE REFERANSLAR: Yanıtında bilgi aktarırken mutlaka ilgili bilginin yanına tam kaynak referansını [Kaynak X: Belge Adı] şeklinde ekle (örnek: [Kaynak 1: Linear Cryptanalysis Method for DES Cipher]).\n"
            "3. Sadece yukarıdaki bağlamda gerçekten bulunan [KAYNAK 1], [KAYNAK 2] vb. numaralarına atıf yap. Asla bağlamda olmayan kaynak numarası veya doküman ismi türetme.\n"
            "4. Kullanıcı soruyu Türkçe sorduysa yanıtını tamamen Türkçe ver. İngilizce sorduysa İngilizce yanıtla.\n"
            "5. Eğer verilen kaynaklar sorunun tam cevabını içermiyorsa, kaynaklarda yer alan kısmı açıkla ve kalan kısmın kaynaklarda bulunmadığını dürüstçe belirt."
        )

        user_prompt = (
            f"Kullanıcı Sorusu: {question}\n\n"
            f"RETRIEVED KNOWLEDGE CONTEXT (ALINAN KAYNAKLAR):\n{context_str}\n\n"
            f"Talimat: Yukarıdaki kaynakları kullanarak soruyu teknik olarak detaylıca yanıtla ve her önemli bilginin yanına [Kaynak X: Belge Başlığı] şeklinde açık kaynak atıflarını ekle."
        )

        # 4. Query Ollama local LLM
        response_text = self.ollama_client.generate_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.2
        )

        return {
            "answer": response_text,
            "sources": sources,
            "retrieved_chunks": search_results
        }

    def summarize_document(self, source: str, model: str = "llama3.1:8b", summary_type: str = "general") -> Dict[str, Any]:
        """Fetch all chunks of a specific document and generate a structured Turkish summary using Ollama."""
        all_records = self.vector_store.collection.get(where={"source": source})
        if not all_records or not all_records.get("documents") or len(all_records["documents"]) == 0:
            return {"error": f"'{source}' isimli doküman veritabanında bulunamadı."}

        docs = all_records["documents"]
        metas = all_records.get("metadatas", [])
        title = metas[0].get("title", source) if metas else source
        doc_type = metas[0].get("type", "Doküman") if metas else "Doküman"

        # Combine text up to limit
        combined_text = "\n\n".join(docs[:15])  # Take up to 15 chunks (~15,000 words max)

        system_prompt = (
            "Sen kıdemli bir Kriptografi ve Kriptoanaliz Uzmanı ve Araştırmacısısın. "
            "Sana verilen makale/RFC dokümanını inceleyerek son derece kaliteli, teknik ve detaylı bir Türkçe özet oluşturacaksın.\n"
            "Özet formatında şu başlıkları mutlaka kullan:\n"
            "1. **Doküman / Makale Künyesi ve Amacı**\n"
            "2. **Temel Kriptoanalitik veya Teknik Yöntem**\n"
            "3. **Saldırı / Protokol Detayları ve Karmaşıklık (Time/Data Complexity)**\n"
            "4. **Kriptografik Etki ve Savunma Yöntemleri (Countermeasures)**"
        )

        prompt_instruction = "Lütfen aşağıdaki doküman içeriğini yukarıdaki kurallara göre detaylıca Türkçe özetle:"
        if summary_type == "attacks":
            prompt_instruction = "Lütfen bu dokümandaki sadece saldırı yöntemlerine, veri/zaman karmaşıklıklarına ve teknik zafiyetlere odaklanarak Türkçe özet çıkar:"
        elif summary_type == "standards":
            prompt_instruction = "Lütfen bu dokümandaki kriptografik protokol standartlarına, parametrelere ve güvenlik kurallarına odaklanarak Türkçe özet çıkar:"

        user_prompt = f"DOKÜMAN BAŞLIĞI: {title} ({source})\n\nDOCUMENT CONTENT:\n{combined_text}\n\n{prompt_instruction}"

        summary_text = self.ollama_client.generate_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.2
        )

        return {
            "source": source,
            "title": title,
            "type": doc_type,
            "chunk_count": len(docs),
            "summary": summary_text
        }

