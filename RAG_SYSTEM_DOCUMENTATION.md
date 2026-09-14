# Yerel Kriptoanaliz ve Kriptografi RAG Sistemi Dokümantasyonu

Bu doküman, `/home/halil/Desktop/RAG` dizininde geliştirilen ve makinenizdeki yerel **Ollama** modelleri ile çalışan **Kriptoanaliz & Kriptografi RAG (Retrieval-Augmented Generation)** sisteminin mimarisini, veri tabanını, bileşenlerini ve kullanım kılavuzunu detaylandırmaktadır.

---

## 1. Sistem Mimarisi ve Genel Bakış

Bu sistem, internet üzerindeki kriptoanaliz teknikleri, akademide en çok atıf almış dönüm noktası makaleler ve IETF Kriptografi RFC standartlarını indeksleyerek yerel Yapay Zeka modellerinizle (**Ollama**) tamamen çevrimdışı ve güvenli bir şekilde sorgulama yapmanızı sağlar.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        WEB ARAYÜZÜ (STUDIO UI)                         │
│             RAG Chat | Landmark Papers | RFC Library | Ingester        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST API
┌───────────────────────────────────▼────────────────────────────────────┐
│                         FastAPI BACKEND (app.py)                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       RAG ENGINE (src/rag_engine.py)                   │
└──────────────┬────────────────────┴────────────────────┬───────────────┘
               │                                         │
┌──────────────▼──────────────┐           ┌──────────────▼───────────────┐
│ VECTOR STORE (ChromaDB)     │           │ OLLAMA API CLIENT            │
│  - nomic-embed-text (768d)  │           │  - qwen3.8:27b               │
│  - Cosine Distance          │           │  - llama3.1:8b               │
│  - 450 Chunks / 12 Docs     │           │  - gemma4:12b                │
└─────────────────────────────┘           └──────────────────────────────┘
```

---

## 2. Proje Dosya Yapısı

| Dosya / Dizin | Görevi ve Açıklaması |
| :--- | :--- |
| **`app.py`** | FastAPI web sunucusu ve REST API uç noktaları (`/api/query`, `/api/stats`, `/api/ingest/*`). |
| **`src/ollama_client.py`** | Ollama API entegrasyonu; `nomic-embed-text` vektör üretimi ve LLM yanıt üretimi. |
| **`src/document_processor.py`** | Paragraf bazlı akıllı metin bölücü (chunker), RFC indirici, web kazıyıcı (scraper), PDF/TXT okuyucu. |
| **`src/vector_store.py`** | **ChromaDB** persistent vektör veri tabanı yöneticisi ve benzerlik araması. |
| **`src/rag_engine.py`** | Vektör araması, kaynak atıf zenginleştirmesi (`[SOURCE 1]`, `[RFC 8446]`) ve prompt orchestrator. |
| **`src/paper_catalog.py`** | En çok atıf alan kriptoanaliz makalelerinin kataloğu. |
| **`src/rfc_fetcher.py`** | Popüler IETF Kriptografi RFC listesi kataloğu. |
| **`templates/index.html`** | Dark-Mode Glassmorphism tasarımına sahip modern HTML5/CSS3/JS arayüzü. |
| **`add_cited_cryptanalysis_papers.py`** | Dünyaca ünlü 7 makaleyi veri tabanına işleyen betik. |
| **`seed_database.py`** | İlk kurulumda temel kriptoanaliz tekniklerini ve RFC'leri yükleyen betik. |
| **`data/chroma_db/`** | Vektör verilerinin kalıcı olarak saklandığı dizin. |

---

## 3. Bilgi Tabanı ve Eklenen Makaleler

Veri tabanında şu anda **450 vektör parçası (chunk)** ve **12 temel doküman** indekslenmiştir:

### A. En Çok Atıf Alan Kriptoanaliz Makaleleri (Landmark Papers)
1. **Eli Biham & Adi Shamir (1991)** — *Differential Cryptanalysis of DES-like Cryptosystems* (3,500+ Atıf)
   - **Analiz**: XOR fark yayılımı (ΔP → ΔC), S-box fark dağıtım tabloları (DDT) ve DES'in $2^{47}$ seçilmiş metinle kırılması.
2. **Mitsuru Matsui (1993)** — *Linear Cryptanalysis Method for DES Cipher* (3,200+ Atıf)
   - **Analiz**: Lineer yaklaşımlar, Matsui Piling-up Lemma, bilinen açık metin saldırıları ve $2^{43}$ metin ile DES kırma.
3. **Paul Kocher, Joshua Jaffe, Benjamin Jun (1999)** — *Differential Power Analysis (DPA)* (4,800+ Atıf)
   - **Analiz**: Yan kanal (Side-Channel) güç tüketim dalgalarının istatistiksel analizi ile kriptografik donanımlardan anahtar çıkarma.
4. **Xiaoyun Wang, Yiqun Lisa Yin, Hongbo Yu (2005)** — *Finding Collisions in Full MD5 and SHA-0 / SHA-1* (2,800+ Atıf)
   - **Analiz**: MD5 ve SHA-1 özet fonksiyonlarının çakışma direncinin kırılması ve mesaj modifikasyon teknikleri.
5. **Daniel Bleichenbacher (1998)** — *Chosen Ciphertext Attacks Against PKCS #1 (Million Packet Attack)* (2,100+ Atıf)
   - **Analiz**: RSA PKCS #1 v1.5 padding oracle hatasından yararlanarak SSL/TLS şifreli metinlerini çözme.
6. **Don Coppersmith (1996)** — *Finding Small Roots of Univariate Modular Equations & RSA Attacks* (1,900+ Atıf)
   - **Analiz**: LLL (Lenstra-Lenstra-Lovász) kafes indirgeme yöntemiyle küçük üslü ($e=3$) RSA ve kısmi anahtar sızıntısı saldırıları.
7. **Serge Vaudenay (2002)** — *Security Flaws in Cipher Block Chaining (CBC) Mode: Padding Oracles* (1,600+ Atıf)
   - **Analiz**: CBC modunda PKCS#7 dolgu doğrulama yanıtlarını sömürerek anahtarsız metin çözme.

### B. İndekslenen IETF RFC Standartları
- **RFC 8446**: The Transport Layer Security (TLS) Protocol Version 1.3
- **RFC 7539**: ChaCha20 and Poly1305 for IETF Protocols
- **RFC 2104**: HMAC: Keyed-Hashing for Message Authentication

---

## 4. Kullanım ve Çalıştırma Rehberi

### Sunucuyu Başlatma
Sunucu arka planda `http://localhost:8000` adresinde çalışmaktadır. Manuel olarak başlatmak için:

```bash
cd /home/halil/Desktop/RAG
source venv/bin/activate
python app.py
```

### Web Arayüzü Özellikleri (`http://localhost:8000`)
1. **RAG Chat Studio**: 
   - İstenilen Ollama modelini (`llama3.1:8b`, `qwen3.8:27b`, `gemma4:12b` vb.) seçme.
   - Getirilecek vektör sayısını (Top-K) ayarlama.
   - Doküman türü filtresi uygulama (yalnızca RFC'ler veya yalnızca Makaleler).
   - Yanıtların altında doğrudan atıfta bulunulan kaynakları (`[SOURCE 1]`) görüntüleme.
2. **Landmark Papers**: Tek tıkla atıf almış 7 büyük makaleyi inceleme ve RAG sorgusu gönderme.
3. **RFC Library**: RFC 1321, 3447, 9180 gibi popüler RFC'leri veya istenen herhangi bir RFC numarasını IETF'ten canlı indirip vektörleştirme.
4. **Web Article Ingester**: Herhangi bir kriptoanaliz makale URL'sini kazıyıp vektör veri tabanına ekleme.
5. **Knowledge Base Manager**: Yerel PDF/TXT yükleme ve indekslenmiş dokümanları silme/yönetme.

---

## 5. Örnek API Kullanımı (curl)

**Veri Tabanı İstatistiklerini Sorgulama:**
```bash
curl -s http://localhost:8000/api/stats
```

**RAG Sorgusu Gönderme:**
```bash
curl -s -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does Differential Cryptanalysis work and how do S-boxes mitigate it?",
    "model": "llama3.1:8b",
    "top_k": 3
  }'
```
