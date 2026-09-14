# Yerel Kriptoanaliz ve Kriptografi RAG Sistemi

Bu proje, yerel **Ollama** modelleriniz (`llama3.1:8b`, `qwen3.8:27b`, `gemma4:12b`, `nemotron-3.5-lightning` vb.) ve `nomic-embed-text` vektör gömme modeli ile çalışan, **Kriptoanaliz Teknikleri**, **Dünyaca Ünlü Atıf Almış Makaleler** ve **IETF Kriptografi RFC Standartları** üzerine odaklanmış Türkçe arayüzlü çevrimdışı ve güvenli bir **RAG (Retrieval-Augmented Generation)** sistemidir.

---

## 1. Sistem Mimarisi ve Ekranlar

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             WEB ARAYÜZÜ (STUDIO UI)                              │
│ RAG Sohbet │ Program Nasıl Çalışır? │ Kütüphane │ Doküman Özetleyici │ Yükleyici    │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ HTTP / REST API
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│                              FastAPI BACKEND (app.py)                            │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│                            RAG ENGINE (src/rag_engine.py)                        │
└──────────────┬─────────────────────────┴─────────────────────────┬───────────────┘
               │                                                   │
┌──────────────▼──────────────┐                     ┌──────────────▼───────────────┐
│ VECTOR STORE (ChromaDB)     │                     │ OLLAMA API CLIENT            │
│  - nomic-embed-text (768d)  │                     │  - qwen3.8:27b               │
│  - Cosine Distance          │                     │  - llama3.1:8b               │
│  - 450 Chunks / 12 Docs     │                     │  - gemma4:12b                │
└─────────────────────────────┘                     └──────────────────────────────┘
```

### Uygulama Sekmeleri

1. 💬 **RAG Sohbet Stüdyosu**: Yerel modellerle doğrudan anlamsal arama ve atıflı yanıt alma.
2. 💡 **Program Nasıl Çalışır?**: Sistemin RAG mimarisini, vektörleştirme ve Ollama işlem adımlarını anlatan rehber.
3. 📚 **Kütüphane (RFC & Makaleler)**: Birleştirilmiş RFC standartları ve dünyaca ünlü atıf almış kriptoanaliz makaleleri kataloğu.
4. 📝 **Doküman Özetleyici**: Veri tabanındaki istenilen makale veya RFC seçilerek yerel yapay zekadan detaylı Türkçe özet oluşturma.
5. 🌐 **Web & Dosya Yükleyici**: Canlı URL kazıma, PDF/TXT yükleme ve doküman yönetimi.

---

## 2. İndekslenen Kütüphane ve Makaleler (450 Chunks, 12 Doküman)

### A. Landmark Kriptoanaliz Makaleleri
1. **Eli Biham & Adi Shamir (1991)** — *Differential Cryptanalysis of DES-like Cryptosystems* (3,500+ Atıf)
2. **Mitsuru Matsui (1993)** — *Linear Cryptanalysis Method for DES Cipher* (3,200+ Atıf)
3. **Paul Kocher, Joshua Jaffe, Benjamin Jun (1999)** — *Differential Power Analysis (DPA)* (4,800+ Atıf)
4. **Xiaoyun Wang, Yiqun Lisa Yin, Hongbo Yu (2005)** — *Finding Collisions in Full MD5 and SHA-0 / SHA-1* (2,800+ Atıf)
5. **Daniel Bleichenbacher (1998)** — *Chosen Ciphertext Attacks Against PKCS #1 (Million Packet Attack)* (2,100+ Atıf)
6. **Don Coppersmith (1996)** — *Finding Small Roots of Univariate Modular Equations & RSA Attacks* (1,900+ Atıf)
7. **Serge Vaudenay (2002)** — *Security Flaws in CBC Mode: Padding Oracles* (1,600+ Atıf)

### B. IETF RFC Kriptografi Standartları
- **RFC 8446**: TLS 1.3 Protokolü
- **RFC 7539**: ChaCha20 ve Poly1305 AEAD
- **RFC 2104**: HMAC (Keyed-Hashing for Message Authentication)

---

## 3. Çalıştırma Rehberi

### Sunucuyu Başlatma
Sistem arka planda aktif çalışmaktadır. Manuel çalıştırmak isterseniz:

```bash
cd /home/halil/Desktop/RAG
source venv/bin/activate
python app.py
```

### Web Arayüzüne Erişim
Tarayıcınızdan şu adrese gidin:
👉 **`http://localhost:8000`**

---

## 4. REST API Uç Noktaları

| Yöntem | Uç Nokta | Açıklama |
| :--- | :--- | :--- |
| `GET` | `/api/stats` | Toplam chunk, doküman, RFC ve makale sayılarını döner. |
| `POST` | `/api/query` | RAG arama ve Ollama LLM yanıtı oluşturur. |
| `POST` | `/api/summarize` | Seçili dokümanı yerel modelle detaylıca Türkçe özetler. |
| `POST` | `/api/ingest/rfc` | Belirtilen RFC numarasını canlı indirip vektörleştirir. |
| `POST` | `/api/ingest/url` | Herhangi bir web makalesi URL'sini kazıyıp vektörleştirir. |
| `POST` | `/api/ingest/file` | PDF/TXT/MD dosyası yükleyip vektörleştirir. |
| `DELETE` | `/api/documents` | Belirtilen dokümanı vektör veri tabanından siler. |
