import sys
import os

# Ensure src module importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ollama_client import OllamaClient
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

CRYPTANALYSIS_TECHNIQUES_DOC = """
# Comprehensive Overview of Cryptanalysis Techniques

## 1. Differential Cryptanalysis
Differential cryptanalysis is a general form of cryptanalysis applicable primarily to block ciphers, but also to stream ciphers and cryptographic hash functions. In the broadest sense, it is the study of how differences in an input pair can affect the resulting difference at the output. In the case of a block cipher, it refers to a set of techniques for tracing differences through the network of transformations, discovering where the cipher exhibits non-random behavior, and exploiting such properties to recover the secret key.

Key Concepts:
- Difference Propagation: Measuring XOR differences Delta X = X1 ^ X2 through Substitution-Permutation Networks (SPN) or Feistel networks.
- Differential Characteristics / Trails: High probability paths through S-boxes.
- Key Recovery: Guessing round keys for the last round and verifying if the expected output difference matches.
- Countermeasures: Designing S-boxes with low maximum differential probability (e.g., AES S-box max diff probability is 4/256).

## 2. Linear Cryptanalysis
Linear cryptanalysis was introduced by Mitsuru Matsui in 1993 to attack DES. It is a known-plaintext attack that finds high-probability linear approximations between parity bits of plaintext, ciphertext, and secret key bits.

Key Concepts:
- Linear Approximations: Expressions of the form P[i1, i2...] ^ C[j1, j2...] = K[k1, k2...] holding with probability p = 1/2 + bias.
- Matsui's Piling-up Lemma: Used to calculate the bias of a linear trail constructed by joining multiple S-box linear approximations.
- Data Complexity: Inversely proportional to the square of the bias (1 / bias^2).
- Countermeasures: Maximizing the minimum number of active S-boxes (branch number) and minimizing S-box linear probability.

## 3. Meet-in-the-Middle Attack (MITM)
Meet-in-the-middle attack is a space-time tradeoff attack against cryptographic schemes that use multiple encryption steps (such as 2DES or 3DES). 

Key Concepts:
- Double DES Attack: Given plaintext P and ciphertext C = E_K2(E_K1(P)). The attacker computes E_K1(P) for all possible 2^56 keys K1 and stores them in a hash table. Then computes D_K2(C) for all possible K2 and searches for collisions in the hash table.
- Time Complexity: Reducible from 2^112 to 2^57 calculations.
- Application to Hash Functions: Used in preimage and collision attacks on hash functions by matching intermediate states from backward and forward computations.

## 4. Side-Channel Attacks (SCA)
Side-channel attacks target physical implementations of cryptographic algorithms rather than mathematical flaws in the algorithms themselves.

Key Variants:
- Simple Power Analysis (SPA) & Differential Power Analysis (DPA): Measuring power consumption during hardware operations. DPA uses statistical correlation across thousands of traces to isolate key bits.
- Timing Attacks: Measuring execution time differences caused by conditional branches (e.g., modular exponentiation in RSA or Montgomery ladder).
- Electromagnetic Attacks (EMA): Measuring EM radiation emitted by CPU/crypto chips.
- Fault Injection Attacks (DFA): Injecting laser, voltage, or clock glitches during computation (e.g., Bellcore attack on RSA-CRT signatures).

## 5. Lattice-Based Cryptanalysis & LLL Reduction
Lattice cryptanalysis uses lattice algorithms (primarily Lenstra-Lenstra-Lovász / LLL lattice reduction and BKZ) to break public key schemes or solve hidden number problems.

Key Applications:
- RSA Low Public Exponent Attack (Coppersmith's Method): Finding small roots of modular polynomials using LLL. Enables solving RSA when e=3 with small message space or partial key exposure.
- DSA / ECDSA Nonce Leakage: If nonces (k) in ECDSA signatures leak even a few bits or have biased PRNG generation, the secret key can be recovered by formulating a Hidden Number Problem (HNP) as a lattice and applying LLL.
- Knapsack Cryptosystems: Completely broken using low-density lattice reduction techniques.

## 6. Padding Oracle Attacks
Padding oracle attacks exploit side-channel leakage (such as error messages or response timing) when a server decrypts CBC-mode ciphertext and validates PKCS#7 padding.

Key Concepts:
- Mechanism: By modifying the second-to-last ciphertext block C_{i-1} and observing whether the server returns a "Padding Invalid" error versus a generic error, an attacker decrypts byte-by-byte without knowing the secret key.
- Target Algorithms: AES-CBC, DES-CBC in TLS/HTTPS, ASP.NET encrypted cookies, SSH.
- Countermeasures: Always use Authenticated Encryption (AEAD) like AES-GCM or ChaCha20-Poly1305, or standard Encrypt-then-MAC (EtM).

## 7. Slide Attacks and Rotational Cryptanalysis
Slide attacks target ciphers with self-similar key schedules (iterated round structure where every round uses the exact same key or repetitive key schedule).

Rotational Cryptanalysis:
- Targets ARX (Addition-Rotation-XOR) symmetric primitives (like Blake2, Chaskey, Speck, Simon).
- Examines how rotational differences (shifting bit-rotations) propagate through modular addition and XOR operations.
"""

def seed_data():
    print("Starting vector database seeding with Cryptanalysis knowledge...")
    ollama_client = OllamaClient()
    vector_store = VectorStore(ollama_client=ollama_client)
    rag_engine = RAGEngine(vector_store, ollama_client)

    # 1. Ingest Cryptanalysis Techniques Reference Doc
    print("Ingesting Cryptanalysis Techniques Guide...")
    doc_meta = {
        "source": "Cryptanalysis Techniques Reference",
        "title": "Comprehensive Guide to Cryptanalysis Techniques & Attacks",
        "url": "https://en.wikipedia.org/wiki/Cryptanalysis",
        "type": "Cryptanalysis Article"
    }
    chunks = rag_engine.processor.chunk_text(CRYPTANALYSIS_TECHNIQUES_DOC, doc_meta)
    added = vector_store.add_chunks(chunks)
    print(f"Ingested {added} chunks of Cryptanalysis Techniques.")

    # 2. Ingest Popular Cryptography RFCs
    rfcs_to_seed = [8446, 7539, 2104]  # TLS 1.3, ChaCha20-Poly1305, HMAC
    for rfc_num in rfcs_to_seed:
        print(f"Fetching and Ingesting RFC {rfc_num}...")
        res = rag_engine.ingest_rfc(rfc_num)
        if res.get("success"):
            print(f"Successfully added RFC {rfc_num} ({res['chunks_added']} chunks).")
        else:
            print(f"Failed to fetch RFC {rfc_num}: {res.get('error')}")

    stats = vector_store.get_stats()
    print("Seeding complete!", stats)

if __name__ == "__main__":
    seed_data()
