import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ollama_client import OllamaClient
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

HIGHLY_CITED_PAPERS = [
    {
        "title": "Differential Cryptanalysis of DES-like Cryptosystems",
        "authors": "Eli Biham and Adi Shamir (CRYPTO 1990 / 1991)",
        "source": "Biham & Shamir (1991) - Differential Cryptanalysis",
        "url": "https://link.springer.com/chapter/10.1007/3-540-38424-3_1",
        "type": "Cryptanalysis Paper",
        "content": """
# Differential Cryptanalysis of DES-like Cryptosystems
Authors: Eli Biham, Adi Shamir
Year: 1990/1991 (CRYPTO '90 / Journal of Cryptology 1991)

## Executive Summary & Technical Impact
This paper introduced Differential Cryptanalysis, one of the most powerful and fundamental general cryptanalytic methods against block ciphers. Biham and Shamir demonstrated that DES (Data Encryption Standard) could be broken faster than exhaustive key search using chosen plaintexts.

## Core Methodology
1. XOR Differences: The attack tracks the propagation of input XOR differences (ΔP = P1 ^ P2) through round transformations to output XOR differences (ΔC = C1 ^ C2).
2. S-box Difference Distribution Tables (DDT): Analyzing the non-uniform distribution of output differences given specific input differences for non-linear S-boxes.
3. Differential Trails & Characteristics: Chaining high-probability S-box differential transitions across multiple rounds.
4. Round-Key Recovery: Using the last-round differential characteristics to filter incorrect subkey guesses and isolate the true subkey.

## Results & Complexity
- Full 16-round DES: Broken with 2^47 chosen plaintexts using differential characteristics.
- 8-round DES: Broken in minutes using only 2^14 chosen plaintexts.
- FEAL-4 & FEAL-8: Completely broken with as few as 8 chosen plaintexts.

## Countermeasures & Legacy
- Standardized the requirement for S-box design in modern ciphers (e.g. AES Rijndael S-box max differential probability of 4/256).
- Proved that DES S-boxes were already partially optimized by NSA against differential attacks.
"""
    },
    {
        "title": "Linear Cryptanalysis Method for DES Cipher",
        "authors": "Mitsuru Matsui (EUROCRYPT 1993)",
        "source": "Matsui (1993) - Linear Cryptanalysis",
        "url": "https://link.springer.com/chapter/10.1007/3-540-48285-7_33",
        "type": "Cryptanalysis Paper",
        "content": """
# Linear Cryptanalysis Method for DES Cipher
Author: Mitsuru Matsui
Year: 1993 (EUROCRYPT '93 / EUROCRYPT '94)

## Executive Summary & Technical Impact
Mitsuru Matsui introduced Linear Cryptanalysis, a known-plaintext attack that finds linear approximations relating plaintext bits, ciphertext bits, and secret key bits. Unlike differential cryptanalysis, it requires only known plaintexts rather than chosen plaintexts.

## Core Methodology
1. Linear Approximations: Finding expressions of the form:
   P[i1, i2...] ^ C[j1, j2...] = K[k1, k2...]
   which hold with probability p = 1/2 + ε (where ε is the bias).
2. Matsui's Piling-up Lemma: Evaluates the total bias of a multi-round linear trail constructed from individual S-box linear approximations:
   ε_{1,2...n} = 2^{n-1} * Π (ε_i)
3. Algorithm 1 & Algorithm 2:
   - Algorithm 1 determines 1 parity bit of the key.
   - Algorithm 2 guesses subkey bits of the outer rounds, computes candidate parity bits, and uses maximum likelihood counting to identify the correct key.

## Data & Time Complexity
- Full 16-round DES: Broken with 2^43 known plaintexts (Matsui experimentally verified this on 12 DEC Alpha workstations in 1994).
- 12-round DES: Requires 2^33 known plaintexts.

## Legacy & Modern Cipher Design
- Established the Wide Trail Strategy used in AES (Rijndael) to maximize active S-boxes and eliminate high-bias linear paths.
"""
    },
    {
        "title": "Chosen Ciphertext Attacks Against Protocols Based on PKCS #1",
        "authors": "Daniel Bleichenbacher (CRYPTO 1998)",
        "source": "Bleichenbacher (1998) - PKCS#1 Padding Oracle",
        "url": "https://link.springer.com/chapter/10.1007/BFb0055716",
        "type": "Cryptanalysis Paper",
        "content": """
# Chosen Ciphertext Attacks Against Protocols Based on PKCS #1 (Bleichenbacher's Million Packet Attack)
Author: Daniel Bleichenbacher
Year: 1998 (CRYPTO '98)

## Executive Summary & Technical Impact
Bleichenbacher demonstrated an adaptive chosen-ciphertext attack against RSA encryption padded with PKCS #1 v1.5. By exploiting error messages returned by SSL/TLS servers validating padding formats, an attacker can decrypt ciphertext without knowing the RSA private key.

## Technical Mechanism
1. PKCS #1 v1.5 RSA Padding Format:
   EB = 00 || 02 || PS || 00 || Data
   Valid padded plaintext M MUST start with bytes 0x00 0x02.
2. Padding Oracle: When an attacker sends a trial ciphertext C' = (C * s^e) mod N to a server:
   - If decrypted M' starts with 0x00 0x02, server responds with SSL error A or continues handshake.
   - If decrypted M' does NOT start with 0x00 0x02, server responds with Padding Error B.
3. Interval Reduction: The condition 2*B <= M' < 3*B (where B = 2^{8*(k-2)}) constrains the possible range of the secret message M. By cleverly picking multipliers s, the attacker systematically narrows down the search interval.

## Attack Complexity & Impact
- Requires ~1 million chosen ciphertexts (hence "Million Packet Attack") to recover a full 1024-bit RSA premaster secret.
- Affected all SSL/TLS servers, IPsec, and S/MIME implementations using PKCS #1 v1.5.

## Re-emergence & Prevention
- Re-emerged as ROBOT (Return of Bleichenbacher's Oracle Threat) in 2017 against HTTPS websites.
- Led to RSA OAEP (Optimal Asymmetric Encryption Padding) in RFC 3447 and the complete deprecation of RSA encryption in TLS 1.3 (RFC 8446).
"""
    },
    {
        "title": "Finding Small Roots of Univariate Modular Equations & RSA Low Exponent Attacks",
        "authors": "Don Coppersmith (1996 / Journal of Cryptology 1997)",
        "source": "Coppersmith (1996) - Lattice RSA Attacks",
        "url": "https://link.springer.com/article/10.1007/s001459900030",
        "type": "Cryptanalysis Paper",
        "content": """
# Finding Small Roots of Univariate Modular Equations & Applications to RSA
Author: Don Coppersmith
Year: 1996 / 1997 (EUROCRYPT '96 & Journal of Cryptology)

## Executive Summary & Technical Impact
Coppersmith formulated a method using LLL (Lenstra-Lenstra-Lovász) lattice reduction to find small integer roots of polynomial equations modulo an integer N of unknown factorization. This revolutionized public-key cryptanalysis of RSA.

## Key Theorems & Mathematical Basis
1. Univariate Coppersmith Theorem: Given a polynomial P(x) of degree d modulo N, one can efficiently find all small integer roots x0 satisfying |x0| < N^{1/d}.
2. Bivariate Coppersmith Theorem: Extends to modular bivariate equations x*y + a*x + b*y + c = 0 mod N.

## Major Cryptanalytic Applications
1. Small Public Exponent e=3 RSA Attack: If the same message M is encrypted to 3 recipients using e=3, or if small message space is used without randomized padding, M can be recovered directly.
2. Padded RSA Message Recovery: If two encrypted messages differ only by a known public header (e.g. timestamp), Coppersmith's Hastad/Franklin-Reiter attack recovers both plaintexts.
3. Partial Private Key Exposure: If an attacker learns just 1/4 of the bits of the RSA secret exponent d (or prime factor p), LLL reduction recovers the entire RSA private key.

## Countermeasures
- Must use randomized padding schemes like RSA-OAEP (RFC 3447).
- Public exponent e=65537 (2^16 + 1) recommended instead of e=3.
"""
    },
    {
        "title": "Finding Collisions in the Full MD5 and SHA-0 / SHA-1 Hash Functions",
        "authors": "Xiaoyun Wang, Yiqun Lisa Yin, Hongbo Yu (CRYPTO 2005)",
        "source": "Wang et al. (2005) - MD5 & SHA-1 Collisions",
        "url": "https://link.springer.com/chapter/10.1007/11535218_2",
        "type": "Cryptanalysis Paper",
        "content": """
# Finding Collisions in the Full MD5 and SHA-0 / SHA-1 Hash Functions
Authors: Xiaoyun Wang, Yiqun Lisa Yin, Hongbo Yu
Year: 2005 (CRYPTO 2005 / EUROCRYPT 2005)

## Executive Summary & Technical Impact
Xiaoyun Wang and her team presented revolutionary collision attacks on MD5, SHA-0, Haval-128, and SHA-1. This shattered the cryptographic community's belief in MD5 and SHA-1 collision resistance and forced the global transition to SHA-256 / SHA-3.

## Technical Mechanism
1. Modular Difference & Differential Trails: Tracking non-linear boolean function differences and modular addition (+ 2^32) differences simultaneously.
2. Message Modification Techniques:
   - Neutral bits and basic message modification: Adjusting message block bits in early rounds (rounds 1-16) to force conditions on intermediate states.
   - Advanced message modification: Correcting differential conditions up to round 22.
3. Collision Search Efficiency:
   - MD5 Collision: Reduced complexity from 2^64 to 2^39 operations (takes seconds on a laptop).
   - SHA-1 Collision: Reduced complexity from 2^80 to 2^63 operations (later realized as SHAttered attack by Google in 2017).

## Real-World Impact
- Flame Malware (2012): Used MD5 collision to forge a fake Microsoft Code Signing Certificate.
- Deprecation of MD5 (RFC 6151) and SHA-1 (RFC 6194) across TLS, SSH, Git, and X.509 certificates.
"""
    },
    {
        "title": "Differential Power Analysis",
        "authors": "Paul Kocher, Joshua Jaffe, Benjamin Jun (CRYPTO 1999)",
        "source": "Kocher et al. (1999) - Differential Power Analysis",
        "url": "https://link.springer.com/chapter/10.1007/3-540-48405-1_25",
        "type": "Cryptanalysis Paper",
        "content": """
# Differential Power Analysis (DPA)
Authors: Paul Kocher, Joshua Jaffe, Benjamin Jun
Year: 1999 (CRYPTO '99)

## Executive Summary & Technical Impact
Introduced Differential Power Analysis (DPA), a physical side-channel attack that extracts cryptographic keys from hardware devices (smart cards, ASIC, microcontrollers) by measuring power consumption during cryptographic execution.

## Technical Mechanism
1. Power Consumption Model: Transistors in CMOS logic draw current when switching states (0 -> 1 or 1 -> 0). Power consumption correlates directly with Hamming weight or Hamming distance of processed data.
2. Data Collection: Attacker records power traces (oscilloscope measurements) for N random plaintext encryptions.
3. Statistical Partitioning & Hypothesis Testing:
   - Guess a subkey bit k.
   - Predict intermediate target bit D(P, k).
   - Split power traces into two sets: S0 (where predicted bit is 0) and S1 (where predicted bit is 1).
   - Compute difference of means: ΔD = Mean(S0) - Mean(S1).
4. Signal-to-Noise Isolation: If the subkey guess is correct, ΔD produces a sharp statistical spike. If incorrect, ΔD approaches 0.

## Impact & Mitigation
- Defeated unshielded DES, AES, RSA, and ECC hardware chips without needing physical micro-probing.
- Countermeasures: Masking (XORing internal states with random nonces), Dual-rail balanced logic, Random dummy cycles / clock jittering.
"""
    },
    {
        "title": "Security Flaws in CBC Mode & Padding Oracles",
        "authors": "Serge Vaudenay (EUROCRYPT 2002)",
        "source": "Vaudenay (2002) - CBC Padding Oracle Attack",
        "url": "https://link.springer.com/chapter/10.1007/3-540-46035-7_35",
        "type": "Cryptanalysis Paper",
        "content": """
# Security Flaws in Cipher Block Chaining (CBC) Mode: Padding Oracles
Author: Serge Vaudenay
Year: 2002 (EUROCRYPT 2002)

## Executive Summary & Technical Impact
Vaudenay showed that any symmetric encryption protocol using Cipher Block Chaining (CBC) mode with standard PKCS#7 / PKCS#5 byte padding is vulnerable to plaintext recovery if the receiver leaks padding validity status.

## Technical Mechanism
1. CBC Decryption Formula:
   P_i = D_K(C_i) ^ C_{i-1}
2. Oracle Exploitation:
   - Attacker manipulates the last byte of C_{i-1} (let's call it C'_{i-1}).
   - Sends modified ciphertext (C'_{i-1}, C_i) to server.
   - Server decrypts D_K(C_i), XORs with C'_{i-1}, and checks if ending byte equals valid PKCS#7 padding 0x01.
   - If padding is valid, D_K(C_i)[last_byte] ^ C'_{i-1}[last_byte] = 0x01.
   - Attacker immediately learns D_K(C_i)[last_byte] = 0x01 ^ C'_{i-1}[last_byte].
3. Byte-by-Byte Decryption: Repeating this for 0x02 0x02, 0x03 0x03 0x03 allows full decryption of any block without knowing key K.

## Impact & Mitigation
- Affected SSH, IPsec, ASP.NET, POODLE attack on SSL 3.0 (2014), Lucky Thirteen timing attack on TLS (2013).
- Mandated standard Authenticated Encryption with Associated Data (AEAD) like AES-GCM or ChaCha20-Poly1305.
"""
    }
]

def add_papers():
    print("Ingesting Highly Cited Cryptanalysis Papers into Vector Database...")
    ollama_client = OllamaClient()
    vector_store = VectorStore(ollama_client=ollama_client)
    rag_engine = RAGEngine(vector_store, ollama_client)

    total_chunks = 0
    for p in HIGHLY_CITED_PAPERS:
        doc_meta = {
            "source": p["source"],
            "title": f"{p['title']} ({p['authors']})",
            "url": p["url"],
            "type": p["type"]
        }
        chunks = rag_engine.processor.chunk_text(p["content"], doc_meta)
        added = vector_store.add_chunks(chunks)
        total_chunks += added
        print(f"✓ Added '{p['title']}' ({added} chunks)")

    stats = vector_store.get_stats()
    print(f"\nSuccessfully added {len(HIGHLY_CITED_PAPERS)} landmark cryptanalysis papers ({total_chunks} new chunks)!")
    print("Updated Vector DB Stats:", stats)

if __name__ == "__main__":
    add_papers()
