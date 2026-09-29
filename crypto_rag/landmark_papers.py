"""Curated landmark cryptanalysis papers.

These entries provide machine-readable, structurally-normalized knowledge
for the most-cited cryptanalysis papers that shaped modern symmetric and
asymmetric cryptography. They are ingested alongside the local PDF corpus
and are referenced by the QA benchmark as ground-truth sources.
"""

from __future__ import annotations

from .ingestion import Document

_L = [
    dict(
        source="Biham & Shamir (1991) - Differential Cryptanalysis",
        title="Differential Cryptanalysis of DES-like Cryptosystems "
              "(Biham, Shamir 1991)",
        url="https://link.springer.com/chapter/10.1007/3-540-38424-3_1",
        type="Cryptanalysis Paper",
        content="""
# Differential Cryptanalysis of DES-like Cryptosystems
Authors: Eli Biham, Adi Shamir. CRYPTO 1990 / Journal of Cryptology 1991.

## Core methodology
1. XOR differences: the attack tracks the propagation of input XOR
   differences (dP = P1 XOR P2) through the round transformations to output
   XOR differences (dC = C1 XOR C2).
2. S-box Difference Distribution Tables (DDT): the non-uniform distribution
   of output differences given an input difference for a non-linear S-box.
3. Differential characteristics: chaining high-probability S-box transitions
   across multiple rounds.
4. Round-key recovery: the last-round characteristic is used to filter wrong
   subkey guesses and isolate the correct subkey.

## Results and complexity
- Full 16-round DES broken with 2^47 chosen plaintexts.
- 8-round DES broken in minutes with 2^14 chosen plaintexts.
- FEAL-4 and FEAL-8 broken with as few as 8 chosen plaintexts.

## Countermeasures and legacy
- Established S-box design criteria; the AES (Rijndael) S-box has maximum
  differential probability 4/256.
- Showed that DES S-boxes were already partially optimized by the NSA
  against differential attacks.
"""),
    dict(
        source="Matsui (1993) - Linear Cryptanalysis",
        title="Linear Cryptanalysis Method for DES Cipher (Matsui 1993)",
        url="https://link.springer.com/chapter/10.1007/3-540-48285-7_33",
        type="Cryptanalysis Paper",
        content="""
# Linear Cryptanalysis Method for DES Cipher
Author: Mitsuru Matsui. EUROCRYPT 1993 / 1994.

## Core methodology
1. Linear approximations: expressions of the form
   P[i1,i2,...] XOR C[j1,j2,...] = K[k1,k2,...] that hold with probability
   p = 1/2 + epsilon, where epsilon is the bias.
2. Matsui's Piling-up Lemma: the total bias of a trail built from n
   independent S-box approximations is epsilon = 2^(n-1) * product(epsilon_i).
3. Algorithm 1 (determines one parity bit of the key) and Algorithm 2
   (guesses outer-round subkey bits and uses maximum-likelihood counting).

## Data and time complexity
- Full 16-round DES broken with 2^43 known plaintexts (experimentally
  verified by Matsui in 1994 on DEC Alpha workstations).
- 12-round DES requires 2^33 known plaintexts.

## Legacy
- Known-plaintext attack; established the Wide Trail Strategy used in the
  AES design to maximize active S-boxes and eliminate high-bias trails.
"""),
    dict(
        source="Bleichenbacher (1998) - PKCS#1 Padding Oracle",
        title="Chosen Ciphertext Attacks Against PKCS #1 "
              "(Bleichenbacher 1998)",
        url="https://link.springer.com/chapter/10.1007/BFb0055716",
        type="Cryptanalysis Paper",
        content="""
# Chosen Ciphertext Attacks Against Protocols Based on PKCS #1
Author: Daniel Bleichenbacher. CRYPTO 1998.

## Mechanism
1. PKCS #1 v1.5 RSA padding: EB = 00 || 02 || PS || 00 || Data; a valid
   decryption must start with bytes 0x00 0x02.
2. Padding oracle: for trial ciphertext C' = (C * s^e) mod N, whether the
   server returns a handshake/padding error reveals whether the decrypted
   message starts with 0x00 0x02.
3. Interval reduction: the condition 2B <= M' < 3B with B = 2^(8(k-2))
   bounds the message M; iteratively chosen multipliers s shrink the
   interval until M is recovered.

## Complexity and impact
- Requires about 1 million chosen ciphertexts to recover a full 1024-bit
  RSA pre-master secret (the "Million Message Attack").
- Affected SSL/TLS, IPsec, and S/MIME implementations using PKCS #1 v1.5.
- Re-emerged as ROBOT in 2017; motivated RSA-OAEP (RFC 3447) and the
  removal of RSA encryption from TLS 1.3 (RFC 8446).
"""),
    dict(
        source="Coppersmith (1996) - Lattice RSA Attacks",
        title="Finding Small Roots of Univariate Modular Equations: "
              "RSA Attacks (Coppersmith 1996)",
        url="https://link.springer.com/article/10.1007/s001459900030",
        type="Cryptanalysis Paper",
        content="""
# Finding Small Roots of Univariate Modular Equations and Applications to RSA
Author: Don Coppersmith. EUROCRYPT 1996 / Journal of Cryptology 1997.

## Theorems
1. Univariate Coppersmith theorem: given a degree-d polynomial P(x) mod N,
   all integer roots x0 with |x0| < N^(1/d) can be found efficiently using
   LLL lattice reduction.
2. Bivariate variant for equations such as x*y + a*x + b*y + c = 0 (mod N).

## Applications
1. Small-exponent attack: with public exponent e=3 and no randomized
   padding, a message encrypted to three recipients can be recovered.
2. Franklin-Reiter / Hastad: recover plaintexts when two messages differ by
   a known linear relation.
3. Partial key exposure: knowing a quarter of the bits of d (or of a factor
   p) recovers the full RSA private key via LLL.

## Countermeasures
- Randomized padding (RSA-OAEP, RFC 3447); e = 65537 instead of e = 3.
"""),
    dict(
        source="Wang et al. (2005) - MD5 & SHA-1 Collisions",
        title="Finding Collisions in the Full MD5 and SHA-0/SHA-1 "
              "(Wang, Yin, Yu 2005)",
        url="https://link.springer.com/chapter/10.1007/11535218_2",
        type="Cryptanalysis Paper",
        content="""
# Finding Collisions in the Full MD5 and SHA-0/SHA-1 Hash Functions
Authors: Xiaoyun Wang, Yiqun Lisa Yin, Hongbo Yu. CRYPTO 2005.

## Mechanism
1. Modular differences and differential trails: tracking both boolean
   function differences and modular (mod 2^32) differences.
2. Message modification: neutral bits force conditions on early-round
   (rounds 1-16) intermediate states; advanced modification corrects
   conditions up to round 22.

## Collision complexity
- MD5: from 2^64 to about 2^39 operations (seconds on a laptop).
- SHA-1: from 2^80 to about 2^63 operations (realized as SHAttered in 2017).

## Impact
- The Flame malware (2012) forged a Microsoft code-signing certificate
  using an MD5 chosen-prefix collision.
- Drove deprecation of MD5 (RFC 6151) and SHA-1 (RFC 6194) across TLS,
  SSH, Git, and X.509.
"""),
    dict(
        source="Kocher et al. (1999) - Differential Power Analysis",
        title="Differential Power Analysis (Kocher, Jaffe, Jun 1999)",
        url="https://link.springer.com/chapter/10.1007/3-540-48405-1_25",
        type="Cryptanalysis Paper",
        content="""
# Differential Power Analysis (DPA)
Authors: Paul Kocher, Joshua Jaffe, Benjamin Jun. CRYPTO 1999.

## Mechanism
1. Power model: CMOS power consumption correlates with the Hamming weight
   (or distance) of processed intermediate data.
2. Collection: record power traces for N random plaintext encryptions.
3. Statistical test: guess a subkey, predict an intermediate bit D(P,k),
   split traces into S0 (predicted 0) and S1 (predicted 1), and compute the
   difference of means. A correct guess produces a sharp statistical spike.

## Impact and mitigation
- Extracted keys from unshielded DES, AES, RSA, and ECC hardware.
- Countermeasures: Boolean masking, dual-rail balanced logic, random
  dummy cycles and clock jitter.
"""),
    dict(
        source="Vaudenay (2002) - CBC Padding Oracle",
        title="Security Flaws in CBC Mode: Padding Oracles (Vaudenay 2002)",
        url="https://link.springer.com/chapter/10.1007/3-540-46035-7_35",
        type="Cryptanalysis Paper",
        content="""
# Security Flaws in Cipher Block Chaining (CBC) Mode: Padding Oracles
Author: Serge Vaudenay. EUROCRYPT 2002.

## Mechanism
1. CBC decryption: P_i = D_K(C_i) XOR C_{i-1}.
2. Oracle exploitation: modify the last byte of C_{i-1}; if the decrypted
   block ends in valid PKCS#7 padding (0x01), then
   D_K(C_i)[last] XOR C'_{i-1}[last] = 0x01, revealing
   D_K(C_i)[last]. Repeat for 0x02 0x02, 0x03 0x03 0x03, ... to decrypt a
   whole block without the key.

## Impact and mitigation
- Affected SSH, IPsec, ASP.NET, POODLE (SSL 3.0, 2014), and Lucky Thirteen
  (TLS, 2013).
- Mandated Authenticated Encryption (AEAD: AES-GCM, ChaCha20-Poly1305).
"""),
    dict(
        source="Goldwasser et al. (1989) - Zero Knowledge",
        title="The Knowledge Complexity of Interactive Proof Systems "
              "(Goldwasser, Micali, Rackoff 1989)",
        url="https://epubs.siam.org/doi/10.1137/0218012",
        type="Cryptography Paper",
        content="""
# The Knowledge Complexity of Interactive Proof Systems
Authors: Shafi Goldwasser, Silvio Micali, Charles Rackoff.
SIAM Journal on Computing 18(1), 1989 (originally STOC 1985).

## Definition
An interactive proof system for a language L is a pair of interactive
machines (P, V): a computationally unbounded prover P and a probabilistic
polynomial-time verifier V. For every x in L, the verifier accepts with
probability at least 1 - 1/|x|^c (completeness); for every x not in L and
every cheating prover P*, the verifier accepts with negligible probability
(soundness).

## Zero knowledge
A zero-knowledge proof is an interactive proof that conveys no knowledge
beyond the validity of the statement: the verifier's view of any interaction
with the honest prover can be simulated by a probabilistic polynomial-time
machine. Formally the distributions of the outputs of the simulator and the
view of a (potentially malicious) verifier are (perfectly, statistically, or
computationally) indistinguishable.

## Variants
- Perfect, statistical and computational zero knowledge depending on the
  indistinguishability notion.
- Honest-verifier zero knowledge (HVZK) if the simulator only needs to fool
  the honest verifier.
- The Fiat-Shamir heuristic transforms public-coin honest-verifier
  zero-knowledge proofs into non-interactive signatures in the random
  oracle model.

## Impact
- Foundation of modern privacy-preserving cryptography (ZK proofs, ZK
  SNARKs/STARKs, anonymous credentials, verifiable computation).
"""),
]


def _to_document(entry: dict) -> Document:
    return Document(
        source=entry["source"],
        title=entry["title"],
        url=entry["url"],
        type=entry["type"],
        content=entry["content"],
    )


LANDMARK_PAPERS = [{"key": e["source"], "document": _to_document(e)}
                   for e in _L]
