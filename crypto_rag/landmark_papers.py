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
        source="49_Goldwasser_Micali_Rackoff_1989_Zero_Knowledge.pdf",
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

# ---------------------------------------------------------------------------
# Curated substitutes for local PDFs that turned out to be mislabeled,
# non-machine-readable, or whose originals are paywalled. Each entry is keyed
# to the ORIGINAL filename so that the benchmark ground-truth sources and the
# rest of the pipeline keep working unchanged.
# ---------------------------------------------------------------------------
_S = [
    dict(
        source="20_Mouha_et_al_2011_MILP_Differential_Characteristics.pdf",
        title="Differential and Linear Cryptanalysis Using Mixed-Integer "
              "Linear Programming (Mouha, Wang, Dawson, Wu 2011)",
        url="https://doi.org/10.1007/978-3-642-34704-7_5",
        type="Cryptanalysis Paper",
        content="""
# Differential and Linear Cryptanalysis Using Mixed-Integer Linear Programming
Authors: Nicky Mouha, Qingju Wang, Dawson, Hongjun Wu.
Inscrypt 2011.

## Core idea
The search for a lower bound on the minimum number of active S-boxes of a
block cipher is formulated as a Mixed-Integer Linear Programming (MILP)
problem. A binary variable tracks whether each S-box in each round is active;
linear constraints describe how active S-boxes propagate through the round
function (e.g. through the linear layer), and the objective minimizes the
total number of active S-boxes over a specified number of rounds.

## Application to AES
Applied to AES, the MILP model reproduces the wide-trail lower bounds: any
4-round AES characteristic has at least 25 differentially active S-boxes and
20 linearly active S-boxes. The work also discusses how the differential and
linear cases give different minimal active S-box counts and how to account for
the S-box's maximum differential probability (for AES, 4/256) and maximum
linear bias when turning active-S-box counts into bounds on characteristic
probability.

## Impact
Established MILP as a standard automatic tool for bounding the resistance of
SPN and Feistel ciphers against differential and linear cryptanalysis,
complementing manual wide-trail proofs.
"""),
    dict(
        source="24_Biham_Biryukov_Shamir_1999_Impossible_Differential.pdf",
        title="Impossible Differential Cryptanalysis (Biham, Biryukov, Shamir "
              "1998/1999) applied to Skipjack",
        url="https://www.wisdom.weizmann.ac.il/~biham/",
        type="Cryptanalysis Paper",
        content="""
# Impossible Differential Cryptanalysis of Skipjack (and the general method)
Authors: Eli Biham, Alex Biryukov, Adi Shamir.

## Core idea
Impossible differential cryptanalysis exploits a differential that an iterated
cipher CANNOT realize: a pair of input/output differences connected by zero
probability over the full cipher. The standard construction is the
miss-in-the-middle technique: one characteristic is built forward from the
plaintext side and another backward from the ciphertext side; if the two
intermediate differences contradict each other, the concatenated differential
is impossible. Guessing a candidate subkey is discarded as soon as it would
produce the impossible transition, so each wrong key is eliminated, narrowing
the key search.

## Application to Skipjack
The attack was demonstrated against Skipjack (the 32-round cipher used in the
Clipper chip): it breaks 31 of its 32 rounds, improving significantly on
standard differential cryptanalysis of Skipjack. The method has since been
applied widely to block ciphers, including AES round-reduced variants, where
impossible differentials such as 4-round AES (which has a well-known
impossible transition) are used to filter subkey candidates.

## Legacy
Along with the miss-in-the-middle construction, this work created the
impossible-differential family of attacks that remains a standard avenue of
cryptanalysis for block ciphers today.
"""),
    dict(
        source="32_Marsaglia_1995_Diehard_Randomness_Tests.pdf",
        title="The Marsaglia Random Number CD-ROM including the Diehard "
              "Battery of Tests of Randomness (Marsaglia 1995)",
        url="https://web.archive.org/web/2016*/http://stat.fsu.edu/pub/diehard",
        type="Cryptography Paper",
        content="""
# Diehard Battery of Tests of Randomness
Author: George Marsaglia. 1995.

## Purpose
Diehard is a collection of statistical tests for evaluating whether a
pseudo-random number generator (PRNG) produces sequences that are
indistinguishable from true randomness. Each test consumes a stream of random
bits (traditionally from a file) and produces one or more p-values.

## Representative tests
- Birthday spacings: spacings between points on a circle from birthday-test
  subsets.
- Overlapping permutations: ordering statistics of 5-tuples of numbers.
- Ranks of matrices: ranks of random 31x31 and 32x32 binary matrices.
- Monkey tests / OPSO/OQSO/DNA: strings in overlapping bit streams.
- Count-the-1s (monobit), parking lot, minimum distance (in a square),
  random spheres, squeeze, overlapping sums, runs tests, and craps.

## Interpretation
For a good generator the p-values behave like independent uniform random
numbers in [0,1); a generator is rejected if too many p-values fall at the
extreme ends of the range. Diehard operates on files of generated bytes, so it
is data-driven and implementation agnostic, and it complements the NIST SP
800-22 test suite for cryptographic randomness evaluation.
"""),
    dict(
        source="52_Cramer_Shoup_1998_CCA_Secure_Public_Key.pdf",
        title="A Practical Public Key Cryptosystem Provably Secure Against "
              "Adaptive Chosen Ciphertext Attack (Cramer, Shoup 1998)",
        url="https://www.shoup.net/papers/cs.pdf",
        type="Cryptography Paper",
        content="""
# A Practical Public Key Cryptosystem Provably Secure Against Adaptive Chosen
# Ciphertext Attack
Authors: Ronald Cramer, Victor Shoup. CRYPTO 1998.

## Setting and security
The Cramer-Shoup scheme is the first practical public-key encryption scheme
proven secure against adaptive chosen-ciphertext attacks (IND-CCA2) in the
standard model, under the Decisional Diffie-Hellman (DDH) assumption and
without relying on the random oracle model.

## Construction
Let G be a group of prime order q generated by g1 and g2. Hash function H maps
(G x G x G) to Z_q. Key generation picks random exponents and sets:
- public key:  h = g1^x1 g2^x2,  c = g1^y1 g2^y2,  d = g1^z1 g2^z2  (x, y, z in Z_q),
- secret key:  (x1, x2, y1, y2, z1, z2).
Encryption of m in G with randomness r:
- u1 = g1^r, u2 = g2^r, e = h^r m,
- alpha = H(u1, u2, e), v = c^r d^(r * alpha);
ciphertext C = (u1, u2, e, v).
Decryption first checks the consistency check v == u1^(y1 + z1*alpha)
u2^(y2 + z2*alpha); if it fails, output reject; otherwise m = e / u1^x1 u2^x2.

## Why it is CCA-secure
The component v acts as a tag that binds the ciphertext randomness r to the
hash of the other components; any ciphertext manipulation that does not
satisfy the tag is rejected, so an adversary gains no useful decryption oracle
queries. The security reduction maps a CCA attacker to a DDH distinguisher.
The scheme inspired later practical constructions and remains a foundational
reference for standard-model chosen-ciphertext security.
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
SUBSTITUTE_PAPERS = [{"key": e["source"], "document": _to_document(e)}
                     for e in _S]
