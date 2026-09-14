FEATURED_CRYPTANALYSIS_PAPERS = [
    {
        "id": "biham_shamir_1991",
        "title": "Differential Cryptanalysis of DES-like Cryptosystems",
        "authors": "Eli Biham & Adi Shamir (1991)",
        "description": "Introduced Differential Cryptanalysis tracking XOR difference propagation through S-boxes. Reduced DES attack to 2^47 chosen plaintexts.",
        "citations": "3500+ Citations",
        "year": 1991
    },
    {
        "id": "matsui_1993",
        "title": "Linear Cryptanalysis Method for DES Cipher",
        "authors": "Mitsuru Matsui (EUROCRYPT 1993)",
        "description": "Introduced Linear Cryptanalysis & Piling-up Lemma. Known-plaintext attack breaking 16-round DES with 2^43 known plaintexts.",
        "citations": "3200+ Citations",
        "year": 1993
    },
    {
        "id": "bleichenbacher_1998",
        "title": "Chosen Ciphertext Attacks Against PKCS #1 (Million Packet Attack)",
        "authors": "Daniel Bleichenbacher (CRYPTO 1998)",
        "description": "Adaptive chosen-ciphertext padding oracle attack against RSA PKCS #1 v1.5 padding in SSL/TLS protocols.",
        "citations": "2100+ Citations",
        "year": 1998
    },
    {
        "id": "coppersmith_1996",
        "title": "Finding Small Roots of Univariate Modular Equations & RSA Attacks",
        "authors": "Don Coppersmith (EUROCRYPT 1996)",
        "description": "LLL lattice reduction methods for finding small roots modulo N. Used to break RSA with low public exponent e=3 or partial key exposure.",
        "citations": "1900+ Citations",
        "year": 1996
    },
    {
        "id": "wang_2005",
        "title": "Collisions in Full MD5 and SHA-0 / SHA-1 Hash Functions",
        "authors": "Xiaoyun Wang, Yiqun Lisa Yin, Hongbo Yu (CRYPTO 2005)",
        "description": "Shattered MD5 & SHA-1 collision resistance using message modification techniques. Reduced MD5 collision to 2^39 operations.",
        "citations": "2800+ Citations",
        "year": 2005
    },
    {
        "id": "kocher_1999",
        "title": "Differential Power Analysis (DPA Side-Channel Attacks)",
        "authors": "Paul Kocher, Joshua Jaffe, Benjamin Jun (CRYPTO 1999)",
        "description": "Physical side-channel attack extracting cryptographic keys by analyzing statistical power consumption traces during chip execution.",
        "citations": "4800+ Citations",
        "year": 1999
    },
    {
        "id": "vaudenay_2002",
        "title": "Security Flaws in CBC Mode & Padding Oracles",
        "authors": "Serge Vaudenay (EUROCRYPT 2002)",
        "description": "CBC-mode padding oracle attack allowing full plaintext recovery by observing error responses. Led to mandatory AEAD ciphers.",
        "citations": "1600+ Citations",
        "year": 2002
    }
]
