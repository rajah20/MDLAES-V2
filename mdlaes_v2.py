"""
mdlaes_ab.py
============
Python implementation of MDLAES-AB:

  Matrix-based Diffusion Layer AES — Abikoye (MDLAES-AB)

Scheme:
  Encryption:
    1. Compute K² = K ⊗ K  in GF(2⁸)      [key-squared matrix]
    2. Compute P′ = P ⊗ K²  in GF(2⁸)     [pre-scramble plaintext]
    3. Encrypt P′ using Abikoye Modified AES with key K → ciphertext C

  Decryption:
    1. Decrypt C using Abikoye Modified AES inverse with key K → P′
    2. Compute K² = K ⊗ K  in GF(2⁸)
    3. Compute (K²)⁻¹ via Gauss-Jordan in GF(2⁸)
    4. Recover P = P′ ⊗ (K²)⁻¹

Depends on abikoye_aes.py (must be in the same directory).

Test vectors (Tables 3 & 4, Abikoye et al. 2019):
    PT   = b"I Love Unilorin!"
    KEY1 = b"dKro9Wahme#dHrn7"
    KEY2 = b"dKro9Wahme#dHsn7"   [bit 112 flipped]
"""

from abikoye_aes import (
    AES_Abikoye, AES_Standard,
    bytes_to_state, state_to_bytes,
    avalanche_effect, flip_bit,
    key_expansion,
)

# ─── GF(2⁸) scalar multiply ──────────────────────────────────────────────────

def gmul(a: int, b: int) -> int:
    """Multiply two bytes in GF(2⁸) using AES irreducible polynomial
    m(x) = x⁸ + x⁴ + x³ + x + 1  (0x11B)."""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xff
        if hi:
            a ^= 0x1b
        b >>= 1
    return p

# ─── GF(2⁸) 4×4 matrix operations ───────────────────────────────────────────

def gf_mat_mul(A, B):
    """4×4 matrix product C = A ⊗ B over GF(2⁸).
    Matrices are list-of-rows:  M[row][col].
    C[row][col] = XOR_k( gmul(A[row][k], B[k][col]) )
    """
    C = [[0]*4 for _ in range(4)]
    for row in range(4):
        for col in range(4):
            v = 0
            for k in range(4):
                v ^= gmul(A[row][k], B[k][col])
            C[row][col] = v
    return C

def gf_mat_inv(M):
    """Compute the inverse of a 4×4 GF(2⁸) matrix using Gauss-Jordan elimination.
    Returns None if the matrix is singular (determinant = 0 in GF(2⁸))."""
    # Augment M with identity
    aug = [[M[r][c] for c in range(4)] + ([1 if c==r else 0 for c in range(4)])
           for r in range(4)]

    for col in range(4):
        # Find pivot
        pivot = None
        for row in range(col, 4):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            return None   # singular
        aug[col], aug[pivot] = aug[pivot], aug[col]

        # Scale pivot row so aug[col][col] = 1
        inv_piv = _gf_inv(aug[col][col])
        aug[col] = [gmul(x, inv_piv) for x in aug[col]]

        # Eliminate column in all other rows
        for row in range(4):
            if row != col and aug[row][col] != 0:
                factor = aug[row][col]
                aug[row] = [aug[row][c] ^ gmul(factor, aug[col][c]) for c in range(8)]

    # Extract right half
    inv = [[aug[r][4+c] for c in range(4)] for r in range(4)]
    return inv

# GF(2⁸) multiplicative inverse via exponentiation (x^254 = x^{-1} for x≠0)
def _gf_inv(x: int) -> int:
    if x == 0:
        raise ZeroDivisionError("GF(2⁸) inverse of 0 is undefined")
    result = 1
    base = x
    exp = 254   # Fermat: x^(256-2) = x^(-1) in GF(2^8)
    while exp:
        if exp & 1:
            result = gmul(result, base)
        base = gmul(base, base)
        exp >>= 1
    return result

def identity_4x4():
    return [[1 if r==c else 0 for c in range(4)] for r in range(4)]

def gf_mat_verify_inverse(M, Minv):
    """Verify M ⊗ Minv == I in GF(2⁸). Returns True if correct."""
    product = gf_mat_mul(M, Minv)
    return product == identity_4x4()

# ─── State ↔ Matrix conversion (AES column-major layout) ─────────────────────
# AES state: state[row][col], bytes stored column-major: byte index = row + 4*col
# We keep matrices in row-major list-of-rows for algebra clarity.

def bytes_to_matrix(b: bytes):
    """Convert 16 bytes to 4×4 matrix in AES column-major order.
    matrix[row][col] = b[row + 4*col]
    This matches the AES state layout used in abikoye_aes.py."""
    return [[b[r + 4*c] for c in range(4)] for r in range(4)]

def matrix_to_bytes(M) -> bytes:
    out = []
    for c in range(4):
        for r in range(4):
            out.append(M[r][c])
    return bytes(out)

# ─── MDLAES-AB ────────────────────────────────────────────────────────────────

class MDLAES_AB:
    """
    MDLAES-AB: GF(2⁸) key-squared matrix pre-diffusion
    combined with Abikoye et al. (2019) Modified AES.

    Encryption (Algorithm 3):
        K²  = GF_MatMul(K_matrix, K_matrix)
        P′  = GF_MatMul(P_matrix, K²)
        C   = Abikoye_Encrypt(P′, K)

    Decryption (Algorithm 4):
        P′  = Abikoye_Decrypt(C, K)
        K²  = GF_MatMul(K_matrix, K_matrix)
        P   = GF_MatMul(P′_matrix, GF_MatInv(K²))
    """

    def __init__(self):
        self._abikoye = AES_Abikoye()

    def _key_matrix(self, key: bytes):
        return bytes_to_matrix(key)

    def _k_squared(self, key: bytes):
        Km = self._key_matrix(key)
        return gf_mat_mul(Km, Km)

    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        assert len(plaintext) == 16 and len(key) == 16

        # Step 1: K² = K ⊗ K
        K2 = self._k_squared(key)

        # Step 2: P′ = P ⊗ K²
        Pm = bytes_to_matrix(plaintext)
        P_prime_m = gf_mat_mul(Pm, K2)
        P_prime = matrix_to_bytes(P_prime_m)

        # Step 3: Abikoye Modified AES on P′ with key K
        return self._abikoye.encrypt(P_prime, key)

    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        assert len(ciphertext) == 16 and len(key) == 16

        # Step 1: Reverse Abikoye Modified AES → P′
        P_prime = self._abikoye.decrypt(ciphertext, key)

        # Step 2: Recompute K²
        K2 = self._k_squared(key)

        # Step 3: (K²)⁻¹
        K2_inv = gf_mat_inv(K2)
        if K2_inv is None:
            raise ValueError("Key matrix K² is singular in GF(2⁸); decryption impossible for this key.")

        # Step 4: P = P′ ⊗ (K²)⁻¹
        P_prime_m = bytes_to_matrix(P_prime)
        Pm = gf_mat_mul(P_prime_m, K2_inv)
        return matrix_to_bytes(Pm)

    def pre_aes_hamming(self, key1: bytes, key2: bytes, plaintext: bytes) -> tuple:
        """Return (hamming_bits, hamming_pct) of P⊗K²₁ vs P⊗K²₂ before AES begins."""
        K2_1 = self._k_squared(key1)
        K2_2 = self._k_squared(key2)
        Pm = bytes_to_matrix(plaintext)
        p1 = matrix_to_bytes(gf_mat_mul(Pm, K2_1))
        p2 = matrix_to_bytes(gf_mat_mul(Pm, K2_2))
        h = sum(bin(a^b).count('1') for a,b in zip(p1,p2))
        return h, h / (len(p1)*8) * 100.0


# ─── Main: reproduce paper results ───────────────────────────────────────────

if __name__ == "__main__":
    import math, time, os

    PT   = b"I Love Unilorin!"
    PT2  = b"I Love Unimorin!"   # bit 88 flipped: 'l' → 'm'
    KEY1 = b"dKro9Wahme#dHrn7"
    KEY2 = b"dKro9Wahme#dHsn7"   # bit 112 flipped

    std = AES_Standard()
    abi = AES_Abikoye()
    mdl = MDLAES_AB()

    print("=" * 70)
    print("  MDLAES-AB — GF(2⁸) Matrix Pre-Diffusion + Abikoye Modified AES")
    print("=" * 70)

    # ── K² intermediate values ────────────────────────────────────────────
    print("\n── Section 5.2: K² Matrix & Pre-AES Diffusion ──")
    Km = bytes_to_matrix(KEY1)
    K2 = mdl._k_squared(KEY1)
    print("  K² (from KEY1):")
    for row in K2:
        print("   ", " ".join(f"{v:02X}" for v in row))

    Pm = bytes_to_matrix(PT)
    P_prime = matrix_to_bytes(gf_mat_mul(Pm, K2))
    print(f"\n  P′ = P ⊗ K²:")
    P_prime_m = bytes_to_matrix(P_prime)
    for row in P_prime_m:
        print("   ", " ".join(f"{v:02X}" for v in row))

    h_bits, h_pct = mdl.pre_aes_hamming(KEY1, KEY2, PT)
    print(f"\n  Pre-AES Hamming distance (bit 112 flip): {h_bits}/128 bits = {h_pct:.1f}%")
    print(f"  (Paper states: 62/128 = 48.4%)")

    # ── Table 3 equivalent: Key-flip avalanche ────────────────────────────
    print("\n── Table 5: Key-Bit-Flip Avalanche Effect (10 positions) ──")
    bit_positions = [112, 8, 24, 40, 56, 72, 88, 100, 116, 120]

    ct_base_std = std.encrypt(PT, KEY1)
    ct_base_abi = abi.encrypt(PT, KEY1)
    ct_base_mdl = mdl.encrypt(PT, KEY1)

    print(f"  {'Bit':>5}  {'Std AES':>10}  {'Abikoye':>10}  {'MDLAES-AB':>10}  {'Δ vs Abi':>10}")
    print("  " + "-"*58)

    results = []
    for bp in bit_positions:
        k2 = flip_bit(KEY1, bp)
        ae_s = avalanche_effect(ct_base_std, std.encrypt(PT, k2))
        ae_a = avalanche_effect(ct_base_abi, abi.encrypt(PT, k2))
        ae_m = avalanche_effect(ct_base_mdl, mdl.encrypt(PT, k2))
        delta = ae_m - ae_a
        results.append((bp, ae_s, ae_a, ae_m))
        print(f"  {bp:>5}  {ae_s:>9.2f}%  {ae_a:>9.2f}%  {ae_m:>9.2f}%  {delta:>+9.2f}pp")

    mean_s = sum(r[1] for r in results) / len(results)
    mean_a = sum(r[2] for r in results) / len(results)
    mean_m = sum(r[3] for r in results) / len(results)
    sd_s = math.sqrt(sum((r[1]-mean_s)**2 for r in results)/len(results))
    sd_a = math.sqrt(sum((r[2]-mean_a)**2 for r in results)/len(results))
    sd_m = math.sqrt(sum((r[3]-mean_m)**2 for r in results)/len(results))
    wins = sum(1 for r in results if r[3] > r[2])

    print("  " + "-"*58)
    print(f"  {'Mean':>5}  {mean_s:>9.2f}%  {mean_a:>9.2f}%  {mean_m:>9.2f}%")
    print(f"  {'σ':>5}  {sd_s:>9.2f}%  {sd_a:>9.2f}%  {sd_m:>9.2f}%")
    print(f"\n  MDLAES-AB wins {wins}/10 positions vs Abikoye")
    print(f"  Improvement vs Abikoye: {mean_m - mean_a:+.2f} pp")
    print(f"  Improvement vs Std AES: {mean_m - mean_s:+.2f} pp")

    # ── Table 4 equivalent: Plaintext-flip avalanche ──────────────────────
    print("\n── Table 6 Summary: Plaintext-Bit-Flip Avalanche Effect ──")
    ct_pt_std1 = std.encrypt(PT,  KEY1)
    ct_pt_std2 = std.encrypt(PT2, KEY1)
    ct_pt_abi1 = abi.encrypt(PT,  KEY1)
    ct_pt_abi2 = abi.encrypt(PT2, KEY1)
    ct_pt_mdl1 = mdl.encrypt(PT,  KEY1)
    ct_pt_mdl2 = mdl.encrypt(PT2, KEY1)

    ae_pt_s = avalanche_effect(ct_pt_std1, ct_pt_std2)
    ae_pt_a = avalanche_effect(ct_pt_abi1, ct_pt_abi2)
    ae_pt_m = avalanche_effect(ct_pt_mdl1, ct_pt_mdl2)

    print(f"  Std AES  PT-flip AE : {ae_pt_s:.4f}%")
    print(f"  Abikoye  PT-flip AE : {ae_pt_a:.4f}%  (paper: 56.25%)")
    print(f"  MDLAES-AB PT-flip AE: {ae_pt_m:.4f}%")
    print(f"  MDLAES-AB gain vs Abikoye: {ae_pt_m - ae_pt_a:+.2f} pp")

    # ── Single-vector bit-112 (paper Table 3 specific) ────────────────────
    print("\n── Exact Bit-112 Comparison (as in Abikoye Table 3) ──")
    ct1_std = std.encrypt(PT, KEY1)
    ct2_std = std.encrypt(PT, KEY2)
    ct1_abi = abi.encrypt(PT, KEY1)
    ct2_abi = abi.encrypt(PT, KEY2)
    ct1_mdl = mdl.encrypt(PT, KEY1)
    ct2_mdl = mdl.encrypt(PT, KEY2)

    print(f"  MDLAES-AB CT (KEY1): {ct1_mdl.hex().upper()}")
    print(f"  MDLAES-AB CT (KEY2): {ct2_mdl.hex().upper()}")
    print(f"  Std AES  AE: {avalanche_effect(ct1_std, ct2_std):.4f}%")
    print(f"  Abikoye  AE: {avalanche_effect(ct1_abi, ct2_abi):.4f}%  (paper: 57.81%)")
    print(f"  MDLAES-AB AE: {avalanche_effect(ct1_mdl, ct2_mdl):.4f}%")

    # ── Decryption correctness ────────────────────────────────────────────
    print("\n── Decryption Correctness ──")
    dec = mdl.decrypt(ct1_mdl, KEY1)
    print(f"  Recovered PT: {dec} — {'OK ✓' if dec == PT else 'FAIL ✗'}")

    # ── K² invertibility ─────────────────────────────────────────────────
    K2 = mdl._k_squared(KEY1)
    K2_inv = gf_mat_inv(K2)
    ok = gf_mat_verify_inverse(K2, K2_inv)
    print(f"  K² invertible for KEY1: {'Yes ✓' if ok else 'No ✗'}")

    K2b = mdl._k_squared(KEY2)
    K2b_inv = gf_mat_inv(K2b)
    ok2 = gf_mat_verify_inverse(K2b, K2b_inv)
    print(f"  K² invertible for KEY2: {'Yes ✓' if ok2 else 'No ✗'}")

    # ── Table 6 Summary ───────────────────────────────────────────────────
    print("\n── Table 6: Extended Summary (paper Table 6 format) ──")
    print(f"  {'Metric':<45}  {'Std AES':>8}  {'Abikoye':>8}  {'MDLAES-AB':>10}")
    print("  " + "-"*80)
    bp112_s = avalanche_effect(ct1_std, ct2_std)
    bp112_a = avalanche_effect(ct1_abi, ct2_abi)
    bp112_m = avalanche_effect(ct1_mdl, ct2_mdl)
    print(f"  {'Key-flip AE — bit 112 (Table 3)':<45}  {bp112_s:>7.2f}%  {bp112_a:>7.2f}%  {bp112_m:>9.2f}%")
    print(f"  {'PT-flip AE — bit 88 (Table 4)':<45}  {ae_pt_s:>7.2f}%  {ae_pt_a:>7.2f}%  {ae_pt_m:>9.2f}%")
    print(f"  {'Mean key-flip AE (10 positions)':<45}  {mean_s:>7.2f}%  {mean_a:>7.2f}%  {mean_m:>9.2f}%")
    print(f"  {'Standard deviation σ':<45}  {sd_s:>7.2f}%  {sd_a:>7.2f}%  {sd_m:>9.2f}%")
    print(f"  {'Positions won vs Abikoye (of 10)':<45}  {'—':>8}  {'—':>8}  {wins:>4}/10     ")

    # ── Execution time ────────────────────────────────────────────────────
    print("\n── Table 7: Execution Time (5-run average) ──")
    sizes = [16, 32, 64, 128]
    RUNS  = 5
    print(f"  {'Size':>6}  {'Std AES (ms)':>14}  {'Abikoye (ms)':>14}  {'MDLAES-AB (ms)':>16}  {'Overhead vs Abi':>18}")
    print("  " + "-"*76)
    for sz in sizes:
        data   = os.urandom(sz)
        blocks = [data[i:i+16] for i in range(0, sz, 16)]

        t0 = time.perf_counter()
        for _ in range(RUNS):
            for blk in blocks: std.encrypt(blk, KEY1)
        t_s = (time.perf_counter()-t0)/RUNS*1000

        t0 = time.perf_counter()
        for _ in range(RUNS):
            for blk in blocks: abi.encrypt(blk, KEY1)
        t_a = (time.perf_counter()-t0)/RUNS*1000

        t0 = time.perf_counter()
        for _ in range(RUNS):
            for blk in blocks: mdl.encrypt(blk, KEY1)
        t_m = (time.perf_counter()-t0)/RUNS*1000

        overhead = (t_m - t_a) / t_a * 100
        print(f"  {sz:>4}B  {t_s:>13.3f}  {t_a:>13.3f}  {t_m:>15.3f}  {overhead:>+16.1f}%")

    # ── State-of-the-art table ────────────────────────────────────────────
    print("\n── Table 8: State-of-the-Art Comparison (paper Table 9 format) ──")
    print(f"  {'No.':<4}  {'Author / Scheme':<38}  {'Conv. AES%':>10}  {'Mod AES%':>10}  {'Diff(pp)':>10}")
    print("  " + "-"*80)
    sota = [
        ("1", "Abikoye et al. (2019) [11]",         49.973, 56.363, 6.39),
        ("2", "Al-Mamun et al. (2017) [12]",         50.78,  52.34,  1.56),
        ("3", "Hamad & Khalaf (2019) [13]",           49.80,  53.12,  3.32),
        ("4", "Hafsa et al. (2021) [15]",              49.51,  53.90,  4.39),
        ("5", "MDLAES-AB (This work)*",               mean_s, mean_m, mean_m-mean_s),
    ]
    for no, auth, c, m, d in sota:
        print(f"  {no:<4}  {auth:<38}  {c:>9.3f}%  {m:>9.3f}%  {d:>+9.2f}pp")
    print("  * Mean across 10 key-bit-flip positions")

    print("\nDone.")
