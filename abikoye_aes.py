"""
abikoye_aes.py
==============
Faithful Python implementation of the Modified AES algorithm described in:

  Abikoye, O.C., Haruna, A.D., Abubakar, A., Akande, N.O., Asani, E.O.
  "Modified Advanced Encryption Standard Algorithm for Information Security"
  Symmetry 2019, 11, 1484.  doi:10.3390/sym11121484

Modifications over standard AES-128:
  1. Modified SubBytes  (Eq. 2–8):  round-key-dependent, per-row XOR before S-Box lookup.
  2. Modified ShiftRows (Eq. 16–26): rank-value-based, randomised row shifts.
  3. Modified InvSubBytes / Modified InvShiftRows for exact decryption recovery.

All other transformations (MixColumns, AddRoundKey, KeyExpansion) are standard AES-128.

Usage
-----
    from abikoye_aes import AES_Abikoye
    aes = AES_Abikoye()
    ct  = aes.encrypt(plaintext_bytes, key_bytes)   # bytes, len 16
    pt  = aes.decrypt(ct, key_bytes)

Test vectors (from Tables 3 & 4 of the paper):
    PT   = b"I Love Unilorin!"
    KEY1 = b"dKro9Wahme#dHrn7"
    KEY2 = b"dKro9Wahme#dHsn7"   # bit 112 flipped
"""

# ─── AES constants ────────────────────────────────────────────────────────────

SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]

INV_SBOX = [0]*256
for i,v in enumerate(SBOX):
    INV_SBOX[v] = i

RCON = [0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36]

def _xtime(a):
    return ((a << 1) ^ 0x1b) & 0xff if (a & 0x80) else (a << 1) & 0xff

def _gmul(a, b):
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

# ─── State helpers (column-major: index = row + 4*col) ───────────────────────

def _bytes_to_state(b):
    return [b[r + 4*c] for c in range(4) for r in range(4)]
    # Actually AES column-major: state[row + 4*col]
    # but the loop above gives state[col*4 + row] — let's be explicit:

def bytes_to_state(b):
    """Convert 16-byte sequence to 4x4 state list, column-major (AES convention).
    state[row][col]"""
    s = [[0]*4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            s[r][c] = b[r + 4*c]
    return s

def state_to_bytes(s):
    out = []
    for c in range(4):
        for r in range(4):
            out.append(s[r][c])
    return bytes(out)

# ─── Standard AES transformations ────────────────────────────────────────────

def key_expansion(key):
    """Standard AES-128 key schedule. Returns list of 11 round keys,
    each a 4x4 state (list of lists)."""
    w = [list(key[4*i:4*i+4]) for i in range(4)]
    for i in range(4, 44):
        temp = w[i-1][:]
        if i % 4 == 0:
            temp = temp[1:] + temp[:1]
            temp = [SBOX[b] for b in temp]
            temp[0] ^= RCON[i//4]
        w.append([a ^ b for a, b in zip(w[i-4], temp)])
    # Package into 11 round keys of shape state[row][col]
    round_keys = []
    for rnd in range(11):
        rk = [[0]*4 for _ in range(4)]
        for c in range(4):
            for r in range(4):
                rk[r][c] = w[rnd*4 + c][r]
        round_keys.append(rk)
    return round_keys

def add_round_key(state, rk):
    s = [[state[r][c] ^ rk[r][c] for c in range(4)] for r in range(4)]
    return s

def mix_columns(state):
    s = [[0]*4 for _ in range(4)]
    for c in range(4):
        s[0][c] = _gmul(0x02, state[0][c]) ^ _gmul(0x03, state[1][c]) ^ state[2][c] ^ state[3][c]
        s[1][c] = state[0][c] ^ _gmul(0x02, state[1][c]) ^ _gmul(0x03, state[2][c]) ^ state[3][c]
        s[2][c] = state[0][c] ^ state[1][c] ^ _gmul(0x02, state[2][c]) ^ _gmul(0x03, state[3][c])
        s[3][c] = _gmul(0x03, state[0][c]) ^ state[1][c] ^ state[2][c] ^ _gmul(0x02, state[3][c])
    return s

def inv_mix_columns(state):
    s = [[0]*4 for _ in range(4)]
    for c in range(4):
        s[0][c] = (_gmul(0x0e,state[0][c]) ^ _gmul(0x0b,state[1][c]) ^
                   _gmul(0x0d,state[2][c]) ^ _gmul(0x09,state[3][c]))
        s[1][c] = (_gmul(0x09,state[0][c]) ^ _gmul(0x0e,state[1][c]) ^
                   _gmul(0x0b,state[2][c]) ^ _gmul(0x0d,state[3][c]))
        s[2][c] = (_gmul(0x0d,state[0][c]) ^ _gmul(0x09,state[1][c]) ^
                   _gmul(0x0e,state[2][c]) ^ _gmul(0x0b,state[3][c]))
        s[3][c] = (_gmul(0x0b,state[0][c]) ^ _gmul(0x0d,state[1][c]) ^
                   _gmul(0x09,state[2][c]) ^ _gmul(0x0e,state[3][c]))
    return s

# ─── Standard SubBytes / ShiftRows (for reference) ───────────────────────────

def std_sub_bytes(state):
    return [[SBOX[state[r][c]] for c in range(4)] for r in range(4)]

def std_inv_sub_bytes(state):
    return [[INV_SBOX[state[r][c]] for c in range(4)] for r in range(4)]

def std_shift_rows(state):
    s = [row[:] for row in state]
    for r in range(1, 4):
        s[r] = s[r][r:] + s[r][:r]
    return s

def std_inv_shift_rows(state):
    s = [row[:] for row in state]
    for r in range(1, 4):
        s[r] = s[r][4-r:] + s[r][:4-r]
    return s

# ─── Abikoye Modified SubBytes (Eq. 2–8) ─────────────────────────────────────

def _xorkeys(rk):
    """Compute XORkey_i = K[i,0] ^ K[i,1] ^ K[i,2] ^ K[i,3] for i in 0..3."""
    return [rk[i][0] ^ rk[i][1] ^ rk[i][2] ^ rk[i][3] for i in range(4)]

def modified_sub_bytes(state, rk):
    """Eq. 7-8: S'[i,j] = SBox[ S[i,j] XOR XORkey_i ]"""
    xk = _xorkeys(rk)
    return [[SBOX[state[r][c] ^ xk[r]] for c in range(4)] for r in range(4)]

def modified_inv_sub_bytes(state, rk):
    """Eq. 14-15: first InvSBox, then XOR with XORkey_i."""
    xk = _xorkeys(rk)
    return [[INV_SBOX[state[r][c]] ^ xk[r] for c in range(4)] for r in range(4)]

# ─── Abikoye Modified ShiftRows (Eq. 16–26) ──────────────────────────────────

def _rank_order(state, rk):
    """Return list of (row_index, shift_amount) sorted by RVal ascending.
    RVal_i = XOR of (S[i,j] ^ K[i,j]) for j in 0..3.
    Rank Number = sort position (1-based), shift = RNo - 1."""
    rvals = []
    for i in range(4):
        sk = [state[i][j] ^ rk[i][j] for j in range(4)]
        rv = sk[0] ^ sk[1] ^ sk[2] ^ sk[3]
        rvals.append((rv, i))
    rvals_sorted = sorted(rvals, key=lambda x: x[0])   # ascending by RVal
    # RNo 1 → shift 0, RNo 2 → shift 1, RNo 3 → shift 2, RNo 4 → shift 3
    shifts = {}
    for rno_minus1, (rv, row_idx) in enumerate(rvals_sorted):
        shifts[row_idx] = rno_minus1   # shift = RNo - 1
    return shifts

def modified_shift_rows(state, rk):
    """Shift row i left by shifts[i] positions, where shifts come from rank values."""
    shifts = _rank_order(state, rk)
    s = []
    for r in range(4):
        sh = shifts[r]
        row = state[r]
        s.append(row[sh:] + row[:sh])
    return s

def modified_inv_shift_rows(state, rk):
    """Shift row i RIGHT by shifts[i] positions (same rank computation)."""
    shifts = _rank_order(state, rk)
    s = []
    for r in range(4):
        sh = shifts[r]
        row = state[r]
        s.append(row[4-sh:] + row[:4-sh])
    return s

# ─── Abikoye Modified AES-128 ────────────────────────────────────────────────

class AES_Abikoye:
    """
    Modified AES-128 per Abikoye et al. (2019).
    Only SubBytes and ShiftRows are modified; MixColumns and AddRoundKey are standard.
    """

    NR = 10  # AES-128 rounds

    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        assert len(plaintext) == 16 and len(key) == 16
        rks = key_expansion(key)
        state = bytes_to_state(plaintext)

        # Pre-round AddRoundKey
        state = add_round_key(state, rks[0])

        # Rounds 1..NR-1
        for rnd in range(1, self.NR):
            state = modified_sub_bytes(state, rks[rnd])
            state = modified_shift_rows(state, rks[rnd])
            state = mix_columns(state)
            state = add_round_key(state, rks[rnd])

        # Final round (no MixColumns)
        state = modified_sub_bytes(state, rks[self.NR])
        state = modified_shift_rows(state, rks[self.NR])
        state = add_round_key(state, rks[self.NR])

        return state_to_bytes(state)

    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        assert len(ciphertext) == 16 and len(key) == 16
        rks = key_expansion(key)
        state = bytes_to_state(ciphertext)

        # Undo final round
        state = add_round_key(state, rks[self.NR])
        state = modified_inv_shift_rows(state, rks[self.NR])
        state = modified_inv_sub_bytes(state, rks[self.NR])

        # Rounds NR-1 downto 1
        for rnd in range(self.NR - 1, 0, -1):
            state = add_round_key(state, rks[rnd])
            state = inv_mix_columns(state)
            state = modified_inv_shift_rows(state, rks[rnd])
            state = modified_inv_sub_bytes(state, rks[rnd])

        # Undo pre-round AddRoundKey
        state = add_round_key(state, rks[0])

        return state_to_bytes(state)


# ─── Standard AES-128 (for comparison baseline) ──────────────────────────────

class AES_Standard:
    NR = 10

    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        assert len(plaintext) == 16 and len(key) == 16
        rks = key_expansion(key)
        state = bytes_to_state(plaintext)
        state = add_round_key(state, rks[0])
        for rnd in range(1, self.NR):
            state = std_sub_bytes(state)
            state = std_shift_rows(state)
            state = mix_columns(state)
            state = add_round_key(state, rks[rnd])
        state = std_sub_bytes(state)
        state = std_shift_rows(state)
        state = add_round_key(state, rks[self.NR])
        return state_to_bytes(state)

    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        assert len(ciphertext) == 16 and len(key) == 16
        rks = key_expansion(key)
        state = bytes_to_state(ciphertext)
        state = add_round_key(state, rks[self.NR])
        state = std_inv_shift_rows(state)
        state = std_inv_sub_bytes(state)
        for rnd in range(self.NR - 1, 0, -1):
            state = add_round_key(state, rks[rnd])
            state = inv_mix_columns(state)
            state = std_inv_shift_rows(state)
            state = std_inv_sub_bytes(state)
        state = add_round_key(state, rks[0])
        return state_to_bytes(state)


# ─── Avalanche effect helper ──────────────────────────────────────────────────

def hamming_distance(b1: bytes, b2: bytes) -> int:
    """Count differing bits between two equal-length byte strings."""
    diff = 0
    for x, y in zip(b1, b2):
        v = x ^ y
        while v:
            diff += v & 1
            v >>= 1
    return diff

def avalanche_effect(b1: bytes, b2: bytes) -> float:
    """Avalanche effect as percentage (Eq. 27 of Abikoye et al.)."""
    h = hamming_distance(b1, b2)
    n = len(b1) * 8
    return (h / n) * 100.0

def flip_bit(data: bytes, bit_pos: int) -> bytes:
    """Flip bit at position bit_pos (0 = MSB of byte 0)."""
    ba = bytearray(data)
    byte_idx = bit_pos // 8
    bit_idx  = 7 - (bit_pos % 8)   # MSB-first within byte
    ba[byte_idx] ^= (1 << bit_idx)
    return bytes(ba)


# ─── Reproduce paper test vectors ────────────────────────────────────────────

if __name__ == "__main__":
    PT   = b"I Love Unilorin!"
    KEY1 = b"dKro9Wahme#dHrn7"
    KEY2 = b"dKro9Wahme#dHsn7"   # bit 112 flipped (r→s in position 13)

    std = AES_Standard()
    mod = AES_Abikoye()

    print("=" * 70)
    print("  Abikoye et al. (2019) — Modified AES — Python Implementation")
    print("=" * 70)

    # ── Table 3: Key-flip avalanche ───────────────────────────────────────
    print("\n── Table 3 Reproduction: Key-bit-flip Avalanche Effect ──")
    ct1_std = std.encrypt(PT, KEY1)
    ct2_std = std.encrypt(PT, KEY2)
    ct1_mod = mod.encrypt(PT, KEY1)
    ct2_mod = mod.encrypt(PT, KEY2)

    ae_std = avalanche_effect(ct1_std, ct2_std)
    ae_mod = avalanche_effect(ct1_mod, ct2_mod)

    print(f"  Plaintext      : {PT.decode()!r}")
    print(f"  KEY1 (hex)     : {KEY1.hex().upper()}")
    print(f"  KEY2 (hex)     : {KEY2.hex().upper()}  [bit 112 flipped]")
    print()
    print(f"  Std AES CT1    : {ct1_std.hex().upper()}")
    print(f"  Std AES CT2    : {ct2_std.hex().upper()}")
    print(f"  Std AES AE     : {ae_std:.4f}%   (paper: 50.78%)")
    print()
    print(f"  Mod AES CT1    : {ct1_mod.hex().upper()}")
    print(f"  Mod AES CT2    : {ct2_mod.hex().upper()}")
    print(f"  Mod AES AE     : {ae_mod:.4f}%   (paper: 57.81%)")

    # ── Table 4: Plaintext-flip avalanche ─────────────────────────────────
    print("\n── Table 4 Reproduction: Plaintext-bit-flip Avalanche Effect ──")
    PT2 = b"I Love Unimorin!"   # bit 88: 'l' -> 'm'
    ct_pt1_std = std.encrypt(PT,  KEY1)
    ct_pt2_std = std.encrypt(PT2, KEY1)
    ct_pt1_mod = mod.encrypt(PT,  KEY1)
    ct_pt2_mod = mod.encrypt(PT2, KEY1)

    ae_pt_std = avalanche_effect(ct_pt1_std, ct_pt2_std)
    ae_pt_mod = avalanche_effect(ct_pt1_mod, ct_pt2_mod)

    print(f"  PT1            : {PT.decode()!r}    hex: {PT.hex().upper()}")
    print(f"  PT2            : {PT2.decode()!r}  hex: {PT2.hex().upper()}")
    print(f"  Key            : {KEY1.decode()!r}")
    print()
    print(f"  Std AES AE     : {ae_pt_std:.4f}%   (paper: 49.21%)")
    print(f"  Mod AES AE     : {ae_pt_mod:.4f}%   (paper: 56.25%)")

    # ── Decryption correctness ────────────────────────────────────────────
    print("\n── Decryption Correctness ──")
    dec_std = std.decrypt(ct1_std, KEY1)
    dec_mod = mod.decrypt(ct1_mod, KEY1)
    print(f"  Std AES decrypt: {dec_std} — {'OK' if dec_std == PT else 'FAIL'}")
    print(f"  Mod AES decrypt: {dec_mod} — {'OK' if dec_mod == PT else 'FAIL'}")

    # ── Multi-position key-flip table ─────────────────────────────────────
    print("\n── Multi-Position Key-Flip Avalanche (10 positions) ──")
    bit_positions = [112, 8, 24, 40, 56, 72, 88, 100, 116, 120]
    ct_base_std = std.encrypt(PT, KEY1)
    ct_base_mod = mod.encrypt(PT, KEY1)

    results = []
    for bp in bit_positions:
        k2 = flip_bit(KEY1, bp)
        ct_f_std = std.encrypt(PT, k2)
        ct_f_mod = mod.encrypt(PT, k2)
        ae_s = avalanche_effect(ct_base_std, ct_f_std)
        ae_m = avalanche_effect(ct_base_mod, ct_f_mod)
        results.append((bp, ae_s, ae_m))
        delta = ae_m - ae_s
        print(f"  Bit {bp:3d}: Std={ae_s:6.2f}%  Mod={ae_m:6.2f}%  Δ={delta:+6.2f}%")

    mean_std = sum(r[1] for r in results) / len(results)
    mean_mod = sum(r[2] for r in results) / len(results)
    import math
    sd_std = math.sqrt(sum((r[1]-mean_std)**2 for r in results)/len(results))
    sd_mod = math.sqrt(sum((r[2]-mean_mod)**2 for r in results)/len(results))
    wins = sum(1 for r in results if r[2] > r[1])
    print(f"\n  Mean Std AES AE : {mean_std:.2f}%  σ={sd_std:.2f}%")
    print(f"  Mean Mod AES AE : {mean_mod:.2f}%  σ={sd_mod:.2f}%")
    print(f"  Mod wins        : {wins} / {len(results)}")

    # ── Execution time ────────────────────────────────────────────────────
    import time, os
    print("\n── Execution Time (5-run average) ──")
    sizes = [16, 32, 64, 128]
    RUNS = 5
    for sz in sizes:
        data = os.urandom(sz)
        blocks = [data[i:i+16] for i in range(0, sz, 16)]
        # Standard AES
        t0 = time.perf_counter()
        for _ in range(RUNS):
            for blk in blocks:
                std.encrypt(blk, KEY1)
        t_std = (time.perf_counter() - t0) / RUNS * 1000
        # Modified AES
        t0 = time.perf_counter()
        for _ in range(RUNS):
            for blk in blocks:
                mod.encrypt(blk, KEY1)
        t_mod = (time.perf_counter() - t0) / RUNS * 1000
        print(f"  {sz:4d}B: Std={t_std:.3f}ms  Mod={t_mod:.3f}ms  overhead={((t_mod-t_std)/t_std*100):+.1f}%")

    print("\nDone.")
