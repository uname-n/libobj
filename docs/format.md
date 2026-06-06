# libobj on-disk format specification

This document specifies the byte-level on-disk format of a libobj database.
It is the **stable contract** of the project: the `obj-core` crate is an
unstable implementation detail with no SemVer guarantee, but the format
described here is frozen for the entire v1.x series. A file written by any
v1.x build opens in any other v1.x build, and a future v2.0 build will reject
v1 files rather than silently misinterpret them.

- **Format version:** `format_major = 1`, `format_minor = 2` (the v1.0 frozen,
  feature-complete baseline).
- **Reader compatibility:** v1.x readers also accept pre-1.0 (`format_major =
  0`) files, read-only; v1.x writers never produce `format_major = 0`.
- **Endianness:** every multi-byte integer on disk is **little-endian**.
- **Checksum:** CRC32C (Castagnoli polynomial) everywhere a checksum appears.

A database is a single file. Optional sidecar files share its path with a
suffix: `<db>-wal` (write-ahead log). The main file is an array of fixed-size
pages; page 0 is the file header, pages 1..N hold data.

---

## 1. Pages and physical layout

| Constant | Value | Notes |
|----------|-------|-------|
| Page size | 4096 bytes | Fixed for `format_major ∈ {0, 1}`. |
| Page trailer | 4 bytes | Per-page CRC32C, last 4 bytes of each non-header page. |
| Encryption overhead | 40 bytes | 24-byte nonce + 16-byte Poly1305 tag. |

**Page 0** is always exactly 4096 bytes and always plaintext — it carries the
signal a reader needs to interpret everything else.

**Pages 1..N** have a physical *stride* that depends on `feature_flags`:

- Unencrypted: stride = 4096. Page `k` begins at byte offset `k * 4096`.
- Encrypted (feature bit 1 set): stride = 4136 (4096 ciphertext + 24 nonce +
  16 tag). Page `k` (k ≥ 1) begins at byte offset `4096 + (k - 1) * 4136`.

Page ids are `NonZeroU64`; the value `0` is the reserved "no page" sentinel
(used by the freelist and the catalog root pointer).

---

## 2. Page 0 — file header

The first 4096-byte page. Magic `OBJF` (ASCII). All fields little-endian.

| Offset | Size | Field | Description |
|-------:|-----:|-------|-------------|
| 0   | 4  | `magic`          | ASCII `OBJF` (`4F 42 4A 46`). |
| 4   | 2  | `format_major`   | `1` for v1.x; reader also accepts `0`. |
| 6   | 2  | `format_minor`   | `2` for `format_major = 1`. |
| 8   | 2  | `page_size`      | `4096`. |
| 10  | 4  | `feature_flags`  | Bit 0 = LZ4 compression, bit 1 = encryption. |
| 14  | 2  | `reserved`       | Must be zero (readers reject non-zero). |
| 16  | 8  | `page_count`     | Total pages in the file, including page 0. |
| 24  | 8  | `root_catalog`   | Catalog root page-id, or `0` if empty. |
| 32  | 8  | `freelist_head`  | First freelist page-id, or `0` if empty. |
| 40  | 16 | `wal_salt`       | Salt tying the file to its WAL generation. |
| 56  | 16 | `file_uuid`      | Stable per-file UUID. |
| 72  | 32 | `kdf_salt`       | HKDF-SHA256 salt; zero unless encryption is on. |
| 104 | …  | `reserved`       | Zero-filled up to the trailer. |
| 4092 | 4 | `header_crc32c`  | CRC32C of bytes `[0 .. 4092]`. |

`header_crc32c` (`PAGE_SIZE - 4`) covers the whole page except itself. On
decode, a mismatch is reported as corruption of page 0.

### Feature flags

| Bit | Meaning |
|----:|---------|
| 0 | Every non-header page is LZ4-compressed (signalled per page in the trailer). |
| 1 | Every non-header page is encrypted with XChaCha20-Poly1305. |

Any bit set on disk that this build does not understand is rejected with an
"unknown feature flag" error: a reader that cannot interpret a flag MUST refuse
to guess. Compression and encryption compose, in that order: **compress first,
encrypt second**.

---

## 3. Page trailer (non-header pages)

The last 4 bytes of every non-header page form the trailer. Two
interpretations exist, selected by `format_minor`:

- **v0 trailer** (`format_minor = 0`): the full 32-bit CRC32C of the page's
  preceding `4092` bytes.
- **v1 trailer** (`format_minor ≥ 1`, compression-capable): **bit 31** is the
  per-page "this page is LZ4-compressed" flag; **bits 0..30** are a 31-bit
  CRC32C of the on-disk page bytes.

For encrypted files, the trailer (and the rest of the logical 4096-byte page)
is part of the plaintext that is compressed/encrypted; the 40-byte nonce+tag
suffix is appended *after* the 4096-byte ciphertext on disk.

---

## 4. Encryption

When feature bit 1 is set:

- **Cipher:** XChaCha20-Poly1305 (AEAD), per non-header page.
- **Per-file key:** `HKDF-SHA256(ikm = user_key, salt = kdf_salt, info =
  "obj-page-encryption-v1")`, where `kdf_salt` is the 32-byte page-0 field.
- **Associated data:** the page's `page_id` (the `kdf_salt` is *not* bound into
  AD — its integrity rests on the page-0 header CRC plus the fact that
  tampering changes the derived key and surfaces as a wrong-key error on first
  decrypt, never as silent plaintext disclosure).
- **Physical stride:** 4136 bytes per page (4096 ciphertext + 24 nonce + 16
  tag). Page 0 is never encrypted.

---

## 5. Write-ahead log (`<db>-wal`)

A sidecar that makes commits crash-atomic. Magic `OBJW` (ASCII).

### WAL header (64 bytes)

| Offset | Size | Field | Description |
|-------:|-----:|-------|-------------|
| 0  | 4 | `magic`        | ASCII `OBJW`. |
| 4  | 2 | `format_major` | Validated against the main file's supported majors. |
| 6  | 2 | `format_minor` | Same supported-minor rule as page 0. |
| 8  | 2 | `page_size`    | `4096`. |
| 12 | 4 | `salt`         | WAL generation salt; rotated on each checkpoint. |

### WAL frame

Each frame is a 64-byte frame header followed by a 4096-byte page body.

- Plaintext frame size: 4160 bytes (`64 + 4096`).
- Encrypted frame size: 4200 bytes (`4160 + 24 nonce + 16 tag`, appended after
  the body).

Frame header layout:

| Offset | Size | Field | Description |
|-------:|-----:|-------|-------------|
| 0  | 8 | `page_id` | Page this frame replaces. |
| 8  | 8 | `lsn`     | Monotonic per-generation log sequence number. |
| 16 | 4 | `salt`    | Must match the WAL header's `salt`. |
| 20 | 1 | `flags`   | Bit 0 = commit marker (last frame of a transaction). |
| 60 | 4 | `crc32c`  | CRC32C of the frame header (CRC field zeroed) ++ body. |

For encrypted WALs the CRC is computed over the **plaintext** body; encryption
is applied after the CRC is stamped.

### Recovery

Recovery is salt-driven and two-pass:

1. If the file is shorter than the header, or the header salt does not match
   the main file's `wal_salt`, the WAL is treated as empty (a stale generation
   from a completed checkpoint).
2. **Pass 1** scans forward to find the last frame whose salt matches and whose
   `commit` flag is set — the durable tail.
3. **Pass 2** re-walks frames up to that commit. Any salt-matching frame with a
   bad CRC in this range is genuine corruption (a hard error); a bad CRC or
   salt mismatch *past* the last commit is a torn tail and is discarded.

The generation salt rotates on every successful checkpoint, so leftover frames
from a previous generation are unambiguously distinguished from live ones
without zeroing the file.

---

## 6. Document records

Every stored document is the value in its collection's primary B+tree: a
16-byte record header immediately followed by a [postcard](https://postcard.jamesmunns.com/)-encoded
payload.

| Offset | Size | Field | Description |
|-------:|-----:|-------|-------------|
| 0  | 4 | `collection_id`   | Pins the record to its collection. |
| 4  | 4 | `type_version`    | Schema/type version for migration. |
| 8  | 4 | `payload_len`     | Length of the postcard payload that follows. |
| 12 | 4 | `payload_crc32c`  | CRC32C of the payload bytes alone. |

The B+tree leaf's own page trailer protects the record's surroundings;
`payload_crc32c` lets a forensic tool verify a single record in isolation.

**Document ids** are per-collection monotonic `NonZeroU64` values. Each
collection's catalog row carries its own `next_id` watermark, so collections
allocate ids independently. The serde representation of an id is its inner
`NonZeroU64`, so ids round-trip through postcard inside user document types.

---

## 7. Versioning and compatibility policy

| `format_major` | `format_minor` | Status |
|---------------:|---------------:|--------|
| 0 | 0, 1, 2 | Pre-1.0 files. v1.x readers accept them read-compatibly. |
| 1 | 2       | The v1.0 frozen wire format. The only valid minor for major 1. |

The format is frozen for the v1.x series: no minor bumps without a `major = 2`
release. A v2.0 build will reject `format_major ∈ {0, 1}` outright rather than
risk misreading them.
