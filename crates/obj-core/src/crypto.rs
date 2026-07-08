//! XChaCha20-Poly1305 page-level AEAD + HKDF-SHA256
//! key derivation. Gated on the `encryption` Cargo feature; without
//! that feature this module does not compile and no symbol from it is
//! reachable.
//!
//! # Layout summary
//!
//! - **Logical page**: 4096 bytes (body + trailer). Encoders never
//!   see encryption — compression sits *below* this layer.
//! - **Physical page** in an encrypted file: `4096 + 24 + 16 = 4136`
//!   bytes = `ciphertext || nonce || tag`.
//! - **Nonce**: 24 random bytes (`XChaCha20`'s extended 192-bit nonce),
//!   generated freshly on every page write from the injected
//!   [`Entropy`] source. No nonce-counter
//!   persistence is required. The 192-bit
//!   width removes the birthday-bound rewrite ceiling that a 96-bit
//!   random nonce would impose under a single per-file key.
//! - **Tag**: 16-byte Poly1305 authentication tag.
//! - **Associated data (AD)**: `page_id.to_le_bytes()` (8 bytes).
//!   Binds the ciphertext to its on-disk slot — an attacker cannot
//!   swap an encrypted page from one slot to another without
//!   detection.
//! - **Key derivation**: HKDF-SHA256, `info = b"obj-page-encryption-v1"`.
//!   The trailing `-v1` is the versioning hook for any future KDF
//!   migration.

#![cfg(feature = "encryption")]
#![forbid(unsafe_code)]

use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{Key, Tag, XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use sha2::Sha256;

use crate::error::{Error, Result};
use crate::platform::env::Entropy;

/// Logical page size in bytes (4 KiB).
pub const LOGICAL_PAGE_SIZE: usize = 4096;
/// AEAD nonce size, in bytes. XChaCha20-Poly1305 uses a 24-byte
/// (192-bit) extended nonce.
pub const NONCE_SIZE: usize = 24;
/// AEAD authentication tag size, in bytes. Poly1305 is 16 bytes.
pub const TAG_SIZE: usize = 16;
/// Per-page on-disk overhead added by the encryption layer.
pub const ENCRYPTION_OVERHEAD: usize = NONCE_SIZE + TAG_SIZE;
/// Physical page size on disk for an encrypted page = ciphertext +
/// nonce + tag.
pub const PHYSICAL_ENCRYPTED_PAGE_SIZE: usize = LOGICAL_PAGE_SIZE + ENCRYPTION_OVERHEAD;
/// Length of the HKDF salt stored plaintext in the page-0 header.
pub const KDF_SALT_SIZE: usize = 32;
/// Length of the user-supplied master key and of the derived per-file
/// page key (both 32 bytes — `XChaCha20` takes a 32-byte key).
pub const KEY_SIZE: usize = 32;

/// HKDF `info` string. The trailing `-v1` is the versioning hook: a
/// future KDF change can bump this constant and derive an entirely
/// fresh key space from the same `(user_key, kdf_salt)` pair without
/// disturbing the on-disk format spec.
pub const HKDF_INFO: &[u8] = b"obj-page-encryption-v1";

/// Derive a 32-byte per-file page-encryption key from the caller's
/// 32-byte master key and the 32-byte `kdf_salt` carried in the
/// page-0 header.
///
/// HKDF-SHA256 with `info = b"obj-page-encryption-v1"`. HKDF-Expand
/// only fails when the requested output length exceeds `255 * HashLen`,
/// and 32 bytes is well under SHA-256's 8160-byte cap, so the error arm
/// is unreachable today. It is nonetheless surfaced as a hard error
/// rather than papered over with a fallback key: a zeroed (or any
/// fixed) key is the worst possible fail-open mode, silently encrypting
/// every page under an attacker-predictable constant. Should a future
/// `KEY_SIZE` or KDF change make the arm reachable, the failure is
/// loud.
///
/// # Errors
///
/// - [`Error::EncryptionKeyInvalid`] if HKDF-Expand fails (only when the
///   requested output length exceeds `255 * HashLen`).
pub fn derive_page_key(
    user_key: &[u8; KEY_SIZE],
    salt: &[u8; KDF_SALT_SIZE],
) -> Result<[u8; KEY_SIZE]> {
    let hk = Hkdf::<Sha256>::new(Some(salt), user_key);
    let mut out = [0u8; KEY_SIZE];
    hk.expand(HKDF_INFO, &mut out)
        .map_err(|_| Error::EncryptionKeyInvalid)?;
    Ok(out)
}

/// Encrypt a 4096-byte logical page into a 4136-byte physical
/// representation. The output layout is `ciphertext || nonce || tag`.
///
/// A fresh 24-byte nonce is drawn from the injected `entropy` source
/// on every call, so callers do not need to track a nonce counter.
/// Production passes
/// [`OsEntropy`] (OS CSPRNG); the DST
/// harness passes a seeded source for reproducible nonces.
///
/// `page_id` is bound to the ciphertext via the AEAD's associated
/// data: an attacker who relocates an encrypted page from slot `N`
/// to slot `M` will see the decryption at slot `M` fail Poly1305
/// verification.
///
/// # Errors
///
/// - [`Error::EncryptionKeyInvalid`] if the AEAD encryptor signals
///   failure. This is structurally unreachable for XChaCha20-Poly1305
///   on inputs of bounded size, but we surface it as a real error
///   rather than `unwrap`.
pub fn encrypt_page(
    key: &[u8; KEY_SIZE],
    page_id: u64,
    plaintext: &[u8; LOGICAL_PAGE_SIZE],
    out: &mut [u8; PHYSICAL_ENCRYPTED_PAGE_SIZE],
    entropy: &dyn Entropy,
) -> Result<()> {
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    entropy.fill_bytes(&mut nonce_bytes);
    let nonce = XNonce::from_slice(&nonce_bytes);

    out[..LOGICAL_PAGE_SIZE].copy_from_slice(plaintext);

    let cipher = XChaCha20Poly1305::new(Key::from_slice(key));
    let ad = page_id.to_le_bytes();
    let tag = cipher
        .encrypt_in_place_detached(nonce, &ad, &mut out[..LOGICAL_PAGE_SIZE])
        .map_err(|_| Error::EncryptionKeyInvalid)?;

    out[LOGICAL_PAGE_SIZE..LOGICAL_PAGE_SIZE + NONCE_SIZE].copy_from_slice(&nonce_bytes);
    out[LOGICAL_PAGE_SIZE + NONCE_SIZE..].copy_from_slice(&tag);
    Ok(())
}

/// Decrypt a 4136-byte physical page into a 4096-byte logical page.
///
/// Layout assumed: `ciphertext || nonce || tag`, exactly as produced
/// by [`encrypt_page`].
///
/// # Errors
///
/// - [`Error::EncryptionKeyInvalid`] if Poly1305 verification fails
///   (wrong key, tampered ciphertext, tampered nonce, mismatched
///   `page_id`, or any other AD / ciphertext alteration).
pub fn decrypt_page(
    key: &[u8; KEY_SIZE],
    page_id: u64,
    ciphertext: &[u8; PHYSICAL_ENCRYPTED_PAGE_SIZE],
    out: &mut [u8; LOGICAL_PAGE_SIZE],
) -> Result<()> {
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    nonce_bytes.copy_from_slice(&ciphertext[LOGICAL_PAGE_SIZE..LOGICAL_PAGE_SIZE + NONCE_SIZE]);
    let nonce = XNonce::from_slice(&nonce_bytes);

    let mut tag_bytes = [0u8; TAG_SIZE];
    tag_bytes.copy_from_slice(&ciphertext[LOGICAL_PAGE_SIZE + NONCE_SIZE..]);
    let tag = Tag::from_slice(&tag_bytes);

    out.copy_from_slice(&ciphertext[..LOGICAL_PAGE_SIZE]);
    let cipher = XChaCha20Poly1305::new(Key::from_slice(key));
    let ad = page_id.to_le_bytes();
    cipher
        .decrypt_in_place_detached(nonce, &ad, out, tag)
        .map_err(|_| Error::EncryptionKeyInvalid)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        decrypt_page, derive_page_key, encrypt_page, KDF_SALT_SIZE, KEY_SIZE, LOGICAL_PAGE_SIZE,
        NONCE_SIZE, PHYSICAL_ENCRYPTED_PAGE_SIZE,
    };
    use crate::platform::env::OsEntropy;

    fn test_key() -> [u8; KEY_SIZE] {
        let mut k = [0u8; KEY_SIZE];
        for (i, b) in k.iter_mut().enumerate() {
            *b = u8::try_from(i & 0xFF).unwrap_or(0);
        }
        k
    }

    fn test_salt() -> [u8; KDF_SALT_SIZE] {
        let mut s = [0u8; KDF_SALT_SIZE];
        for (i, b) in s.iter_mut().enumerate() {
            *b = u8::try_from((i ^ 0xA5) & 0xFF).unwrap_or(0);
        }
        s
    }

    fn test_plaintext() -> [u8; LOGICAL_PAGE_SIZE] {
        let mut p = [0u8; LOGICAL_PAGE_SIZE];
        for (i, b) in p.iter_mut().enumerate() {
            *b = u8::try_from(i & 0xFF).unwrap_or(0);
        }
        p
    }

    #[test]
    fn derive_page_key_is_deterministic() {
        let k = test_key();
        let s = test_salt();
        assert_eq!(
            derive_page_key(&k, &s).expect("derive"),
            derive_page_key(&k, &s).expect("derive")
        );
    }

    #[test]
    fn derive_page_key_changes_with_salt() {
        let k = test_key();
        let s1 = test_salt();
        let mut s2 = s1;
        s2[0] ^= 0xFF;
        assert_ne!(
            derive_page_key(&k, &s1).expect("derive"),
            derive_page_key(&k, &s2).expect("derive")
        );
    }

    #[test]
    fn derive_page_key_changes_with_user_key() {
        let s = test_salt();
        let k1 = test_key();
        let mut k2 = k1;
        k2[0] ^= 0x55;
        assert_ne!(
            derive_page_key(&k1, &s).expect("derive"),
            derive_page_key(&k2, &s).expect("derive")
        );
    }

    #[test]
    fn round_trip_round_trips() {
        let key = derive_page_key(&test_key(), &test_salt()).expect("derive");
        let pt = test_plaintext();
        let mut ct = [0u8; PHYSICAL_ENCRYPTED_PAGE_SIZE];
        encrypt_page(&key, 7, &pt, &mut ct, &OsEntropy).expect("encrypt");
        let mut decrypted = [0u8; LOGICAL_PAGE_SIZE];
        decrypt_page(&key, 7, &ct, &mut decrypted).expect("decrypt");
        assert_eq!(decrypted, pt);
    }

    #[test]
    fn wrong_key_fails_decryption() {
        let key = derive_page_key(&test_key(), &test_salt()).expect("derive");
        let mut wrong_user = test_key();
        wrong_user[0] ^= 0x42;
        let wrong_key = derive_page_key(&wrong_user, &test_salt()).expect("derive");
        let pt = test_plaintext();
        let mut ct = [0u8; PHYSICAL_ENCRYPTED_PAGE_SIZE];
        encrypt_page(&key, 7, &pt, &mut ct, &OsEntropy).expect("encrypt");
        let mut decrypted = [0u8; LOGICAL_PAGE_SIZE];
        let err =
            decrypt_page(&wrong_key, 7, &ct, &mut decrypted).expect_err("wrong key must fail");
        assert!(matches!(err, crate::error::Error::EncryptionKeyInvalid));
    }

    #[test]
    fn bit_flip_in_ciphertext_fails_poly1305() {
        let key = derive_page_key(&test_key(), &test_salt()).expect("derive");
        let pt = test_plaintext();
        let mut ct = [0u8; PHYSICAL_ENCRYPTED_PAGE_SIZE];
        encrypt_page(&key, 11, &pt, &mut ct, &OsEntropy).expect("encrypt");
        ct[100] ^= 0x01;
        let mut decrypted = [0u8; LOGICAL_PAGE_SIZE];
        let err = decrypt_page(&key, 11, &ct, &mut decrypted).expect_err("bit flip must fail");
        assert!(matches!(err, crate::error::Error::EncryptionKeyInvalid));
    }

    #[test]
    fn bit_flip_in_nonce_fails_poly1305() {
        let key = derive_page_key(&test_key(), &test_salt()).expect("derive");
        let pt = test_plaintext();
        let mut ct = [0u8; PHYSICAL_ENCRYPTED_PAGE_SIZE];
        encrypt_page(&key, 3, &pt, &mut ct, &OsEntropy).expect("encrypt");
        ct[LOGICAL_PAGE_SIZE] ^= 0x40;
        let mut decrypted = [0u8; LOGICAL_PAGE_SIZE];
        let err = decrypt_page(&key, 3, &ct, &mut decrypted).expect_err("nonce flip must fail");
        assert!(matches!(err, crate::error::Error::EncryptionKeyInvalid));
    }

    #[test]
    fn bit_flip_in_tag_fails_poly1305() {
        let key = derive_page_key(&test_key(), &test_salt()).expect("derive");
        let pt = test_plaintext();
        let mut ct = [0u8; PHYSICAL_ENCRYPTED_PAGE_SIZE];
        encrypt_page(&key, 3, &pt, &mut ct, &OsEntropy).expect("encrypt");
        ct[PHYSICAL_ENCRYPTED_PAGE_SIZE - 1] ^= 0x80;
        let mut decrypted = [0u8; LOGICAL_PAGE_SIZE];
        let err = decrypt_page(&key, 3, &ct, &mut decrypted).expect_err("tag flip must fail");
        assert!(matches!(err, crate::error::Error::EncryptionKeyInvalid));
    }

    #[test]
    fn wrong_page_id_fails_decryption() {
        let key = derive_page_key(&test_key(), &test_salt()).expect("derive");
        let pt = test_plaintext();
        let mut ct = [0u8; PHYSICAL_ENCRYPTED_PAGE_SIZE];
        encrypt_page(&key, 42, &pt, &mut ct, &OsEntropy).expect("encrypt");
        let mut decrypted = [0u8; LOGICAL_PAGE_SIZE];
        let err =
            decrypt_page(&key, 43, &ct, &mut decrypted).expect_err("swapped page-id must fail");
        assert!(matches!(err, crate::error::Error::EncryptionKeyInvalid));
    }

    #[test]
    fn fresh_nonce_per_encryption() {
        let key = derive_page_key(&test_key(), &test_salt()).expect("derive");
        let pt = test_plaintext();
        let mut ct1 = [0u8; PHYSICAL_ENCRYPTED_PAGE_SIZE];
        let mut ct2 = [0u8; PHYSICAL_ENCRYPTED_PAGE_SIZE];
        encrypt_page(&key, 1, &pt, &mut ct1, &OsEntropy).expect("encrypt 1");
        encrypt_page(&key, 1, &pt, &mut ct2, &OsEntropy).expect("encrypt 2");
        assert_ne!(ct1[..LOGICAL_PAGE_SIZE], ct2[..LOGICAL_PAGE_SIZE]);
        assert_ne!(
            ct1[LOGICAL_PAGE_SIZE..LOGICAL_PAGE_SIZE + NONCE_SIZE],
            ct2[LOGICAL_PAGE_SIZE..LOGICAL_PAGE_SIZE + NONCE_SIZE]
        );
    }
}
