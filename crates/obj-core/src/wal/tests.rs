//! WAL append / commit / recovery tests.
//!
//! These exercise the [`Wal`] API in isolation, without going through
//! the pager. The pager-WAL integration tests live in
//! `crates/obj-core/src/pager/tests.rs`.

use tempfile::TempDir;

use crate::pager::page::{Page, PageId};
use crate::wal::{Lsn, Wal, WalConfig};

fn page_with(byte: u8) -> Page {
    let mut p = Page::zeroed();
    p.as_bytes_mut()[0] = byte;
    p.as_bytes_mut()[1000] = byte.wrapping_mul(3);
    p
}

fn id(n: u64) -> PageId {
    PageId::new(n).expect("non-zero")
}

#[test]
fn create_fresh_writes_header_only() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
    assert_eq!(wal.committed_frames(), 0);
    assert_eq!(wal.next_lsn(), Lsn::ONE);
}

#[test]
fn append_and_commit_one_frame() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
    let mut txn = wal.begin_txn();
    txn.append(id(1), &page_with(0xAA)).expect("append");
    let lsn = txn.commit().expect("commit");
    assert_eq!(lsn, Lsn::new(1));
    assert_eq!(wal.committed_frames(), 1);
}

#[test]
fn group_commit_assigns_consecutive_lsns() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
    let mut txn = wal.begin_txn();
    for n in 1u8..=4 {
        txn.append(id(u64::from(n)), &page_with(n)).expect("append");
    }
    let last_lsn = txn.commit().expect("commit");
    assert_eq!(last_lsn, Lsn::new(4));
    assert_eq!(wal.committed_frames(), 4);
    assert_eq!(wal.next_lsn(), Lsn::new(5));
}

#[test]
fn empty_txn_is_noop() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
    let txn = wal.begin_txn();
    let lsn = txn.commit().expect("empty commit");
    assert_eq!(lsn, Lsn::ZERO);
    assert_eq!(wal.committed_frames(), 0);
}

#[test]
fn reset_after_checkpoint_rotates_salt_and_truncates() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
    let old_salt = wal.salt();
    let mut txn = wal.begin_txn();
    txn.append(id(1), &page_with(0xAA)).expect("append");
    let _ = txn.commit().expect("commit");
    wal.reset_after_checkpoint().expect("reset");
    assert_ne!(wal.salt(), old_salt);
    assert_eq!(wal.committed_frames(), 0);
    assert_eq!(wal.next_lsn(), Lsn::ONE);
}

#[test]
fn recover_two_committed_txns_with_torn_tail() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a1");
        t.append(id(2), &page_with(0xBB)).expect("a2");
        t.commit().expect("commit 1");
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xCC)).expect("a3");
        t.append(id(3), &page_with(0xDD)).expect("a4");
        t.commit().expect("commit 2");
        wal.salt()
    };
    {
        use std::fs::OpenOptions;
        use std::io::Write as _;
        let mut f = OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open wal rw");
        f.write_all(&[0xFFu8; 100]).expect("torn tail bytes");
    }
    let recovered =
        Wal::open_for_recovery(&path, salt, WalConfig::default().size_limit).expect("recover");
    assert_eq!(recovered.committed_frames, 4);
    let p1 = recovered.view.get(&id(1)).expect("p1");
    assert_eq!(p1.as_bytes()[0], 0xCC);
    let p2 = recovered.view.get(&id(2)).expect("p2");
    assert_eq!(p2.as_bytes()[0], 0xBB);
    let p3 = recovered.view.get(&id(3)).expect("p3");
    assert_eq!(p3.as_bytes()[0], 0xDD);
}

#[test]
fn recover_empty_wal_with_header_only() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let salt = {
        let wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        wal.salt()
    };
    let recovered =
        Wal::open_for_recovery(&path, salt, WalConfig::default().size_limit).expect("recover");
    assert_eq!(recovered.committed_frames, 0);
    assert!(recovered.view.is_empty());
}

#[test]
fn recover_no_wal_file_is_empty() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("absent.obj-wal");
    let recovered = Wal::open_for_recovery(&path, 0xCAFE_BABE, WalConfig::default().size_limit)
        .expect("recover");
    assert_eq!(recovered.committed_frames, 0);
    assert!(recovered.view.is_empty());
}

#[test]
fn recover_stale_salt_is_treated_as_empty() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a");
        t.commit().expect("commit");
    }
    let recovered = Wal::open_for_recovery(&path, 0xDEAD_BEEF, WalConfig::default().size_limit)
        .expect("recover");
    assert_eq!(recovered.committed_frames, 0);
    assert!(recovered.view.is_empty());
}

#[test]
fn recover_corrupted_tail_after_last_commit_truncates() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a1");
        t.commit().expect("commit 1");
        let mut t = wal.begin_txn();
        t.append(id(2), &page_with(0xBB)).expect("a2");
        t.commit().expect("commit 2");
        wal.salt()
    };
    {
        use std::fs::OpenOptions;
        use std::io::{Seek, SeekFrom, Write as _};
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open wal rw");
        let corrupt_offset = 64u64 + 4160u64 + 64u64 + 50u64;
        f.seek(SeekFrom::Start(corrupt_offset)).expect("seek");
        f.write_all(&[0xAB]).expect("corrupt");
    }
    let recovered =
        Wal::open_for_recovery(&path, salt, WalConfig::default().size_limit).expect("recover");
    assert_eq!(recovered.committed_frames, 1);
    assert!(recovered.view.contains_key(&id(1)));
    assert!(!recovered.view.contains_key(&id(2)));
}

#[test]
fn wal_size_limit_rejects_overflow() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let config = WalConfig {
        size_limit: 4096 + 4160,
        ..WalConfig::default()
    };
    let mut wal = Wal::create_fresh(&path, config).expect("create");
    let mut txn = wal.begin_txn();
    txn.append(id(1), &page_with(0x11)).expect("first append");
    let err = txn.append(id(2), &page_with(0x22));
    assert!(err.is_err(), "second append must hit the size limit");
}

/// Flipping a byte in the FIRST committed transaction's frame body
/// must surface as `Error::WalCorruption`, NOT silently truncate at
/// the bad CRC. The two-pass walk catches the mid-WAL CRC mismatch
/// and refuses to guess.
#[test]
fn recover_corrupted_first_frame_surfaces_wal_corruption() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a1");
        t.commit().expect("commit 1");
        let mut t = wal.begin_txn();
        t.append(id(2), &page_with(0xBB)).expect("a2");
        t.commit().expect("commit 2");
        wal.salt()
    };
    {
        use std::fs::OpenOptions;
        use std::io::{Seek, SeekFrom, Write as _};
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open wal rw");
        let corrupt_offset = 64u64 + 64u64 + 50u64;
        f.seek(SeekFrom::Start(corrupt_offset)).expect("seek");
        f.write_all(&[0xAB]).expect("corrupt");
    }
    let err = Wal::open_for_recovery(&path, salt, WalConfig::default().size_limit)
        .expect_err("must surface WalCorruption");
    match err {
        crate::Error::WalCorruption { frame_offset } => {
            assert_eq!(frame_offset, 64);
        }
        other => panic!("expected WalCorruption, got {other:?}"),
    }
}

#[test]
fn recover_torn_tail_byte_past_last_commit_recovers_cleanly() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a1");
        t.commit().expect("commit 1");
        let mut t = wal.begin_txn();
        t.append(id(2), &page_with(0xBB)).expect("a2");
        t.commit().expect("commit 2");
        wal.salt()
    };
    {
        use std::fs::OpenOptions;
        use std::io::Write as _;
        let mut f = OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open wal append");
        f.write_all(&[0xFFu8; 200]).expect("append torn bytes");
    }
    let recovered =
        Wal::open_for_recovery(&path, salt, WalConfig::default().size_limit).expect("recover");
    assert_eq!(recovered.committed_frames, 2);
    assert!(recovered.view.contains_key(&id(1)));
    assert!(recovered.view.contains_key(&id(2)));
}

#[test]
fn recover_mid_wal_crc_in_multi_frame_txn_surfaces_corruption() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a1");
        t.append(id(2), &page_with(0xBB)).expect("a2");
        t.commit().expect("commit 1");
        let mut t = wal.begin_txn();
        t.append(id(3), &page_with(0xCC)).expect("a3");
        t.commit().expect("commit 2");
        wal.salt()
    };
    {
        use std::fs::OpenOptions;
        use std::io::{Seek, SeekFrom, Write as _};
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open wal rw");
        let corrupt_offset = 64u64 + 64u64 + 200u64;
        f.seek(SeekFrom::Start(corrupt_offset)).expect("seek");
        f.write_all(&[0x55]).expect("corrupt");
    }
    let err = Wal::open_for_recovery(&path, salt, WalConfig::default().size_limit)
        .expect_err("must surface WalCorruption");
    assert!(matches!(err, crate::Error::WalCorruption { .. }));
}

/// A bit-flip in the body of the LAST frame — past the previous
/// commit marker — must be treated as an ordinary torn tail on an
/// ENCRYPTED WAL, exactly as the plaintext path treats a bad-CRC
/// tail. The salt lives in the plaintext frame header and is excluded
/// from the AEAD associated data, so the flipped tail frame still
/// matches the generation salt while Poly1305 fails. Before the fix
/// this misfired as `Error::EncryptionKeyInvalid` and aborted
/// recovery; now the committed prefix recovers cleanly.
#[cfg(feature = "encryption")]
#[test]
fn encrypted_tail_frame_bit_flip_recovers_committed_prefix() {
    use crate::platform::FileHandle;
    use crate::wal::frame::{FRAME_HEADER_SIZE, FRAME_SIZE_ENCRYPTED, WAL_HEADER_SIZE};

    let key = [0x2Au8; 32];
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("enc.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        wal.set_key(Some(key));
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a1");
        t.commit().expect("commit 1");
        let mut t = wal.begin_txn();
        t.append(id(2), &page_with(0xBB)).expect("a2");
        t.commit().expect("commit 2 (becomes torn tail)");
        wal.salt()
    };
    // Flip one byte inside the SECOND frame's ciphertext body (past its
    // 64-byte plaintext header, so the generation salt stays intact).
    {
        use std::fs::OpenOptions;
        use std::io::{Read as _, Seek, SeekFrom, Write as _};
        let frame1 = (WAL_HEADER_SIZE + FRAME_SIZE_ENCRYPTED) as u64;
        let flip_at = frame1 + FRAME_HEADER_SIZE as u64 + 128;
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open wal rw");
        f.seek(SeekFrom::Start(flip_at)).expect("seek");
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte).expect("read body byte");
        byte[0] ^= 0x40;
        f.seek(SeekFrom::Start(flip_at)).expect("seek back");
        f.write_all(&byte).expect("flip body byte");
    }
    let file = FileHandle::open_or_create(&path).expect("reopen wal");
    let recovered = Wal::<FileHandle>::open_for_recovery_with_key(
        &file,
        salt,
        WalConfig::default().size_limit,
        Some(key),
    )
    .expect("torn tail must recover, not report EncryptionKeyInvalid");
    assert_eq!(recovered.committed_frames, 1);
    let p1 = recovered.view.get(&id(1)).expect("committed prefix present");
    assert_eq!(p1.as_bytes()[0], 0xAA);
    assert!(
        !recovered.view.contains_key(&id(2)),
        "torn tail frame must be discarded"
    );
}

/// A torn write that persisted the plaintext frame header but left the
/// ciphertext body only partially written (the file is truncated
/// mid-body of the final frame) must also recover the committed prefix
/// on an encrypted WAL rather than aborting with
/// `Error::EncryptionKeyInvalid`.
#[cfg(feature = "encryption")]
#[test]
fn encrypted_torn_tail_partial_body_recovers_committed_prefix() {
    use crate::platform::FileHandle;
    use crate::wal::frame::{FRAME_HEADER_SIZE, FRAME_SIZE_ENCRYPTED, WAL_HEADER_SIZE};

    let key = [0x7Cu8; 32];
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("enc.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        wal.set_key(Some(key));
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a1");
        t.commit().expect("commit 1");
        let mut t = wal.begin_txn();
        t.append(id(2), &page_with(0xBB)).expect("a2");
        t.commit().expect("commit 2 (torn below)");
        wal.salt()
    };
    // Truncate so the final frame keeps its 64-byte plaintext header
    // plus a partial body — a power-loss-during-append signature.
    {
        let frame1 = (WAL_HEADER_SIZE + FRAME_SIZE_ENCRYPTED) as u64;
        let truncate_to = frame1 + FRAME_HEADER_SIZE as u64 + 256;
        let file = FileHandle::open_or_create(&path).expect("open wal");
        file.set_len(truncate_to).expect("truncate mid-body");
    }
    let file = FileHandle::open_or_create(&path).expect("reopen wal");
    let recovered = Wal::<FileHandle>::open_for_recovery_with_key(
        &file,
        salt,
        WalConfig::default().size_limit,
        Some(key),
    )
    .expect("partial tail must recover, not report EncryptionKeyInvalid");
    assert_eq!(recovered.committed_frames, 1);
    assert_eq!(
        recovered
            .view
            .get(&id(1))
            .expect("committed prefix present")
            .as_bytes()[0],
        0xAA
    );
    assert!(!recovered.view.contains_key(&id(2)));
}

/// A bit-flip in the ciphertext body of a committed frame that sits
/// **before** the last commit marker must be classified as
/// `Error::WalCorruption`, NOT `Error::EncryptionKeyInvalid`: the later
/// commit marker decrypts and CRC-validates under the same key, which
/// proves the key is correct — so the only explanation for the earlier
/// frame failing to decrypt is corruption of a committed frame. Before
/// the fix this fail-closed path mislabeled the cause as a wrong key.
#[cfg(feature = "encryption")]
#[test]
fn encrypted_committed_frame_corruption_before_marker_is_wal_corruption() {
    use crate::platform::FileHandle;
    use crate::wal::frame::{FRAME_HEADER_SIZE, WAL_HEADER_SIZE};

    let key = [0x3Bu8; 32];
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("enc.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        wal.set_key(Some(key));
        // Two independent commits: frame 0 (id 1) and frame 1 (id 2).
        // Frame 1 is the last commit marker; frame 0 precedes it.
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a1");
        t.commit().expect("commit 1");
        let mut t = wal.begin_txn();
        t.append(id(2), &page_with(0xBB)).expect("a2");
        t.commit().expect("commit 2 (last marker)");
        wal.salt()
    };
    // Flip one byte inside the FIRST frame's ciphertext body (past its
    // 64-byte plaintext header so the generation salt stays intact).
    // The second frame's commit marker is left untouched, so it still
    // decrypts — proving the key is correct.
    {
        use std::fs::OpenOptions;
        use std::io::{Read as _, Seek, SeekFrom, Write as _};
        let flip_at = WAL_HEADER_SIZE as u64 + FRAME_HEADER_SIZE as u64 + 128;
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open wal rw");
        f.seek(SeekFrom::Start(flip_at)).expect("seek");
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte).expect("read body byte");
        byte[0] ^= 0x40;
        f.seek(SeekFrom::Start(flip_at)).expect("seek back");
        f.write_all(&byte).expect("flip body byte");
    }
    let file = FileHandle::open_or_create(&path).expect("reopen wal");
    let err = Wal::<FileHandle>::open_for_recovery_with_key(
        &file,
        salt,
        WalConfig::default().size_limit,
        Some(key),
    )
    .expect_err("corrupt committed frame must surface an error");
    let expected_offset = WAL_HEADER_SIZE as u64;
    assert!(
        matches!(err, crate::Error::WalCorruption { frame_offset } if frame_offset == expected_offset),
        "decrypt failure before a valid commit marker must be WalCorruption, not EncryptionKeyInvalid; got {err:?}"
    );
}

/// A committed WAL opened with the WRONG key: every frame fails
/// Poly1305 verification, so ZERO frames decrypt and no commit marker is
/// ever found. A salt-matching decrypt failure with nothing decrypted is
/// the genuine wrong-key smoking gun and must STILL surface as
/// `Error::EncryptionKeyInvalid`. This is the case that stays unchanged:
/// the no-marker branch escalates only when `any_frame_decrypted` is
/// false (contrast `encrypted_torn_uncommitted_no_marker_recovers`,
/// where an intact frame proves the key and the same no-marker shape is
/// discarded instead).
#[cfg(feature = "encryption")]
#[test]
fn encrypted_no_commit_marker_decrypt_failure_is_wrong_key() {
    use crate::platform::FileHandle;

    let right_key = [0x11u8; 32];
    let wrong_key = [0x22u8; 32];
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("enc.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        wal.set_key(Some(right_key));
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a1");
        t.commit().expect("commit 1");
        wal.salt()
    };
    // Open with the WRONG key: no frame decrypts, so no commit marker is
    // ever found and the whole generation is undecryptable.
    let file = FileHandle::open_or_create(&path).expect("reopen wal");
    let err = Wal::<FileHandle>::open_for_recovery_with_key(
        &file,
        salt,
        WalConfig::default().size_limit,
        Some(wrong_key),
    )
    .expect_err("wrong key must be refused");
    assert!(
        matches!(err, crate::Error::EncryptionKeyInvalid),
        "no commit marker + decrypt failure must stay EncryptionKeyInvalid; got {err:?}"
    );
}

/// The torn-UNCOMMITTED-tail case that the #1/#19 fix mis-handled: a
/// post-checkpoint first transaction whose commit marker never reached
/// disk AND one of whose non-final frames is bit-flipped, while another
/// non-final frame stays intact. Because `reset_after_checkpoint`
/// truncates the WAL to header-only, this generation legitimately has NO
/// commit marker — yet the intact frame decrypts and CRC-validates,
/// proving the key correct. Recovery must therefore DISCARD the
/// uncommitted generation and open cleanly (committed prefix empty), NOT
/// abort with `Error::EncryptionKeyInvalid`.
#[cfg(feature = "encryption")]
#[test]
fn encrypted_torn_uncommitted_no_marker_recovers() {
    use crate::platform::FileHandle;
    use crate::wal::frame::{FRAME_HEADER_SIZE, FRAME_SIZE_ENCRYPTED, WAL_HEADER_SIZE};

    let key = [0x5Du8; 32];
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("enc.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        wal.set_key(Some(key));
        // A committed transaction, then a checkpoint reset — this rotates
        // the salt and truncates the WAL back to header-only, so the next
        // transaction re-enters the "no commit marker yet" window.
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("pre-checkpoint");
        t.commit().expect("commit pre-checkpoint");
        wal.reset_after_checkpoint().expect("checkpoint reset");
        // First post-checkpoint transaction: three staged frames. On
        // disk this is frame 0 (commit=false), frame 1 (commit=false),
        // frame 2 (commit=true — the marker).
        let mut t = wal.begin_txn();
        t.append(id(10), &page_with(0xB0)).expect("f0");
        t.append(id(11), &page_with(0xB1)).expect("f1");
        t.append(id(12), &page_with(0xB2)).expect("f2 (marker)");
        t.commit().expect("commit post-checkpoint");
        wal.salt()
    };
    // Simulate power loss: the commit marker (frame 2) never persisted —
    // truncate it off — and frame 0's ciphertext body is torn (bit-flip
    // past its 64-byte plaintext header so the salt stays intact). Frame
    // 1 is left intact, so it still decrypts and proves the key correct.
    {
        use std::fs::OpenOptions;
        use std::io::{Read as _, Seek, SeekFrom, Write as _};
        let keep = WAL_HEADER_SIZE as u64 + 2 * FRAME_SIZE_ENCRYPTED as u64;
        let file = FileHandle::open_or_create(&path).expect("open wal");
        file.set_len(keep).expect("drop unsynced marker frame");

        let flip_at = WAL_HEADER_SIZE as u64 + FRAME_HEADER_SIZE as u64 + 128;
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open wal rw");
        f.seek(SeekFrom::Start(flip_at)).expect("seek");
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte).expect("read body byte");
        byte[0] ^= 0x40;
        f.seek(SeekFrom::Start(flip_at)).expect("seek back");
        f.write_all(&byte).expect("flip body byte");
    }
    let file = FileHandle::open_or_create(&path).expect("reopen wal");
    let recovered = Wal::<FileHandle>::open_for_recovery_with_key(
        &file,
        salt,
        WalConfig::default().size_limit,
        Some(key),
    )
    .expect("torn uncommitted tail with a proven key must recover, not report EncryptionKeyInvalid");
    assert_eq!(
        recovered.committed_frames, 0,
        "no commit marker → nothing committed"
    );
    assert!(recovered.view.is_empty(), "uncommitted generation discarded");
    assert!(!recovered.view.contains_key(&id(10)));
    assert!(!recovered.view.contains_key(&id(11)));
    assert!(!recovered.view.contains_key(&id(12)));
}

/// Write a multi-frame transaction that INCLUDES a page-0
/// (header) frame, then recover it. Proves the `is_header` bool +
/// the reused frame scratch produce a WAL whose header frame is
/// emitted with `wire_page_id == 0` (recovered into `Recovered.header`)
/// while the regular frames recover into `Recovered.view` — i.e. the
/// allocation-hygiene refactor preserves on-disk semantics.
#[test]
fn recover_multi_frame_txn_including_header_frame() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("db.obj-wal");
    let salt = {
        let mut wal = Wal::create_fresh(&path, WalConfig::default()).expect("create");
        let mut t = wal.begin_txn();
        t.append(id(1), &page_with(0xAA)).expect("a1");
        t.append_header(&page_with(0x11)).expect("hdr");
        t.append(id(2), &page_with(0xBB)).expect("a2");
        t.commit().expect("commit");
        wal.salt()
    };
    let recovered =
        Wal::open_for_recovery(&path, salt, WalConfig::default().size_limit).expect("recover");
    assert_eq!(recovered.committed_frames, 3);
    let p1 = recovered.view.get(&id(1)).expect("p1");
    assert_eq!(p1.as_bytes()[0], 0xAA);
    let p2 = recovered.view.get(&id(2)).expect("p2");
    assert_eq!(p2.as_bytes()[0], 0xBB);
    let header = recovered.header.expect("header recovered");
    assert_eq!(header.as_bytes()[0], 0x11);
    assert_eq!(
        recovered.view.get(&id(1)).expect("p1 again").as_bytes()[0],
        0xAA
    );
}
