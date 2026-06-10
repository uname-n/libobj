

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "libobj.h"

#define DOC_COUNT 3

static const char *PAYLOADS[DOC_COUNT] = {
    "first-doc",
    "second-doc",
    "third-doc-with-different-length",
};


static const char *tmpdir_path(void) {
    const char *t = getenv("TMPDIR");
    if (t == NULL) t = getenv("TEMP");
    if (t == NULL) t = getenv("TMP");
    if (t == NULL) t = ".";
    return t;
}


static int compose_path(char *out, size_t cap, const char *dir, const char *file) {
    size_t needed = strlen(dir) + 1 + strlen(file) + 1;
    if (needed > cap) return 1;

    snprintf(out, cap, "%s/%s", dir, file);
    return 0;
}


static int fail(const char *step, obj_error_t err) {
    fprintf(stderr, "smoke: step '%s' failed: obj_strerror: %s (code %d)\n",
            step, obj_strerror(err), (int)err);
    return 1;
}


static void remove_if_exists(const char *path) {

    (void)remove(path);
}

int main(void) {
    const char *dir = tmpdir_path();
    char db_path[512];
    char backup_path[512];
    if (compose_path(db_path, sizeof db_path, dir, "obj-c-smoke.obj") != 0) {
        fprintf(stderr, "smoke: tempdir path too long\n");
        return 1;
    }
    if (compose_path(backup_path, sizeof backup_path, dir, "obj-c-smoke-backup.obj") != 0) {
        fprintf(stderr, "smoke: tempdir path too long\n");
        return 1;
    }
    remove_if_exists(db_path);
    remove_if_exists(backup_path);

    obj_error_t code;


    obj_db_t *db = NULL;
    code = obj_open(db_path, &db);
    if (code != OBJ_OK) return fail("obj_open", code);


    obj_write_txn_t *wtxn = NULL;
    code = obj_txn_begin_write(db, &wtxn);
    if (code != OBJ_OK) {
        obj_close(db);
        return fail("obj_txn_begin_write", code);
    }
    uint64_t ids[DOC_COUNT];
    for (int i = 0; i < DOC_COUNT; i++) {
        const char *payload = PAYLOADS[i];
        size_t len = strlen(payload);
        code = obj_doc_insert_raw(wtxn, "smoke",
                              (const uint8_t *)payload, len,
                              &ids[i]);
        if (code != OBJ_OK) {
            obj_txn_rollback(wtxn);
            obj_close(db);
            return fail("obj_doc_insert_raw", code);
        }
    }
    code = obj_txn_commit(wtxn);
    if (code != OBJ_OK) {
        obj_close(db);
        return fail("obj_txn_commit", code);
    }


    obj_read_txn_t *rtxn = NULL;
    code = obj_txn_begin_read(db, &rtxn);
    if (code != OBJ_OK) {
        obj_close(db);
        return fail("obj_txn_begin_read", code);
    }
    for (int i = 0; i < DOC_COUNT; i++) {
        uint8_t *out_payload = NULL;
        size_t out_len = 0;
        code = obj_doc_get(rtxn, "smoke", ids[i],
                           &out_payload, &out_len);
        if (code != OBJ_OK) {
            obj_txn_end_read(rtxn);
            obj_close(db);
            return fail("obj_doc_get", code);
        }
        const char *expected = PAYLOADS[i];
        size_t expected_len = strlen(expected);
        if (out_len != expected_len ||
            memcmp(out_payload, expected, out_len) != 0) {
            fprintf(stderr,
                    "smoke: payload mismatch at id=%llu: got %zu bytes, expected %zu\n",
                    (unsigned long long)ids[i], out_len, expected_len);
            obj_buf_free(out_payload);
            obj_txn_end_read(rtxn);
            obj_close(db);
            return 1;
        }
        obj_buf_free(out_payload);
    }
    obj_txn_end_read(rtxn);


    code = obj_txn_begin_read(db, &rtxn);
    if (code != OBJ_OK) {
        obj_close(db);
        return fail("obj_txn_begin_read (iter)", code);
    }
    obj_iter_t *iter = NULL;
    code = obj_iter_all(rtxn, "smoke", &iter);
    if (code != OBJ_OK) {
        obj_txn_end_read(rtxn);
        obj_close(db);
        return fail("obj_iter_all", code);
    }
    int seen = 0;
    while (1) {
        uint64_t id = 0;
        uint8_t *p = NULL;
        size_t plen = 0;
        code = obj_iter_next(iter, &id, &p, &plen);
        if (code == OBJ_ERR_NOT_FOUND) break;
        if (code != OBJ_OK) {
            obj_iter_free(iter);
            obj_txn_end_read(rtxn);
            obj_close(db);
            return fail("obj_iter_next", code);
        }
        obj_buf_free(p);
        seen++;
        if (seen > DOC_COUNT) break;
    }
    obj_iter_free(iter);
    obj_txn_end_read(rtxn);
    if (seen != DOC_COUNT) {
        fprintf(stderr, "smoke: iter_all visited %d docs, expected %d\n",
                seen, DOC_COUNT);
        obj_close(db);
        return 1;
    }


    obj_integrity_report_t *report = NULL;
    code = obj_integrity_check(db, &report);
    if (code != OBJ_OK || !obj_integrity_report_is_ok(report)) {
        size_t nf = obj_integrity_report_failure_count(report);
        fprintf(stderr,
                "smoke: integrity_check failed: code=%d, %zu failure(s)\n",
                (int)code, nf);
        obj_integrity_report_free(report);
        obj_close(db);
        return 1;
    }
    obj_integrity_report_free(report);


    code = obj_backup_to(db, backup_path);
    if (code != OBJ_OK) {
        obj_close(db);
        return fail("obj_backup_to", code);
    }
    obj_db_t *backup_db = NULL;
    code = obj_open(backup_path, &backup_db);
    if (code != OBJ_OK) {
        obj_close(db);
        return fail("obj_open (backup)", code);
    }
    obj_integrity_report_t *backup_report = NULL;
    code = obj_integrity_check(backup_db, &backup_report);
    if (code != OBJ_OK || !obj_integrity_report_is_ok(backup_report)) {
        size_t nf = obj_integrity_report_failure_count(backup_report);
        fprintf(stderr,
                "smoke: backup integrity check failed: code=%d, %zu failure(s)\n",
                (int)code, nf);
        obj_integrity_report_free(backup_report);
        obj_close(backup_db);
        obj_close(db);
        return 1;
    }
    obj_integrity_report_free(backup_report);
    obj_close(backup_db);


    obj_stat_t stat;
    code = obj_stat(db, &stat);
    if (code != OBJ_OK) {
        obj_close(db);
        return fail("obj_stat", code);
    }
    if (stat.collection_count < 1 || stat.page_count == 0 || stat.page_size == 0) {
        fprintf(stderr,
                "smoke: stat sanity failed (collections=%llu pages=%llu page_size=%u)\n",
                (unsigned long long)stat.collection_count,
                (unsigned long long)stat.page_count,
                (unsigned)stat.page_size);
        obj_close(db);
        return 1;
    }


    obj_close(db);

    remove_if_exists(db_path);
    remove_if_exists(backup_path);
    printf("OBJ_C_SMOKE_OK\n");
    return 0;
}
