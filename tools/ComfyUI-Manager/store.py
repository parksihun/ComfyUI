"""Store: one SQLite file (data/manager.db) for what the library knows about each file, the jobs that were run,
and which job made which file, so a file can be asked for its job and a job for its files.

Files are kept per result folder (root), so looking at another server's folder does not throw away what was read
from the first one. Everything in `files` can be rebuilt from the files themselves; `jobs`, `job_files` and `marks`
(favourite, tags and note the user gave a file) cannot.
"""
import json
import os
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    root TEXT NOT NULL, rel TEXT NOT NULL COLLATE NOCASE, mtime REAL, size INTEGER, entry TEXT NOT NULL,
    PRIMARY KEY (root, rel));
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, time REAL, kind TEXT, state TEXT, root TEXT,
    scenario_dir TEXT, source_rel TEXT COLLATE NOCASE, entry TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS jobs_time ON jobs (time DESC);
CREATE INDEX IF NOT EXISTS jobs_scenario ON jobs (scenario_dir);
CREATE INDEX IF NOT EXISTS jobs_source ON jobs (root, source_rel);
CREATE TABLE IF NOT EXISTS job_files (
    job_id TEXT NOT NULL, n INTEGER NOT NULL, root TEXT NOT NULL, rel TEXT NOT NULL COLLATE NOCASE,
    PRIMARY KEY (job_id, n));
CREATE INDEX IF NOT EXISTS job_files_rel ON job_files (root, rel);
CREATE TABLE IF NOT EXISTS marks (
    root TEXT NOT NULL, rel TEXT NOT NULL COLLATE NOCASE, favorite INTEGER NOT NULL DEFAULT 0,
    tags TEXT NOT NULL DEFAULT '[]', note TEXT NOT NULL DEFAULT '', PRIMARY KEY (root, rel));
CREATE TABLE IF NOT EXISTS workflow_index (
    server TEXT NOT NULL, name TEXT NOT NULL, modified INTEGER, entry TEXT NOT NULL, PRIMARY KEY (server, name));
"""


def root_key(folder):
    """A result folder as it is written in the store."""
    return os.path.normcase(os.path.normpath(folder))


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.lock = threading.Lock()      # the library scans in a worker thread, the web handlers run in the loop
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock, self.db:
            self.db.executescript(SCHEMA)

    def _all(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def _run(self, statements):
        with self.lock, self.db:
            for sql, args in statements:
                self.db.execute(sql, args)

    # ---- files: the library's index of a result folder ---------------------------------------------
    def file_stats(self, root):
        return {r["rel"]: (r["mtime"], r["size"]) for r in self._all("SELECT rel, mtime, size FROM files WHERE root = ?", (root,))}

    def files(self, root):
        return {r["rel"]: json.loads(r["entry"]) for r in self._all("SELECT rel, entry FROM files WHERE root = ?", (root,))}

    def put_file(self, root, rel, entry):
        self._run([("INSERT OR REPLACE INTO files (root, rel, mtime, size, entry) VALUES (?, ?, ?, ?, ?)",
                    (root, rel, entry["mtime"], entry["size"], json.dumps(entry, ensure_ascii=False)))])

    def drop_files(self, root, rels):
        self._run([("DELETE FROM files WHERE root = ? AND rel = ?", (root, rel)) for rel in rels])

    def rename_file(self, root, old, new):
        """The file keeps what is known about it and stays connected to its job."""
        self._run([("DELETE FROM files WHERE root = ? AND rel = ?", (root, new)),
                   ("UPDATE files SET rel = ? WHERE root = ? AND rel = ?", (new, root, old)),
                   ("UPDATE job_files SET rel = ? WHERE root = ? AND rel = ?", (new, root, old)),
                   ("DELETE FROM marks WHERE root = ? AND rel = ?", (root, new)),
                   ("UPDATE marks SET rel = ? WHERE root = ? AND rel = ?", (new, root, old)),
                   ("UPDATE jobs SET source_rel = ? WHERE root = ? AND source_rel = ?", (new, root, old))])

    # ---- marks: what the user said about a file ---------------------------------------------------------
    def marks(self, root):
        """{path in lower case: {'favorite': bool, 'tags': [..], 'note': str}} for the files that have any."""
        return {r["rel"].lower(): {"favorite": bool(r["favorite"]), "tags": json.loads(r["tags"]), "note": r["note"]}
                for r in self._all("SELECT rel, favorite, tags, note FROM marks WHERE root = ?", (root,))}

    def mark(self, root, rel):
        found = self._all("SELECT favorite, tags, note FROM marks WHERE root = ? AND rel = ?", (root, rel))
        if not found:
            return {"favorite": False, "tags": [], "note": ""}
        return {"favorite": bool(found[0]["favorite"]), "tags": json.loads(found[0]["tags"]), "note": found[0]["note"]}

    def set_mark(self, root, rel, favorite=None, tags=None, note=None):
        """Change the given ones of favourite / tags / note; a file left with none of them has no row."""
        now = self.mark(root, rel)
        if favorite is not None:
            now["favorite"] = bool(favorite)
        if tags is not None:
            now["tags"] = tags
        if note is not None:
            now["note"] = note
        if now["favorite"] or now["tags"] or now["note"]:
            self._run([("INSERT OR REPLACE INTO marks (root, rel, favorite, tags, note) VALUES (?, ?, ?, ?, ?)",
                        (root, rel, int(now["favorite"]), json.dumps(now["tags"], ensure_ascii=False), now["note"]))])
        else:
            self._run([("DELETE FROM marks WHERE root = ? AND rel = ?", (root, rel))])
        return now

    def forget_files(self, root, rels):
        """Files that were deleted on purpose: nothing is kept about them, and their jobs no longer point at them."""
        self._run([(f"DELETE FROM {table} WHERE root = ? AND rel = ?", (root, rel))
                   for rel in rels for table in ("files", "marks", "job_files")])

    # ---- what is known about the workflows a ComfyUI server keeps (can be read from the server again) -----
    def workflow_index(self, server):
        """{name: (modified, entry or None)}; the entry is what the library needs to recognise the workflow."""
        return {r["name"]: (r["modified"], json.loads(r["entry"]))
                for r in self._all("SELECT name, modified, entry FROM workflow_index WHERE server = ?", (server,))}

    def put_workflows(self, server, items, gone=()):
        self._run([("INSERT OR REPLACE INTO workflow_index (server, name, modified, entry) VALUES (?, ?, ?, ?)",
                    (server, name, modified, json.dumps(entry, ensure_ascii=False))) for name, modified, entry in items]
                  + [("DELETE FROM workflow_index WHERE server = ? AND name = ?", (server, name)) for name in gone])

    # ---- jobs and what they made ---------------------------------------------------------------------
    def add_job(self, entry, root, scenario_dir, source_rel, rels):
        """rels: for each of entry['outputs'], its path inside the result folder, or None when it is not kept there."""
        self._run([("INSERT OR REPLACE INTO jobs (id, time, kind, state, root, scenario_dir, source_rel, entry) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (entry["id"], entry["time"], entry["kind"], entry["state"], root, scenario_dir or "", source_rel or "",
                     json.dumps(entry, ensure_ascii=False))),
                   ("DELETE FROM job_files WHERE job_id = ?", (entry["id"],))]
                  + [("INSERT INTO job_files (job_id, n, root, rel) VALUES (?, ?, ?, ?)", (entry["id"], n, root, rel))
                     for n, rel in enumerate(rels) if rel])

    def _jobs(self, where="", args=(), limit=500):
        rows = self._all(f"SELECT * FROM jobs {where} ORDER BY time DESC LIMIT ?", (*args, limit))
        found = [dict(r, entry=json.loads(r["entry"]), files={}) for r in rows]
        by_id = {j["id"]: j for j in found}
        for start in range(0, len(found), 400):
            ids = [j["id"] for j in found[start:start + 400]]
            for r in self._all(f"SELECT job_id, n, rel FROM job_files WHERE job_id IN ({','.join('?' * len(ids))})", ids):
                by_id[r["job_id"]]["files"][r["n"]] = r["rel"]
        return found

    def jobs(self, limit=500):
        """Newest first. Each: the columns, 'entry' (what the page shows) and 'files' {output number: path now}."""
        return self._jobs(limit=limit)

    def job_count(self):
        return self._all("SELECT COUNT(*) AS n FROM jobs")[0]["n"]

    def job_of(self, root, rel):
        """The job that made this file (the latest one, should a name have been used twice)."""
        found = self._jobs("WHERE id IN (SELECT job_id FROM job_files WHERE root = ? AND rel = ?)", (root, rel), 1)
        return found[0] if found else None

    def jobs_like(self, kind, limit=20, **columns):
        """Jobs of one kind with these column values, e.g. jobs_like('video', scenario_dir=...)."""
        where = " AND ".join(["kind = ?"] + [f"{name} = ?" for name in columns])
        return self._jobs("WHERE " + where, (kind, *columns.values()), limit)

    def rels_with_job(self, root):
        return {r["rel"].lower() for r in self._all("SELECT DISTINCT rel FROM job_files WHERE root = ?", (root,))}
