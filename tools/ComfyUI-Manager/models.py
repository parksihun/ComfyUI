"""Model files: the model folder of the connected ComfyUI as this PC sees it.

Easy-Install keeps the models in a top-level folder `model` (ComfyUI\\models is only there for ComfyUI's own
subfolders), next to `output`, so the folder is found the way the scenario folder is: beside the result folder.
Nothing is read from inside the files; the list is the folders and files with their size and date, plus what the
library index knows (which result files were made with the model) and what ComfyUI lists (a model file ComfyUI
does not list is in the wrong folder or has an extension it does not load).
"""
import os
import re
import shutil

MODEL_EXT = (".safetensors", ".gguf", ".ckpt", ".sft", ".pt", ".pth")


def _lower(name):
    return name.replace("\\", "/").lower()


class Models:
    def __init__(self, root_of, store, library):
        self.root_of = root_of      # () -> the model folder (depends on the result folder picked)
        self.store, self.library = store, library

    @property
    def root(self):
        return self.root_of()

    def resolve(self, ref, folders=False):
        """'model:<folder>/<file>' -> absolute path inside the model folder (nothing outside it); 'model:' alone is
        the folder itself when folders are allowed."""
        kind, _, rel = (ref or "").partition(":")
        base = os.path.realpath(self.root)
        if kind != "model" or not os.path.isdir(base) or (not rel and not folders):
            raise ValueError("잘못된 파일 경로입니다")
        full = os.path.realpath(os.path.join(base, rel)) if rel else base
        if os.path.commonpath([full, base]) != base or not (os.path.isfile(full) or (folders and os.path.isdir(full))):
            raise ValueError("파일을 찾을 수 없습니다: " + rel)
        return full

    def inside(self, path):
        """Whether the absolute path is in the model folder."""
        base = os.path.realpath(self.root)
        return os.path.isdir(base) and os.path.commonpath([os.path.realpath(path), base]) == base

    def used_by(self):
        """{file name in lower case: number of library files made with a model of that name}."""
        counts = {}
        for entry in self.store.files(self.library.root).values():
            names = {m["name"] for m in entry.get("models", [])} | {l["name"] for l in entry.get("loras", [])}
            for name in {_lower(n).rsplit("/", 1)[-1] for n in names}:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def listing(self, known=None):
        """(folders, files, free bytes). `known`: (the model folder names ComfyUI has, the relative names it lists over
        all of them), lower case; None when ComfyUI could not be asked. A model file in a folder ComfyUI has but not
        in its lists is flagged (known False); a folder ComfyUI does not have (a custom node's own) is not judged.
        Placeholder files (put_..._here) are left out."""
        base = self.root
        if not os.path.isdir(base):
            raise ValueError("모델 폴더가 없습니다: " + base)
        used = self.used_by()
        folders, files = [], []
        for folder in sorted(os.listdir(base), key=str.lower):
            top = os.path.join(base, folder)
            if folder.startswith(".") or not os.path.isdir(top):
                continue
            judged = known is not None and folder.lower() in known[0]
            count = size = 0
            for here, dirs, names in os.walk(top):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in sorted(names, key=str.lower):
                    if name.startswith(".") or (name.startswith("put_") and name.endswith("_here")):
                        continue
                    full = os.path.join(here, name)
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    rel = os.path.relpath(full, top).replace("\\", "/")
                    is_model = name.lower().endswith(MODEL_EXT)
                    files.append({"ref": "model:" + folder + "/" + rel, "folder": folder, "name": rel, "size": st.st_size,
                                  "mtime": st.st_mtime, "used": used.get(name.lower(), 0),
                                  "known": None if not judged or not is_model else _lower(rel) in known[1]})
                    count += 1
                    size += st.st_size
            folders.append({"name": folder, "count": count, "size": size})
        try:
            free = shutil.disk_usage(base).free
        except OSError:
            free = None
        return folders, files, free

    def rename(self, ref, name):
        """Another name in the same folder (the extension stays). Returns the new ref."""
        full = self.resolve(ref)
        folder, old = os.path.split(full)
        stem, ext = os.path.splitext(old)
        name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", name or "").strip()
        if name.lower().endswith(ext.lower()):
            name = name[:-len(ext)]
        name = name.strip(" .")
        if not name:
            raise ValueError("새 이름을 입력하세요")
        dst = os.path.join(folder, name + ext)
        if os.path.exists(dst) and os.path.normcase(full) != os.path.normcase(dst):
            raise ValueError("같은 이름의 파일이 이미 있습니다: " + name + ext)
        os.rename(full, dst)
        return "model:" + os.path.relpath(dst, os.path.realpath(self.root)).replace("\\", "/")

    def delete(self, ref):
        """Remove the file for good. Returns its name."""
        if os.path.isdir(self.resolve(ref, folders=True)):
            raise ValueError("폴더는 지우지 않습니다. 안의 파일을 골라 지우세요: " + ref.partition(":")[2])
        full = self.resolve(ref)
        os.remove(full)
        return os.path.basename(full)


def list_dir(path):
    """One folder of this PC for the other pane: its subfolders and files. '' is the list of drives."""
    if not path:
        if os.name != "nt":
            return {"path": "", "parent": None, "dirs": [{"name": "/", "path": "/"}], "files": []}
        import ctypes
        import string
        mask = ctypes.windll.kernel32.GetLogicalDrives()
        roots = [c + ":\\" for i, c in enumerate(string.ascii_uppercase) if mask >> i & 1]
        return {"path": "", "parent": None, "dirs": [{"name": r[:2], "path": r} for r in roots], "files": []}
    path = os.path.abspath(path)
    dirs, files = [], []
    with os.scandir(path) as entries:
        for e in entries:
            if e.name.startswith((".", "$")):
                continue
            try:
                if e.is_dir():
                    dirs.append({"name": e.name, "path": os.path.join(path, e.name)})
                elif e.is_file():
                    st = e.stat()
                    files.append({"name": e.name, "path": os.path.join(path, e.name), "size": st.st_size, "mtime": st.st_mtime})
            except OSError:
                pass
    parent = os.path.dirname(path)
    return {"path": path, "parent": "" if parent == path else parent,
            "dirs": sorted(dirs, key=lambda d: d["name"].lower()), "files": sorted(files, key=lambda f: f["name"].lower())}


class Copy:
    """One copy job: files and folders (whole) into a folder, run in a thread, asked for its progress. A file that is
    already there is left alone and reported; a file half copied when the job is stopped or fails is removed."""
    CHUNK = 8 << 20

    def __init__(self, sources, dest):
        self.sources, self.dest = sources, dest
        self.total = self.done = 0
        self.current, self.copied, self.skipped, self.errors = "", [], [], []
        self.state, self.stop = "planning", False

    def plan(self):
        """(source file, target file) for everything to copy, and the bytes."""
        pairs = []
        for src in self.sources:
            if os.path.isdir(src):
                if os.path.commonpath([os.path.realpath(self.dest), os.path.realpath(src)]) == os.path.realpath(src):
                    raise ValueError("폴더를 자기 자신 안으로 복사할 수 없습니다: " + os.path.basename(src))
                for here, dirs, names in os.walk(src):
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for n in names:
                        full = os.path.join(here, n)
                        pairs.append((full, os.path.join(self.dest, os.path.basename(src), os.path.relpath(full, src))))
            else:
                pairs.append((src, os.path.join(self.dest, os.path.basename(src))))
        self.total = sum(os.path.getsize(s) for s, _ in pairs)
        return pairs

    def run(self):
        try:
            pairs = self.plan()
            self.state = "running"
            for src, dst in pairs:
                if self.stop:
                    break
                name = os.path.relpath(dst, self.dest)
                self.current = name
                if os.path.exists(dst):
                    self.skipped.append({"name": name, "why": "이미 있음" + ("" if os.path.getsize(dst) == os.path.getsize(src) else " (크기가 다름)")})
                    self.done += os.path.getsize(src)
                    continue
                try:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(src, "rb") as fin, open(dst, "wb") as fout:
                        while not self.stop:
                            chunk = fin.read(self.CHUNK)
                            if not chunk:
                                break
                            fout.write(chunk)
                            self.done += len(chunk)
                    if self.stop:
                        os.remove(dst)
                        break
                    shutil.copystat(src, dst)
                    self.copied.append(name)
                except OSError as e:
                    self.errors.append({"name": name, "error": e.strerror or str(e)})
                    try:
                        os.remove(dst)
                    except OSError:
                        pass
            self.state = "cancelled" if self.stop else "done"
        except (OSError, ValueError) as e:
            self.errors.append({"name": self.current, "error": getattr(e, "strerror", None) or str(e)})
            self.state = "error"
        self.current = ""

    def progress(self):
        return {"state": self.state, "total": self.total, "done": self.done, "current": self.current, "dest": self.dest,
                "copied": self.copied, "skipped": self.skipped, "errors": self.errors}

