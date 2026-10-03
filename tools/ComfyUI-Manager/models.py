"""Model files: the model folder of the connected ComfyUI as this PC sees it.

Easy-Install keeps the models in a top-level folder `model` (ComfyUI\\models is only there for ComfyUI's own
subfolders), next to `output`, so the folder is found the way the scenario folder is: beside the result folder.
Nothing is read from inside the files; the list is the folders and files with their size and date, plus what the
library index knows (which result files were made with the model) and what ComfyUI lists (a model file ComfyUI
does not list is in the wrong folder or has an extension it does not load).
"""
import json
import os
import re
import shutil
import struct

MODEL_EXT = (".safetensors", ".gguf", ".ckpt", ".sft", ".pt", ".pth")


def _lower(name):
    return name.replace("\\", "/").lower()


def within(path, base):
    """Whether `path` is `base` or below it. Paths on different drives are not (os.path.commonpath raises there)."""
    base = os.path.realpath(base)
    try:
        return os.path.commonpath([os.path.realpath(path), base]) == base
    except ValueError:
        return False


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
        if not within(full, base) or not (os.path.isfile(full) or (folders and os.path.isdir(full))):
            raise ValueError("파일을 찾을 수 없습니다: " + rel)
        return full

    def inside(self, path):
        """Whether the absolute path is in the model folder."""
        base = os.path.realpath(self.root)
        return os.path.isdir(base) and within(path, base)

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


# ---------------------------------------------------------------------------------------------- #
# downloaded models: what kind a file is (from its own header) and which model folder it belongs in
# ---------------------------------------------------------------------------------------------- #
# the architectures ComfyUI-GGUF loads as diffusion models; any other GGUF is a text encoder / LLM
GGUF_IMAGE = {"flux", "sd1", "sdxl", "sd3", "aura", "ltxv", "hyvid", "wan", "hidream", "cosmos", "lumina2", "qwen_image",
              "chroma", "z_image", "pig"}
# where a kind goes: the first of these folders that the model folder has
PLACES = {"lora": ["loras"], "checkpoint": ["checkpoints"], "diffusion": ["diffusion_models", "unet"], "gguf_diffusion": ["unet", "diffusion_models"],
          "vae": ["vae"], "text": ["text_encoders", "clip"], "clip_vision": ["clip_vision"], "controlnet": ["controlnet"],
          "upscale": ["upscale_models"], "latent_upscale": ["latent_upscale_models", "upscale_models"], "embedding": ["embeddings"],
          "pulid": ["pulid"], "ipadapter": ["ipadapter"]}
KIND_KO = {"pulid": "PuLID", "ipadapter": "IP-Adapter","lora": "LoRA", "checkpoint": "체크포인트", "diffusion": "diffusion 모델", "gguf_diffusion": "diffusion 모델 (GGUF)", "vae": "VAE",
           "text": "텍스트 인코더", "clip_vision": "CLIP vision", "controlnet": "ControlNet", "upscale": "업스케일 모델",
           "latent_upscale": "latent 업스케일 모델", "embedding": "임베딩", "": "알 수 없음"}


def safetensors_header(path, limit=96 << 20):
    """The JSON header of a .safetensors file: {'__metadata__': {...}, tensor name: {...}}. None when it is not one."""
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        if not 0 < size <= limit:
            return None
        try:
            head = json.loads(f.read(size))
        except ValueError:
            return None
    return head if isinstance(head, dict) else None


def gguf_architecture(path):
    """general.architecture of a GGUF file ('' when it does not say within its first keys)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return ""
        f.read(4 + 8)      # version, tensor count
        count = struct.unpack("<Q", f.read(8))[0]

        def text():
            return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")

        def skip(kind):
            if kind == 8:
                text()
            elif kind == 9:
                inner, n = struct.unpack("<IQ", f.read(12))
                if n > 100000:
                    raise ValueError("long array")
                for _ in range(n):
                    skip(inner)
            else:
                f.read(sizes[kind])

        try:
            for _ in range(min(count, 40)):
                key, kind = text(), struct.unpack("<I", f.read(4))[0]
                if key == "general.architecture" and kind == 8:
                    return text().lower()
                skip(kind)
        except (ValueError, KeyError, struct.error):
            pass
    return ""


def kind_by_name(name):
    low = name.lower()
    if "lora" in low or "lokr" in low:
        return "lora"
    if "clip_vision" in low or "clip-vision" in low:
        return "clip_vision"
    if "controlnet" in low or "control_" in low:
        return "controlnet"
    if "vae" in low:
        return "vae"
    if "spatial-upscaler" in low or "temporal-upscaler" in low or "latent_upscal" in low:
        return "latent_upscale"
    if re.search(r"esrgan|upscal|swinir|realesr|[_\-]x[248][_\-.]|^[248]x[_\-]", low):
        return "upscale"
    if re.search(r"t5|umt5|clip_[lg]|text_encoder|text_proj|gemma|qwen_\d|llama", low):
        return "text"
    if "pulid" in low:
        return "pulid"
    if "ip-adapter" in low or "ip_adapter" in low or "ipadapter" in low:
        return "ipadapter"
    return ""


def classify(path):
    """(kind, why) of a model file. A .safetensors file tells by the names of its tensors and its metadata, a GGUF file
    by its architecture; the other formats (pickles) cannot be opened safely and go by their name."""
    name = os.path.basename(path)
    low = name.lower()
    try:
        if low.endswith(".gguf"):
            arch = gguf_architecture(path)
            if arch in GGUF_IMAGE:
                return "gguf_diffusion", "GGUF, 구조 " + arch
            if arch:
                return "text", "GGUF, 구조 " + arch
        elif low.endswith((".safetensors", ".sft")):
            head = safetensors_header(path)
            if head:
                meta = head.get("__metadata__") or {}
                keys = [k for k in head if k != "__metadata__"]
                has = lambda *parts: any(p in k for k in keys for p in parts)      # noqa: E731
                starts = lambda *parts: any(k.startswith(parts) for k in keys)      # noqa: E731
                if (str(meta.get("modelspec.architecture", "")).endswith("/lora") or "ss_network_module" in meta or "ss_network_dim" in meta
                        or has("lora_down", "lora_up", "lora_A", "lora_B", ".lora.", "lokr_", "hada_")):
                    return "lora", "LoRA 가중치 (" + (str(meta.get("ss_base_model_version") or meta.get("modelspec.architecture") or "lora_up/down 텐서")) + ")"
                if starts("control_model.", "controlnet_") or has("input_hint_block", "controlnet_cond_embedding"):
                    return "controlnet", "ControlNet 텐서"
                unet = starts("model.diffusion_model.")
                if unet and starts("first_stage_model.", "cond_stage_model.", "conditioner.", "text_encoders.", "vae."):
                    return "checkpoint", "모델 + VAE / 텍스트 인코더가 함께 든 파일"
                if unet:
                    return "diffusion", "diffusion 모델만 든 파일"
                if (has("text_model.encoder.layers", "encoder.block.", ".SelfAttention.", "language_model.")
                        or starts("model.layers.", "layers.0.self_attn")):      # CLIP text, T5, an LLM (also one that sees: gemma)
                    return "text", "텍스트 인코더 텐서"
                if has("vision_model.encoder.layers"):
                    return "clip_vision", "CLIP vision 텐서"
                if starts("decoder.") and all(k.startswith(("encoder.", "decoder.", "quant_conv", "post_quant_conv", "conv1.", "conv2.",
                                                           "per_channel_statistics", "vae.", "latents_")) for k in keys):
                    return "vae", "VAE 인코더 / 디코더 텐서"
                if kind_by_name(name) == "latent_upscale":
                    return "latent_upscale", "이름으로 판단"
                if starts("double_blocks.", "single_blocks.", "transformer_blocks.", "blocks.", "joint_blocks.", "input_blocks.", "down_blocks.",
                          "noise_refiner.", "context_refiner.", "x_embedder.", "img_in.", "patch_embedding.", "time_embedding.", "layers.",
                          "patchify_proj.", "pos_embed."):
                    return "diffusion", "diffusion 모델(트랜스포머 / UNet) 텐서"
                if len(keys) <= 4 and os.path.getsize(path) < (8 << 20):
                    return "embedding", "작은 임베딩 파일"
    except (OSError, ValueError, struct.error):
        pass
    kind = kind_by_name(name)
    if not kind and low.endswith(".ckpt"):
        kind = "checkpoint"
    return kind, ("이름으로 판단" if kind else "파일에서 종류를 알아내지 못했습니다")


# other names a downloaded folder may have for a model folder ('lora/HIGH/x' belongs in loras/HIGH)
FOLDER_ALIAS = {"lora": "loras", "checkpoint": "checkpoints", "ckpt": "checkpoints", "text_encoder": "text_encoders",
                "diffusion_model": "diffusion_models", "embedding": "embeddings", "upscale_model": "upscale_models",
                "controlnets": "controlnet", "vaes": "vae"}


def incoming(folder, model_root, limit=500):
    """The model files under a folder of this PC (three levels deep), each with its kind and the model folder it
    belongs in ('' when none fits or the kind is unknown), and whether a file of that name is already there.
    `sub` is the folders the file sits in that go along with it: what follows a folder named like a model folder
    (.../lora/HIGH/x -> HIGH), else the folders below the scanned one."""
    have = sorted((d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d)) and not d.startswith(".")), key=str.lower)
    lower = {d.lower(): d for d in have}
    named = set(lower) | set(FOLDER_ALIAS) | {"model", "models"}
    items = []
    base = os.path.abspath(folder)
    tail = [p for p in re.split(r"[\\/]+", base) if p][-3:]      # the scanned folder may itself be .../lora/HIGH
    for here, dirs, names in os.walk(base):
        depth = os.path.relpath(here, base).count(os.sep) + (0 if here == base else 1)
        dirs[:] = [] if depth >= 3 else sorted(d for d in dirs if not d.startswith((".", "$")))
        for name in sorted(names, key=str.lower):
            if not name.lower().endswith(MODEL_EXT) or len(items) >= limit:
                continue
            full = os.path.join(here, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            kind, why = classify(full)
            place = next((lower[p] for p in PLACES.get(kind, []) if p in lower), "")
            rel = os.path.relpath(full, base).replace("\\", "/")
            below = rel.split("/")[:-1]
            parts = tail + below
            at = max((i for i, p in enumerate(parts) if p.lower() in named), default=-1)
            sub = "/".join(parts[at + 1:] if at >= 0 else below)
            if not place and at >= 0:      # the kind is not known, but it sat in a folder named like a model folder
                hint = FOLDER_ALIAS.get(parts[at].lower(), parts[at].lower())
                if hint in lower:
                    place, why = lower[hint], "들어 있던 폴더 이름(" + parts[at] + ")으로 판단"
            items.append({"path": full, "name": rel, "sub": sub, "size": st.st_size, "mtime": st.st_mtime,
                          "kind": KIND_KO.get(kind, kind), "why": why, "folder": place,
                          "exists": bool(place) and os.path.exists(os.path.join(model_root, place, sub, name)),
                          "exists_flat": bool(place) and os.path.exists(os.path.join(model_root, place, name))})
    return {"path": base, "items": items, "folders": have}


class Copy:
    """One copy job: files and folders (whole) into a folder, run in a thread, asked for its progress. A file that is
    already there is left alone and reported; a file half copied when the job is stopped or fails is removed.
    With `pairs` it is a list of (file, where it goes) instead; with `move` the original is removed once it is there."""
    CHUNK = 8 << 20

    def __init__(self, sources, dest, pairs=None, move=False, tidy=None):
        self.sources, self.dest, self.pairs, self.move = sources, dest, pairs, move
        self.tidy = tidy      # after a move: the folder below which the folders left empty are removed
        self.total = self.done = 0
        self.current, self.copied, self.skipped, self.errors = "", [], [], []
        self.state, self.stop = "planning", False

    def plan(self):
        """(source file, target file) for everything to copy, and the bytes."""
        if self.pairs is not None:
            self.total = sum(os.path.getsize(s) for s, _ in self.pairs)
            return self.pairs
        pairs = []
        for src in self.sources:
            if os.path.isdir(src):
                if within(self.dest, src):
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
                    if self.move:      # on the same drive a move is a rename: at once, whatever the size
                        size = os.path.getsize(src)
                        try:
                            os.rename(src, dst)
                            self.done += size
                            self.copied.append(name)
                            continue
                        except OSError:
                            pass      # another drive: copy, then remove the original
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
                    if self.move:
                        os.remove(src)
                except OSError as e:
                    self.errors.append({"name": name, "error": e.strerror or str(e)})
                    try:
                        os.remove(dst)
                    except OSError:
                        pass
            if self.move and self.tidy:      # the folders the files were moved out of go too, once nothing is left in them
                base = os.path.realpath(self.tidy)
                for folder in sorted({os.path.dirname(os.path.realpath(s)) for s, _ in pairs}, key=len, reverse=True):
                    while folder != base and within(folder, base):
                        try:
                            os.rmdir(folder)      # only an empty folder goes
                        except OSError:
                            break
                        folder = os.path.dirname(folder)
            self.state = "cancelled" if self.stop else "done"
        except (OSError, ValueError) as e:
            self.errors.append({"name": self.current, "error": getattr(e, "strerror", None) or str(e)})
            self.state = "error"
        self.current = ""

    def progress(self):
        return {"state": self.state, "total": self.total, "done": self.done, "current": self.current, "dest": self.dest, "move": self.move,
                "copied": self.copied, "skipped": self.skipped, "errors": self.errors}

