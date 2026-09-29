"""
ComfyUI-WanChain
Wan 2.2 First-Last-Frame 구간 연결용 보조 노드.

- WanChain Load Image (Optional): 이미지를 비워둘 수 있는 Load Image. 비어 있으면 IMAGE=None, has_image=False.
- WanChain Collect Segments: 구간 1부터 '첫 이미지가 있는 구간'까지만 실행하고 결과를 이어 붙임.
  (lazy 입력이라 쓰지 않는 구간의 샘플러는 아예 실행되지 않음)
"""
import os
import hashlib

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
import comfy.utils

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # 구버전 ComfyUI
    from comfy_execution.graph import ExecutionBlocker

NONE = "(none)"
MAX_SEG = 5


def _input_files():
    d = folder_paths.get_input_directory()
    files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    try:
        files = folder_paths.filter_files_content_types(files, ["image"])
    except Exception:
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
        files = [f for f in files if f.lower().endswith(exts)]
    return sorted(files)


class WanChainLoadImageOptional:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ([NONE] + _input_files(), {"image_upload": True})}}

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "has_image")
    FUNCTION = "load"
    CATEGORY = "WanChain"

    def load(self, image):
        if not image or image == NONE:
            return (None, False)
        path = folder_paths.get_annotated_filepath(image)
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return (torch.from_numpy(arr)[None,], True)

    @classmethod
    def IS_CHANGED(cls, image):
        if not image or image == NONE:
            return NONE
        m = hashlib.sha256()
        with open(folder_paths.get_annotated_filepath(image), "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not image or image == NONE:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return f"이미지 파일을 찾을 수 없습니다: {image}"
        return True


class WanChainCollect:
    @classmethod
    def INPUT_TYPES(cls):
        req = {f"has{i}": ("BOOLEAN", {"forceInput": True}) for i in range(1, MAX_SEG + 1)}
        req["drop_duplicate_first_frame"] = ("BOOLEAN", {"default": True})
        opt = {f"seg{i}": ("IMAGE", {"lazy": True}) for i in range(1, MAX_SEG + 1)}
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("IMAGE",) + ("IMAGE",) * MAX_SEG + ("INT",)
    RETURN_NAMES = ("merged",) + tuple(f"seg{i}" for i in range(1, MAX_SEG + 1)) + ("count",)
    FUNCTION = "collect"
    CATEGORY = "WanChain"

    @staticmethod
    def _count(kw):
        n = 0
        for i in range(1, MAX_SEG + 1):
            if kw.get(f"has{i}"):
                n += 1
            else:
                break
        return n

    def check_lazy_status(self, **kw):
        n = self._count(kw)
        return [f"seg{i}" for i in range(1, n + 1) if kw.get(f"seg{i}") is None]

    def collect(self, drop_duplicate_first_frame=True, **kw):
        n = self._count(kw)
        if n == 0:
            raise ValueError("구간 1의 첫 이미지를 넣어주세요.")
        segs = [kw[f"seg{i}"] for i in range(1, n + 1)]
        h, w = segs[0].shape[1], segs[0].shape[2]
        parts = []
        for i, t in enumerate(segs):
            if t.shape[1] != h or t.shape[2] != w:
                t = comfy.utils.common_upscale(t.movedim(-1, 1), w, h, "bilinear", "center").movedim(1, -1)
            if i > 0 and drop_duplicate_first_frame and t.shape[0] > 1:
                t = t[1:]
            parts.append(t)
        merged = torch.cat(parts, dim=0)
        outs = [kw[f"seg{i}"] if i <= n else ExecutionBlocker(None) for i in range(1, MAX_SEG + 1)]
        print(f"[WanChain] {n}개 구간 실행, 합친 프레임 {merged.shape[0]}장")
        return (merged, *outs, n)


NODE_CLASS_MAPPINGS = {
    "WanChainLoadImageOptional": WanChainLoadImageOptional,
    "WanChainCollect": WanChainCollect,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanChainLoadImageOptional": "WanChain Load Image (Optional)",
    "WanChainCollect": "WanChain Collect Segments",
}
