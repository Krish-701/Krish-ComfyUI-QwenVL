"""Install a vision-capable llama-cpp-python wheel into the current ComfyUI Python."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import urllib.request

GITHUB_RELEASES = "https://api.github.com/repos/JamePeng/llama-cpp-python/releases"


def _py_tag() -> str:
    v = sys.version_info
    return f"cp{v.major}{v.minor}"


def _os_tag() -> str | None:
    system = platform.system()
    is_64 = sys.maxsize > 2**32
    if system == "Windows" and is_64:
        return "win_amd64"
    if system == "Linux" and is_64:
        return "linux_x86_64"
    if system == "Darwin" and platform.machine() == "arm64":
        return "macosx_11_0_arm64"
    return None


def _cuda_tag() -> str:
    try:
        import torch

        if torch.cuda.is_available() and torch.version.cuda:
            return "cu" + str(torch.version.cuda).replace(".", "")
    except Exception:
        pass
    return ""


def llama_cpp_vision_ok() -> bool:
    try:
        from llama_cpp import Llama  # noqa: F401
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler  # noqa: F401

        return True
    except Exception:
        try:
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler  # noqa: F401

            return True
        except Exception:
            return False


def _score_wheel(name: str, py_tag: str, os_tag: str, cuda_tag: str) -> int:
    lower = name.lower()
    if not name.endswith(".whl"):
        return -1
    if py_tag not in name or os_tag not in name:
        return -1
    if cuda_tag and cuda_tag in name:
        return 100
    if "cpu" in lower or "basic" in lower:
        return 15
    if "cu" in lower:
        return 40
    return 20


def _pick_wheel_url(releases: list, py_tag: str, os_tag: str, cuda_tag: str) -> tuple[str | None, str]:
    best_url = None
    best_name = ""
    best_score = -1
    for release in releases:
        for asset in release.get("assets") or []:
            name = asset.get("name") or ""
            url = asset.get("browser_download_url") or ""
            score = _score_wheel(name, py_tag, os_tag, cuda_tag)
            if score > best_score and url:
                best_score = score
                best_url = url
                best_name = name
                if score >= 100:
                    return best_url, best_name
    return best_url, best_name


def _pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print(f"[QwenVL] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def ensure_llama_cpp_vision(force: bool = False) -> bool:
    """Install JamePeng vision llama-cpp-python into this interpreter if needed."""
    if llama_cpp_vision_ok() and not force:
        return True

    py_tag = _py_tag()
    os_tag = _os_tag()
    cuda_tag = _cuda_tag()
    print("[QwenVL] llama_cpp vision backend missing — installing into ComfyUI Python:")
    print(f"[QwenVL]   python = {sys.executable}")
    print(f"[QwenVL]   tags   = {py_tag} {os_tag} {cuda_tag or 'cpu/auto'}")

    if not os_tag:
        print(f"[QwenVL] Unsupported platform: {platform.system()} {platform.machine()}")
        return False

    req = urllib.request.Request(GITHUB_RELEASES, headers={"User-Agent": "Krish-ComfyUI-QwenVL"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            releases = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"[QwenVL] Could not fetch JamePeng releases: {exc}")
        return False

    url, name = _pick_wheel_url(releases, py_tag, os_tag, cuda_tag)
    if not url:
        print(f"[QwenVL] No matching wheel for {py_tag} {os_tag} {cuda_tag}.")
        print("[QwenVL] See https://github.com/JamePeng/llama-cpp-python/releases")
        return False

    print(f"[QwenVL] Installing wheel: {name}")
    try:
        _pip("install", "diskcache>=5.6.2", "jinja2>=3.1.0")
        _pip("install", "--upgrade", "--force-reinstall", "--no-deps", "--no-cache-dir", url)
    except subprocess.CalledProcessError as exc:
        print(f"[QwenVL] pip install failed: {exc}")
        return False

    import importlib

    importlib.invalidate_caches()
    ok = llama_cpp_vision_ok()
    print("[QwenVL] llama_cpp vision install " + ("OK" if ok else "FAILED — restart ComfyUI and retry"))
    return ok
