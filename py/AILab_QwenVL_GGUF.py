# ComfyUI-QwenVL (GGUF)
# GGUF nodes powered by llama.cpp for Qwen-VL models, including Qwen3-VL and Qwen2.5-VL.
# Provides vision-capable GGUF inference and prompt execution.
#
# Models are loaded via llama-cpp-python and configured through gguf_models.json.
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import base64
import gc
import io
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

import folder_paths
from AILab_OutputCleaner import OutputCleanConfig, clean_model_output, split_response_and_reasoning

from AILab_Utils import (
    PLUGIN_DIR,
    GGUF_CONFIG_PATH,
    CUSTOM_MODELS_PATH,
    ENABLE_DISABLE,
    TOKEN_PRESETS,
    parse_token_preset,
    safe_dirname,
    resolve_base_dir,
    find_local_gguf_file,
    model_name_to_filename_candidates,
    filter_kwargs_for_callable,
    load_system_prompts,
    parse_gguf_repos,
    tensor_to_base64_png,
    sample_video_frames,
    resolve_safe_video_max_side,
    collect_labeled_images,
    collect_labeled_videos,
    collect_audio_notes,
    build_media_user_prompt,
    compose_system_prompt,
    resolve_user_prompt,
    format_history_block,
    flag_enabled,
)

_prompts = load_system_prompts()
PRESET_PROMPTS = _prompts["preset_prompts"]
SYSTEM_PROMPTS = _prompts["qwenvl_prompts"]


@dataclass(frozen=True)
class GGUFVLResolved:
    display_name: str
    repo_id: str | None
    alt_repo_ids: list[str]
    author: str | None
    repo_dirname: str
    model_filename: str
    mmproj_filename: str | None
    context_length: int
    image_max_tokens: int
    n_batch: int
    gpu_layers: int
    top_k: int
    pool_size: int


def _load_gguf_vl_catalog():
    flattened: dict[str, dict] = {}
    seen_display_names: set[str] = set()
    base_dir = "LLM/GGUF"

    if GGUF_CONFIG_PATH.exists():
        try:
            with open(GGUF_CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            base_dir = data.get("base_dir") or base_dir
            for key in ["qwenVL_model", "Qwen_model"]:
                repos = data.get(key) or {}
                if isinstance(repos, dict):
                    parse_gguf_repos(repos, flattened, seen_display_names)
            legacy_models = data.get("models") or {}
            if isinstance(legacy_models, dict):
                for name, entry in legacy_models.items():
                    if isinstance(entry, dict):
                        flattened[name] = entry
        except Exception as exc:
            print(f"[QwenVL] gguf_models.json load failed: {exc}")

    # Merge custom_models.json (central custom config)
    if CUSTOM_MODELS_PATH.exists():
        try:
            with open(CUSTOM_MODELS_PATH, "r", encoding="utf-8") as fh:
                custom_data = json.load(fh) or {}
            for key in ["gguf_models", "gguf_vl_models", "gguf_text_models"]:
                repos = custom_data.get(key) or {}
                if isinstance(repos, dict) and repos:
                    parse_gguf_repos(repos, flattened, seen_display_names, overwrite_existing=True)
        except Exception as exc:
            print(f"[QwenVL] custom_models.json (GGUF) skipped: {exc}")

    return {"base_dir": base_dir, "models": flattened}


def reload_gguf_vl_catalog():
    global GGUF_VL_CATALOG
    GGUF_VL_CATALOG = _load_gguf_vl_catalog()


GGUF_VL_CATALOG = _load_gguf_vl_catalog()


GGUF_VL_CATALOG = _load_gguf_vl_catalog()


def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)

    allowed: set[str] = set()
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(p.name)
    return {k: v for k, v in kwargs.items() if k in allowed}


# Aliases for shared utilities from AILab_Utils
_tensor_to_base64_png = tensor_to_base64_png
_resolve_safe_video_max_side = resolve_safe_video_max_side
_sample_video_frames = sample_video_frames


def _pick_device(device_choice: str) -> str:
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_choice.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if device_choice == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _download_single_file(repo_ids: list[str], filename: str, target_path: Path):
    if target_path.exists():
        print(f"[QwenVL] Using cached file: {target_path}")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for repo_id in repo_ids:
        print(f"[QwenVL] Downloading {filename} from {repo_id} -> {target_path}")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(target_path.parent),
                local_dir_use_symlinks=False,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                downloaded_path.replace(target_path)
            if target_path.exists():
                print(f"[QwenVL] Download complete: {target_path}")
            break
        except Exception as exc:
            last_exc = exc
            print(f"[QwenVL] hf_hub_download failed from {repo_id}: {exc}")
    else:
        raise FileNotFoundError(f"[QwenVL] Download failed for {filename}: {last_exc}")

    if not target_path.exists():
        raise FileNotFoundError(f"[QwenVL] File not found after download: {target_path}")


def _resolve_model_entry(model_name: str) -> GGUFVLResolved:
    catalog = _load_gguf_vl_catalog()
    all_models = catalog.get("models") or {}
    entry = all_models.get(model_name) or {}
    if not entry:
        wanted = model_name_to_filename_candidates(model_name)
        for candidate in all_models.values():
            filename = candidate.get("filename")
            if filename and Path(filename).name in wanted:
                entry = candidate
                break

    repo_id = entry.get("repo_id")
    alt_repo_ids = entry.get("alt_repo_ids") or []

    author = entry.get("author") or entry.get("publisher")
    repo_dirname = entry.get("repo_dirname") or (repo_id.split("/")[-1] if isinstance(repo_id, str) and "/" in repo_id else model_name)

    model_filename = entry.get("filename")
    mmproj_filename = entry.get("mmproj_filename")

    if not model_filename:
        raise ValueError(f"[QwenVL] gguf_models.json entry missing 'filename' for: {model_name}")

    def _int(name: str, default: int) -> int:
        value = entry.get(name, default)
        try:
            return int(value)
        except Exception:
            return default

    return GGUFVLResolved(
        display_name=model_name,
        repo_id=repo_id,
        alt_repo_ids=[str(x) for x in alt_repo_ids if x],
        author=str(author) if author else None,
        repo_dirname=safe_dirname(str(repo_dirname)),
        model_filename=str(model_filename),
        mmproj_filename=str(mmproj_filename) if mmproj_filename else None,
        context_length=_int("context_length", 8192),
        image_max_tokens=_int("image_max_tokens", 4096),
        n_batch=_int("n_batch", 512),
        gpu_layers=_int("gpu_layers", -1),
        top_k=_int("top_k", 0),
        pool_size=_int("pool_size", 4194304),
    )


class QwenVLGGUFBase:
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        self.conversation_memory = []
        self.last_response = ""
        self.last_reasoning = ""

    def clear(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_backend(self):
        try:
            from llama_cpp import Llama  # noqa: F401
            return
        except Exception:
            pass

        print("[QwenVL] llama_cpp not found in this ComfyUI Python. Auto-installing vision wheel...")
        try:
            from llama_cpp_install import ensure_llama_cpp_vision

            installed = ensure_llama_cpp_vision()
        except Exception as exc:
            raise RuntimeError(
                "[QwenVL] llama_cpp auto-install failed. "
                "Run docs/install_llama_cpp_vision.py with ComfyUI's python_embeded/python.exe. "
                f"Details: {exc}"
            ) from exc

        if not installed:
            raise RuntimeError(
                "[QwenVL] llama_cpp is not available in this ComfyUI Python.\n"
                f"  Interpreter: {sys.executable}\n"
                "  The node tried to auto-install a vision wheel and it did not succeed.\n"
                "  Close ComfyUI, then run:\n"
                f"    {sys.executable} custom_nodes/Krish-ComfyUI-QwenVL/docs/install_llama_cpp_vision.py"
            )

        try:
            from llama_cpp import Llama  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "[QwenVL] llama_cpp installed but still cannot import. Fully restart ComfyUI, then retry."
            ) from exc

    def _load_model(
        self,
        model_name: str,
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
    ):
        self._load_backend()

        resolved = _resolve_model_entry(model_name)
        base_dir = resolve_base_dir(GGUF_VL_CATALOG.get("base_dir") or "LLM/GGUF")

        author_dir = safe_dirname(resolved.author or "")
        repo_dir = safe_dirname(resolved.repo_dirname)
        target_dir = base_dir / author_dir / repo_dir

        existing_model = find_local_gguf_file(resolved.model_filename, target_dir)
        if existing_model:
            model_path = existing_model
        else:
            model_path = target_dir / Path(resolved.model_filename).name

        existing_mmproj = find_local_gguf_file(resolved.mmproj_filename, target_dir, allow_recursive=False)
        if existing_mmproj:
            mmproj_path = existing_mmproj
        else:
            mmproj_path = target_dir / Path(resolved.mmproj_filename).name if resolved.mmproj_filename else None

        repo_ids: list[str] = []
        if resolved.repo_id:
            repo_ids.append(resolved.repo_id)
        repo_ids.extend(resolved.alt_repo_ids)

        if not model_path.exists():
            if not repo_ids:
                raise FileNotFoundError(f"[QwenVL] GGUF model not found locally and no repo_id provided: {model_path}")
            _download_single_file(repo_ids, resolved.model_filename, model_path)

        # Smart mmproj resolution:
        # 1. If configured filename exists in model's directory, use it.
        # 2. If configured filename does not exist, try downloading it from model repo.
        # 3. If no filename configured or download failed, search model's own target_dir locally for *mmproj*.gguf.
        # 4. If still not found, search remote repo_ids for *mmproj*.gguf, download, and use.
        if mmproj_path is not None and not mmproj_path.exists():
            if repo_ids:
                try:
                    _download_single_file(repo_ids, resolved.mmproj_filename, mmproj_path)
                except Exception as exc:
                    print(f"[QwenVL] Configured mmproj download failed ({resolved.mmproj_filename}): {exc}")

        if mmproj_path is None or not mmproj_path.exists():
            # Check ONLY model's own target_dir for any matching mmproj (never hijack from other model folders)
            local_mmprojs = []
            if target_dir.exists():
                local_mmprojs = list(target_dir.glob("*mmproj*.gguf"))
            if local_mmprojs:
                model_stem = Path(resolved.model_filename).stem.lower()
                matched_local = None
                for lm in local_mmprojs:
                    lm_name = lm.name.lower()
                    if "q8" in model_stem and "q8" in lm_name:
                        matched_local = lm
                        break
                    elif "f16" in lm_name or "bf16" in lm_name:
                        if matched_local is None:
                            matched_local = lm
                mmproj_path = matched_local or local_mmprojs[0]
                print(f"[QwenVL] Auto-detected local visual projector: {mmproj_path.name}")

        if (mmproj_path is None or not mmproj_path.exists()) and repo_ids:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                for rid in repo_ids:
                    repo_files = api.list_repo_files(repo_id=rid)
                    mmproj_files = [f for f in repo_files if "mmproj" in f.lower() and f.endswith(".gguf")]
                    if mmproj_files:
                        model_stem = Path(resolved.model_filename).stem.lower()
                        chosen_mmproj = mmproj_files[0]
                        for mf in mmproj_files:
                            mf_lower = mf.lower()
                            if "q8" in model_stem and "q8" in mf_lower:
                                chosen_mmproj = mf
                                break
                            elif "f16" in mf_lower or "bf16" in mf_lower:
                                chosen_mmproj = mf

                        target_mmproj_file = target_dir / Path(chosen_mmproj).name
                        print(f"[QwenVL] Auto-discovering visual projector from {rid}: {chosen_mmproj}")
                        _download_single_file([rid], chosen_mmproj, target_mmproj_file)
                        if target_mmproj_file.exists():
                            mmproj_path = target_mmproj_file
                            print(f"[QwenVL] Successfully auto-downloaded visual projector: {mmproj_path.name}")
                            break
            except Exception as exc:
                print(f"[QwenVL] Remote mmproj auto-discovery skipped: {exc}")

        device_kind = _pick_device(device)

        n_ctx = int(ctx) if ctx is not None else resolved.context_length
        n_batch_val = int(n_batch) if n_batch is not None else resolved.n_batch
        top_k_val = int(top_k) if top_k is not None else resolved.top_k
        pool_size_val = int(pool_size) if pool_size is not None else resolved.pool_size

        if device_kind == "cuda":
            n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
        else:
            n_gpu_layers = 0

        img_max = int(image_max_tokens) if image_max_tokens is not None else resolved.image_max_tokens

        has_mmproj = mmproj_path is not None and mmproj_path.exists()

        signature = (
            str(model_path),
            str(mmproj_path) if has_mmproj else "",
            n_ctx,
            n_batch_val,
            n_gpu_layers,
            img_max,
            top_k_val,
            pool_size_val,
        )
        if self.llm is not None and self.current_signature == signature:
            return

        self.clear()

        from llama_cpp import Llama

        self.chat_handler = None
        if has_mmproj:
            handler_cls = None
            try:
                from llama_cpp.llama_chat_format import Qwen3VLChatHandler

                handler_cls = Qwen3VLChatHandler
            except ImportError:
                try:
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler

                    handler_cls = Qwen25VLChatHandler
                except ImportError:
                    raise RuntimeError(
                        "[QwenVL] Missing Qwen VL chat handler in llama_cpp. Install the correct fork/wheel. See docs/GGUF_MANUAL_INSTALL.md"
                    )

            mmproj_kwargs = {
                "clip_model_path": str(mmproj_path),
                "image_max_tokens": img_max,
                "force_reasoning": True,
                "verbose": False,
            }
            mmproj_kwargs = filter_kwargs_for_callable(getattr(handler_cls, "__init__", handler_cls), mmproj_kwargs)
            if "image_max_tokens" not in mmproj_kwargs:
                print(
                    "[QwenVL] Warning: installed llama_cpp chat handler does not support image_max_tokens; "
                    "image token budget will be controlled by ctx only."
                )
            self.chat_handler = handler_cls(**mmproj_kwargs)

        llm_kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch_val,
            "swa_full": True,
            "verbose": False,
            "pool_size": pool_size_val,
            "top_k": top_k_val,
        }
        if has_mmproj and self.chat_handler is not None:
            llm_kwargs["chat_handler"] = self.chat_handler
            llm_kwargs["image_min_tokens"] = 1024
            llm_kwargs["image_max_tokens"] = img_max

        print(f"[QwenVL] Loading GGUF: {model_path.name} (device={device_kind}, gpu_layers={n_gpu_layers}, ctx={n_ctx})")
        llm_kwargs_filtered = filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), llm_kwargs)
        if has_mmproj and self.chat_handler is not None and "chat_handler" not in llm_kwargs_filtered:
            print(
                "[QwenVL] Warning: installed llama_cpp Llama() does not accept chat_handler; images will be ignored. "
                "Update llama-cpp-python to a multimodal-capable build."
            )
        if device_kind == "cuda" and n_gpu_layers == 0:
            print("[QwenVL] Warning: device=cuda selected but n_gpu_layers=0; model will run on CPU.")
        self.llm = Llama(**llm_kwargs_filtered)
        self.current_signature = signature

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        labeled_b64: list | None = None,
        stream: bool = False,
    ) -> str:
        labeled = labeled_b64 or []
        if not labeled and images_b64:
            labeled = [("image", "Image", img) for img in images_b64 if img]

        if labeled and self.chat_handler is not None:
            content = []
            for _kind, label, img in labeled:
                if not img:
                    continue
                content.append({"type": "text", "text": f"[{label}]"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
            content.append({"type": "text", "text": user_prompt})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            if labeled and self.chat_handler is None:
                print("[QwenVL] Warning: Image provided but model has no visual projector (mmproj); running in text-only mode.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        start = time.perf_counter()
        completion_kwargs = dict(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repeat_penalty=float(repetition_penalty),
            seed=int(seed),
            stop=["<|im_end|>", "<|im_start|>"],
        )
        extra_reasoning = ""
        if stream:
            pieces = []
            reasoning_pieces = []
            for part in self.llm.create_chat_completion(**completion_kwargs, stream=True):
                delta = (part.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    pieces.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning_pieces.append(delta["reasoning_content"])
            elapsed = max(time.perf_counter() - start, 1e-6)
            print(f"[QwenVL] Stream complete in {elapsed:.2f}s")
            raw = "".join(pieces)
            extra_reasoning = "".join(reasoning_pieces)
        else:
            result = self.llm.create_chat_completion(**completion_kwargs)
            elapsed = max(time.perf_counter() - start, 1e-6)

            usage = result.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(completion_tokens, int) and completion_tokens > 0:
                tok_s = completion_tokens / elapsed
                if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                    print(
                        f"[QwenVL] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, "
                        f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s"
                    )
                else:
                    print(f"[QwenVL] Tokens: completion={completion_tokens}, time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")

            message = (result.get("choices") or [{}])[0].get("message", {}) or {}
            raw = message.get("content", "")
            extra_reasoning = message.get("reasoning_content") or ""
        cleaned, reasoning = split_response_and_reasoning(str(raw or ""), extra_reasoning, mode="text")
        return cleaned, reasoning

    def run(
        self,
        model_name: str,
        preset_prompt: str,
        custom_prompt: str,
        image,
        video,
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        keep_model_loaded: bool,
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
        video_frame_size: str = "auto",
    ):
        torch.manual_seed(int(seed))

        prompt = resolve_user_prompt(preset_prompt, custom_prompt, "", SYSTEM_PROMPTS)

        resolved = _resolve_model_entry(model_name)
        effective_ctx = int(ctx) if ctx is not None else resolved.context_length

        images_b64: list[str] = []
        if image is not None:
            img = _tensor_to_base64_png(image, max_side=1280)
            if img:
                images_b64.append(img)
        if video is not None:
            video_max_side = _resolve_safe_video_max_side(
                video,
                frame_count=int(frame_count),
                ctx=effective_ctx,
                video_frame_size=video_frame_size,
            )
            for frame in _sample_video_frames(video, int(frame_count)):
                img = _tensor_to_base64_png(frame, max_side=video_max_side)
                if img:
                    images_b64.append(img)

        try:
            self._load_model(
                model_name=model_name,
                device=device,
                ctx=ctx,
                n_batch=n_batch,
                gpu_layers=gpu_layers,
                image_max_tokens=image_max_tokens,
                top_k=top_k,
                pool_size=pool_size,
            )
            if images_b64 and self.chat_handler is None:
                print("[QwenVL] Warning: images provided but this model entry has no mmproj_file; images will be ignored")
            text, reasoning = self._invoke(
                system_prompt=compose_system_prompt(),
                user_prompt=prompt,
                images_b64=images_b64 if self.chat_handler is not None else [],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
            return (text, reasoning)
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_QwenVL_GGUF(QwenVLGGUFBase):
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        catalog = _load_gguf_vl_catalog()
        all_models = catalog.get("models") or {}
        model_keys = sorted(list(all_models.keys())) or ["(edit gguf_models.json)"]
        default_model = model_keys[0]

        prompts = PRESET_PROMPTS or ["none"]
        default_prompt = "none" if "none" in prompts else prompts[0]

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model}),
                "preset_prompt": (prompts, {"default": default_prompt}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 16384, "min": 64, "max": 65536}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "reasoning_content")
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(
        self,
        model_name,
        preset_prompt,
        custom_prompt,
        max_tokens,
        keep_model_loaded,
        seed,
        image=None,
        video=None,
    ):
        return self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            frame_count=16,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device="auto",
            ctx=None,
            n_batch=None,
            gpu_layers=None,
            image_max_tokens=None,
            top_k=None,
            pool_size=None,
            video_frame_size="auto",
        )


class AILab_QwenVL_GGUF_Advanced(QwenVLGGUFBase):
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        catalog = _load_gguf_vl_catalog()
        all_models = catalog.get("models") or {}
        model_keys = sorted(list(all_models.keys())) or ["(edit gguf_models.json)"]
        default_model = model_keys[0]

        prompts = PRESET_PROMPTS or ["none"]
        default_prompt = "none" if "none" in prompts else prompts[0]

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model}),
                "device": (device_options, {"default": "auto"}),
                "preset_prompt": (prompts, {"default": default_prompt}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "token_preset": (TOKEN_PRESETS, {"default": "16k"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "is_memory": (ENABLE_DISABLE, {"default": "disable"}),
                "is_tools_in_sys_prompt": (ENABLE_DISABLE, {"default": "disable"}),
                "is_locked": (ENABLE_DISABLE, {"default": "disable"}),
                "main_brain": (ENABLE_DISABLE, {"default": "enable"}),
                "max_length": ("INT", {"default": 16384, "min": 64, "max": 65536}),
                "max_tokens": ("INT", {"default": 16384, "min": 64, "max": 65536}),
                "conversation_rounds": ("INT", {"default": 1, "min": 1, "max": 32}),
                "historical_record": ("STRING", {"default": "", "multiline": True}),
                "is_enable": ("BOOLEAN", {"default": True}),
                "stream": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64}),
                "video_frame_size": (["auto", "384", "448", "512", "768", "original"], {"default": "auto"}),
                "ctx": ("INT", {"default": 65536, "min": 1024, "max": 262144, "step": 512}),
                "n_batch": ("INT", {"default": 512, "min": 64, "max": 32768, "step": 64}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
                "image_max_tokens": ("INT", {"default": 16384, "min": 256, "max": 65536, "step": 256}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 32768}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "video": ("IMAGE",),
                "video1": ("IMAGE",),
                "video2": ("IMAGE",),
                "video3": ("IMAGE",),
                "audio": ("AUDIO",),
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "audio3": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "reasoning_content")
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(
        self,
        model_name,
        device,
        preset_prompt,
        system_prompt,
        user_prompt,
        custom_prompt,
        token_preset,
        temperature,
        is_memory,
        is_tools_in_sys_prompt,
        is_locked,
        main_brain,
        max_length,
        max_tokens,
        conversation_rounds,
        historical_record,
        is_enable,
        stream,
        top_p,
        repetition_penalty,
        frame_count,
        video_frame_size,
        ctx,
        n_batch,
        gpu_layers,
        image_max_tokens,
        top_k,
        pool_size,
        keep_model_loaded,
        seed,
        image=None,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        video=None,
        video1=None,
        video2=None,
        video3=None,
        audio=None,
        audio1=None,
        audio2=None,
        audio3=None,
    ):
        if not is_enable:
            return ("", "")
        if flag_enabled(is_locked) and self.last_response:
            return (self.last_response, self.last_reasoning)

        labeled_images = collect_labeled_images(
            image=image, image1=image1, image2=image2, image3=image3, image4=image4, image5=image5
        )
        labeled_videos = collect_labeled_videos(video=video, video1=video1, video2=video2, video3=video3)
        audio_notes = collect_audio_notes(audio=audio, audio1=audio1, audio2=audio2, audio3=audio3)
        user_text = resolve_user_prompt(preset_prompt, custom_prompt, user_prompt, SYSTEM_PROMPTS)
        history = self.conversation_memory if flag_enabled(is_memory) else []
        history_block = format_history_block(historical_record, history, conversation_rounds)
        prompt_text = build_media_user_prompt(
            (history_block + "\n\n" + user_text).strip() if history_block else user_text,
            [lab for lab, _ in labeled_images],
            [lab for lab, _ in labeled_videos],
            audio_notes,
        )
        sys_text = compose_system_prompt(
            system_prompt=system_prompt,
            main_brain=main_brain,
            is_tools_in_sys_prompt=is_tools_in_sys_prompt,
        )
        token_budget = max(parse_token_preset(token_preset), int(max_length or 0), int(max_tokens or 0))
        torch.manual_seed(int(seed))
        resolved = _resolve_model_entry(model_name)
        effective_ctx = max(int(ctx) if ctx is not None else resolved.context_length, token_budget)

        labeled_b64 = []
        for label, tensor in labeled_images:
            img = _tensor_to_base64_png(tensor, max_side=1280)
            if img:
                labeled_b64.append(("image", label, img))
        for label, tensor in labeled_videos:
            video_max_side = _resolve_safe_video_max_side(
                tensor,
                frame_count=int(frame_count),
                ctx=effective_ctx,
                video_frame_size=video_frame_size,
            )
            for i, frame in enumerate(_sample_video_frames(tensor, int(frame_count)), start=1):
                img = _tensor_to_base64_png(frame, max_side=video_max_side)
                if img:
                    labeled_b64.append(("video", f"{label} frame {i}", img))

        try:
            self._load_model(
                model_name=model_name,
                device=device,
                ctx=effective_ctx,
                n_batch=n_batch,
                gpu_layers=gpu_layers,
                image_max_tokens=max(int(image_max_tokens or 0), min(token_budget, 65536)),
                top_k=top_k,
                pool_size=pool_size,
            )
            if labeled_b64 and self.chat_handler is None:
                print("[QwenVL] Warning: media provided but this model has no mmproj; images/video frames will be ignored")
            text, reasoning = self._invoke(
                system_prompt=sys_text,
                user_prompt=prompt_text,
                images_b64=[],
                max_tokens=token_budget,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                labeled_b64=labeled_b64 if self.chat_handler is not None else [],
                stream=bool(stream),
            )
            self.last_response = text
            self.last_reasoning = reasoning
            if flag_enabled(is_memory):
                self.conversation_memory.append({"user": user_text, "assistant": text})
            return (text, reasoning)
        finally:
            if not keep_model_loaded:
                self.clear()


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_GGUF": AILab_QwenVL_GGUF,
    "AILab_QwenVL_GGUF_Advanced": AILab_QwenVL_GGUF_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_GGUF": "QwenVL (GGUF)",
    "AILab_QwenVL_GGUF_Advanced": "QwenVL Advanced (GGUF)",
}
