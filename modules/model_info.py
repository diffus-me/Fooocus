#!/usr/bin/env python3

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Literal

import requests
from pydantic import BaseModel


def _get_binary_path(sha256: str) -> Path:
    from settings import settings

    sha256 = sha256.lower()
    return Path(settings.binary_dir) / sha256[:2] / sha256[2:4] / sha256[4:6] / sha256


def _download_file(
    url: str,
    headers: dict[str, str] | None,
    params: dict[str, str] | None,
    path: Path,
    chunk_size: int,
) -> str:
    hash_calculator = hashlib.sha256()
    response = requests.get(url, headers=headers, params=params, stream=True)
    response.raise_for_status()
    with path.open("wb") as fp:
        for data in response.iter_content(chunk_size=chunk_size):
            hash_calculator.update(data)
            fp.write(data)

    return hash_calculator.hexdigest()


def _download_file_with_retry(
    url: str,
    headers: dict[str, str] | None,
    params: dict[str, str] | None,
    path: Path,
    sha256: str | None,
    *,
    chunk_size: int = 512 * 1024 * 1024,
    retries: int = 3,
) -> str:
    error: Exception | None = None
    local_sha256 = None

    for _ in range(retries):
        try:
            local_sha256 = _download_file(url, headers, params, path, chunk_size)
        except requests.HTTPError as _error:
            if _error.response.status_code == 404:
                raise

            error = _error
            continue
        except Exception as _error:
            error = _error
            continue

        if sha256 is None or local_sha256 == sha256:
            return local_sha256

    if error is not None:
        raise error

    raise ValueError(f"Download sha256 mismatch, target sha256: '{sha256}', downloaded sha256: '{local_sha256}'")


class ModelInfo(BaseModel):
    model_type: Literal["checkpoint", "embedding", "lora"]
    name: str
    sha256: str
    url: str
    base: Literal[
        "SDXL",
        "SDV1",
        "Playground v2",
        "Pony",
        "SD 3",
        "SD3",
        "SDXL 1.0",
        "SDXL Distilled",
        "SDXL Hyper",
        "SDXL Turbo",
        "Flux.1 D",
        "Flux.1 S",
        "Flux",
        "SSD",
    ] | None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @property
    def name_for_extra(self) -> str:
        return os.path.splitext(self.name)[0]

    @property
    def filename(self) -> str:
        return str(_get_binary_path(self.sha256))

    @property
    def is_safetensors(self) -> bool:
        return os.path.splitext(self.name)[-1].lower() == ".safetensors"

    def check_file_existence(self) -> None:
        assert os.path.exists(self.filename), f"Model '{self.name}' does not exist"

    def download_model(self) -> None:
        from settings import settings

        path = Path(self.filename)
        if path.exists():
            return

        print(f"Start downloading model '{self.name}' from url '{self.url}' to '{path}'")
        path.parent.mkdir(exist_ok=True, parents=True)
        try:
            _download_file_with_retry(
                self.url,
                None,
                {"token": settings.civitai_token} if "civitai" in self.url else None,
                path,
                self.sha256,
            )
        except Exception:
            path.unlink()
            print(f"Model '{self.name}' downloaded failed")
            return

        print(f"Model '{self.name}' downloaded successfully")


class AllModelInfo:
    def __init__(self, path: str) -> None:
        with open(path) as fp:
            source = json.load(fp)

        self._models: list[ModelInfo] = []
        self.checkpoint_models: dict[str, ModelInfo] = {}
        self.lora_models: dict[str, ModelInfo] = {}
        self.embedding_models: dict[str, ModelInfo] = {}

        for item in source:
            model = ModelInfo(**item)
            self._models.append(model)
            match model.model_type:
                case "checkpoint":
                    self.checkpoint_models[model.name] = model
                case "lora":
                    self.lora_models[model.name] = model
                case "embedding":
                    self.embedding_models[model.name] = model
                    self.embedding_models[model.name_for_extra] = model
                case _:
                    raise ValueError(f"Unknown model type: '{model.model_type}'")

    def check_file_existence(self) -> None:
        for model in self._models:
            model.check_file_existence()

    def download_models(self) -> None:
        for model in self._models:
            model.download_model()

    def get_model(self, name: str) -> ModelInfo | None:
        if name in self.checkpoint_models:
            return self.checkpoint_models[name]
        if name in self.lora_models:
            return self.lora_models[name]
        if name in self.embedding_models:
            return self.embedding_models[name]
        return None


_all_model_info: AllModelInfo | None = None


def get_all_model_info() -> AllModelInfo:
    global _all_model_info

    if _all_model_info is None:
        from settings import settings

        _all_model_info = AllModelInfo(settings.models_db_path)

    return _all_model_info


def download_models() -> None:
    get_all_model_info().download_models()


def update_model_list() -> None:
    from modules import config

    all_model_info = get_all_model_info()

    config.model_filenames = sorted(
        model.name for model in all_model_info.checkpoint_models.values() if model.base in set(["SDXL", "Flux.1 D", "Flux.1 S", "Flux", "SSD", "SD3"])
    )
    config.refiner_model_filenames = sorted(all_model_info.checkpoint_models.keys())
    config.lora_filenames = sorted(all_model_info.lora_models.keys())


_KEYS = ["vae_approx", "upscale_models", "inpaint", "controlnet", "clip_vision"]


def _sync_fooocus_expansion(models_dir: Path) -> Path:
    _SUBDIR = "prompt_expansion/fooocus_expansion/"

    source_dir = Path("./models/") / _SUBDIR
    target_dir = models_dir / _SUBDIR
    target_dir.mkdir(exist_ok=True, parents=True)

    for file in source_dir.iterdir():
        if not file.is_file():
            continue

        target_path = target_dir / file.name
        if target_path.exists():
            continue

        shutil.copy(file, target_path)

    return target_dir


def overwrite_config(config_path: str) -> None:
    print(f"Overwriting config file '{config_path}'")

    models_dir = Path(os.environ["MODELS_DIR"])
    configs = {}

    for key in _KEYS:
        path = models_dir / key
        path.mkdir(exist_ok=True, parents=True)
        configs[f"path_{key}"] = str(path)

    configs["path_fooocus_expansion"] = str(_sync_fooocus_expansion(models_dir))

    with open(config_path, "w") as fp:
        json.dump(configs, fp)
