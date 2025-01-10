import asyncio
import base64
import copy
import hashlib
import imghdr
import io
import json
import logging
import mimetypes
import os
import socket
import time
import traceback
import uuid
from datetime import datetime, timezone
from functools import cache
from typing import Annotated, Callable, Any, Literal
from urllib.parse import urlparse

import aiofiles
import aiofiles.os
import aiohttp
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel, Field

from modules import script_callbacks
import modules.advanced_parameters as advanced_parameters
import modules.config
import modules.flags as flags
from modules.model_info import get_all_model_info
import modules.style_sorter as style_sorter
from modules.util import get_shape_ceil, setup_logging
from modules import system_monitor
from modules.config import convert_ratio, read_preset_and_update_config
from modules.database import (
    create_tables,
    favorite_an_image,
    get_db,
    insert_focus_task_record,
    insert_sync_task_record,
    like_an_image,
    query_focus_task_record_with_status,
    share_an_image,
    unfavorite_an_image,
    unlike_an_image,
    unshare_an_image,
    update_focus_task_record,
)
from settings import settings


database_created = False

setup_logging()
logger = logging.getLogger("default")


class Status(BaseModel):
    percentage: int = 0
    title: str = ""
    images: list[np.ndarray] = []
    image_filepaths: list[str] = []
    is_nsfw: list[bool] = []

    class Config:
        arbitrary_types_allowed = True


class QueuingStatus(BaseModel):
    position: int = 0
    total: int = 0


class Progress(BaseModel):
    flag: str
    task_id: str
    status: Status
    queuing_status: QueuingStatus | None = None


class LoraConfig(BaseModel):
    lora_model: str = Field(
        default=modules.config.default_loras[0][0],
        description=f'LoRA model name. Options are: {["None"] + modules.config.lora_filenames}',
    )
    lora_weight: float = Field(default=modules.config.default_loras[0][1], description="LoRA weight.")


class ImageSource(BaseModel):
    image_url: str | None = Field(
        default=None,
        description=("The url to the image. " "image_url and encoded_image at least one must be provided."),
    )
    encoded_image: str | None = Field(
        default=None,
        description=(
            "Base64 encoded image. " "If both image_url and encoded_image are provided, encoded_image will be used."
        ),
    )


class ImageResult(ImageSource):
    image_id: str = Field(description="Image id.")


class RemoteImageSource(ImageSource):
    image_filepath: str | None = None
    image_np: np.ndarray | None = None

    class Config:
        arbitrary_types_allowed = True

class ControlConfig(BaseModel):
    ip_image: ImageSource | None = Field(default=None, description="Image prompt for generation.")
    ip_stop: float = Field(default=flags.default_parameters[flags.default_ip][0], description="Stop at for controlnet.")
    ip_weight: float = Field(
        default=flags.default_parameters[flags.default_ip][1], description="Weight for controlnet."
    )
    ip_type: str = Field(default=flags.default_ip, description=f"Image prompt type. Options are: {flags.ip_list}")


class InpaintInputImage(BaseModel):
    image: ImageSource = Field(description="Image for inpaint.")
    mask: ImageSource = Field(description="Mask for inpaint.")


class DescribeImageInfo(BaseModel):
    mode: str = Field(
        default=flags.desc_type_photo,
        description=f"Mode of the description task.Options are [{flags.desc_type_photo}, {flags.desc_type_anime}]",
    )
    image: ImageSource = Field(description="Image for description.")


class DescribeImageResult(BaseModel):
    task_id: str = Field(description="Task uuid.")
    image_id: str | None = Field(default=None, description="Image id.")
    prompt: str | None = Field(default=None, description="Image url.")
    styles: list[str] = Field(default=list(), description="Description of the image.")


class AdvancedOptions(BaseModel):
    disable_preview: bool = Field(default=False, description="Disable preview during generation.")
    adm_scaler_positive: float = Field(
        default=1.5, description="The scaler multiplied to positive ADM (use 1.0 to disable). "
    )
    adm_scaler_negative: float = Field(
        default=0.8, description="The scaler multiplied to negative ADM (use 1.0 to disable). "
    )
    adm_scaler_end: float = Field(default=0.3, description="When to end the guidance from positive/negative ADM. ")
    adaptive_cfg: float = Field(
        default=modules.config.default_cfg_tsnr,
        description="Enabling Fooocus's implementation of CFG mimicking for TSNR (effective when real CFG > mimicked CFG).",
    )
    sampler_name: str = Field(
        default=modules.config.default_sampler, description=f"Sampler. Options are: {flags.sampler_list}"
    )
    scheduler_name: str = Field(
        default=modules.config.default_scheduler, description=f"Scheduler. Options are: {flags.scheduler_list}"
    )
    generate_image_grid: bool = Field(
        default=False,
        description="(Experimental) This may cause performance problems on some computers and certain internet conditions.",
    )
    overwrite_step: int = Field(
        default=modules.config.default_overwrite_step,
        description="Forced Overwrite of Sampling Step. Set as -1 to disable. For developer debugging.",
    )
    overwrite_switch: int = Field(
        default=modules.config.default_overwrite_switch,
        description="Forced Overwrite of Refiner Switch Step. Set as -1 to disable. For developer debugging.",
    )
    overwrite_width: int = Field(
        default=-1,
        description="Forced Overwrite of Generating Width. Set as -1 to disable. For developer debugging. Results will be worse for non-standard numbers that SDXL is not trained on.",
    )
    overwrite_height: int = Field(
        default=-1,
        description="Forced Overwrite of Generating Height. Set as -1 to disable. For developer debugging. Results will be worse for non-standard numbers that SDXL is not trained on.",
    )
    overwrite_vary_strength: float = Field(
        default=-1,
        description='Forced Overwrite of Denoising Strength of "Vary". Set as negative number to disable. For developer debugging.',
    )
    overwrite_upscale_strength: float = Field(
        default=-1,
        description='Forced Overwrite of Denoising Strength of "Upscale". Set as negative number to disable. For developer debugging.',
    )
    mixing_image_prompt_and_vary_upscale: bool = Field(
        default=False, description="Mixing Image Prompt and Vary/Upscale"
    )
    mixing_image_prompt_and_inpaint: bool = Field(default=False, description="Mixing Image Prompt and Inpaint")
    debugging_cn_preprocessor: bool = Field(default=False, description="Debug Preprocessors")
    skipping_cn_preprocessor: bool = Field(default=False, description="Skip Preprocessors")
    controlnet_softness: float = Field(
        default=0.25, description="Similar to the Control Mode in A1111 (use 0.0 to disable). "
    )
    canny_low_threshold: int = Field(default=64, description="Canny Low Threshold")
    canny_high_threshold: int = Field(default=128, description="Canny High Threshold")
    refiner_swap_method: str = Field(default="joint", description="Refiner swap method")
    freeu_enabled: bool = Field(default=False, description="Enabled")
    freeu_b1: float = Field(default=1.01, description="B1")
    freeu_b2: float = Field(default=1.02, description="B2")
    freeu_s1: float = Field(default=0.99, description="S1")
    freeu_s2: float = Field(default=0.95, description="S2")
    debugging_inpaint_preprocessor: bool = Field(default=False, description="Debug Inpaint Preprocessing")
    inpaint_disable_initial_latent: bool = Field(default=False, description="Disable initial latent in inpaint")
    inpaint_engine: str = Field(
        default=modules.config.default_inpaint_engine_version,
        description=f"Inpaint Engine. Options are: {flags.inpaint_engine_versions}",
    )
    inpaint_strength: float = Field(
        default=1.0,
        description="Inpaint Denoising Strength. Same as the denoising strength in A1111 inpaint. Only used in inpaint, not used in outpaint. (Outpaint always use 1.0)",
    )
    inpaint_respective_field: float = Field(
        default=0.618,
        description='Inpaint Respective Field. The area to inpaint. Value 0 is same as "Only Masked" in A1111. Value 1 is same as "Whole Image" in A1111. Only used in inpaint, not used in outpaint. (Outpaint always use 1.0)',
    )
    inpaint_mask_upload_checkbox: bool = Field(default=False, description="Enable Mask Upload.")
    invert_mask_checkbox: bool = Field(default=False, description="Invert Mask.")
    inpaint_erode_or_dilate: int = Field(
        default=0,
        description=(
            "Mask Erode or Dilate."
            "Positive value will make white area in the mask larger, "
            "negative value will make white area smaller."
            "(default is 0, always process before any mask invert)"
        ),
    )


class GenerationOption(BaseModel):
    task_id: str = Field(description="Task id for generation.")
    task_type: Literal["sdxl", "sd3", "flux"] | None = Field(description="Task type for generation.")
    prompt: str = Field(description="Prompt for generation.")
    negative_prompt: str = Field(
        default=modules.config.default_prompt_negative, description="Negative prompt for generation."
    )
    style_selections: list[str] = Field(
        default=copy.deepcopy(modules.config.default_styles),
        description=f"Styles for generation. Options are: {copy.deepcopy(style_sorter.all_styles)}",
    )
    performance_selection: str = Field(
        default=modules.config.default_performance,
        description=f"Performance for generation. Options are: {flags.performance_selections}",
    )
    aspect_ratios_selection: str = Field(
        default=modules.config.default_aspect_ratio,
        description=f"width × height. Options are: {modules.config.available_aspect_ratios}",
    )
    image_number: int = Field(default=modules.config.default_image_number, description="Number of images to generate.")
    image_seed: int = Field(default=-1, description="Seed for generation. -1 means random.")
    sharpness: float = Field(
        default=modules.config.default_sample_sharpness,
        description=(
            "Image Sharpness. Higher value means image and texture are sharper. Min 0.0, Max 30.0."
            "Use as Clip Skip in Flux and SD3 mode. Min 1, max 5."),
    )
    guidance_scale: float = Field(
        default=modules.config.default_cfg_scale,
        description="Guidance Scale. Higher value means style is cleaner, vivider, and more artistic. Min 1.0, Max 30.0.",
    )
    base_model: str = Field(
        default=modules.config.default_base_model_name,
        description=f"Base Model (SDXL only). Options are: {modules.config.model_filenames}",
    )
    refiner_model: str = Field(
        default=modules.config.default_refiner_model_name,
        description=f"Refiner (SDXL or SD 1.5). Options are: {modules.config.refiner_model_filenames}",
    )
    refiner_switch: float = Field(
        default=modules.config.default_refiner_switch,
        description="Refiner Switch At. Use 0.4 for SD1.5 realistic models; or 0.667 for SD1.5 anime models; or 0.8 for XL-refiners; or any value for switching two SDXL models. Min 0.1, Max 1.0.",
    )
    loras: list[LoraConfig] = Field(default=[], description="LoRA configs.")
    input_image_checkbox: bool = Field(default=False, description="Whether to use input image.")
    current_tab: str = Field(default="uov", description="Current tab.")
    uov_method: str = Field(default=flags.disabled, description=f"Upscale or Variation. Options are: {flags.uov_list}")
    uov_input_image: ImageSource | None = Field(
        default=None, description="Input image for upscale, variation or image prompt."
    )
    outpaint_selections: list[str] = Field(
        default=[], description="Outpaint directions. 'Left', 'Right', 'Top', 'Bottom'"
    )
    inpaint_input_image: InpaintInputImage | None = Field(default=None, description="Input image for inpaint.")
    inpaint_additional_prompt: str = Field(default="", description="Describe what you want to inpaint.")
    inpaint_mask_image: ImageSource | None = Field(default=None, description="Mask image for inpaint.")
    ip_ctrls: list[ControlConfig] = Field(default=[], description="ControlNet configs.")
    advanced_options: AdvancedOptions = Field(
        default=AdvancedOptions(), description="Advanced settings for generation."
    )

    def get_target_resolution(self) -> tuple[int, int]:
        if "x" in self.aspect_ratios_selection:
            width, height = self.aspect_ratios_selection.replace("x", " ").split(" ")[:2]
        elif "*" in self.aspect_ratios_selection:
            width, height = self.aspect_ratios_selection.replace("*", " ").split(" ")[:2]
        else:
            raise ValueError(f"invalid aspect_ratio {self.aspect_ratios_selection}")
        if self.advanced_options.overwrite_width > 0:
            width = self.advanced_options.overwrite_width

        if self.advanced_options.overwrite_height > 0:
            height = self.advanced_options.overwrite_height
        return int(width), int(height)

    def get_steps(self) -> int:
        steps = 30
        if self.performance_selection == "Speed":
            steps = 30
        elif self.performance_selection == "Quality":
            steps = 60
        elif self.performance_selection == "Extreme Speed":
            steps = 8
        if self.advanced_options.overwrite_step > 0:
            steps = self.advanced_options.overwrite_step
        return steps

    def get_steps_coefficient(self) -> int:
        mapping = {
            "Speed": 2,
            "Quality": 4,
            "Extreme Speed": 1,
            "Turbo": 1,
        }
        return mapping[self.performance_selection]


class FocusTask(BaseModel):
    task_id: str = Field(description="Task uuid.")
    status: str = Field(description="Status of the current task.")
    created_at: datetime = Field(description="Created time of the current task.")


class FocusTasks(BaseModel):
    tasks: list[FocusTask] = Field(default=[], description="Tasks of the current user.")


class OptionList(BaseModel):
    options: list[str] = Field(default=[], description="Options for the field.")
    default: str | None = Field(default=None, description="Default value for the field.")
    default_list: list[str] = Field(default=[], description="Default value list for the field.")


class ImagePromptOptions(BaseModel):
    stop: float = Field(description="Stop at for controlnet.")
    weight: float = Field(description="Weight for controlnet.")


class LoraOptions(BaseModel):
    lora_name: OptionList = Field(description="Lora namei and possible selections.")
    lora_weight: float = Field(description="Lora weight.")


class StyleOption(BaseModel):
    style_name: str = Field(description="Style name.")
    style_preview: str = Field(description="Style preview images.")


class StyleOptions(BaseModel):
    options: list[StyleOption] = Field(description="Style options.")
    default_list: list[str] = Field(default=[], description="Default value list for the field.")


class DefaultSD3Options(BaseModel):
    base_models: OptionList = Field(description="Avaialable SD3 checkpoints.")
    aspect_ratios: OptionList = Field(description="Aspect ratio options.")


class DefaultOptions(BaseModel):
    performances: OptionList = Field(description="Performance options.")
    aspect_ratios: OptionList = Field(description="Aspect ratio options.")
    styles: StyleOptions = Field(description="Style options.")
    base_models: OptionList = Field(description="Avaialable SD checkpoints.")
    refiner_models: OptionList = Field(description="Avaialable refiners.")
    refiner_switch: float = Field(description="When to switch to a refiner model. Value needs to be between 0.0 ~ 1.0")
    loras: list[LoraOptions] = Field(description="Lora options.")
    uovs: OptionList = Field(description="Upscale or variation options.")
    ip_types: OptionList = Field(description="Image prompt Control Types.")
    ip_default_options: dict[str, ImagePromptOptions] = Field(description="Image prompt default options.")
    num_image_prompts: int = Field(description="Number of image prompts.")
    content_types: OptionList = Field(description="Content types for describe image.")
    cfg_scale: float = Field(description="Guidance scale.")
    sample_sharpness: float = Field(description="Image sharpness.")
    sampler: OptionList = Field(description="Sampler options.")
    scheduler: OptionList = Field(description="Scheduler options.")
    prompt: str = Field(description="Default prompt for generation.")
    negative_prompt: str = Field(description="Default negative prompt for generation.")
    steps: int | None = Field(description="Default steps for generation.", default=None)
    presets: OptionList = Field(description="Presets for settings for a specific genre.")
    sd3: DefaultSD3Options = Field(description="SD3 default options")


class Like(BaseModel):
    image_id: str = Field(description="Image id.")
    like: bool = Field(description="Like or unlike the image.")


class Favorite(BaseModel):
    image_id: str = Field(description="Image id.")
    favorite: bool = Field(description="Favorite or unfavorite the image.")


class Share(BaseModel):
    image_id: str = Field(description="Image id.")
    share: bool = Field(description="Share or unshare the image.")


def encode_filepath_with_base64(filepath: str) -> str:
    return base64.b64encode(filepath.encode("utf-8")).decode("utf-8")


def decode_filepath_from_base64(encoded_filepath: str) -> str:
    return base64.b64decode(encoded_filepath.encode("utf-8")).decode("utf-8")


def is_base64_image(img_str: str) -> tuple[bool, str, int | None, int | None, str | None]:
    """
    Check if a string is a base64 encoded image and return its dimensions and MIME type.

    :param str s: a string to check
    :return: tuple (is_image, removed_schema_str, width, height, mime) or (False, img_str, None, None, None) if the string is not a valid image
    """
    try:
        mime = None
        # Check if the string has the embedded schema and remove it
        removed_schema_str = img_str
        if ";base64," in img_str:
            mime, removed_schema_str = img_str.split(";base64,")
            mime = mime.split(":")[1] if "data:" in mime else None

        # Decode the base64 string
        decoded = base64.b64decode(removed_schema_str)

        # Open the image and get its size
        image = Image.open(io.BytesIO(decoded))
        width, height = image.size

        # If mime type is not available in the string, guess it using imghdr
        if mime is None:
            mime = imghdr.what(None, h=decoded)
            mime = "image/" + mime if mime else None

        return True, removed_schema_str, width, height, mime
    except Exception as e:
        logger.exception(f"Failed to check if the string is a valid image: {e}")
        return False, img_str, None, None, None


async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    output_path: str,
    headers: dict[str, str] = dict(),
    append_ext: bool = False,
) -> tuple[str | None, str | None]:
    if url.startswith("file://"):
        output_path = url.removeprefix("file://")
        if os.path.exists(output_path):
            async with aiofiles.open(output_path, mode="rb") as f:
                data = await f.read()
            return output_path, base64.b64encode(data).decode("utf-8")
        else:
            logger.error(f"Download failed for image {url} with status code 404: File not found")
            return None, None
    async with session.get(url, headers=headers) as resp:
        if resp.status == 200:
            content_type = resp.headers.get("content-type", None)
            if content_type and content_type.startswith("image/"):
                data = await resp.read()
                dirname = os.path.dirname(output_path)
                if not await aiofiles.os.path.exists(dirname):
                    await aiofiles.os.makedirs(dirname, exist_ok=True)
                if append_ext:
                    ext = mimetypes.guess_extension(content_type)
                    if ext and (not output_path.endswith(ext)):
                        output_path = f"{output_path}{ext}"
                async with aiofiles.open(output_path, mode="wb") as f:
                    await f.write(data)
                return output_path, base64.b64encode(data).decode("utf-8")
        try:
            resp_message = await resp.text()
            logger.error(f"Download failed for image {url} with status code {resp.status}: {resp_message}")
        except:
            logger.error(f"Download failed for image {url} with status code {resp.status} and cannot process image")
        return None, None


def remove_schema(base64_str: str) -> str:
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    return base64_str


async def save_base64_image_to_file(encoded_image: str, output_path: str) -> str:
    output_dir = os.path.dirname(output_path)
    if not await aiofiles.os.path.exists(output_dir):
        await aiofiles.os.makedirs(output_dir, exist_ok=True)
    decoded_image = base64.b64decode(remove_schema(encoded_image))
    async with aiofiles.open(output_path, "wb") as f:
        await f.write(decoded_image)
    return output_path


async def save_numpy_image_to_file(np_image: np.ndarray, output_path: str, format: str = "JPEG") -> str:
    output_dir = os.path.dirname(output_path)
    if not await aiofiles.os.path.exists(output_dir):
        await aiofiles.os.makedirs(output_dir, exist_ok=True)
    image = Image.fromarray(np_image)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    async with aiofiles.open(output_path, "wb") as f:
        await f.write(buffer.getvalue())
    return output_path


def get_exception(exception_class: Callable, status_code: int, msg: str) -> WebSocketException | HTTPException:
    if status_code == 400:
        if exception_class == WebSocketException:
            return WebSocketException(code=status.WS_1003_UNSUPPORTED_DATA, reason=msg)
        else:
            return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    if exception_class == WebSocketException:
        return WebSocketException(code=status_code, reason=msg)
    else:
        return HTTPException(status_code=status_code, detail=msg)


async def verify_image(
    session: aiohttp.ClientSession,
    image: ImageSource,
    user_id: str,
    with_schema: bool = False,
    subdir: str | None = None,
    exception: Callable = WebSocketException,
) -> RemoteImageSource:
    if not image.image_url and not image.encoded_image:
        raise get_exception(
            exception, status_code=400, msg="Either image_url or encoded_image must be provided for init_img."
        )
    image_name = str(uuid.uuid4())
    local_image = RemoteImageSource()
    if subdir:
        output_path = f"{settings.api_image_dir}/{user_id}/{subdir}/{image_name}"
    else:
        output_path = f"{settings.api_image_dir}/{user_id}/{image_name}"
    if image.image_url and not image.encoded_image:
        local_image.image_filepath, local_image.encoded_image = await download_image(
            session, image.image_url, output_path, append_ext=True
        )
    if not local_image.encoded_image and not image.encoded_image:
        raise get_exception(exception, status_code=400, msg=f"Failed to download image from {image.image_url}.")
    mime = None
    if image.encoded_image:
        is_img, image.encoded_image, _, _, mime = is_base64_image(image.encoded_image)
        if not is_img:
            raise get_exception(exception, status_code=400, msg="Failed to decode image from encoded_image str.")
        local_image.encoded_image = image.encoded_image
        if mime:
            ext = mimetypes.guess_extension(mime)
            if ext and (not output_path.endswith(ext)):
                output_path = f"{output_path}{ext}"
        local_image.image_filepath = await save_base64_image_to_file(local_image.encoded_image, output_path)
        local_image.image_np = base64_to_numpy_array(local_image.encoded_image)
    if with_schema:
        if mime is None and local_image.image_filepath:
            mime, _ = mimetypes.guess_type(local_image.image_filepath)
        if mime is None:
            raise get_exception(exception, status_code=400, msg="Failed to detect the mime type of image.")
        image.encoded_image = f"data:{mime};base64,{image.encoded_image}"
    return local_image


def base64_to_numpy_array(base64_str: str) -> np.ndarray:
    """
    Convert a base64 encoded image with a prefix to a NumPy array.
    """
    # Find the start of the base64 string
    base64_str = base64_str.split(",")[-1]

    # Decode the base64 string
    image_data = base64.b64decode(base64_str)

    # Convert the bytes to a PIL image
    image = Image.open(io.BytesIO(image_data))

    # Convert the PIL image to RGB mode, dropping the alpha channel if present
    image = image.convert("RGB")

    # Convert the PIL image to a NumPy array
    numpy_array = np.array(image)

    return numpy_array


def numpy_array_to_base64(numpy_array: np.ndarray, format="JPEG", with_schema: bool = False) -> str:
    """
    Convert a NumPy array to a base64 encoded image.

    Parameters:
    - numpy_array: numpy.ndarray
        The NumPy array to convert.
    - format: str, optional
        The format of the image to encode (default is JPEG).

    Returns:
    - str
        The base64 encoded string of the image.
    """
    # Convert the NumPy array to a PIL image
    image = Image.fromarray(numpy_array)

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format=format)

    # Get the raw bytes from the buffer
    image_bytes = buffer.getvalue()

    # Encode the bytes to base64 and return
    base64_str = base64.b64encode(image_bytes).decode("utf-8")

    if with_schema:
        return f"data:image/{format.lower()};base64,{base64_str}"
    return base64_str


def convert_advanced_options_to_list(advanced_options: AdvancedOptions) -> list:
    return [
        advanced_options.disable_preview,
        advanced_options.adm_scaler_positive,
        advanced_options.adm_scaler_negative,
        advanced_options.adm_scaler_end,
        advanced_options.adaptive_cfg,
        advanced_options.sampler_name,
        advanced_options.scheduler_name,
        advanced_options.generate_image_grid,
        advanced_options.overwrite_step,
        advanced_options.overwrite_switch,
        advanced_options.overwrite_width,
        advanced_options.overwrite_height,
        advanced_options.overwrite_vary_strength,
        advanced_options.overwrite_upscale_strength,
        advanced_options.mixing_image_prompt_and_vary_upscale,
        advanced_options.mixing_image_prompt_and_inpaint,
        advanced_options.debugging_cn_preprocessor,
        advanced_options.skipping_cn_preprocessor,
        advanced_options.controlnet_softness,
        advanced_options.canny_low_threshold,
        advanced_options.canny_high_threshold,
        advanced_options.refiner_swap_method,
        advanced_options.freeu_enabled,
        advanced_options.freeu_b1,
        advanced_options.freeu_b2,
        advanced_options.freeu_s1,
        advanced_options.freeu_s2,
        advanced_options.debugging_inpaint_preprocessor,
        advanced_options.inpaint_disable_initial_latent,
        advanced_options.inpaint_engine,
        advanced_options.inpaint_strength,
        advanced_options.inpaint_respective_field,
        advanced_options.inpaint_mask_upload_checkbox,
        advanced_options.invert_mask_checkbox,
        advanced_options.inpaint_erode_or_dilate,
    ]


class GenerationProgress(BaseModel):
    task_id: str = Field(description="Task uuid.")
    status: str = Field(description="Status of the current task.")
    progress: int = Field(default=0, description="Progress of the current task. From 0 ~ 100")
    message: str = Field(default="", description="Message of the current task.")
    is_url: bool = Field(default=False, description="Whether the result is a url.")
    images: list[ImageResult] = Field(default=[], description="Preview or result images")
    is_nsfw: list[bool] = Field(default=[], description="Whether the result images are nsfw.")
    queue_length: int | None = Field(default=None, description="Queue length of the current task.")
    queue_position: int | None = Field(default=None, description="Queue position of the current task.")


def get_user_subdir(user_id: str) -> str:
    h = hashlib.sha256()
    h.update(user_id.encode("utf-8"))
    encoded_user_path = h.hexdigest()
    # same user data in 4 level folders, to prevent a folder has too many subdir
    return f"{encoded_user_path[:2]}/{encoded_user_path[2:4]}/{encoded_user_path[4:6]}/{encoded_user_path}"


async def process_result_images(
    progress: Progress, is_url: bool, user_id: str, start_time: datetime
) -> list[ImageResult]:
    images = []
    if progress.status.images:
        for idx, image in enumerate(progress.status.images):
            if image is not None:
                image_id = ""
                if len(progress.status.image_filepaths) == len(progress.status.images):
                    filepath = progress.status.image_filepaths[idx]
                    if filepath:
                        image_id = encode_filepath_with_base64(filepath)

                if is_url:
                    rel_filepath = os.path.join(
                        "fooocus/outputs/",
                        get_user_subdir(user_id),
                        f"{start_time.strftime('%Y-%m-%d')}/{progress.task_id}-{progress.flag}-{progress.status.percentage}-{idx}.jpeg",
                    )
                    output_path = f"{settings.api_image_dir}/{rel_filepath}"
                    output_url = f"{settings.s3_prefix}/{rel_filepath}"
                    await save_numpy_image_to_file(image, output_path)
                    images.append(ImageResult(image_url=output_url, image_id=image_id))
                else:
                    images.append(
                        ImageResult(encoded_image=numpy_array_to_base64(image, with_schema=True), image_id=image_id)
                    )
    return images


def extract_queue_length(progress: Progress) -> tuple[int | None, int | None]:
    if progress.queuing_status:
        return progress.queuing_status.position, progress.queuing_status.total
    return None, None


async def extract_progress(progress: Progress, is_url: bool, user_id: str, start_time: datetime) -> GenerationProgress:
    images = await process_result_images(progress, is_url, user_id, start_time)
    queue_position, queue_length = extract_queue_length(progress)
    return GenerationProgress(
        task_id=progress.task_id,
        status=progress.flag,
        progress=progress.status.percentage,
        message=progress.status.title,
        is_url=is_url,
        images=images,
        queue_length=queue_length,
        queue_position=queue_position,
        is_nsfw=progress.status.is_nsfw,
    )


def strip_encoded_image_from_generation_option(generation_params: GenerationOption) -> GenerationOption:
    generation_params = copy.deepcopy(generation_params)
    if generation_params.uov_input_image:
        generation_params.uov_input_image.encoded_image = ""
        if isinstance(generation_params.uov_input_image, RemoteImageSource):
            generation_params.uov_input_image.image_np = None

    if generation_params.inpaint_input_image:
        generation_params.inpaint_input_image.image.encoded_image = ""
        if isinstance(generation_params.inpaint_input_image.image, RemoteImageSource):
            generation_params.inpaint_input_image.image.image_np = None

        generation_params.inpaint_input_image.mask.encoded_image = ""
        if isinstance(generation_params.inpaint_input_image.mask, RemoteImageSource):
            generation_params.inpaint_input_image.mask.image_np = None

    for ip_ctrl in generation_params.ip_ctrls:
        if ip_ctrl.ip_image:
            ip_ctrl.ip_image.encoded_image = ""
            if isinstance(ip_ctrl.ip_image, RemoteImageSource):
                ip_ctrl.ip_image.image_np = None

    return generation_params


def strip_encoded_image(image_source: ImageSource) -> ImageSource:
    image_source = copy.deepcopy(image_source)
    image_source.encoded_image = ""
    return image_source


async def update_database(
    progress: Progress, previous_status: str | None, user_id: str, generation_params: GenerationOption | None = None
) -> str:
    async with get_db() as db:
        if previous_status is None and generation_params is not None:
            hostname = socket.gethostname()
            server_ip = socket.gethostbyname(hostname)
            await insert_focus_task_record(
                db,
                user_id,
                progress.task_id,
                progress.flag,
                strip_encoded_image_from_generation_option(generation_params).json(),
                hostname,
                server_ip,
            )
        if previous_status != progress.flag:
            if progress.flag == "finish":
                await update_focus_task_record(
                    db, progress.task_id, progress.flag, json.dumps(progress.status.image_filepaths)
                )
            else:
                await update_focus_task_record(db, progress.task_id, progress.flag)
    return progress.flag


def get_hostname_and_port_from_url(url: str) -> str:
    if url:
        parsed_origin = urlparse(url)
        hostname = parsed_origin.hostname or ""
        port = parsed_origin.port or ""
        return f"{hostname}:{port}" if hostname and port else hostname
    return ""


def get_hostname(request: Request, hostname_from_setting: str) -> str:
    if hostname_from_setting:
        return hostname_from_setting
    hostname = get_hostname_and_port_from_url(request.headers.get("origin", ""))
    if hostname:
        return hostname
    hostname = get_hostname_and_port_from_url(request.headers.get("referer", ""))
    if hostname:
        return hostname
    return ""


async def list_presets(path: str) -> list[str]:
    if await aiofiles.os.path.exists(path):
        return [
            filename.removesuffix(".json") for filename in await aiofiles.os.listdir(path) if filename.endswith(".json")
        ]
    return []


@cache
def get_preset(preset: str, task_type: str | None = None) -> dict[str, Any]:
    config_dict = copy.deepcopy(modules.config.config_dict)
    config_dict = read_preset_and_update_config(preset, config_dict, settings.preset_dir)
    if task_type and task_type.lower() not in config_dict.get("task_type", "").lower():
        logger.error(f"Preset {preset} is for {config_dict.get('task_type', '')} tasks, not {task_type} tasks.")
        raise HTTPException(status_code=400, detail=f"Preset {preset} is designed for {config_dict.get('task_type', '')} tasks.")
    return config_dict


async def get_presets_for_task_types(path: str, task_type: Literal["sdxl", "sd3", "flux"] = "sdxl"):
    presets = await list_presets(path)
    filtered_presets = []
    for preset in presets:
        config_dict = await asyncio.to_thread(get_preset, preset)
        if task_type.lower() in config_dict.get("task_type", "").lower():
            filtered_presets.append(preset)
    return filtered_presets


def create_api(
    app: FastAPI,
    generate_clicked: Callable,
    refresh_seed: Callable,
    recover_task: Callable,
    stop_clicked: Callable,
    skip_clicked: Callable,
    trigger_describe: Callable,
) -> FastAPI:
    from modules.sdxl_pipeline import get_all_samplers, get_all_schedulers
    from modules.sdxl_ruined_styles import load_styles
    app.mount("/api/focus/static", StaticFiles(directory="static"), name="static")
    app.mount("/api/focus/styles", StaticFiles(directory="sdxl_styles"), name="styles")

    templates = Jinja2Templates(directory="templates", autoescape=False, auto_reload=True)

    async def prepare_args_for_generate(config: GenerationOption, user_id: str) -> list:
        async with aiohttp.ClientSession() as http_session:
            lora_ctrls = []

            for i, (n, v) in enumerate(modules.config.default_loras):
                if i < len(config.loras):
                    n = config.loras[i].lora_model
                    v = config.loras[i].lora_weight
                lora_ctrls += [n, v]

            ip_ctrls = []
            for i in range(4):
                if i < len(config.ip_ctrls):
                    image = config.ip_ctrls[i].ip_image
                    if image:
                        image_source = await verify_image(http_session, image, user_id, subdir="fooocus/inputs")
                        config.ip_ctrls[i].ip_image = image_source
                        if image_source.image_np is not None:
                            ip_ctrls += [
                                image_source.image_np,
                                config.ip_ctrls[i].ip_stop,
                                config.ip_ctrls[i].ip_weight,
                                config.ip_ctrls[i].ip_type,
                            ]
                            continue
                default_end, default_weight = flags.default_parameters[flags.default_ip]
                ip_ctrls += [None, default_end, default_weight, flags.default_ip]

            uov_input_image = None
            if config.uov_input_image:
                uov_input_image_source = await verify_image(
                    http_session, config.uov_input_image, user_id, subdir="fooocus/inputs"
                )
                config.uov_input_image = uov_input_image_source
                uov_input_image = uov_input_image_source.image_np

            inpaint_input_image = None
            inpaint_mask = None
            if config.inpaint_input_image:
                inpaint_input_image_source = await verify_image(
                    http_session, config.inpaint_input_image.image, user_id, subdir="fooocus/inputs"
                )
                config.inpaint_input_image.image = inpaint_input_image_source
                inpaint_input_image = inpaint_input_image_source.image_np

                inpaint_mask_source = await verify_image(
                    http_session, config.inpaint_input_image.mask, user_id, subdir="fooocus/inputs"
                )
                config.inpaint_input_image.mask = inpaint_mask_source
                inpaint_mask = inpaint_mask_source.image_np

            inpaint_mask_image = None
            if config.inpaint_mask_image:
                inpaint_mask_image_source = await verify_image(
                    http_session, config.inpaint_mask_image, user_id, subdir="fooocus/inputs"
                )
                inpaint_mask_image = inpaint_mask_image_source.image_np

            if config.image_seed < 0:
                image_seed = refresh_seed(True, config.image_seed)
            else:
                image_seed = refresh_seed(False, config.image_seed)

            inpaint_input = None
            if inpaint_input_image is not None and inpaint_mask is not None:
                inpaint_input = {"image": inpaint_input_image, "mask": inpaint_mask}

            ctrls = [
                config.prompt,
                config.negative_prompt,
                config.style_selections,
                config.performance_selection,
                config.aspect_ratios_selection,
                config.image_number,
                image_seed,
                config.sharpness,
                config.guidance_scale,
            ]

            ctrls += [config.base_model, config.refiner_model, config.refiner_switch] + lora_ctrls
            ctrls += [config.input_image_checkbox, config.current_tab]
            ctrls += [config.uov_method, uov_input_image]
            ctrls += [config.outpaint_selections, inpaint_input, config.inpaint_additional_prompt, inpaint_mask_image]
            ctrls += ip_ctrls
            return ctrls

    @app.websocket("/api/focus/ws/generate")
    async def generate_image_socket(
        websocket: WebSocket,
        task_id: str | None = None,
        task_type: Literal["sdxl", "sd3", "flux"] | None = None,
        is_url: bool = False,
        user_id: Annotated[str | None, Header()] = "local",
    ):
        if user_id is None:
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Could not identify user.")
        start_time = datetime.now(timezone.utc)
        await websocket.accept()
        previous_status = None
        output_dir = None
        if user_id != "local" and settings.output_base_dir:
            output_dir = os.path.join(settings.output_base_dir, get_user_subdir(user_id), "outputs", "focus")
        try:
            if task_id:
                async for progress in recover_task(task_id):
                    previous_status = await update_database(progress, previous_status, user_id)
                    generate_progress = await extract_progress(progress, is_url, user_id, start_time)
                    await websocket.send_json(generate_progress.dict())
            else:
                data = await websocket.receive_text()
                generation_option = GenerationOption(**json.loads(data))
                advanced_parameters.set_all_advanced_parameters(
                    *convert_advanced_options_to_list(generation_option.advanced_options)
                )
                request_headers = dict(websocket.headers)
                request_headers["x-session-hash"] = str(uuid.uuid4())
                request_headers["x-task-id"] = generation_option.task_id
                task_id = generation_option.task_id
                if task_type and task_type != generation_option.task_type:
                    raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="task_type from query & generation_option are different.")
                script_callbacks.before_task_callback(task_id)
                args = await prepare_args_for_generate(generation_option, user_id)
                function_name, decoded_params = _get_consume_args(generation_option)
                async with system_monitor.monitor_call_context(
                    request_headers=request_headers,
                    api_name=function_name,
                    function_name=function_name,
                    task_id=generation_option.task_id,
                    is_intermediate=False,
                ) as task_logger:
                    result_dict = {}
                    result_images = []
                    async with system_monitor.monitor_call_context(
                        request_headers=request_headers,
                        api_name=function_name,
                        function_name=function_name,
                        decoded_params=decoded_params,
                    ) as step_logger:
                        try:
                            async for progress in generate_clicked(
                                *args,
                                base_dir=output_dir,
                                task_id=generation_option.task_id,
                                metadata=request_headers,
                                task_type=generation_option.task_type,
                            ):
                                previous_status = await update_database(
                                    progress, previous_status, user_id, generation_option
                                )
                                generate_progress = await extract_progress(progress, is_url, user_id, start_time)
                                result_dict = generate_progress.dict()
                                await websocket.send_json(result_dict)
                                result_images = progress.status.image_filepaths
                            step_logger(result_images, result_dict.get('status', 'failed') == 'failed')

                            task_logger(result_images, result_dict.get('status', 'failed') == 'failed')
                        except Exception as e:
                            step_logger({
                                "exception": traceback.format_exc(),
                            }, True)
                            task_logger({
                                "exception": traceback.format_exc(),
                            }, True)
                            raise
        except WebSocketDisconnect:
            print("Client disconnected")
        except system_monitor.MonitorException as error:
            match (error.status_code, error.code):
                case (402, "WEBUIFE-01010001"):
                    reason = "INSUFFICIENT_CREDITS"
                case (402, "WEBUIFE-01010003"):
                    reason = "INSUFFICIENT_DAILY_CREDITS"
                case (429, "WEBUIFE-01010004"):
                    reason = "REACH_CONCURRENCY_LIMIT"
                case _:
                    logger.error(
                        f"mismatched status_code({error.status_code}) "
                        f"and code({error.code}) in 'MonitorException'"
                    )
                    raise

            await websocket.send_json(
                GenerationProgress(
                    task_id=error.task_id,
                    status=f"MonitorException.{reason}",
                    message=str(error),
                    progress=100,
                ).dict()
            )
            raise
        except Exception as e:
            logger.exception(f'un-handled prediction exception: {e.__str__()}')
            await websocket.send_json(
                GenerationProgress(
                    task_id=task_id or "",
                    status=f"failed",
                    message=str(e),
                    progress=100,
                ).dict()
            )
            raise
        finally:
            script_callbacks.after_task_callback(task_id)
            await websocket.close()

    @app.post("/api/focus/stop", response_class=JSONResponse)
    async def stop_task(task_id: str, user_id: Annotated[str | None, Header()] = "local"):
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not identify user.")
        task_stopped = stop_clicked(task_id)
        if task_stopped:
            return {"status": "success"}
        return {"status": "failed"}

    @app.post("/api/focus/skip", response_class=JSONResponse)
    async def skip_task(task_id: str, user_id: Annotated[str | None, Header()] = "local"):
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not identify user.")
        task_skipped = skip_clicked(task_id)
        if task_skipped:
            return {"status": "success"}
        return {"status": "failed"}

    @app.get("/api/focus/task_records", response_model=FocusTasks, response_class=JSONResponse)
    async def get_focus_task_record_for_status(task_status: str, user_id: Annotated[str | None, Header()] = "local"):
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not identify user.")
        async with get_db() as db:
            records = await query_focus_task_record_with_status(db, user_id, task_status)
            records_response = FocusTasks(
                tasks=[
                    FocusTask(task_id=record.task_id, status=record.status, created_at=record.created_at)
                    for record in records
                ]
            )
            return records_response

    @app.post("/api/focus/describe/local", response_model=DescribeImageResult, response_class=JSONResponse)
    async def describe_image_local(image_info: DescribeImageInfo, user_id: Annotated[str | None, Header()] = "local"):
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not identify user.")
        if image_info.mode not in [flags.desc_type_photo, flags.desc_type_anime]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid mode.")
        async with aiohttp.ClientSession() as http_session:
            task_id = str(uuid.uuid4())

            image_source = await verify_image(
                http_session, image_info.image, user_id, subdir="fooocus/inputs", exception=HTTPException
            )
            if not image_source.encoded_image:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Image is invalid or cannot be accessed."
                )
            prompt, styles = await asyncio.to_thread(
                trigger_describe, image_info.mode, base64_to_numpy_array(image_source.encoded_image)
            )
            image_id = None
            if image_source.image_filepath:
                image_id = encode_filepath_with_base64(image_source.image_filepath)
            result = DescribeImageResult(task_id=task_id, image_id=image_id, prompt=prompt, styles=styles)
            image_info.image = strip_encoded_image(image_info.image)
            async with get_db() as db:
                hostname = socket.gethostname()
                server_ip = socket.gethostbyname(hostname)
                await insert_sync_task_record(
                    db,
                    user_id,
                    task_id,
                    "describe-local",
                    "finished" if image_id else "failed",
                    image_info.json(),
                    hostname,
                    server_ip,
                    result.json(),
                )
            return result

    def get_checkpoints_for_task_type(task_type: Literal["sdxl", "sd3", "flux"] = "sdxl"):
        all_model_info = get_all_model_info()
        return [
            model_name for model_name, model_info in all_model_info.checkpoint_models.items()
            if model_info.base is not None and task_type.lower() in model_info.base.lower()
        ]

    def get_loras_for_task_type(task_type: Literal["sdxl", "sd3", "flux"] = "sdxl"):
        all_model_info = get_all_model_info()
        return ["None"] + [
            model_name for model_name, model_info in all_model_info.lora_models.items()
            if model_info.base is not None and task_type.lower() in model_info.base.lower()
        ]

    @app.get("/api/focus/default_options", response_model=DefaultOptions, response_class=JSONResponse)
    async def get_default_options(
        request: Request,
        preset: str = "default",
        task_type: Literal["sdxl", "sd3", "flux"] = "sdxl",
        user_id: Annotated[str | None, Header()] = "local"
    ):
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not identify user.")
        available_presets = await get_presets_for_task_types(settings.preset_dir, task_type)
        config_dict = await asyncio.to_thread(get_preset, preset, task_type)
        if task_type == "sd3" or task_type == "flux":
            lora_options = [
                LoraOptions(
                    lora_name=OptionList(default=lora_name, options=get_loras_for_task_type(task_type)),
                    lora_weight=lora_weight,
                )
                for lora_name, lora_weight in config_dict.get("default_loras", [])
            ]
            performance = "Speed"
            performance_selections = ["Speed", "Quality"]
            if performance not in performance_selections:
                performance_selections.append(performance)
            return DefaultOptions(
                performances=OptionList(
                    default=performance,
                    options=performance_selections,
                ),
                aspect_ratios=OptionList(
                    default=convert_ratio(config_dict.get("default_aspect_ratio", modules.config.default_aspect_ratio)),
                    options=[
                        convert_ratio(x)
                        for x in config_dict.get("available_aspect_ratios", modules.config.available_aspect_ratios)
                    ],
                ),
                styles=StyleOptions(
                    default_list=list(config_dict.get("default_styles", [])),
                    options=[
                        StyleOption(
                            style_name=style_name,
                            style_preview="")
                        for style_name in load_styles()],
                ),
                base_models=OptionList(
                    default=config_dict.get("default_model"),
                    options=config_dict.get("model_filenames", get_checkpoints_for_task_type(task_type)),
                ),
                refiner_models=OptionList(
                    default=config_dict.get("default_refiner", modules.config.default_refiner_model_name),
                    options=["None"] + config_dict.get("refiner_filenames", get_checkpoints_for_task_type(task_type)),
                ),
                refiner_switch=config_dict.get("default_refiner_switch", 1),
                loras=lora_options,
                uovs=OptionList(default=flags.disabled, options=[flags.disabled]),
                ip_types=OptionList(default="", options=[]),
                ip_default_options={},
                num_image_prompts=0,
                content_types=OptionList(
                    default="", options=[]
                ),
                cfg_scale=config_dict.get("default_cfg_scale", modules.config.default_cfg_scale),
                sample_sharpness=config_dict.get("default_clip_skip", 1.0),
                sampler=OptionList(
                    default=config_dict.get("default_sampler", modules.config.default_sampler), options=get_all_samplers()
                ),
                scheduler=OptionList(
                    default=config_dict.get("default_scheduler", modules.config.default_scheduler),
                    options=get_all_schedulers(),
                ),
                prompt=config_dict.get("default_prompt", modules.config.default_prompt),
                negative_prompt=config_dict.get("default_prompt_negative", modules.config.default_prompt_negative),
                steps=config_dict.get("default_steps", None),
                presets=OptionList(
                    default=preset if preset in available_presets else available_presets[0], options=available_presets),
                sd3=DefaultSD3Options(
                    base_models=OptionList(
                        default=modules.config.sd3_config["baseModel"],
                        options=modules.config.sd3_config["baseModels"],
                    ),
                    aspect_ratios=OptionList(
                        default=modules.config.sd3_config["aspectRatio"],
                        options=modules.config.sd3_config["aspectRatios"],
                    ),
                )
            )
        lora_options = [
            LoraOptions(
                lora_name=OptionList(default=lora_name, options=get_loras_for_task_type(task_type)),
                lora_weight=lora_weight,
            )
            for lora_name, lora_weight in config_dict.get("default_loras", modules.config.default_loras)
        ]
        performance = config_dict.get("default_performance", modules.config.default_performance)
        performance_selections = copy.deepcopy(flags.performance_selections)
        if performance not in performance_selections:
            performance_selections.append(performance)
        return DefaultOptions(
            performances=OptionList(
                default=performance,
                options=performance_selections,
            ),
            aspect_ratios=OptionList(
                default=convert_ratio(config_dict.get("default_aspect_ratio", modules.config.default_aspect_ratio)),
                options=[
                    convert_ratio(x)
                    for x in config_dict.get("available_aspect_ratios", modules.config.available_aspect_ratios)
                ],
            ),
            styles=StyleOptions(
                default_list=list(config_dict.get("default_styles", modules.config.default_styles)),
                options=[
                    StyleOption(
                        style_name=style_name,
                        style_preview=f"/api/focus/styles/samples/{style_name.lower().replace(' ', '_')}.jpg")
                    for style_name in copy.deepcopy(style_sorter.all_styles)],
            ),
            base_models=OptionList(
                default=config_dict.get("default_model", modules.config.default_base_model_name),
                options=config_dict.get("model_filenames", get_checkpoints_for_task_type(task_type)),
            ),
            refiner_models=OptionList(
                default=config_dict.get("default_refiner", modules.config.default_refiner_model_name),
                options=["None"] + config_dict.get("refiner_filenames", get_checkpoints_for_task_type(task_type)),
            ),
            refiner_switch=config_dict.get("default_refiner_switch", modules.config.default_refiner_switch),
            loras=lora_options,
            uovs=OptionList(default=flags.disabled, options=flags.uov_list),
            ip_types=OptionList(default=flags.default_ip, options=flags.ip_list),
            ip_default_options={
                ip_type: ImagePromptOptions(
                    stop=flags.default_parameters[ip_type][0], weight=flags.default_parameters[ip_type][1]
                )
                for ip_type in flags.ip_list
            },
            num_image_prompts=4,
            content_types=OptionList(
                default=flags.desc_type_photo, options=[flags.desc_type_photo, flags.desc_type_anime]
            ),
            cfg_scale=config_dict.get("default_cfg_scale", modules.config.default_cfg_scale),
            sample_sharpness=config_dict.get("default_sample_sharpness", modules.config.default_sample_sharpness),
            sampler=OptionList(
                default=config_dict.get("default_sampler", modules.config.default_sampler), options=flags.sampler_list
            ),
            scheduler=OptionList(
                default=config_dict.get("default_scheduler", modules.config.default_scheduler),
                options=flags.scheduler_list,
            ),
            prompt=config_dict.get("default_prompt", modules.config.default_prompt),
            negative_prompt=config_dict.get("default_prompt_negative", modules.config.default_prompt_negative),
            steps=config_dict.get("default_steps", None),
            presets=OptionList(default=preset if preset in available_presets else "default", options=available_presets),
            sd3=DefaultSD3Options(
                base_models=OptionList(
                    default=modules.config.sd3_config["baseModel"],
                    options=modules.config.sd3_config["baseModels"],
                ),
                aspect_ratios=OptionList(
                    default=modules.config.sd3_config["aspectRatio"],
                    options=modules.config.sd3_config["aspectRatios"],
                ),
            )
        )

    @app.post("/api/focus/like", response_class=JSONResponse)
    async def like_or_unlike_an_image(like: Like, user_id: Annotated[str | None, Header()] = "local"):
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not identify user.")
        async with get_db() as db:
            if like.like:
                await like_an_image(db, user_id, like.image_id)
            else:
                await unlike_an_image(db, user_id, like.image_id)
            return {
                "success": True,
                "liked": like.like,
            }

    @app.post("/api/focus/favorite", response_class=JSONResponse)
    async def favorite_or_unfavorite_an_image(favorite: Favorite, user_id: Annotated[str | None, Header()] = "local"):
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not identify user.")
        async with get_db() as db:
            if favorite.favorite:
                await favorite_an_image(db, user_id, favorite.image_id)
            else:
                await unfavorite_an_image(db, user_id, favorite.image_id)
            return {
                "success": True,
                "favorited": favorite.favorite,
            }

    @app.post("/api/focus/share", response_class=JSONResponse)
    async def share_or_unshare_an_image(share: Share, user_id: Annotated[str | None, Header()] = "local"):
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not identify user.")
        async with get_db() as db:
            if share.share:
                await share_an_image(db, user_id, share.image_id)
            else:
                await unshare_an_image(db, user_id, share.image_id)
            return {
                "success": True,
                "shared": share.share,
            }

    @app.post("/api/focus/upload/")
    async def upload_image(
        subdir: str = "", file: UploadFile = File(...), user_id: Annotated[str | None, Header()] = "local"
    ):
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not identify user.")
        if user_id != "local" and settings.output_base_dir:
            output_dir = os.path.join(settings.output_base_dir, get_user_subdir(user_id), "inputs", "focus")
            if subdir:
                output_dir = os.path.join(output_dir, subdir)
        else:
            output_dir = os.path.join("./inputs")
        os.makedirs(output_dir, exist_ok=True)
        try:
            if not file.filename:
                file.filename = str(uuid.uuid4())
            else:
                file.filename = str(uuid.uuid4()) + file.filename
            file_path = os.path.join(output_dir, file.filename)

            async with aiofiles.open(file_path, "wb") as out_file:
                while content := await file.read(1024):  # Read 1024 bytes at a time
                    await out_file.write(content)

            return {"filename": file_path}
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": str(e)})

    @app.get("/ui", response_class=HTMLResponse)
    async def vue_ui(
        request: Request,
        user_id: Annotated[str | None, Header()] = None,
    ):
        global database_created
        if not database_created:
            await create_tables()
            database_created = True

        encoded_user_id = ""
        if user_id:
            encoded_user_id = base64.b64encode(user_id.encode('utf-8')).decode('utf-8')

        return templates.TemplateResponse(
            "ui.html",
            {
                "request": request,
                "base64_encoded_user_id": encoded_user_id,
                "user_tier": request.headers["user-tier"],
                "js_version": time.time(),
                "app_version": os.path.getmtime("static/js/app.js"),
            }
        )

    return app


def _get_image_resolution(image: ImageSource) -> tuple[int, int]:
    assert isinstance(image, RemoteImageSource) and image.image_np is not None
    width, height, _ = image.image_np.shape
    return width, height


def _get_image_options_consume_args(config: GenerationOption) -> tuple[str, dict[str, Any]] | None:
    if not config.input_image_checkbox:
        return None

    if config.current_tab == "uov":
        if not config.uov_input_image:
            return None

        width, height = _get_image_resolution(config.uov_input_image)
        if "Vary" in config.uov_method:
            shape_ceil = get_shape_ceil(height, width)
            if shape_ceil <= 1024:
                width = 1024
                height = 1024
            elif shape_ceil >= 2048:
                width = 2048
                height = 2048

            return "fooocus.vary", {
                "width": width,
                "height": height,
                "steps_coefficient": config.get_steps_coefficient(),
                "image_number": config.image_number,
            }

        if "Upscale" in config.uov_method:
            scale = 1.5 if "1.5" in config.uov_method else 2.0
            width *= scale
            height *= scale

            is_fast = "Fast" in config.uov_method

            shape_ceil = get_shape_ceil(height, width)
            if shape_ceil <= 1024:
                width = 1024
                height = 1024
            elif shape_ceil > 2800:
                is_fast = True

            steps_coefficient = 0
            image_number = 1
            if not is_fast:
                steps_coefficient = config.get_steps_coefficient()
                image_number = config.image_number

            return "fooocus.upscale", {
                "width": width,
                "height": height,
                "steps_coefficient": steps_coefficient,
                "image_number": image_number,
            }

        return None

    if config.current_tab == "inpaint":
        if not config.inpaint_input_image:
            return None

        width, height = _get_image_resolution(config.inpaint_input_image.image)

        if config.outpaint_selections:
            scale = 0
            if "Top" in config.outpaint_selections:
                scale += 0.3
            if "Bottom" in config.outpaint_selections:
                scale += 0.3
            height += height * scale

            scale = 0
            if "Left" in config.outpaint_selections:
                scale += 0.3
            if "Right" in config.outpaint_selections:
                scale += 0.3
            width += height * scale

        return "fooocus.inpaint", {
            "width": int(width),
            "height": int(height),
            "steps_coefficient": config.get_steps_coefficient(),
            "image_number": config.image_number,
        }

    if config.current_tab == "ip":
        ip_ctrls = sum(bool(ip_ctrl.ip_image) for ip_ctrl in config.ip_ctrls)
        width, height = config.get_target_resolution()

        return "fooocus", {
            "width": width,
            "height": height,
            "steps_coefficient": config.get_steps_coefficient(),
            "ip_ctrls": ip_ctrls,
            "image_number": config.image_number,
        }

    return None

def _get_consume_args(config: GenerationOption) -> tuple[str, dict[str, Any]]:
    args = _get_image_options_consume_args(config)
    if args:
        return args

    width, height = config.get_target_resolution()
    if config.task_type == "flux":
        return "fooocus.flux", {
            "width": width,
            "height": height,
            "steps_coefficient": config.get_steps_coefficient(),
            "ip_ctrls": 0,
            "image_number": config.image_number,
            "ratio": 2.0,
        }
    elif config.task_type == "sd3":
        return "fooocus.sd3", {
            "width": width,
            "height": height,
            "steps_coefficient": config.get_steps_coefficient(),
            "ip_ctrls": 0,
            "image_number": config.image_number,
            "ratio": 1.5,
        }
    else:
        return "fooocus", {
            "width": width,
            "height": height,
            "steps_coefficient": config.get_steps_coefficient(),
            "ip_ctrls": 0,
            "image_number": config.image_number,
            "ratio": 1.0,
        }
