import gc
import logging
import threading
from modules import script_callbacks
from typing import Any, Literal
from PIL import Image
import functools

from modules.model_info import get_all_model_info
from modules.util import setup_logging

setup_logging()
logger = logging.getLogger("default")


class AsyncTask:
    def __init__(
        self,
        task_id,
        args,
        base_dir: str | None = None,
        metadata: dict[str, Any] | None = None,
        task_type: Literal["sdxl", "sd3", "flux"] | None = None,
    ):
        self.task_id = task_id
        self.args = args
        self.yields = []
        self.results = []
        self.result_paths = []
        self.is_nsfw = []
        self.base_dir: str | None = base_dir
        self.metadata = metadata
        self.task_type = task_type

async_tasks: list[AsyncTask] = []
running_task: AsyncTask | None = None
finished_tasks: list[AsyncTask] = []
stop_or_skipped_tasks: list[AsyncTask] = []


def worker():
    global async_tasks
    global running_task
    global finished_tasks

    import traceback
    import math
    import numpy as np
    import torch
    import time
    import shared
    import random
    import copy
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.config
    import modules.patch
    from ldm_patched.modules.model_management import (
        InterruptProcessingException, processing_interrupted, throw_exception_if_processing_interrupted)
    import extras.preprocessors as preprocessors
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants
    import modules.advanced_parameters as advanced_parameters
    import extras.ip_adapter as ip_adapter
    import extras.face_crop
    import fooocus_version

    from modules.sdxl_styles import apply_style, apply_wildcards, fooocus_expansion
    from modules.private_logger import log
    from extras.expansion import safe_str
    from modules.util import remove_empty_str, HWC3, resize_image, \
        get_image_shape_ceil, set_image_shape_ceil, get_shape_ceil, resample_image, erode_or_dilate
    from modules.upscaler import perform_upscale
    from modules.prompt_processing import process_metadata, process_prompt, parse_loras
    from modules.util import TimeIt, memory_context_manager
    import modules.pipelines
    import modules.controlnet

    try:
        async_gradio_app = shared.gradio_root
        if async_gradio_app:
            flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
            if async_gradio_app.share:
                flag += f''' or {async_gradio_app.share_url}'''
            logger.info(flag)
    except Exception as e:
        logger.exception(e)


    all_models = get_all_model_info()

    def progressbar(async_task, number, text, preview_image = None, status = "preview"):
        logger.info(f'[Fooocus] {text}')
        async_task.yields.append([status, (number, text, preview_image)])

    def yield_result(
        async_task,
        imgs,
        do_not_show_finished_images=False,
        img_paths: str | list | None = None,
        is_nsfw: bool | list[bool] | None = None
    ):
        if not isinstance(imgs, list):
            imgs = [imgs]

        async_task.results = async_task.results + imgs

        if img_paths is not None:
            if not isinstance(img_paths, list):
                img_paths = [img_paths]
            async_task.result_paths = async_task.result_paths + img_paths


        if is_nsfw is not None:
            if not isinstance(is_nsfw, list):
                is_nsfw = [is_nsfw]
            async_task.is_nsfw = async_task.is_nsfw + is_nsfw

        if do_not_show_finished_images:
            return

        async_task.yields.append(['results', async_task.results])
        return

    def build_image_wall(async_task):
        if not advanced_parameters.generate_image_grid:
            return

        results = async_task.results

        if len(results) < 2:
            return

        for img in results:
            if not isinstance(img, np.ndarray):
                return
            if img.ndim != 3:
                return

        H, W, C = results[0].shape

        for img in results:
            Hn, Wn, Cn = img.shape
            if H != Hn:
                return
            if W != Wn:
                return
            if C != Cn:
                return

        cols = float(len(results)) ** 0.5
        cols = int(math.ceil(cols))
        rows = float(len(results)) / float(cols)
        rows = int(math.ceil(rows))

        wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(results):
                    img = results[y * cols + x]
                    wall[y * H:y * H + H, x * W:x * W + W, :] = img

        # must use deep copy otherwise gradio is super laggy. Do not use list.append() .
        async_task.results = async_task.results + [wall]
        return

    @torch.no_grad()
    @torch.inference_mode()
    def handler(async_task):
        execution_start_time = time.perf_counter()

        args = async_task.args
        args.reverse()

        prompt = args.pop()
        negative_prompt = args.pop()
        style_selections = args.pop()
        performance_selection = args.pop()
        aspect_ratios_selection = args.pop()
        image_number = args.pop()
        image_seed = args.pop()
        sharpness = args.pop()
        guidance_scale = args.pop()
        base_model_name = args.pop()
        refiner_model_name = args.pop()
        refiner_switch = args.pop()
        loras = [[str(args.pop()), float(args.pop())] for _ in range(5)]
        input_image_checkbox = args.pop()
        current_tab = args.pop()
        uov_method = args.pop()
        uov_input_image = args.pop()
        outpaint_selections = args.pop()
        inpaint_input_image = args.pop()
        inpaint_additional_prompt = args.pop()
        inpaint_mask_image_upload = args.pop()

        cn_tasks = {x: [] for x in flags.ip_list}
        for _ in range(4):
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        outpaint_selections = [o.lower() for o in outpaint_selections]
        base_model_additional_loras = []
        raw_style_selections = copy.deepcopy(style_selections)
        uov_method = uov_method.lower()

        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0

        if base_model_name == refiner_model_name:
            logger.info(f'Refiner disabled because base model and refiner are same.')
            refiner_model_name = 'None'

        assert performance_selection in ['Speed', 'Quality', 'Extreme Speed', 'Turbo']

        steps = 30

        if performance_selection == 'Speed':
            steps = 30

        if performance_selection == 'Quality':
            steps = 60

        if performance_selection == 'Turbo':
            steps = 5

        if performance_selection == 'Extreme Speed':
            logger.info('Enter LCM mode.')
            progressbar(async_task, 1, 'Downloading LCM components ...')
            loras += [(modules.config.downloading_sdxl_lcm_lora(), 1.0)]

            if refiner_model_name != 'None':
                logger.info(f'Refiner disabled in LCM mode.')

            refiner_model_name = 'None'
            sampler_name = advanced_parameters.sampler_name = 'lcm'
            scheduler_name = advanced_parameters.scheduler_name = 'lcm'
            modules.patch.sharpness = sharpness = 0.0
            cfg_scale = guidance_scale = 1.0
            modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg = 1.0
            refiner_switch = 1.0
            modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive = 1.0
            modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative = 1.0
            modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end = 0.0
            steps = 8

        modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg
        logger.info(f'[Parameters] Adaptive CFG = {modules.patch.adaptive_cfg}')

        modules.patch.sharpness = sharpness
        logger.info(f'[Parameters] Sharpness = {modules.patch.sharpness}')

        modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive
        modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative
        modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end
        logger.info(f'[Parameters] ADM Scale = '
              f'{modules.patch.positive_adm_scale} : '
              f'{modules.patch.negative_adm_scale} : '
              f'{modules.patch.adm_scaler_end}')

        cfg_scale = float(guidance_scale)
        logger.info(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = aspect_ratios_selection.replace('x', ' ').split(' ')[:2]
        width, height = int(width), int(height)

        skip_prompt_processing = False
        refiner_swap_method = advanced_parameters.refiner_swap_method

        inpaint_worker.current_task = None
        inpaint_parameterized = advanced_parameters.inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        seed = int(image_seed)
        logger.info(f'[Parameters] Seed = {seed}')

        sampler_name = advanced_parameters.sampler_name
        scheduler_name = advanced_parameters.scheduler_name

        goals = []
        tasks = []

        if input_image_checkbox:
            if (current_tab == 'uov' or (
                    current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_vary_upscale)) \
                    and uov_method != flags.disabled and uov_input_image is not None:
                uov_input_image = HWC3(uov_input_image)
                if 'vary' in uov_method:
                    goals.append('vary')
                elif 'upscale' in uov_method:
                    goals.append('upscale')
                    if 'fast' in uov_method:
                        skip_prompt_processing = True
                    else:
                        steps = 18

                        if performance_selection == 'Speed':
                            steps = 18

                        if performance_selection == 'Quality':
                            steps = 36

                        if performance_selection == 'Extreme Speed':
                            steps = 8

                        if performance_selection == 'Turbo':
                            steps = 5

                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
            if (current_tab == 'inpaint' or (
                    current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_inpaint)) \
                    and isinstance(inpaint_input_image, dict):
                inpaint_image = inpaint_input_image['image']
                inpaint_mask = inpaint_input_image['mask'][:, :, 0]

                if advanced_parameters.inpaint_mask_upload_checkbox:
                    if isinstance(inpaint_mask_image_upload, np.ndarray):
                        if inpaint_mask_image_upload.ndim == 3:
                            H, W, C = inpaint_image.shape
                            inpaint_mask_image_upload = resample_image(inpaint_mask_image_upload, width=W, height=H)
                            inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
                            inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
                            inpaint_mask = np.maximum(inpaint_mask, inpaint_mask_image_upload)

                if int(advanced_parameters.inpaint_erode_or_dilate) != 0:
                    inpaint_mask = erode_or_dilate(inpaint_mask, advanced_parameters.inpaint_erode_or_dilate)

                if advanced_parameters.invert_mask_checkbox:
                    inpaint_mask = 255 - inpaint_mask

                inpaint_image = HWC3(inpaint_image)
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
                    if inpaint_parameterized:
                        progressbar(async_task, 1, 'Downloading inpainter ...')
                        inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                            advanced_parameters.inpaint_engine)
                        base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                        logger.info(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                        if refiner_model_name == 'None':
                            use_synthetic_refiner = True
                            refiner_switch = 0.5
                    else:
                        inpaint_head_model_path, inpaint_patch_model_path = None, None
                        logger.info(f'[Inpaint] Parameterized inpaint is disabled.')
                    if inpaint_additional_prompt != '':
                        if prompt == '':
                            prompt = inpaint_additional_prompt
                        else:
                            prompt = inpaint_additional_prompt + '\n' + prompt
                    goals.append('inpaint')
            if current_tab == 'ip' or \
                    advanced_parameters.mixing_image_prompt_and_inpaint or \
                    advanced_parameters.mixing_image_prompt_and_vary_upscale:
                goals.append('cn')
                progressbar(async_task, 1, 'Downloading control models ...')
                if len(cn_tasks[flags.cn_canny]) > 0:
                    controlnet_canny_path = modules.config.downloading_controlnet_canny()
                if len(cn_tasks[flags.cn_cpds]) > 0:
                    controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
                if len(cn_tasks[flags.cn_ip]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
                if len(cn_tasks[flags.cn_ip_face]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                        'face')
                progressbar(async_task, 1, 'Loading control models ...')

        # Load or unload CNs
        pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        switch = int(round(steps * refiner_switch))

        if advanced_parameters.overwrite_step > 0:
            steps = advanced_parameters.overwrite_step

        if advanced_parameters.overwrite_switch > 0:
            switch = advanced_parameters.overwrite_switch

        if advanced_parameters.overwrite_width > 0:
            width = advanced_parameters.overwrite_width

        if advanced_parameters.overwrite_height > 0:
            height = advanced_parameters.overwrite_height

        logger.info(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        logger.info(f'[Parameters] Steps = {steps} - {switch}')

        progressbar(async_task, 1, 'Initializing ...')

        if not skip_prompt_processing:

            prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
            negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            if prompt == '':
                # disable expansion when empty since it is not meaningful and influences image prompt
                use_expansion = False

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

            progressbar(async_task, 3, 'Loading models ...')
            pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name,
                                        loras=loras, base_model_additional_loras=base_model_additional_loras,
                                        use_synthetic_refiner=use_synthetic_refiner)

            progressbar(async_task, 3, 'Processing prompts ...')
            tasks = []
            for i in range(image_number):
                task_seed = (seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not
                task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future

                task_prompt = apply_wildcards(prompt, task_rng)
                task_negative_prompt = apply_wildcards(negative_prompt, task_rng)
                task_extra_positive_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_positive_prompts]
                task_extra_negative_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_negative_prompts]

                positive_basic_workloads = []
                negative_basic_workloads = []

                if use_style:
                    for s in style_selections:
                        p, n = apply_style(s, positive=task_prompt)
                        positive_basic_workloads = positive_basic_workloads + p
                        negative_basic_workloads = negative_basic_workloads + n
                else:
                    positive_basic_workloads.append(task_prompt)

                negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

                positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
                negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

                positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
                negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

                tasks.append(dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    expansion='',
                    c=None,
                    uc=None,
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                    log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
                ))

            if use_expansion:
                for i, t in enumerate(tasks):
                    progressbar(async_task, 5, f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    logger.info(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            for i, t in enumerate(tasks):
                progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if len(goals) > 0:
            progressbar(async_task, 13, 'Image processing ...')

        if 'vary' in goals:
            if 'subtle' in uov_method:
                denoising_strength = 0.5
            if 'strong' in uov_method:
                denoising_strength = 0.85
            if advanced_parameters.overwrite_vary_strength > 0:
                denoising_strength = advanced_parameters.overwrite_vary_strength

            shape_ceil = get_image_shape_ceil(uov_input_image)
            if shape_ceil < 1024:
                logger.info(f'[Vary] Image is resized because it is too small.')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                logger.info(f'[Vary] Image is resized because it is too big.')
                shape_ceil = 2048

            uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            logger.info(f'Final resolution is {str((height, width))}.')

        if 'upscale' in goals:
            H, W, C = uov_input_image.shape
            progressbar(async_task, 13, f'Upscaling image from {str((H, W))} ...')
            uov_input_image = perform_upscale(uov_input_image)
            logger.info(f'Image upscaled.')

            if '1.5x' in uov_method:
                f = 1.5
            elif '2x' in uov_method:
                f = 2.0
            else:
                f = 1.0

            shape_ceil = get_shape_ceil(H * f, W * f)

            if shape_ceil < 1024:
                logger.info(f'[Upscale] Image is resized because it is too small.')
                uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
                shape_ceil = 1024
            else:
                uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)

            image_is_super_large = shape_ceil > 2800

            if 'fast' in uov_method:
                direct_return = True
            elif image_is_super_large:
                logger.info('Image is too large. Directly returned the SR image. '
                      'Usually directly return SR image at 4K resolution '
                      'yields better results than SDXL diffusion.')
                direct_return = True
            else:
                direct_return = False

            if direct_return:
                d = {"Upscale By": f, "Upscale Mode": "Fast"}
                is_nsfw, target_image, logged_image_path = log(uov_input_image, d, async_task=async_task)
                yield_result(async_task, target_image, do_not_show_finished_images=True, img_paths=str(logged_image_path), is_nsfw=is_nsfw)
                return

            tiled = True
            denoising_strength = 0.382

            if advanced_parameters.overwrite_upscale_strength > 0:
                denoising_strength = advanced_parameters.overwrite_upscale_strength

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(
                vae=candidate_vae,
                pixels=initial_pixels, tiled=True)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            logger.info(f'Final resolution is {str((height, width))}.')

        if 'inpaint' in goals:
            if len(outpaint_selections) > 0:
                H, W, C = inpaint_image.shape
                if 'top' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                          constant_values=255)
                if 'bottom' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                          constant_values=255)

                H, W, C = inpaint_image.shape
                if 'left' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(H * 0.3), 0]], mode='constant',
                                          constant_values=255)
                if 'right' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(H * 0.3)]], mode='constant',
                                          constant_values=255)

                inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
                advanced_parameters.inpaint_strength = 1.0
                advanced_parameters.inpaint_respective_field = 1.0

            denoising_strength = advanced_parameters.inpaint_strength

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=advanced_parameters.inpaint_respective_field
            )

            if advanced_parameters.debugging_inpaint_preprocessor:
                yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(),
                             do_not_show_finished_images=True)
                return

            progressbar(async_task, 13, 'VAE Inpaint encoding ...')

            inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
            inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
            inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

            candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            latent_inpaint, latent_mask = core.encode_vae_inpaint(
                mask=inpaint_pixel_mask,
                vae=candidate_vae,
                pixels=inpaint_pixel_image)

            latent_swap = None
            if candidate_vae_swap is not None:
                progressbar(async_task, 13, 'VAE SD15 encoding ...')
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap,
                    pixels=inpaint_pixel_fill)['samples']

            progressbar(async_task, 13, 'VAE encoding ...')
            latent_fill = core.encode_vae(
                vae=candidate_vae,
                pixels=inpaint_pixel_fill)['samples']

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

            if inpaint_parameterized:
                pipeline.final_unet = inpaint_worker.current_task.patch(
                    inpaint_head_model_path=inpaint_head_model_path,
                    inpaint_latent=latent_inpaint,
                    inpaint_latent_mask=latent_mask,
                    model=pipeline.final_unet
                )

            if not advanced_parameters.inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}

            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            logger.info(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            for task in cn_tasks[flags.cn_canny]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = preprocessors.canny_pyramid(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_cpds]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = preprocessors.cpds(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_ip]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_ip_face]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = extras.face_crop.crop_image(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return

            all_ip_tasks = cn_tasks[flags.cn_ip] + cn_tasks[flags.cn_ip_face]

            if len(all_ip_tasks) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

        if advanced_parameters.freeu_enabled:
            logger.info(f'FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                advanced_parameters.freeu_b1,
                advanced_parameters.freeu_b2,
                advanced_parameters.freeu_s1,
                advanced_parameters.freeu_s2
            )

        all_steps = steps * image_number

        logger.info(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        logger.info(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        logger.info(f'Preparation time: {preparation_time:.2f} seconds')

        final_sampler_name = sampler_name
        final_scheduler_name = scheduler_name

        if scheduler_name == 'lcm':
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_refiner_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            logger.info('Using lcm scheduler.')

        async_task.yields.append(['preview', (13, 'Moving model to GPU ...', None)])

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * steps + step
            async_task.yields.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}-th Sampling',
                y)])

        for current_task_id, task in enumerate(tasks):
            execution_start_time = time.perf_counter()

            try:
                positive_cond, negative_cond = task['c'], task['uc']

                if 'cn' in goals:
                    for cn_flag, cn_path in [
                        (flags.cn_canny, controlnet_canny_path),
                        (flags.cn_cpds, controlnet_cpds_path)
                    ]:
                        for cn_img, cn_stop, cn_weight in cn_tasks[cn_flag]:
                            positive_cond, negative_cond = core.apply_controlnet(
                                positive_cond, negative_cond,
                                pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)

                imgs = pipeline.process_diffusion(
                    positive_cond=positive_cond,
                    negative_cond=negative_cond,
                    steps=steps,
                    switch=switch,
                    width=width,
                    height=height,
                    image_seed=task['task_seed'],
                    callback=callback,
                    sampler_name=final_sampler_name,
                    scheduler_name=final_scheduler_name,
                    latent=initial_latent,
                    denoise=denoising_strength,
                    tiled=tiled,
                    cfg_scale=cfg_scale,
                    refiner_swap_method=refiner_swap_method
                )

                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                img_paths = []
                target_images = []
                is_nsfw_list = []
                for x in imgs:
                    meta = {
                        'Prompt': task['log_positive_prompt'],
                        'Negative Prompt': task['log_negative_prompt'],
                        'Styles': str(raw_style_selections),
                        'Performance': performance_selection,
                        'Resolution': str((width, height)),
                        'Base Model': base_model_name,
                        'Base Model Hash': all_models.checkpoint_models[base_model_name].sha256,
                        'Refiner Model': refiner_model_name,
                        "Refiner Model Hash": all_models.checkpoint_models[refiner_model_name]
                        if refiner_model_name != "None"
                        else None,
                        'Refiner Switch': refiner_switch,
                        "LoRAs": [{
                            "name": name,
                            "weight": weight,
                            "hash": all_models.lora_models[name].sha256,
                        } for name, weight in loras if name != "None"],
                        'CFG Scale': guidance_scale,
                        'Sharpness': sharpness,
                        'Sampler': sampler_name,
                        'Scheduler': scheduler_name,
                        'Seed': str(task['task_seed']),
                        'Version': fooocus_version.version
                    }

                    is_nsfw, target_image, logged_image_path = log(x, meta, async_task=async_task)
                    img_paths.append(str(logged_image_path))
                    target_images.append(target_image)
                    is_nsfw_list.append(is_nsfw)

                yield_result(
                    async_task,
                    target_images,
                    do_not_show_finished_images=(current_task_id == len(tasks) - 1),
                    img_paths=img_paths,
                    is_nsfw=is_nsfw_list,
                )
            except InterruptProcessingException as e:
                if shared.last_stop == 'skip':
                    logger.info('Task skipped')
                    async_task.yields.append(['skipped', (100 / len(tasks) * (current_task_id + 1), "Task skipped")])
                    continue
                else:
                    logger.info('Task stopped')
                    async_task.yields.append(['stopped', "Task stopped"])
                    break
            finally:
                shared.state["preview_grid"] = None
                shared.state["preview_total"] = 0
                shared.state["preview_count"] = 0

            execution_time = time.perf_counter() - execution_start_time
            logger.info(f'Generating and saving time: {execution_time:.2f} seconds')

        return


    def focus_handler(async_task):
        focus_handler(async_task)
        build_image_wall(async_task)
        pipeline.prepare_text_encoder(async_call=True)


    @torch.no_grad()
    @torch.inference_mode()
    def ruined_handler(async_task):

        args = async_task.args
        args.reverse()

        gen_data = {}

        if advanced_parameters.overwrite_step > 0:
            gen_data["custom_steps"] = advanced_parameters.overwrite_step
        gen_data["sampler_name"] = advanced_parameters.sampler_name
        gen_data["scheduler"] = advanced_parameters.scheduler_name
        gen_data["clip_skip"] = 1
        gen_data["custom_width"] = advanced_parameters.overwrite_width
        gen_data["custom_height"] = advanced_parameters.overwrite_height


        gen_data["prompt"] = args.pop()
        gen_data["negative"] = args.pop()
        gen_data["style_selection"] = args.pop()
        gen_data["performance_selection"] = args.pop()
        gen_data["aspect_ratios_selection"] = args.pop()
        gen_data["image_number"] = args.pop()
        gen_data["seed"] = args.pop()
        gen_data["_sharpness"] = args.pop()
        if task.task_type != 'flux' and int(gen_data["_sharpness"]) >= 1 and int(gen_data["_sharpness"]) <= 5:
            gen_data["clip_skip"] = int(gen_data["_sharpness"])
        gen_data["cfg"] = args.pop()
        gen_data["base_model_name"] = args.pop()
        gen_data["_refiner_model_name"] = args.pop()
        gen_data["_refiner_switch"] = args.pop()
        gen_data["loras"] = [(str(args.pop()), float(args.pop())) for _ in range(5)]
        gen_data["_input_image_checkbox"] = args.pop()
        gen_data["_current_tab"] = args.pop()
        gen_data["_uov_method"] = args.pop()
        gen_data["_uov_input_image"] = args.pop()
        gen_data["_outpaint_selections"] = args.pop()
        gen_data["_inpaint_input_image"] = args.pop()
        gen_data["_inpaint_additional_prompt"] = args.pop()
        gen_data["_inpaint_mask_image_upload"] = args.pop()

        gen_data["cn_selection"] = gen_data["cn_type"] = "None"
        if gen_data["_input_image_checkbox"] and gen_data["_current_tab"] == "uov" and gen_data["_uov_input_image"]:
            if "vary" in gen_data["_uov_method"].lower():
                gen_data["cn_selection"] = gen_data["cn_type"] = "img2img"
                gen_data["start"] = 0.06
                gen_data["denoise"] = 0.64
                gen_data["input_image"] = gen_data["_uov_input_image"]
            elif "upscale" in gen_data["_uov_method"].lower():
                gen_data["cn_selection"] = gen_data["cn_type"] = "upscale"
                gen_data["cn_upscale"] = "4x-UltraSharp.pth"
                gen_data["input_image"] = gen_data["_uov_input_image"]

        gen_data["inpaint_toggle"] = False
        gen_data["inpaint_view"] = {}
        if gen_data["_input_image_checkbox"] and gen_data["_current_tab"] == "inpaint" and gen_data["_inpaint_input_image"] and gen_data["_inpaint_mask_image_upload"]:
            gen_data["inpaint_toggle"] = True
            gen_data["inpaint_view"]["mask"] = gen_data["_inpaint_mask_image_upload"]
            gen_data["inpaint_view"]["image"] = gen_data["_inpaint_input_image"]

        if gen_data["_input_image_checkbox"] and gen_data["_current_tab"] == "ip":
            for _ in range(4):
                cn_img = args.pop()
                cn_stop = args.pop()
                cn_weight = args.pop()
                cn_type = args.pop()
                if cn_img is not None:
                    gen_data["cn_selection"] = gen_data["cn_type"] = cn_type
                    if cn_type.lower() == 'canny':
                        gen_data["cn_edge_low"] = 0.2
                        gen_data["cn_edge_high"] = 0.8
                    gen_data["cn_start"] = 0.0
                    gen_data["cn_stop"] = cn_stop
                    gen_data["cn_strength"] = cn_weight
                    gen_data["input_image"] = cn_img


        gen_data["generate_forever"] = False
        gen_data["obp_assume_direct_control"] = False
        #gen_data["OBP_preset"]
        #gen_data["obp_insanitylevel"]
        #gen_data["obp_subject"]
        #gen_data["obp_artist"]
        #gen_data["obp_chosensubjectsubtypeobject"]
        #gen_data["obp_chosensubjectsubtypehumanoid"]
        #gen_data["obp_chosensubjectsubtypeconcept"]
        #gen_data["obp_chosengender"]
        #gen_data["obp_imagetype"]
        #gen_data["obp_imagemodechance"]
        #gen_data["obp_givensubject"]
        #gen_data["obp_smartsubject"]
        #gen_data["obp_givenoutfit"]
        #gen_data["obp_prefixprompt"]
        #gen_data["obp_suffixprompt"]
        #gen_data["obp_giventypeofimage"]
        #gen_data["obp_antistring"]
        #gen_data["OBP_modeltype"]
        #gen_data["OBP_promptenhance"]

        if gen_data["negative"]:
            gen_data["auto_negative"] = False
        else:
            gen_data["auto_negative"] = True

        gen_data["lora_keywords"] = ""
        model_info = all_models.get_model(gen_data["base_model_name"])
        assert model_info is not None, f"Model {gen_data['base_model_name']} not found"

        gen_data = process_metadata(gen_data)

        shared.state["preview_grid"] = None
        shared.state["preview_total"] = max(gen_data["image_number"], 1)
        shared.state["preview_count"] = 0

        ruined_pipeline = modules.pipelines.update(gen_data)
        if ruined_pipeline == None:
            logger.info(f"ERROR: No pipeline")
            return
        if isinstance(ruined_pipeline, modules.pipelines.NoPipeLine):
            logger.info(f"ERROR: No pipeline")
            return

        try:
            # See if ruined_pipeline wants to pre-parse gen_data
            _parse_gen_data = getattr(ruined_pipeline, "parse_gen_data", None)
            if callable(_parse_gen_data):
                gen_data = ruined_pipeline.parse_gen_data(gen_data)
        except:
            pass

        image_number = gen_data["image_number"]

        loras = gen_data["loras"]

        parsed_loras, pos_stripped, neg_stripped = parse_loras(
            gen_data["prompt"], gen_data["negative"]
        )
        loras.extend(parsed_loras)

        progressbar(async_task, 1, f"Loading base model: {gen_data['base_model_name']}")
        _load_base_model = getattr(ruined_pipeline, "load_base_model", None)
        if not callable(_load_base_model):
            logger.info(f"ERROR: No load_base_model for pipeline")
            return
        gen_data["modelhash"] = ruined_pipeline.load_base_model(gen_data["base_model_name"])
        progressbar(async_task, 1, "Loading LoRA models ...")
        ruined_pipeline.load_loras(loras)

        if (
            gen_data["performance_selection"]
            == shared.performance_settings.CUSTOM_PERFORMANCE
        ):
            steps = gen_data["custom_steps"]
        else:
            perf_options = shared.performance_settings.get_perf_options(
                gen_data["performance_selection"]
            ).copy()
            perf_options.update(gen_data)
            gen_data = perf_options

        # TODO: Put this in config
        if model_info.base and model_info.base.lower() == "flux.1 s":
            if gen_data["performance_selection"].lower() == "speed":
                gen_data["custom_steps"] = 10
            elif gen_data["performance_selection"].lower() == "quality":
                gen_data["custom_steps"] = 20

        steps = gen_data["custom_steps"]

        if (
            gen_data["aspect_ratios_selection"]
            == shared.resolution_settings.CUSTOM_RESOLUTION
        ):
            width, height = (gen_data["custom_width"], gen_data["custom_height"])
        else:
            if gen_data["aspect_ratios_selection"] in shared.resolution_settings.aspect_ratios:
                width, height = shared.resolution_settings.aspect_ratios[
                    gen_data["aspect_ratios_selection"]
                ]
            else:
                a, b = gen_data["aspect_ratios_selection"].replace('x', ' ').split(' ')[:2]
                width, height = int(a), int(b)

        if "width" in gen_data:
            width = gen_data["width"]
        if "height" in gen_data:
            height = gen_data["height"]

        if gen_data.get("cn_selection", "").lower() == "img2img" or gen_data.get("cn_type", "").lower() == "img2img":
            if gen_data["input_image"]:
                width = gen_data["input_image"].width
                height = gen_data["input_image"].height
            else:
                logger.info(f"WARNING: CheatCode selected but no Input image selected. Ignoring PowerUp!")
                gen_data["cn_selection"] = "None"
                gen_data["cn_type"] = "None"

        seed = gen_data["seed"]

        max_seed = 2**32
        if not isinstance(seed, int) or seed < 0:
            seed = random.randint(0, max_seed)
        seed = seed % max_seed

        all_steps = steps * max(image_number, 1)

        def callback(step, x0, x, total_steps, y):

            if processing_interrupted():
                shared.state["interrupted"] = True
                throw_exception_if_processing_interrupted()

            if isinstance(y, Image.Image):
                y = np.array(y)

            # If we only generate 1 image, skip the last preview
            if (
                (not gen_data["generate_forever"])
                and shared.state["preview_total"] == 1
                and steps == step
            ):
                return

            done_steps = i * steps + step

            async_task.yields.append([
                "preview",
                (
                    int(
                        100
                        * (done_steps / all_steps)
                    ),
                    f'Step {step}/{total_steps} in the {i + 1}-th Sampling',
                    y
                )
            ])

        for i in range(max(image_number, 1)):
            p_txt, n_txt = process_prompt(
                gen_data["style_selection"], pos_stripped, neg_stripped, gen_data
            )
            start_step = 0
            denoise = None
            with TimeIt("Pipeline process"):
                try:
                    imgs = ruined_pipeline.process(
                        p_txt,
                        n_txt,
                        gen_data.get("input_image", None),
                        modules.controlnet.get_settings(gen_data),
                        gen_data.get("main_view", None),
                        steps,
                        width,
                        height,
                        seed,
                        start_step,
                        denoise,
                        gen_data["cfg"],
                        gen_data["sampler_name"],
                        gen_data["scheduler"],
                        gen_data["clip_skip"],
                        callback=callback,
                        gen_data=gen_data,
                        progressbar=functools.partial(progressbar, async_task),
                    )

                    img_paths = []
                    target_images = []
                    is_nsfw_list = []
                    for x in imgs:
                        meta = {
                            "Prompt": gen_data["prompt"],
                            "Negative Prompt": gen_data["negative"],
                            "Styles": gen_data["style_selection"],
                            "Performance": gen_data["performance_selection"],
                            "Resolution": str((width, height)),
                            "Base Model": gen_data["base_model_name"],
                            "Base Model Hash": model_info.sha256,
                            "Refiner Model": gen_data["_refiner_model_name"],
                            "Refiner Model Hash": all_models.checkpoint_models[
                                gen_data["_refiner_model_name"]
                            ]
                            if gen_data["_refiner_model_name"] != "None"
                            else None,
                            "Refiner Switch": gen_data["_refiner_switch"],
                            "LoRAs": [{
                                "name": name,
                                "weight": weight,
                                "hash": all_models.lora_models[name].sha256,
                            } for name, weight in loras if name != "None"],
                            "CFG Scale": gen_data["cfg"],
                            "Clip Skip": gen_data["clip_skip"],
                            "Sampler": gen_data["sampler_name"],
                            "Scheduler": gen_data["scheduler"],
                            "Seed": str(seed),
                            "Version": fooocus_version.version,
                        }
                        is_nsfw, target_image, logged_image_path = log(x, meta, async_task=async_task)
                        img_paths.append(str(logged_image_path))
                        target_images.append(target_image)
                        is_nsfw_list.append(is_nsfw)

                        shared.state["preview_count"] += 1

                    seed += 1
                    yield_result(
                        async_task,
                        target_images,
                        do_not_show_finished_images=(i == max(image_number, 1) - 1),
                        img_paths=img_paths,
                        is_nsfw=is_nsfw_list,
                    )
                except InterruptProcessingException as e:
                    if shared.last_stop == 'skip':
                        logger.info('User skipped')
                        async_task.yields.append(['skipped', (100 / max(image_number, 1) * (i + 1), "User skipped")])
                        continue
                    else:
                        logger.info('User stopped')
                        async_task.yields.append(['stopped', "User stopped"])
                        break

        return

    script_callbacks.app_ready_callback()
    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)
            try:
                running_task = task
                script_callbacks.before_task_callback(task.task_id)
                if task.task_type == 'sd3' or task.task_type == 'flux':
                    pipeline.clear_pipeline()
                    with memory_context_manager("Ruined Handler"):
                        ruined_handler(task)
                else:
                    modules.pipelines.clear_pipeline()
                    with memory_context_manager("Focus Handler"):
                        handler(task)
                task.yields.append(['finish', task.results])
            except Exception as e:
                traceback.print_exc()
                task.yields.append(['failed', e.__str__()])
            finally:
                finished_tasks.append(task)
                script_callbacks.after_task_callback(task.task_id)
                running_task = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()


def start():
    threading.Thread(target=worker, daemon=True).start()


if __name__ == '__main__':
    start()
