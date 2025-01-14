import gradio as gr
import random
import os
import json
import time
import shared
from datetime import datetime, timedelta
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.advanced_parameters as advanced_parameters
import modules.style_sorter as style_sorter
import modules.meta_parser
import args_manager
import copy
import logging
import uuid
import asyncio
from typing import Any, Literal
import yaml

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules import script_callbacks
from modules.util import setup_logging

from fastapi import FastAPI
from api import Status, QueuingStatus, Progress, create_api


setup_logging()
logger = logging.getLogger("default")


async def generate_clicked(
    *args,
    base_dir: str | None = None,
    task_id: str = '',
    metadata: dict[str, Any] | None = None,
    task_type: Literal["sdxl", "sd3", "flux"] | None = None,
):
    import ldm_patched.modules.model_management as model_management
    if not task_id:
        task_id = str(uuid.uuid4())

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    # outputs=[progress_html, progress_window, progress_gallery, gallery]

    execution_start_time = time.perf_counter()
    task = worker.AsyncTask(
        task_id=task_id, args=list(args), base_dir=base_dir, metadata=metadata, task_type=task_type)
    finished = False

    yield Progress(flag='preparing', task_id=task_id, status=Status(percentage=1, title='Waiting for task to start ...', images=[]))

    logger.info(f"generate_clicked, got a task {task.task_id} from client, push it to worker.async_tasks")
    worker.async_tasks.append(task)

    started = False
    last_update_time = datetime.now()
    while not finished:
        await asyncio.sleep(0.01)
        if not started:
            current_time = datetime.now()
            if (current_time - last_update_time) >= timedelta(seconds=1):
                last_update_time = current_time
                queue_status = QueuingStatus()
                for idx, task in enumerate(worker.async_tasks):
                    if task.task_id == task_id:
                        queue_status.position = idx + 1
                        queue_status.total = len(worker.async_tasks)
                        break
                if queue_status.position > 0:
                    yield Progress(
                        flag='queueing',
                        task_id=task_id,
                        status=Status(percentage=1, title=f'Waiting in the queue {queue_status.position}/{queue_status.total}', images=[]),
                        queuing_status=queue_status)
                else:
                    started = True
        if len(task.yields) > 0:
            started = True
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                current_time = datetime.now()
                if (current_time - last_update_time) < timedelta(seconds=0.2):
                    continue
                last_update_time = current_time
                percentage, title, image = product
                yield Progress(
                    flag='preview', task_id=task_id, status=Status(percentage=percentage, title=title, images=[image] if image is not None else []))
            if flag == 'results':
                yield Progress(
                    flag='results', task_id=task_id, status=Status(percentage=100, title='Results', images=product, image_filepaths=task.result_paths, is_nsfw=task.is_nsfw))
            if flag == 'finish':
                yield Progress(
                    flag='finish', task_id=task_id, status=Status(percentage=100, title='Finished', images=product, image_filepaths=task.result_paths, is_nsfw=task.is_nsfw))
                finished = True
            if flag == 'skipped':
                percentage, title = product
                yield Progress(
                    flag='skipped', task_id=task_id, status=Status(percentage=percentage, title=title, images=[]))
            if flag == 'stopped':
                yield Progress(
                    flag='stopped', task_id=task_id, status=Status(percentage=100, title=product, images=[]))
            if flag == 'failed':
                yield Progress(
                    flag='failed', task_id=task_id, status=Status(percentage=100, title='Failed: ' + product, images=[]))
                finished = True

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


async def recover_task(task_id: str):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    execution_start_time = time.perf_counter()
    task = None
    finished = False
    started = False

    for finished_task in worker.finished_tasks:
        if finished_task.task_id == task_id:
            task = finished_task
            break
    for stopped_task in worker.stop_or_skipped_tasks:
        if stopped_task.task_id == task_id:
            task = stopped_task
            break
    if task:
        finished = True
        flag, product = task.yields.pop()
        if flag == "failed":
            yield Progress(
                flag='failed', task_id=task_id, status=Status(percentage=100, title='Failed: ' + product, images=[]))
        else:
            yield Progress(
                flag='finish', task_id=task_id, status=Status(percentage=100, title='Finished', images=task.results, image_filepaths=task.result_paths))
        task.yields = []

    if worker.running_task and worker.running_task.task_id == task_id:
        started = True
        task = worker.running_task

    for waiting_task in worker.async_tasks:
        if waiting_task.task_id == task_id:
            task = waiting_task
            break

    if task is None:
        yield Progress(
            flag='unfound', task_id=task_id, status=Status(percentage=0, title=f'Task {task_id} could not be found', images=[]))
        execution_time = time.perf_counter() - execution_start_time
        print(f'Total time: {execution_time:.2f} seconds')
        return

    last_update_time = datetime.now()
    while not finished:
        await asyncio.sleep(0.01)
        if not started:
            current_time = datetime.now()
            if (current_time - last_update_time) >= timedelta(seconds=1):
                last_update_time = current_time
                queue_status = QueuingStatus()
                for idx, task in enumerate(worker.async_tasks):
                    if task.task_id == task_id:
                        queue_status.position = idx + 1
                        queue_status.total = len(worker.async_tasks)
                        break
                if queue_status.position > 0:
                    yield Progress(
                        flag='queueing',
                        task_id=task_id,
                        status=Status(percentage=1, title=f'Waiting in the queue {queue_status.position}/{queue_status.total}', images=[]),
                        queuing_status=queue_status)
                else:
                    started = True
        if len(task.yields) > 0:
            started = True
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                current_time = datetime.now()
                if (current_time - last_update_time) < timedelta(seconds=0.2):
                    continue
                last_update_time = current_time
                percentage, title, image = product
                yield Progress(
                    flag='preview', task_id=task_id, status=Status(percentage=percentage, title=title, images=[image] if image is not None else []))
            if flag == 'results':
                yield Progress(
                    flag='results', task_id=task_id, status=Status(percentage=100, title='Results', images=product, image_filepaths=task.result_paths, is_nsfw=task.is_nsfw))
            if flag == 'finish':
                yield Progress(
                    flag='finish', task_id=task_id, status=Status(percentage=100, title='Finished', images=product, image_filepaths=task.result_paths, is_nsfw=task.is_nsfw))
                finished = True
            if flag == 'skipped':
                percentage, title = product
                yield Progress(
                    flag='skipped', task_id=task_id, status=Status(percentage=percentage, title=title, images=[]))
            if flag == 'stopped':
                yield Progress(
                    flag='stopped', task_id=task_id, status=Status(percentage=100, title=product, images=[]))
            if flag == 'failed':
                yield Progress(
                    flag='failed', task_id=task_id, status=Status(percentage=100, title='Failed: ' + product, images=[]))
                finished = True

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


async def generate_clicked_gradio(*args):
    async for progress in generate_clicked(*args):
        if progress.flag == "preparing":
            yield (
                gr.update(value=progress.task_id),
                gr.update(
                    visible=True,
                    value=modules.html.make_progress_html(
                        progress.status.percentage, progress.status.title
                    ),
                ),
                gr.update(visible=True, value=None),
                gr.update(visible=False, value=None),
                gr.update(visible=False),
            )
        elif progress.flag == "queueing":
            yield (
                gr.update(value=progress.task_id),
                gr.update(
                    visible=True,
                    value=modules.html.make_progress_html(
                        progress.status.percentage, progress.status.title
                    ),
                ),
                gr.update(visible=True, value=None),
                gr.update(visible=False, value=None),
                gr.update(visible=False),
            )
        elif progress.flag == "preview":
            yield (
                gr.update(value=progress.task_id),
                gr.update(
                    visible=True,
                    value=modules.html.make_progress_html(
                        progress.status.percentage, progress.status.title
                    ),
                ),
                gr.update(visible=True, value=progress.status.images[0])
                if len(progress.status.images) > 0
                else gr.update(),
                gr.update(),
                gr.update(visible=False),
            )
        elif progress.flag == "skipped":
            yield (
                gr.update(value=progress.task_id),
                gr.update(
                    visible=True,
                    value=modules.html.make_progress_html(
                        progress.status.percentage, progress.status.title
                    ),
                ),
                gr.update(),
                gr.update(),
                gr.update(visible=False),
            )
        elif progress.flag == "results":
            yield (
                gr.update(value=progress.task_id),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True, value=progress.status.images),
                gr.update(visible=False),
            )
        elif progress.flag == "stopped":
            yield (
                gr.update(value=progress.task_id),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(),
            )
        elif progress.flag == "finish":
            yield (
                gr.update(value=progress.task_id),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True, value=progress.status.images),
            )
        elif progress.flag == "failed":
            yield (
                gr.update(value=progress.task_id),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(),
            )
    return


reload_javascript()


def stop_clicked(task_id: str):
    if worker.running_task and worker.running_task.task_id == task_id:
        import ldm_patched.modules.model_management as model_management
        shared.last_stop = 'stop'
        model_management.interrupt_current_processing()
        return True
    task = None
    for task in worker.async_tasks:
        if task.task_id == task_id:
            break
    if task:
        try:
            worker.async_tasks.remove(task)
            worker.stop_or_skipped_tasks.append(task)
            task.yields.append(['stopped', "User stopped"])
            task.yields.append(['finish', []])
        except ValueError:
            return False
        return True
    for task in worker.finished_tasks:
        if task.task_id == task_id:
            return True
    for task in worker.stop_or_skipped_tasks:
        if task.task_id == task_id:
            return True
    return False

def skip_clicked(task_id: str):
    if worker.running_task and worker.running_task.task_id == task_id:
        import ldm_patched.modules.model_management as model_management
        shared.last_stop = 'skip'
        model_management.interrupt_current_processing()
        return True
    task = None
    for task in worker.async_tasks:
        if task.task_id == task_id:
            break
    if task:
        try:
            worker.async_tasks.remove(task)
            worker.stop_or_skipped_tasks.append(task)
            task.yields.append(['skipped', (0, "User skipped")])
            task.yields.append(['finish', []])
        except ValueError:
            return False
        return True
    for task in worker.finished_tasks:
        if task.task_id == task_id:
            return True
    for task in worker.stop_or_skipped_tasks:
        if task.task_id == task_id:
            return True
    return False


def refresh_seed(r, seed_string):
    if r:
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
    else:
        try:
            seed_value = int(seed_string)
            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                return seed_value
        except ValueError:
            pass
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)


def trigger_describe(mode, img):
    if mode == flags.desc_type_photo:
        from extras.interrogate import default_interrogator as default_interrogator_photo
        return default_interrogator_photo(img), ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"]
    if mode == flags.desc_type_anime:
        from extras.wd14tagger import default_interrogator as default_interrogator_anime
        return default_interrogator_anime(img), ["Fooocus V2", "Fooocus Masterpiece"]
    return mode, ["Fooocus V2"]


def make_gradio_ui(root_block: gr.Blocks):
    with gr.Row():
        with gr.Column(scale=2):
            task_id = gr.Textbox(label='Task ID', value='', visible=False, elem_id='task_id')
            with gr.Row():
                progress_window = grh.Image(label='Preview', show_label=True, visible=False, height=768,
                                            elem_classes=['main_view'])
                progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain',
                                              height=768, visible=False, elem_classes=['main_view', 'image_gallery'])
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', visible=True, height=768,
                                 elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                 elem_id='final_gallery')
            with gr.Row(elem_classes='type_row'):
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        container=False, autofocus=True, elem_classes='type_row', lines=1024)

                    default_prompt = modules.config.default_prompt
                    if isinstance(default_prompt, str) and default_prompt != '':
                        root_block.load(lambda: default_prompt, outputs=prompt)

                with gr.Column(scale=3, min_width=0):
                    generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                    load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', visible=False)
                    stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

                    def stop_clicked_gradio(task_id):
                        stop_clicked(task_id)
                        return [gr.update(interactive=False)] * 2

                    def skip_clicked_gradio(task_id):
                        skip_clicked(task_id)
                        return

                    stop_button.click(stop_clicked_gradio, inputs=[task_id], outputs=[skip_button, stop_button],
                                      queue=False, show_progress=False, _js='cancelGenerateForever')
                    skip_button.click(skip_clicked_gradio, inputs=[task_id], queue=False, show_progress=False)
            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=False, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
            with gr.Row(visible=False) as image_input_panel:
                with gr.Tabs():
                    with gr.TabItem(label='Upscale or Variation') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='Drag above image to here', source='upload', type='numpy')
                            with gr.Column():
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=flags.disabled)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Document</a>')
                    with gr.TabItem(label='Image Prompt') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for _ in range(4):
                                with gr.Column():
                                    ip_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False, height=300)
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=False) as ad_col:
                                        with gr.Row():
                                            default_end, default_weight = flags.default_parameters[flags.default_ip]

                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=default_end)
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=default_weight)
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=flags.default_ip, container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        ip_advanced = gr.Checkbox(label='Advanced', value=False, container=False)
                        gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Document</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)
                    with gr.TabItem(label='Inpaint or Outpaint') as inpaint_tab:
                        with gr.Row():
                            inpaint_input_image = grh.Image(label='Drag inpaint or outpaint image to here', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas')
                            inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy', height=500, visible=False)

                        with gr.Row():
                            inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                            outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                            inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.flags.inpaint_option_default, label='Method')
                        example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts, label='Additional Prompt Quick List', components=[inpaint_additional_prompt], visible=False)
                        gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Document</a>')
                        example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)
                    with gr.TabItem(label='Describe') as desc_tab:
                        with gr.Row():
                            with gr.Column():
                                desc_input_image = grh.Image(label='Drag any image to here', source='upload', type='numpy')
                            with gr.Column():
                                desc_method = gr.Radio(
                                    label='Content Type',
                                    choices=[flags.desc_type_photo, flags.desc_type_anime],
                                    value=flags.desc_type_photo)
                                desc_btn = gr.Button(value='Describe this Image into Prompt')
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Document</a>')
            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            down_js = "() => {viewer_to_bottom();}"

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                        outputs=image_input_panel, queue=False, show_progress=False, _js=switch_js)
            ip_advanced.change(lambda: None, queue=False, show_progress=False, _js=down_js)

            current_tab = gr.Textbox(value='uov', visible=False)
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            desc_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox) as advanced_column:
            with gr.Tab(label='Setting'):
                performance_selection = gr.Radio(label='Performance',
                                                 choices=modules.flags.performance_selections,
                                                 value=modules.config.default_performance)
                aspect_ratios_selection = gr.Radio(label='Aspect Ratios', choices=modules.config.available_aspect_ratios,
                                                   value=modules.config.default_aspect_ratio, info='width × height',
                                                   elem_classes='aspect_ratios')
                image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)
                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.",
                                             info='Describing what you do not want to see.', lines=2,
                                             elem_id='negative_prompt',
                                             value=modules.config.default_prompt_negative)
                seed_random = gr.Checkbox(label='Random', value=True)
                image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False) # workaround for https://github.com/gradio-app/gradio/issues/5354

                def random_checked(r):
                    return gr.update(visible=not r)

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                   queue=False, show_progress=False)

                if not args_manager.args.disable_image_log:
                    gr.HTML(f'<a href="file={get_current_html_path()}" target="_blank">\U0001F4DA History Log</a>')

            with gr.Tab(label='Style'):
                style_search_bar = gr.Textbox(show_label=False, container=False,
                                              placeholder="\U0001F50E Type here to search styles ...",
                                              value="",
                                              label='Search Styles')
                style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=copy.deepcopy(style_sorter.all_styles),
                                                    value=copy.deepcopy(modules.config.default_styles),
                                                    label='Selected Styles',
                                                    elem_classes=['style_selections'])
                gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)

                root_block.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                                outputs=style_selections)

                style_search_bar.change(style_sorter.search_styles,
                                        inputs=[style_selections, style_search_bar],
                                        outputs=style_selections,
                                        queue=False,
                                        show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

                gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                       inputs=style_selections,
                                                       outputs=style_selections,
                                                       queue=False,
                                                       show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

            with gr.Tab(label='Model'):
                with gr.Group():
                    with gr.Row():
                        base_model = gr.Dropdown(label='Base Model (SDXL only)', choices=modules.config.model_filenames, value=modules.config.default_base_model_name, show_label=True)
                        refiner_model = gr.Dropdown(label='Refiner (SDXL or SD 1.5)', choices=['None'] + modules.config.model_filenames, value=modules.config.default_refiner_model_name, show_label=True)

                    refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                               info='Use 0.4 for SD1.5 realistic models; '
                                                    'or 0.667 for SD1.5 anime models; '
                                                    'or 0.8 for XL-refiners; '
                                                    'or any value for switching two SDXL models.',
                                               value=modules.config.default_refiner_switch,
                                               visible=modules.config.default_refiner_model_name != 'None')

                    refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                                         inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)

                with gr.Group():
                    lora_ctrls = []

                    for i, (n, v) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=['None'] + modules.config.lora_filenames, value=n)
                            lora_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.01, value=v,
                                                    elem_classes='lora_weight')
                            lora_ctrls += [lora_model, lora_weight]

                with gr.Row():
                    model_refresh = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
            with gr.Tab(label='Advanced'):
                guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                           value=modules.config.default_cfg_scale,
                                           info='Higher value means style is cleaner, vivider, and more artistic.')
                sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                                      value=modules.config.default_sample_sharpness,
                                      info='Higher value means image and texture are sharper.')
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Document</a>')
                dev_mode = gr.Checkbox(label='Developer Debug Mode', value=False, container=False)

                with gr.Column(visible=False) as dev_tools:
                    with gr.Tab(label='Debug Tools'):
                        adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=1.5, info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
                        adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=0.8, info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
                        adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                                   step=0.001, value=0.3,
                                                   info='When to end the guidance from positive/negative ADM. ')

                        refiner_swap_method = gr.Dropdown(label='Refiner swap method', value='joint',
                                                          choices=['joint', 'separate', 'vae'])

                        adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=modules.config.default_cfg_tsnr,
                                                 info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                      '(effective when real CFG > mimicked CFG).')
                        sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                   value=modules.config.default_sampler)
                        scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                     value=modules.config.default_scheduler)

                        generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                                          info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
                                                          value=False)

                        overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                   minimum=-1, maximum=200, step=1,
                                                   value=modules.config.default_overwrite_step,
                                                   info='Set as -1 to disable. For developer debugging.')
                        overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                     minimum=-1, maximum=200, step=1,
                                                     value=modules.config.default_overwrite_switch,
                                                     info='Set as -1 to disable. For developer debugging.')
                        overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                                    minimum=-1, maximum=2048, step=1, value=-1,
                                                    info='Set as -1 to disable. For developer debugging. '
                                                         'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                                     minimum=-1, maximum=2048, step=1, value=-1,
                                                     info='Set as -1 to disable. For developer debugging. '
                                                          'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                            minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                            info='Set as negative number to disable. For developer debugging.')
                        overwrite_upscale_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Upscale"',
                                                               minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                               info='Set as negative number to disable. For developer debugging.')
                        disable_preview = gr.Checkbox(label='Disable Preview', value=False,
                                                      info='Disable preview during generation.')

                    with gr.Tab(label='Control'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                        skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')

                        mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                                                                           value=False)
                        mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                                      value=False)

                        controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                        step=0.001, value=0.25,
                                                        info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')

                        with gr.Tab(label='Canny'):
                            canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                            step=1, value=64)
                            canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1, maximum=255,
                                                             step=1, value=128)

                    with gr.Tab(label='Inpaint'):
                        debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
                        inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
                        inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                     value=modules.config.default_inpaint_engine_version,
                                                     choices=flags.inpaint_engine_versions,
                                                     info='Version of Fooocus inpaint model')
                        inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                     minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                          'Only used in inpaint, not used in outpaint. '
                                                          '(Outpaint always use 1.0)')
                        inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                             minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                             info='The area to inpaint. '
                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                  'Only used in inpaint, not used in outpaint. '
                                                                  '(Outpaint always use 1.0)')
                        inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                            minimum=-64, maximum=64, step=1, value=0,
                                                            info='Positive value will make white area in the mask larger, '
                                                                 'negative value will make white area smaller.'
                                                                 '(default is 0, always process before any mask invert)')
                        inpaint_mask_upload_checkbox = gr.Checkbox(label='Enable Mask Upload', value=False)
                        invert_mask_checkbox = gr.Checkbox(label='Invert Mask', value=False)

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                        inpaint_mask_upload_checkbox.change(lambda x: gr.update(visible=x),
                                                           inputs=inpaint_mask_upload_checkbox,
                                                           outputs=inpaint_mask_image, queue=False, show_progress=False)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                adps = [disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name,
                        scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width, overwrite_height,
                        overwrite_vary_strength, overwrite_upscale_strength,
                        mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint,
                        debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness,
                        canny_low_threshold, canny_high_threshold, refiner_swap_method]
                adps += freeu_ctrls
                adps += inpaint_ctrls

                def dev_mode_checked(r):
                    return gr.update(visible=r)


                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                queue=False, show_progress=False)

                def model_refresh_clicked():
                    modules.config.update_all_model_names()
                    results = []
                    results += [gr.update(choices=modules.config.model_filenames), gr.update(choices=['None'] + modules.config.model_filenames)]
                    for i in range(5):
                        results += [gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
                    return results

                model_refresh.click(model_refresh_clicked, [], [base_model, refiner_model] + lora_ctrls,
                                    queue=False, show_progress=False)

        performance_selection.change(lambda x: [gr.update(interactive=x != 'Extreme Speed')] * 11 +
                                               [gr.update(visible=x != 'Extreme Speed')] * 1,
                                     inputs=performance_selection,
                                     outputs=[
                                         guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                         scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt
                                     ], queue=False, show_progress=False)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                 queue=False, show_progress=False) \
            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)

        def inpaint_mode_change(mode):
            assert mode in modules.flags.inpaint_options

            # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            # inpaint_disable_initial_latent, inpaint_engine,
            # inpaint_strength, inpaint_respective_field

            if mode == modules.flags.inpaint_option_detail:
                return [
                    gr.update(visible=True), gr.update(visible=False, value=[]),
                    gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
                    False, 'None', 0.5, 0.0
                ]

            if mode == modules.flags.inpaint_option_modify:
                return [
                    gr.update(visible=True), gr.update(visible=False, value=[]),
                    gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
                    True, modules.config.default_inpaint_engine_version, 1.0, 0.0
                ]

            return [
                gr.update(visible=False, value=''), gr.update(visible=True),
                gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
                False, modules.config.default_inpaint_engine_version, 1.0, 0.618
            ]

        inpaint_mode.input(inpaint_mode_change, inputs=inpaint_mode, outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)

        ctrls = [
            prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number, image_seed, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
        ctrls += ip_ctrls

        state_is_generating = gr.State(False)

        def parse_meta(raw_prompt_txt, is_generating):
            loaded_json = None
            try:
                if '{' in raw_prompt_txt:
                    if '}' in raw_prompt_txt:
                        if ':' in raw_prompt_txt:
                            loaded_json = json.loads(raw_prompt_txt)
                            assert isinstance(loaded_json, dict)
            except:
                loaded_json = None

            if loaded_json is None:
                if is_generating:
                    return gr.update(), gr.update(), gr.update()
                else:
                    return gr.update(), gr.update(visible=True), gr.update(visible=False)

            return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)

        prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

        load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating], outputs=[
            advanced_checkbox,
            image_number,
            prompt,
            negative_prompt,
            style_selections,
            performance_selection,
            aspect_ratios_selection,
            overwrite_width,
            overwrite_height,
            sharpness,
            guidance_scale,
            adm_scaler_positive,
            adm_scaler_negative,
            adm_scaler_end,
            base_model,
            refiner_model,
            refiner_switch,
            sampler_name,
            scheduler_name,
            seed_random,
            image_seed,
            generate_button,
            load_parameter_button
        ] + lora_ctrls, queue=False, show_progress=False)

        generate_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True),
                              outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating]) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(advanced_parameters.set_all_advanced_parameters, inputs=adps) \
            .then(fn=generate_clicked_gradio, inputs=ctrls, outputs=[task_id, progress_html, progress_window, progress_gallery, gallery]) \
            .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
                  outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

        desc_btn.click(trigger_describe, inputs=[desc_method, desc_input_image],
                       outputs=[prompt, style_selections], show_progress=True, queue=True)


def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


# dump_default_english_config()


def launch_uvicorn_server(app: FastAPI, log_level: str = "info"):
    import uvicorn
    from gradio.networking import Server
    port = getattr(app, 'port', args_manager.args.port)
    host = getattr(app, 'host', args_manager.args.listen)
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = Server(config)
    server.run_in_thread()


def launch_app_without_gradio(server_port: int, lifespan = None) -> FastAPI:
    from starlette.middleware.cors import CORSMiddleware
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    if lifespan is None:
        app = FastAPI()
    else:
        app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
    setattr(app, 'port', server_port)
    setattr(app, 'host', args_manager.args.listen)
    return app


def launch_app_with_gradio(server_port: int):
    title = f'Fooocus {fooocus_version.version}'

    if isinstance(args_manager.args.preset, str):
        title += ' ' + args_manager.args.preset
    shared.gradio_root = gr.Blocks(
        title=title,
        css=modules.html.css).queue()
    with shared.gradio_root:
        make_gradio_ui(shared.gradio_root)
        app, _, _ = shared.gradio_root.launch(
            inbrowser=False,
            server_name=args_manager.args.listen,
            server_port=server_port,
            share=args_manager.args.share,
            auth=check_auth if args_manager.args.share and auth_enabled else None,
            blocked_paths=[constants.AUTH_FILENAME],
            prevent_thread_lock=True,
            app_kwargs={
                "docs_url": "/docs",
                "redoc_url": "/redoc",
            },
        )
        return app


async def block_thread(app: FastAPI):
    logger.info("Starting the async loop and waiting on server")
    called_worker_start = False
    if args_manager.args.no_gradio:
        launch_uvicorn_server(app)
    try:
        while True:
            if not called_worker_start and args_manager.args.lazy and len(worker.async_tasks) > 0:
                worker.start()
                called_worker_start = True
            await asyncio.sleep(1)
    except (KeyboardInterrupt, OSError):
        logger.info("Keyboard interruption in main thread... closing server.")
        if shared.gradio_root:
            shared.gradio_root.close()
    finally:
        script_callbacks.app_stopped_callback()


def start(server_port: int = 0):
    style_sorter.try_load_sorted_styles(
        style_names=legal_style_names,
        default_selected=modules.config.default_styles)

    if server_port == 0:
        server_port = args_manager.args.port

    script_callbacks.before_ui_callback()
    if args_manager.args.no_gradio:
        app = launch_app_without_gradio(server_port)
    else:
        app = launch_app_with_gradio(server_port)

    create_api(app, generate_clicked, refresh_seed, recover_task, stop_clicked, skip_clicked, trigger_describe)
    script_callbacks.app_started_callback(shared.gradio_root, app)

    # start async worker thread if necessary
    if not args_manager.args.lazy:
        worker.start()

    asyncio.run(block_thread(app))


if __name__ == '__main__':
    start()
