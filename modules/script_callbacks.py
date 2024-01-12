import inspect
import logging
from collections import namedtuple
from typing import Optional, Any

from fastapi import FastAPI
from gradio import Blocks

logger = logging.getLogger(__name__)
exception_records = []


class ImageSaveParams:
    def __init__(self, image, filename, task_metadata: Optional[dict[str, Any]]):
        self.image = image
        """the PIL image itself"""

        self.filename = filename
        """name of file that the image would be saved to"""

        self.task_metadata = task_metadata
        """task metadata from request"""


ScriptCallback = namedtuple("ScriptCallback", ["script", "callback"])

callback_map = dict(
    callbacks_app_started=[],
    callbacks_before_ui=[],
    callbacks_main_loop=[],
    callbacks_before_task=[],
    callbacks_after_task=[],
    callbacks_image_saved=[],
)


def report_exception(e: Exception, c, job):
    logger.exception(f"Error executing callback {job} for {c.script}: {e.__str__()}")


def clear_callbacks():
    for callback_list in callback_map.values():
        callback_list.clear()


def invoke_callbacks(callback_name: str, *args):
    for c in callback_map[callback_name]:
        try:
            c.callback(*args)
        except Exception as e:
            report_exception(e, c, callback_name)


def add_callback(callbacks, fun):
    stack = [x for x in inspect.stack() if x.filename != __file__]
    filename = stack[0].filename if stack else 'unknown file'

    callbacks.append(ScriptCallback(filename, fun))


def app_started_callback(demo: Optional[Blocks], app: FastAPI):
    invoke_callbacks('callbacks_app_started', demo, app)


def before_ui_callback():
    invoke_callbacks('callbacks_before_ui')


def before_task_callback(task_id: str):
    invoke_callbacks('callbacks_before_task', task_id)


def after_task_callback(task_id: str):
    invoke_callbacks('callbacks_after_task', task_id)


def main_loop_callback():
    invoke_callbacks('callbacks_main_loop')


def image_saved_callback(params: ImageSaveParams):
    invoke_callbacks('callbacks_image_saved', params)


def on_app_started(callback):
    """register a function to be called when the webui started, the gradio `Block` component and
    fastapi `FastAPI` object are passed as the arguments"""
    add_callback(callback_map['callbacks_app_started'], callback)


def on_before_ui(callback):
    """register a function to be called before the UI is created."""

    add_callback(callback_map['callbacks_before_ui'], callback)


def on_before_task(callback):
    add_callback(callback_map['callbacks_before_task'], callback)


def on_after_task(callback):
    add_callback(callback_map['callbacks_after_task'], callback)


def on_main_loop(callback):
    add_callback(callback_map['callbacks_main_loop'], callback)


def on_image_saved(callback):
    """register a function to be called after an image is saved to a file.
    The callback is called with one argument:
        - params: ImageSaveParams - parameters the image was saved with. Changing fields in this object does nothing.
    """
    add_callback(callback_map['callbacks_image_saved'], callback)
