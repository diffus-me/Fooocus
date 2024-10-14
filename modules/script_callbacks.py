import inspect
import logging
from collections import namedtuple
from typing import Optional, Any, TYPE_CHECKING

from fastapi import FastAPI
from gradio import Blocks

from modules.util import setup_logging

if TYPE_CHECKING:
    from modules.async_worker import AsyncTask


setup_logging()
logger = logging.getLogger("default")
exception_records = []


class ImageSaveParams:
    def __init__(self, image, filename, task: "AsyncTask", pnginfo: dict[str, Any]):
        self.image = image
        """the PIL image itself"""

        self.filename = filename
        """name of file that the image would be saved to"""

        self.task = task
        """task from request"""

        self.pnginfo = pnginfo


ScriptCallback = namedtuple("ScriptCallback", ["script", "callback"])

callback_map = dict(
    callbacks_app_started=[],
    callbacks_app_ready=[],
    callbacks_app_stopped=[],
    callbacks_before_ui=[],
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


def app_ready_callback():
    invoke_callbacks('callbacks_app_ready')


def app_stopped_callback():
    invoke_callbacks('callbacks_app_stopped')


def before_ui_callback():
    invoke_callbacks('callbacks_before_ui')


def before_task_callback(task_id: str):
    invoke_callbacks('callbacks_before_task', task_id)


def after_task_callback(task_id: str):
    invoke_callbacks('callbacks_after_task', task_id)


def image_saved_callback(params: ImageSaveParams):
    invoke_callbacks('callbacks_image_saved', params)


def on_app_started(callback):
    """register a function to be called when the webui started, the gradio `Block` component and
    fastapi `FastAPI` object are passed as the arguments"""
    add_callback(callback_map['callbacks_app_started'], callback)


def on_app_ready(callback):
    """register a function to be called when the app is ready to process tasks"""
    add_callback(callback_map['callbacks_app_ready'], callback)


def on_app_stopped(callback):
    add_callback(callback_map['callbacks_app_stopped'], callback)


def on_before_ui(callback):
    """register a function to be called before the UI is created."""

    add_callback(callback_map['callbacks_before_ui'], callback)


def on_before_task(callback):
    add_callback(callback_map['callbacks_before_task'], callback)


def on_after_task(callback):
    add_callback(callback_map['callbacks_after_task'], callback)


def on_image_saved(callback):
    """register a function to be called after an image is saved to a file.
    The callback is called with one argument:
        - params: ImageSaveParams - parameters the image was saved with. Changing fields in this object does nothing.
    """
    add_callback(callback_map['callbacks_image_saved'], callback)
