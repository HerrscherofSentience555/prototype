from __future__ import annotations

import gradio as gr

from prototype.task_manager import LocalTaskManager
from prototype.ui.artifacts_tab import build_artifacts_tab
from prototype.ui.compare_tab import build_compare_tab
from prototype.ui.create_tab import build_create_tab
from prototype.ui.logs_tab import build_logs_tab
from prototype.ui.monitor_tab import build_monitor_tab


def build_app() -> gr.Blocks:
    task_manager = LocalTaskManager()
    with gr.Blocks(title="TorchRec Prototype") as demo:
        gr.Markdown(
            """
            # TorchRec Training Prototype
            Single-machine prototype for local TorchRec job configuration, launch, logs, and metrics.
            """
        )
        build_create_tab(task_manager)
        build_logs_tab(task_manager)
        build_monitor_tab(task_manager)
        build_artifacts_tab(task_manager)
        build_compare_tab(task_manager)
    return demo


if __name__ == "__main__":
    build_app().launch()
