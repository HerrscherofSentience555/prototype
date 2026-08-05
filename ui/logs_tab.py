from __future__ import annotations

import gradio as gr

from prototype.task_manager import LocalTaskManager


def _job_id_from_label(value: str | None) -> str:
    if not value:
        return ""
    return value.split(" [", 1)[0]


def _tail_text(text: str, tail_lines: float | None) -> str:
    if not text.strip():
        return ""
    count = int(tail_lines or 0)
    if count <= 0:
        return text
    return "\n".join(text.splitlines()[-count:])


def build_logs_tab(task_manager: LocalTaskManager) -> None:
    with gr.Tab("Logs"):
        job_id = gr.Dropdown(label="Job", choices=[])
        refresh_status = gr.Textbox(label="Available Jobs")
        tail_lines = gr.Number(label="Tail Lines", value=200, precision=0)
        log_view = gr.Textbox(label="launcher.log", lines=18)
        rank_log_view = gr.Textbox(label="train-rank0.log", lines=18)
        command_view = gr.Code(label="command.json", language="json")
        run_bundle = gr.File(label="Download Run Bundle")

        def refresh_jobs():
            jobs = task_manager.list_jobs()
            labels = [f"{job.job_id} [{job.status}]" for job in jobs]
            summary = "\n".join(f"{job.job_id} [{job.status}]" for job in jobs) or "No jobs yet."
            selected = labels[0] if labels else None
            return gr.update(choices=labels, value=selected), summary

        def read_logs(job_id_value: str, tail_lines_value: float | None) -> tuple[str, str, str]:
            job_id_text = _job_id_from_label(job_id_value)
            if not job_id_text:
                return "", "", ""
            launcher_log = task_manager.read_text_file(job_id_text, "launcher.log")
            rank_log = task_manager.read_text_file(job_id_text, "train-rank0.log")
            command = task_manager.read_text_file(job_id_text, "command.json")
            return (
                _tail_text(launcher_log, tail_lines_value),
                _tail_text(rank_log, tail_lines_value),
                command,
            )

        def read_logs_for_change(job_id_value: str) -> tuple[str, str, str]:
            if not job_id_value:
                return "", "", ""
            return read_logs(job_id_value, 200)

        def bundle_path(job_id_value: str):
            job_id_text = _job_id_from_label(job_id_value)
            if not job_id_text:
                return None
            path = task_manager.get_run_dir(job_id_text) / "artifacts" / "run-artifacts.zip"
            return str(path) if path.exists() else None

        refresh_button = gr.Button("Refresh Jobs")
        refresh_logs_button = gr.Button("Refresh Logs")
        refresh_button.click(
            lambda: refresh_jobs(),
            outputs=[job_id, refresh_status],
        )
        refresh_logs_button.click(
            read_logs,
            inputs=[job_id, tail_lines],
            outputs=[log_view, rank_log_view, command_view],
        )
        gr.Button("Download Run Bundle").click(bundle_path, inputs=job_id, outputs=run_bundle)
        job_id.change(read_logs_for_change, inputs=job_id, outputs=[log_view, rank_log_view, command_view])
