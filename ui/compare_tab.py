from __future__ import annotations

import json

import gradio as gr
import pandas as pd

from prototype.runner.run_tracking import compare_runs, export_comparison_csv
from prototype.task_manager import LocalTaskManager


def _job_id_from_label(value: str) -> str:
    return value.split(" [", 1)[0]


def build_compare_tab(task_manager: LocalTaskManager) -> None:
    with gr.Tab("Compare Runs"):
        jobs = gr.CheckboxGroup(label="Runs", choices=[])
        metric = gr.Textbox(label="Sort Metric", value="auc")
        status = gr.Textbox(label="Compare Status", value="Refresh jobs to compare runs.")
        table = gr.Dataframe(label="Comparison")
        best_view = gr.Code(label="Best Run", language="json")

        def refresh_job_choices():
            labels = [f"{job.job_id} [{job.status}]" for job in task_manager.list_jobs()]
            return gr.update(choices=labels, value=labels[:2])

        def compare(job_labels: list[str], metric_name: str):
            if not job_labels:
                return pd.DataFrame(), "{}", "Select at least one run."
            job_ids = [_job_id_from_label(label) for label in job_labels]
            comparison = compare_runs(task_manager.runs_dir, job_ids, metric_name.strip() or "auc")
            output_path = task_manager.runs_dir / "comparison-latest.csv"
            export_comparison_csv(comparison, output_path)
            return (
                pd.DataFrame(comparison["runs"]),
                json.dumps(comparison["best_run"], ensure_ascii=False, indent=2),
                f"Compared {len(job_ids)} runs. CSV exported to {output_path}",
            )

        gr.Button("Refresh Jobs").click(refresh_job_choices, outputs=jobs)
        gr.Button("Compare").click(compare, inputs=[jobs, metric], outputs=[table, best_view, status])
