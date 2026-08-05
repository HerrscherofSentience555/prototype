from __future__ import annotations

import json

import gradio as gr
import pandas as pd

from prototype.task_manager import LocalTaskManager


METRIC_COLUMNS = ["timestamp", "step", "metric", "value"]


def _job_id_from_label(value: str | None) -> str:
    if not value:
        return ""
    return value.split(" [", 1)[0]


def _empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=METRIC_COLUMNS)


def _prepare_plot_data(metrics: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    selected = metrics[metrics["metric"].isin(names)].copy()
    if selected.empty:
        return pd.DataFrame(columns=[*METRIC_COLUMNS, "plot_step"])
    selected = selected.reset_index(drop=True)
    selected["plot_step"] = selected["step"]
    missing_step = selected["plot_step"].isna()
    selected.loc[missing_step, "plot_step"] = selected.index[missing_step] + 1
    return selected


def build_monitor_tab(task_manager: LocalTaskManager) -> None:
    with gr.Tab("Monitor"):
        job_id = gr.Dropdown(label="Job", choices=[])
        metric_status = gr.Textbox(label="Metric Status", value="Refresh jobs to load metrics.")
        metrics_table = gr.Dataframe(label="Recent Metrics")
        loss_plot = gr.LinePlot(
            label="Train Loss",
            x="plot_step",
            y="value",
            color="metric",
        )
        auc_plot = gr.LinePlot(
            label="AUC",
            x="plot_step",
            y="value",
            color="metric",
        )
        throughput_plot = gr.LinePlot(
            label="Throughput",
            x="plot_step",
            y="value",
            color="metric",
        )
        step_time_plot = gr.LinePlot(
            label="Step Time",
            x="plot_step",
            y="value",
            color="metric",
        )
        stage_timing_plot = gr.LinePlot(
            label="Stage Timing",
            x="plot_step",
            y="value",
            color="metric",
        )
        resource_table = gr.Dataframe(label="Recent Resource Metrics")
        resource_summary = gr.Code(label="resource-summary.json", language="json")

        def refresh_job_choices():
            labels = [f"{job.job_id} [{job.status}]" for job in task_manager.list_jobs()]
            selected = labels[0] if labels else None
            return gr.update(choices=labels, value=selected)

        def read_metrics(
            job_id_value: str,
        ) -> tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            str,
        ]:
            job_id_text = _job_id_from_label(job_id_value)
            if not job_id_text:
                empty = _empty_metrics()
                return empty, empty, empty, empty, empty, empty, "Select a job first."
            raw = task_manager.read_text_file(job_id_text, "metrics.jsonl")
            if not raw.strip():
                empty = _empty_metrics()
                return empty, empty, empty, empty, empty, empty, "No metrics.jsonl records for this job yet."
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            metrics = pd.DataFrame(rows)
            recent = metrics.tail(20)
            loss = _prepare_plot_data(metrics, ["train_loss", "loss"])
            auc_names = [
                metric
                for metric in metrics["metric"].dropna().unique().tolist()
                if "auc" in str(metric).lower()
            ]
            auc = _prepare_plot_data(metrics, auc_names)
            throughput = _prepare_plot_data(
                metrics,
                ["samples_per_second", "batches_per_second", "throughput_iter_per_sec"],
            )
            step_time = _prepare_plot_data(metrics, ["step_time_seconds"])
            stage_timing_names = [
                "data_wait_seconds",
                "h2d_seconds",
                "input_distribution_seconds",
                "embedding_lookup_seconds",
                "dense_forward_seconds",
                "backward_seconds",
                "optimizer_seconds",
            ]
            stage_timing = _prepare_plot_data(metrics, stage_timing_names)
            return recent, loss, auc, throughput, step_time, stage_timing, f"Loaded {len(metrics)} metric records."

        def read_resources(job_id_value: str) -> tuple[pd.DataFrame, str]:
            job_id_text = _job_id_from_label(job_id_value)
            if not job_id_text:
                return pd.DataFrame(), "{}"
            raw = task_manager.read_text_file(job_id_text, "resource-metrics.jsonl")
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()] if raw.strip() else []
            summary = task_manager.read_text_file(job_id_text, "artifacts/resource-summary.json") or "{}"
            return pd.DataFrame(rows).tail(20), summary

        gr.Button("Refresh Jobs").click(lambda: refresh_job_choices(), outputs=job_id)
        gr.Button("Refresh Metrics").click(
            read_metrics,
            inputs=job_id,
            outputs=[
                metrics_table,
                loss_plot,
                auc_plot,
                throughput_plot,
                step_time_plot,
                stage_timing_plot,
                metric_status,
            ],
        )
        gr.Button("Refresh Resources").click(
            read_resources,
            inputs=job_id,
            outputs=[resource_table, resource_summary],
        )
        job_id.change(
            read_metrics,
            inputs=job_id,
            outputs=[
                metrics_table,
                loss_plot,
                auc_plot,
                throughput_plot,
                step_time_plot,
                stage_timing_plot,
                metric_status,
            ],
        )
        job_id.change(read_resources, inputs=job_id, outputs=[resource_table, resource_summary])
