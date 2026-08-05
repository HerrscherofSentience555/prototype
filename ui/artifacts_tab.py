from __future__ import annotations

import gradio as gr

from prototype.task_manager import LocalTaskManager


def _job_id_from_label(value: str | None) -> str:
    if not value:
        return ""
    return value.split(" [", 1)[0]


def _format_file_list(title: str, files: list[str]) -> str:
    if not files:
        return f"{title}: no files"
    return title + ":\n" + "\n".join(f"- {file}" for file in files)


def build_artifacts_tab(task_manager: LocalTaskManager) -> None:
    with gr.Tab("Artifacts"):
        job_id = gr.Dropdown(label="Job", choices=[])
        run_dir_view = gr.Textbox(label="Run Directory")
        state_view = gr.Code(label="state.json", language="json")
        config_view = gr.Code(label="resolved-config.yaml", language="yaml")
        evaluation_view = gr.Code(label="evaluation.json", language="json")
        capability_view = gr.Code(label="artifacts/v1-capability-report.json", language="json")
        model_contract_view = gr.Code(label="artifacts/torchrec-model-contract.json", language="json")
        data_plan_view = gr.Code(label="artifacts/torchrec-data-plan.json", language="json")
        batch_materialization_view = gr.Code(
            label="artifacts/torchrec-batch-materialization.json",
            language="json",
        )
        embedding_configs_view = gr.Code(label="artifacts/torchrec-embedding-configs.json", language="json")
        runtime_smoke_view = gr.Code(label="artifacts/torchrec-runtime-smoke.json", language="json")
        sharding_readiness_view = gr.Code(
            label="artifacts/torchrec-sharding-plan-readiness.json",
            language="json",
        )
        training_plan_view = gr.Code(label="artifacts/torchrec-training-plan.json", language="json")
        files_view = gr.Textbox(label="Artifact Files", lines=12)
        stop_result = gr.Textbox(label="Stop Result")

        def refresh_job_choices():
            labels = [f"{job.job_id} [{job.status}]" for job in task_manager.list_jobs()]
            selected = labels[0] if labels else None
            return gr.update(choices=labels, value=selected)

        def read_artifacts(job_id_value: str) -> tuple[str, ...]:
            job_id_text = _job_id_from_label(job_id_value)
            if not job_id_text:
                return "", "", "", "", "", "", "", "", "", "", "", ""
            checkpoints = task_manager.list_files(job_id_text, "checkpoints")
            profiles = task_manager.list_files(job_id_text, "profiles")
            artifacts = task_manager.list_files(job_id_text, "artifacts")
            file_summary = "\n\n".join(
                [
                    _format_file_list("checkpoints", checkpoints),
                    _format_file_list("profiles", profiles),
                    _format_file_list("artifacts", artifacts),
                ]
            )
            return (
                str(task_manager.get_run_dir(job_id_text)),
                task_manager.read_text_file(job_id_text, "state.json"),
                task_manager.read_text_file(job_id_text, "resolved-config.yaml"),
                task_manager.read_text_file(job_id_text, "evaluation.json"),
                task_manager.read_text_file(job_id_text, "artifacts/v1-capability-report.json"),
                task_manager.read_text_file(job_id_text, "artifacts/torchrec-model-contract.json"),
                task_manager.read_text_file(job_id_text, "artifacts/torchrec-data-plan.json"),
                task_manager.read_text_file(job_id_text, "artifacts/torchrec-batch-materialization.json"),
                task_manager.read_text_file(job_id_text, "artifacts/torchrec-embedding-configs.json"),
                task_manager.read_text_file(job_id_text, "artifacts/torchrec-runtime-smoke.json"),
                task_manager.read_text_file(job_id_text, "artifacts/torchrec-sharding-plan-readiness.json"),
                task_manager.read_text_file(job_id_text, "artifacts/torchrec-training-plan.json"),
                file_summary,
            )

        def stop_job(job_id_value: str) -> str:
            job_id_text = _job_id_from_label(job_id_value)
            if not job_id_text:
                return "Select a job first."
            job = task_manager.stop_job(job_id_text)
            if job.status in {"SUCCEEDED", "FAILED"}:
                return f"Job {job.job_id} is {job.status}; no additional stop action was needed."
            return f"Job {job.job_id} marked as {job.status}"

        gr.Button("Refresh Jobs").click(refresh_job_choices, outputs=job_id)
        gr.Button("Refresh Artifacts").click(
            read_artifacts,
            inputs=job_id,
            outputs=[
                run_dir_view,
                state_view,
                config_view,
                evaluation_view,
                capability_view,
                model_contract_view,
                data_plan_view,
                batch_materialization_view,
                embedding_configs_view,
                runtime_smoke_view,
                sharding_readiness_view,
                training_plan_view,
                files_view,
            ],
        )
        job_id.change(
            read_artifacts,
            inputs=job_id,
            outputs=[
                run_dir_view,
                state_view,
                config_view,
                evaluation_view,
                capability_view,
                model_contract_view,
                data_plan_view,
                batch_materialization_view,
                embedding_configs_view,
                runtime_smoke_view,
                sharding_readiness_view,
                training_plan_view,
                files_view,
            ],
        )
        gr.Button("Stop Job").click(stop_job, inputs=job_id, outputs=stop_result)
