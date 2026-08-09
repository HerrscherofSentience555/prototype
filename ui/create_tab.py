from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
from pydantic import ValidationError

from prototype.config import (
    BackendName,
    EmbeddingPlacement,
    PrecisionMode,
    PrototypeConfig,
    RunMode,
    RuntimePlatform,
)
from prototype.local_settings import load_local_settings
from prototype.runner.data_validation import DataValidationError, validate_parquet_dataset
from prototype.runner.parquet_converter import convert_parquet_to_criteo_numpy
from prototype.task_manager import LocalTaskManager


def build_create_tab(task_manager: LocalTaskManager) -> None:
    local_settings = load_local_settings()
    with gr.Tab("Create Job"):
        gr.Markdown("Configure a local TorchRec prototype job and launch it.")

        with gr.Row():
            job_name = gr.Textbox(label="Job Name", value="torchrec-job")
            mode = gr.Dropdown(
                label="Mode",
                choices=[mode.value for mode in RunMode],
                value=RunMode.COLD_START.value,
            )
            backend = gr.Dropdown(
                label="Backend",
                choices=[backend.value for backend in BackendName],
                value=local_settings.defaults.backend,
            )
        with gr.Row():
            runtime_platform = gr.Dropdown(
                label="Runtime Platform",
                choices=[platform.value for platform in RuntimePlatform],
                value=local_settings.runtime.platform,
            )
            dlrm_root = gr.Textbox(label="DLRM Root", value=local_settings.paths.dlrm_root)
            python_env = gr.Textbox(label="Python Env", value=local_settings.runtime.python_env)
            wsl_distribution = gr.Textbox(
                label="WSL Distribution",
                value=local_settings.runtime.wsl_distribution,
            )
        with gr.Row():
            model_file = gr.Textbox(label="Model File", value=local_settings.paths.default_model_file)
            model_config_file = gr.Textbox(label="Model Config File", value="")
        with gr.Row():
            num_embeddings = gr.Number(label="Num Embeddings", value=0, precision=0)
            embedding_dim = gr.Number(label="Embedding Dim", value=0, precision=0)
            dense_arch_layer_sizes = gr.Textbox(label="Dense Arch Layer Sizes", value="")
            over_arch_layer_sizes = gr.Textbox(label="Over Arch Layer Sizes", value="")
        with gr.Row():
            data_format = gr.Dropdown(
                label="Data Format",
                choices=["random", "criteo_binary", "synthetic_multihot", "parquet"],
                value=local_settings.defaults.data_format,
            )
            batch_size = gr.Number(label="Batch Size", value=local_settings.defaults.batch_size, precision=0)
        with gr.Row():
            criteo_binary_path = gr.Textbox(
                label="Criteo Binary Path",
                value=local_settings.paths.criteo_binary_path,
            )
            synthetic_multi_hot_path = gr.Textbox(
                label="Synthetic Multi-Hot Path",
                value=local_settings.paths.synthetic_multi_hot_path,
            )
            dataset_name = gr.Dropdown(
                label="Dataset Name",
                choices=["criteo_1t", "criteo_kaggle"],
                value=local_settings.defaults.dataset_name,
            )
        with gr.Row():
            train_path = gr.Textbox(label="Train Path", value=local_settings.paths.parquet_train_path)
            validation_path = gr.Textbox(
                label="Validation Path",
                value=local_settings.paths.parquet_validation_path,
            )
            test_path = gr.Textbox(label="Test Path", value=local_settings.paths.parquet_test_path)
            schema_path = gr.Textbox(label="Schema Path", value=local_settings.paths.parquet_schema_path)
        with gr.Row():
            parquet_output_path = gr.Textbox(
                label="Parquet Conversion Output",
                value=local_settings.paths.parquet_conversion_output,
            )
        with gr.Row():
            parquet_status = gr.Textbox(label="Parquet Tool Status", value="")
        parquet_profile = gr.Code(label="Parquet Data Profile", language="json")
        parquet_manifest = gr.Code(label="Parquet Conversion Manifest", language="json")
        with gr.Row():
            test_batch_size = gr.Number(
                label="Test Batch Size",
                value=local_settings.defaults.test_batch_size,
                precision=0,
            )
            pin_memory = gr.Checkbox(label="Pin Memory", value=False)
            mmap_mode = gr.Checkbox(label="MMAP Mode", value=False)
        with gr.Row():
            epochs = gr.Number(label="Epochs", value=1, precision=0)
            max_steps = gr.Number(label="Max Steps", value=local_settings.defaults.max_steps, precision=0)
            learning_rate = gr.Number(label="Learning Rate", value=local_settings.defaults.learning_rate)
            nproc = gr.Number(
                label="Processes per Node",
                value=local_settings.defaults.nproc_per_node,
                precision=0,
            )
        with gr.Row():
            gpu_ids = gr.Textbox(label="GPU IDs", value=local_settings.defaults.gpu_ids)
            embedding_placement = gr.Dropdown(
                label="Embedding Placement",
                choices=[placement.value for placement in EmbeddingPlacement],
                value=EmbeddingPlacement.DEVICE.value,
            )
            cache_load_factor = gr.Number(label="Cache Load Factor", value=0.2)
        with gr.Row():
            embedding_precision = gr.Dropdown(
                label="Embedding Precision",
                choices=[mode.value for mode in PrecisionMode],
                value=PrecisionMode.FP32.value,
            )
            dense_precision = gr.Dropdown(
                label="Dense Compute Precision",
                choices=[mode.value for mode in PrecisionMode],
                value=PrecisionMode.FP32.value,
            )
            comm_forward_precision = gr.Dropdown(
                label="Comm Forward Precision",
                choices=[mode.value for mode in PrecisionMode],
                value=PrecisionMode.FP32.value,
            )
            comm_backward_precision = gr.Dropdown(
                label="Comm Backward Precision",
                choices=[mode.value for mode in PrecisionMode],
                value=PrecisionMode.FP32.value,
            )
        with gr.Row():
            checkpoint_load_path = gr.Textbox(label="Checkpoint Load Path", value="")
            save_checkpoints = gr.Checkbox(label="Save Checkpoints", value=True)
            save_optimizer = gr.Checkbox(label="Save Optimizer", value=True)
            checkpoint_save_every = gr.Number(label="Checkpoint Save Every Steps", value=100, precision=0)
            checkpoint_keep_last = gr.Number(label="Checkpoint Keep Last", value=3, precision=0)
        with gr.Row():
            profile_enabled = gr.Checkbox(label="Profile Enabled", value=False)
            profile_start_step = gr.Number(label="Profile Start Step", value=100, precision=0)
            profile_end_step = gr.Number(label="Profile End Step", value=120, precision=0)
            profile_record_shapes = gr.Checkbox(label="Profile Record Shapes", value=True)
            profile_memory = gr.Checkbox(label="Profile Memory", value=True)
        validate_output = gr.Code(label="Resolved YAML", language="yaml")
        launch_output = gr.Textbox(label="Launch Result")

        def optional_int(value: float | None) -> int | None:
            if value is None:
                return None
            int_value = int(value)
            return int_value if int_value > 0 else None

        def optional_text(value: str | None) -> str | None:
            if value is None:
                return None
            stripped = value.strip()
            return stripped or None

        def parse_gpu_ids(value: str | None) -> list[int]:
            text = value.strip() if value else "0"
            return [int(part.strip()) for part in text.split(",") if part.strip()]

        def build_config(
            job_name_value: str,
            mode_value: str,
            backend_value: str,
            runtime_platform_value: str,
            dlrm_root_value: str,
            python_env_value: str,
            wsl_distribution_value: str,
            model_file_value: str,
            model_config_file_value: str,
            num_embeddings_value: float | None,
            embedding_dim_value: float | None,
            dense_arch_layer_sizes_value: str,
            over_arch_layer_sizes_value: str,
            data_format_value: str,
            batch_size_value: float,
            criteo_binary_path_value: str,
            synthetic_multi_hot_path_value: str,
            dataset_name_value: str,
            train_path_value: str,
            validation_path_value: str,
            test_path_value: str,
            schema_path_value: str,
            test_batch_size_value: float | None,
            pin_memory_value: bool,
            mmap_mode_value: bool,
            epochs_value: float,
            max_steps_value: float | None,
            learning_rate_value: float,
            nproc_value: float,
            gpu_ids_value: str,
            embedding_placement_value: str,
            cache_load_factor_value: float,
            embedding_precision_value: str,
            dense_precision_value: str,
            comm_forward_precision_value: str,
            comm_backward_precision_value: str,
            checkpoint_load_path_value: str,
            save_checkpoints_value: bool,
            save_optimizer_value: bool,
            checkpoint_save_every_value: float,
            checkpoint_keep_last_value: float,
            profile_enabled_value: bool,
            profile_start_step_value: float,
            profile_end_step_value: float,
            profile_record_shapes_value: bool,
            profile_memory_value: bool,
        ) -> PrototypeConfig:
            return PrototypeConfig(
                job_name=job_name_value,
                mode=mode_value,
                backend={
                    "name": backend_value,
                    "runtime_platform": runtime_platform_value,
                    "dlrm_root": dlrm_root_value,
                    "python_env": python_env_value,
                    "wsl_distribution": wsl_distribution_value,
                },
                nproc_per_node=int(nproc_value),
                model={
                    "file": model_file_value,
                    "config_file": optional_text(model_config_file_value),
                    "num_embeddings": optional_int(num_embeddings_value),
                    "embedding_dim": optional_int(embedding_dim_value),
                    "dense_arch_layer_sizes": optional_text(dense_arch_layer_sizes_value),
                    "over_arch_layer_sizes": optional_text(over_arch_layer_sizes_value),
                },
                device={
                    "gpu_ids": parse_gpu_ids(gpu_ids_value),
                    "embedding_placement": embedding_placement_value,
                    "cache_load_factor": float(cache_load_factor_value),
                },
                precision={
                    "embedding": embedding_precision_value,
                    "dense_compute": dense_precision_value,
                    "comm_forward": comm_forward_precision_value,
                    "comm_backward": comm_backward_precision_value,
                },
                data={
                    "format": data_format_value,
                    "batch_size": int(batch_size_value),
                    "test_batch_size": optional_int(test_batch_size_value),
                    "criteo_binary_path": optional_text(criteo_binary_path_value),
                    "synthetic_multi_hot_path": optional_text(synthetic_multi_hot_path_value),
                    "dataset_name": dataset_name_value,
                    "train_path": optional_text(train_path_value),
                    "validation_path": optional_text(validation_path_value),
                    "test_path": optional_text(test_path_value),
                    "schema_path": optional_text(schema_path_value),
                    "pin_memory": bool(pin_memory_value),
                    "mmap_mode": bool(mmap_mode_value),
                },
                training={
                    "epochs": int(epochs_value),
                    "max_steps": optional_int(max_steps_value),
                    "learning_rate": float(learning_rate_value),
                },
                checkpoint={
                    "enabled": bool(save_checkpoints_value),
                    "load_path": optional_text(checkpoint_load_path_value),
                    "save_optimizer": bool(save_optimizer_value),
                    "save_every_n_steps": int(checkpoint_save_every_value),
                    "keep_last": int(checkpoint_keep_last_value),
                },
                profile={
                    "enabled": bool(profile_enabled_value),
                    "start_step": int(profile_start_step_value),
                    "end_step": int(profile_end_step_value),
                    "record_shapes": bool(profile_record_shapes_value),
                    "profile_memory": bool(profile_memory_value),
                },
            )

        def validate_config(*values) -> str:
            try:
                config = build_config(*values)
            except ValidationError as exc:
                return f"# Config validation failed\n\n{exc}"
            return config.to_yaml()

        def validate_parquet_config(*values) -> tuple[str, str]:
            try:
                config = build_config(*values)
                profile = validate_parquet_dataset(config)
            except (ValidationError, DataValidationError, ImportError) as exc:
                return f"Parquet validation failed: {type(exc).__name__}: {exc}", "{}"
            return "Parquet validation succeeded.", json.dumps(profile, ensure_ascii=False, indent=2)

        def convert_parquet_config(*values) -> tuple[str, str, object, str]:
            *config_values, output_dir_value = values
            try:
                config = build_config(*config_values)
                output_dir = Path(output_dir_value).expanduser()
                manifest = convert_parquet_to_criteo_numpy(config, output_dir)
            except (ValidationError, DataValidationError, ImportError, OSError, ValueError) as exc:
                return (
                    f"Parquet conversion failed: {type(exc).__name__}: {exc}",
                    "{}",
                    gr.update(),
                    "",
                )
            return (
                f"Parquet conversion succeeded. Output: {output_dir}",
                json.dumps(manifest, ensure_ascii=False, indent=2),
                gr.update(value="criteo_binary"),
                str(output_dir),
            )

        def launch_job(*values) -> str:
            try:
                config = build_config(*values)
            except ValidationError as exc:
                return f"Config validation failed:\n{exc}"
            job = task_manager.create_job(config)
            try:
                launched = task_manager.launch_job(job.job_id)
            except Exception as exc:
                return f"Job {job.job_id} was created but launch failed:\n{type(exc).__name__}: {exc}"
            return (
                f"Job {launched.job_id} launched\n"
                f"Status: {launched.status}\n"
                f"Backend: {config.backend.name.value}\n"
                f"PID: {launched.pid}\n"
                f"Run directory: {launched.run_dir}"
            )

        inputs = [
            job_name,
            mode,
            backend,
            runtime_platform,
            dlrm_root,
            python_env,
            wsl_distribution,
            model_file,
            model_config_file,
            num_embeddings,
            embedding_dim,
            dense_arch_layer_sizes,
            over_arch_layer_sizes,
            data_format,
            batch_size,
            criteo_binary_path,
            synthetic_multi_hot_path,
            dataset_name,
            train_path,
            validation_path,
            test_path,
            schema_path,
            test_batch_size,
            pin_memory,
            mmap_mode,
            epochs,
            max_steps,
            learning_rate,
            nproc,
            gpu_ids,
            embedding_placement,
            cache_load_factor,
            embedding_precision,
            dense_precision,
            comm_forward_precision,
            comm_backward_precision,
            checkpoint_load_path,
            save_checkpoints,
            save_optimizer,
            checkpoint_save_every,
            checkpoint_keep_last,
            profile_enabled,
            profile_start_step,
            profile_end_step,
            profile_record_shapes,
            profile_memory,
        ]
        parquet_inputs = [*inputs, parquet_output_path]
        gr.Button("Validate Parquet").click(
            validate_parquet_config,
            inputs=inputs,
            outputs=[parquet_status, parquet_profile],
        )
        gr.Button("Convert Parquet").click(
            convert_parquet_config,
            inputs=parquet_inputs,
            outputs=[parquet_status, parquet_manifest, data_format, criteo_binary_path],
        )
        gr.Button("Validate Config").click(validate_config, inputs=inputs, outputs=validate_output)
        gr.Button("Launch Job").click(launch_job, inputs=inputs, outputs=launch_output)
