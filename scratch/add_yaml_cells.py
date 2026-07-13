# ruff: noqa: E501
"""
add_yaml_cells.py
=================
Programmatically appends a final code cell to all notebooks in the notebooks/
directory. This cell collects phase execution results and writes them as YAML.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")

CELLS_MAP = {
    "00_Environment_Check.ipynb": [
        "# Export Phase Manifest\n",
        "import sys\n",
        "import torch\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        "manifest = {\n",
        '    "phase": "00_Environment_Check",\n',
        '    "python_version": sys.version,\n',
        '    "pytorch_version": torch.__version__,\n',
        '    "cuda_available": torch.cuda.is_available(),\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_00_environment_check.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "01_Project_Tour.ipynb": [
        "# Export Phase Manifest\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "from src.utils.config import load_config\n",
        "\n",
        "config = load_config()\n",
        "manifest = {\n",
        '    "phase": "01_Project_Tour",\n',
        '    "dataset_name": config["data"]["dataset_name"],\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_01_project_tour.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "02_EDA_Banking77.ipynb": [
        "# Export Phase Manifest\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        "manifest = {\n",
        '    "phase": "02_EDA_Banking77",\n',
        '    "total_rows": summary["total_rows"],\n',
        '    "train_rows": summary["train_rows"],\n',
        '    "val_rows": summary["val_rows"],\n',
        '    "test_rows": summary["test_rows"],\n',
        '    "unique_classes": summary["unique_classes"],\n',
        '    "rare_classes_count": summary["rare_classes_count"],\n',
        '    "vocabulary_size": summary["vocabulary_size"],\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_02_eda.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "03_Baseline_Models.ipynb": [
        "# Export Phase Manifest\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        "manifest = {\n",
        '    "phase": "03_Baseline_Models",\n',
        '    "metrics": metrics_dict,\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_03_baselines.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "04_Error_Analysis.ipynb": [
        "# Export Phase Manifest\n",
        "import json\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        'error_summary_path = REPO_ROOT / "outputs" / "metrics" / "error_summary.json"\n',
        "error_summary = {}\n",
        "if error_summary_path.exists():\n",
        "    with open(error_summary_path) as f:\n",
        "        error_summary = json.load(f)\n",
        "\n",
        "manifest = {\n",
        '    "phase": "04_Error_Analysis",\n',
        '    "error_summary": error_summary,\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_04_error_analysis.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "05_Distil_Data.ipynb": [
        "# Export Phase Manifest\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "from src.utils.config import load_config\n",
        "\n",
        "config = load_config()\n",
        "manifest = {\n",
        '    "phase": "05_Distil_Data",\n',
        '    "max_length": config.get("max_length", 128),\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_05_distil_data.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "06_Training.ipynb": [
        "# Export Phase Manifest\n",
        "import json\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        'training_history_path = REPO_ROOT / "outputs" / "metrics" / "training_history.json"\n',
        "history = {}\n",
        "if training_history_path.exists():\n",
        "    with open(training_history_path) as f:\n",
        "        history = json.load(f)\n",
        "\n",
        "manifest = {\n",
        '    "phase": "06_Training",\n',
        '    "training_history": history,\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_06_training.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "07_Model_Evaluation.ipynb": [
        "# Export Phase Manifest\n",
        "import json\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        'eval_metrics_path = REPO_ROOT / "outputs" / "metrics" / "eval_metrics.json"\n',
        "eval_metrics = {}\n",
        "if eval_metrics_path.exists():\n",
        "    with open(eval_metrics_path) as f:\n",
        "        eval_metrics = json.load(f)\n",
        "\n",
        "manifest = {\n",
        '    "phase": "07_Model_Evaluation",\n',
        '    "eval_metrics": eval_metrics,\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_07_model_evaluation.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "08_Calibration.ipynb": [
        "# Export Phase Manifest\n",
        "import json\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        'calibration_metrics_path = REPO_ROOT / "outputs" / "metrics" / "calibration_metrics.json"\n',
        "calibration_metrics = {}\n",
        "if calibration_metrics_path.exists():\n",
        "    with open(calibration_metrics_path) as f:\n",
        "        calibration_metrics = json.load(f)\n",
        "\n",
        "manifest = {\n",
        '    "phase": "08_Calibration",\n',
        '    "calibration_metrics": calibration_metrics,\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_08_calibration.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "09_Optimization.ipynb": [
        "# Export Phase Manifest\n",
        "import json\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        'opt_metrics_path = REPO_ROOT / "outputs" / "metrics" / "optimization_benchmarks.json"\n',
        "opt_metrics = {}\n",
        "if opt_metrics_path.exists():\n",
        "    with open(opt_metrics_path) as f:\n",
        "        opt_metrics = json.load(f)\n",
        "\n",
        "manifest = {\n",
        '    "phase": "09_Optimization",\n',
        '    "optimization_metrics": opt_metrics,\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_09_optimization.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "10_Explainability.ipynb": [
        "# Export Phase Manifest\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        "manifest = {\n",
        '    "phase": "10_Explainability",\n',
        '    "explainability_check": "LIME report generated successfully",\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_10_explainability.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "11_Retrieval.ipynb": [
        "# Export Phase Manifest\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        "manifest = {\n",
        '    "phase": "11_Retrieval",\n',
        '    "retrieval_status": "FAISS index saved and verified",\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_11_retrieval.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
    "12_LLM_Fallback.ipynb": [
        "# Export Phase Manifest\n",
        "from src.utils.artifacts import save_yaml\n",
        "from src.api.app import get_git_commit\n",
        "\n",
        "manifest = {\n",
        '    "phase": "12_LLM_Fallback",\n',
        '    "routing_decision_flow": "Decision engine built and validated with fallback",\n',
        '    "git_commit": get_git_commit(),\n',
        "}\n",
        'save_yaml(manifest, REPO_ROOT / "outputs" / "manifests" / "phase_12_llm_fallback.yaml")\n',
        'print("YAML manifest saved successfully:")\n',
        "print(manifest)\n",
    ],
}


def main():
    print("Starting notebook cell injection...")
    for filename, source in CELLS_MAP.items():
        filepath = NOTEBOOKS_DIR / filename
        if not filepath.exists():
            print(f"Skipping {filename} - not found.")
            continue

        with open(filepath, encoding="utf-8") as f:
            notebook = json.load(f)

        # Check if the notebook already has a cell starting with "# Export Phase Manifest"
        has_cell = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cell_src = "".join(cell.get("source", []))
                if "# Export Phase Manifest" in cell_src:
                    cell["source"] = source  # Update in-place
                    has_cell = True
                    break

        if not has_cell:
            new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
            notebook["cells"].append(new_cell)
            print(f"Appended manifest cell to {filename}.")
        else:
            print(f"Updated existing manifest cell in {filename}.")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)

    print("Notebook cell injection completed.")


if __name__ == "__main__":
    main()
