import json
import pathlib
import subprocess
import sys


def test_train_rl_token_cli_smoke(tmp_path: pathlib.Path) -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    output_root = tmp_path / "rl_token"
    command = [
        sys.executable,
        "scripts/train_rl_token_pytorch.py",
        "--config-name",
        "debug",
        "--exp-name",
        "pytest_smoke",
        "--device",
        "cpu",
        "--num-train-steps",
        "1",
        "--log-interval",
        "1",
        "--save-interval",
        "1",
        "--feature-source",
        "image_embeddings",
        "--skip-norm-stats",
        "--output-root",
        str(output_root),
    ]

    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr or result.stdout

    checkpoint_dir = output_root / "debug" / "pytest_smoke" / "1"
    assert (checkpoint_dir / "model.safetensors").is_file()
    assert (checkpoint_dir / "optimizer.pt").is_file()

    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["step"] == 1
    assert metadata["args"]["config_name"] == "debug"
    assert metadata["args"]["feature_source"] == "image_embeddings"