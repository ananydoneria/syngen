from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kagglehub
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gan_ae_full.trainer import TrainConfig, train_pipeline
from generator import build_backend, generate_synthetic, write_csv
from prompt_parser.parse_router import parse_user_prompt


@dataclass
class KaggleCase:
    name: str
    dataset_ref: str
    csv_filename: str
    checkpoint_dir: str
    parse_mode: str
    max_rows: int | None = None


CASES = [
    KaggleCase(
        name="stroke_prediction",
        dataset_ref="fedesoriano/stroke-prediction-dataset",
        csv_filename="healthcare-dataset-stroke-data.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_stroke",
        parse_mode="hybrid",
        max_rows=12000,
    ),
    KaggleCase(
        name="heart_failure",
        dataset_ref="fedesoriano/heart-failure-prediction",
        csv_filename="heart.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_heart_failure",
        parse_mode="hybrid",
        max_rows=5000,
    ),
    KaggleCase(
        name="pima_diabetes",
        dataset_ref="uciml/pima-indians-diabetes-database",
        csv_filename="diabetes.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_pima_diabetes",
        parse_mode="hybrid",
        max_rows=5000,
    ),
    KaggleCase(
        name="credit_card_fraud",
        dataset_ref="mlg-ulb/creditcardfraud",
        csv_filename="creditcard.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_credit_card_fraud",
        parse_mode="hybrid",
        max_rows=50000,
    ),
    KaggleCase(
        name="adult_income",
        dataset_ref="uciml/adult-census-income",
        csv_filename="adult.csv",
        checkpoint_dir="checkpoints/full_gan_ae_kaggle_adult_income",
        parse_mode="hybrid",
        max_rows=20000,
    ),
]

# Significantly longer, higher-quality default training for validation runs.
DEFAULT_AE_EPOCHS = 180
DEFAULT_GAN_EPOCHS = 320
DEFAULT_BATCH_SIZE = 512


def main() -> int:
    args = _build_parser().parse_args()
    if args.preflight_only:
        return _run_preflight(expect_gpu=args.expect_gpu)

    out_root = Path("kaggle_eval/output")
    out_root.mkdir(parents=True, exist_ok=True)
    cases_report: list[dict[str, Any]] = []

    for case in CASES:
        print(f"[INFO] Processing case: {case.name}")
        original_csv = _download_case(case)
        source_df = _load_df(original_csv)
        original_df = _cap_rows(source_df, case.max_rows)
        train_csv = original_csv
        if len(original_df) < len(source_df):
            train_csv = out_root / f"{case.name}_train_input.csv"
            original_df.to_csv(train_csv, index=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_cfg = TrainConfig(
            ae_epochs=args.ae_epochs,
            gan_epochs=args.gan_epochs,
            batch_size=args.batch_size,
            device=device,
            seed=42,
        )
        train_report = train_pipeline(str(train_csv), case.checkpoint_dir, train_cfg)

        prompt = _derive_prompt(case.name, original_df)
        spec = parse_user_prompt(prompt, mode=case.parse_mode)
        spec.n_rows = len(original_df)
        backend = build_backend("gan_ae", case.checkpoint_dir)
        synthetic_rows = generate_synthetic(spec, backend=backend)
        synthetic_path = out_root / f"{case.name}_synthetic.csv"
        write_csv(synthetic_rows, str(synthetic_path))
        synthetic_df = pd.read_csv(synthetic_path)

        metrics = _compare_datasets(original_df, synthetic_df)
        prompt_path = out_root / f"{case.name}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        case_record = {
            "case": case.name,
            "dataset_ref": case.dataset_ref,
            "original_csv": str(train_csv),
            "synthetic_csv": str(synthetic_path),
            "checkpoint_dir": case.checkpoint_dir,
            "prompt": prompt,
            "training_report": train_report,
            "metrics": metrics,
        }
        cases_report.append(case_record)
        print(f"[INFO] Completed {case.name}")

    final_report = {
        "cases": cases_report,
        "summary": _summarize(cases_report),
    }
    report_json = out_root / "kaggle_validation_report.json"
    report_md = out_root / "kaggle_validation_report.md"
    report_json.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    report_md.write_text(_to_markdown(final_report), encoding="utf-8")
    print(f"[DONE] Report written: {report_json}")
    print(f"[DONE] Report written: {report_md}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kaggle -> Prompt -> Synthetic -> Validation pipeline.")
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run runtime/data checks only; do not start training.",
    )
    p.add_argument(
        "--expect-gpu",
        action="store_true",
        help="Fail preflight if CUDA GPU is unavailable.",
    )
    p.add_argument(
        "--ae-epochs",
        type=int,
        default=DEFAULT_AE_EPOCHS,
        help="Autoencoder epochs for each Kaggle case.",
    )
    p.add_argument(
        "--gan-epochs",
        type=int,
        default=DEFAULT_GAN_EPOCHS,
        help="Latent GAN epochs for each Kaggle case.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
    )
    return p


def _run_preflight(expect_gpu: bool) -> int:
    results: dict[str, Any] = {}
    try:
        import torch  # type: ignore

        results["torch_version"] = torch.__version__
        results["cuda_available"] = bool(torch.cuda.is_available())
        results["cuda_device_count"] = int(torch.cuda.device_count())
        results["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        print(json.dumps({"ok": False, "stage": "torch", "error": str(exc)}, indent=2))
        return 1

    if expect_gpu and not results["cuda_available"]:
        print(json.dumps({"ok": False, "stage": "gpu", "details": results}, indent=2))
        return 1

    # Verify Kaggle access by downloading one known small dataset.
    try:
        p = kagglehub.dataset_download(CASES[0].dataset_ref)
        results["kaggle_download_ok"] = True
        results["kaggle_sample_path"] = str(p)
    except Exception as exc:
        print(json.dumps({"ok": False, "stage": "kaggle_download", "details": results, "error": str(exc)}, indent=2))
        return 1

    out_root = Path("kaggle_eval/output")
    out_root.mkdir(parents=True, exist_ok=True)
    test_file = out_root / "preflight_ok.txt"
    test_file.write_text("ok", encoding="utf-8")
    results["write_ok"] = test_file.exists()
    results["ready_to_start_training"] = True
    print(json.dumps({"ok": True, "preflight": results}, indent=2))
    return 0


def _download_case(case: KaggleCase) -> Path:
    base = Path(kagglehub.dataset_download(case.dataset_ref))
    csv_path = base / case.csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found for {case.name}: {csv_path}")
    return csv_path


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == object:
            series = df[col].astype(str).str.strip()
            # Convert numeric-like string columns (e.g., Telco TotalCharges)
            maybe = pd.to_numeric(series.replace("", np.nan), errors="coerce")
            if maybe.notna().mean() > 0.9:
                df[col] = maybe.fillna(maybe.median())
            else:
                df[col] = series
    return df


def _cap_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True)


def _derive_prompt(case_name: str, df: pd.DataFrame) -> str:
    n = len(df)
    if case_name == "stroke_prediction":
        return (
            f"Generate {n} healthcare records age 40-75 with high glucose and hypertension mix, "
            "mostly non smoker; industry=healthcare_v1; strict=false; seed=42"
        )
    if case_name == "heart_failure":
        return (
            f"Generate {n} healthcare records age 45-80 with high risk score and mixed smoker status, "
            "mostly hypertensive; industry=healthcare_v1; strict=false; seed=42"
        )
    if case_name == "pima_diabetes":
        return (
            f"Generate {n} healthcare records age 30-70 with bmi 22-42 and high glucose, "
            "70% diabetic; industry=healthcare_v1; strict=false; seed=42"
        )
    if case_name == "credit_card_fraud":
        return (
            f"Generate {n} finance records with varied transaction_amount and risk_score, "
            "mostly non defaulted customers; industry=finance_v1; strict=false; seed=42"
        )
    if case_name == "adult_income":
        return (
            f"Generate {n} finance records age 25-60 with medium to high income, "
            "mostly employed; industry=finance_v1; strict=false; seed=42"
        )
    return f"Generate {n} records; industry=healthcare_v1; strict=false; seed=42"


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().replace(" ", "").replace("_", "")
        if k in norm:
            return norm[k]
    raise KeyError(f"Could not find any of columns: {candidates}")


def _iqr_range(df: pd.DataFrame, col: str) -> tuple[float, float]:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    return (round(q1, 2), round(q3, 2))


def _majority_label(series: pd.Series) -> str:
    vals = series.astype(str).str.strip().str.lower().replace({"1": "yes", "0": "no"})
    top = vals.value_counts().index[0]
    return "yes" if top in {"yes", "true", "1"} else "no"


def _compare_datasets(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict[str, Any]:
    common = [c for c in real_df.columns if c in syn_df.columns]
    numeric = [c for c in common if pd.api.types.is_numeric_dtype(real_df[c])]
    categorical = [c for c in common if c not in numeric]

    num_scores = []
    num_details = []
    for col in numeric:
        a = pd.to_numeric(real_df[col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(syn_df[col], errors="coerce").dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        ks = float(ks_2samp(a, b).statistic)
        wd = float(wasserstein_distance(a, b))
        scale = float(np.std(a)) if float(np.std(a)) > 1e-8 else 1.0
        wd_norm = wd / scale
        score = max(0.0, 1.0 - min(1.0, 0.5 * ks + 0.5 * min(1.0, wd_norm)))
        num_scores.append(score)
        num_details.append({"column": col, "ks": ks, "wasserstein_norm": wd_norm, "score": score})

    cat_scores = []
    cat_details = []
    for col in categorical:
        a = real_df[col].astype(str).str.lower().value_counts(normalize=True)
        b = syn_df[col].astype(str).str.lower().value_counts(normalize=True)
        idx = sorted(set(a.index).union(set(b.index)))
        av = np.array([float(a.get(i, 0.0)) for i in idx])
        bv = np.array([float(b.get(i, 0.0)) for i in idx])
        tv = 0.5 * float(np.abs(av - bv).sum())
        score = max(0.0, 1.0 - min(1.0, tv))
        cat_scores.append(score)
        cat_details.append({"column": col, "tv_distance": tv, "score": score})

    overall = float(np.mean(num_scores + cat_scores)) if (num_scores or cat_scores) else 0.0
    return {
        "common_columns": len(common),
        "numeric_columns": len(numeric),
        "categorical_columns": len(categorical),
        "numeric_score_avg": float(np.mean(num_scores)) if num_scores else 0.0,
        "categorical_score_avg": float(np.mean(cat_scores)) if cat_scores else 0.0,
        "overall_similarity_score": overall,
        "numeric_details": num_details,
        "categorical_details": cat_details,
    }


def _summarize(cases_report: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [c["metrics"]["overall_similarity_score"] for c in cases_report]
    return {
        "num_cases": len(cases_report),
        "overall_similarity_avg": float(np.mean(scores)) if scores else 0.0,
        "best_case": max(cases_report, key=lambda c: c["metrics"]["overall_similarity_score"])["case"]
        if cases_report
        else None,
        "worst_case": min(cases_report, key=lambda c: c["metrics"]["overall_similarity_score"])["case"]
        if cases_report
        else None,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Kaggle Validation Report", ""]
    s = report["summary"]
    lines.append(f"- Cases: {s['num_cases']}")
    lines.append(f"- Overall similarity average: {s['overall_similarity_avg']:.4f}")
    lines.append(f"- Best case: {s['best_case']}")
    lines.append(f"- Worst case: {s['worst_case']}")
    lines.append("")
    for c in report["cases"]:
        m = c["metrics"]
        lines.append(f"## {c['case']}")
        lines.append(f"- Dataset: `{c['dataset_ref']}`")
        lines.append(f"- Prompt: `{c['prompt']}`")
        lines.append(f"- Checkpoint: `{c['checkpoint_dir']}`")
        lines.append(f"- Similarity score: {m['overall_similarity_score']:.4f}")
        lines.append(f"- Numeric avg: {m['numeric_score_avg']:.4f}, Categorical avg: {m['categorical_score_avg']:.4f}")
        lines.append(f"- Synthetic CSV: `{c['synthetic_csv']}`")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
