from __future__ import annotations

import json
import os
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from checkpoint_registry import (
    DEFAULT_REGISTRY_PATH,
    load_registry,
    resolve_checkpoint_for_profile,
    validate_checkpoint_dir,
)
from generator import build_backend, generate_synthetic, write_csv
from prompt_parser.parse_router import parse_user_prompt
from prompt_parser.parser import PromptParseException

HARD_CODED_BACKEND = "gan_ae"
HARD_CODED_REGISTRY = DEFAULT_REGISTRY_PATH


def _timestamped_csv_path(base_path: str | None) -> str:
    raw = base_path.strip() if base_path else "output/synthetic_data.csv"
    path = Path(raw)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    return str(path)


class PromptGeneratorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Syngen")
        self.root.geometry("760x520")
        self.root.minsize(700, 480)

        self.prompt_var = tk.StringVar()
        self.parse_mode_var = tk.StringVar(value="hybrid")
        self.out_var = tk.StringVar(value="output/synthetic_data.csv")
        self.last_csv_path: str | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=14)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text="Syngen: Prompt-to-CSV Generator", font=("Segoe UI", 14, "bold"))
        title.pack(anchor="w")

        ttk.Label(container, text="Enter your prompt:").pack(anchor="w", pady=(12, 4))
        self.prompt_text = tk.Text(container, height=7, wrap="word")
        self.prompt_text.pack(fill=tk.X)
        self.prompt_text.insert(
            "1.0",
            "Generate 1200 healthcare patients age 40-75 with high glucose and hypertension, mostly non smoker",
        )

        options = ttk.Frame(container)
        options.pack(fill=tk.X, pady=(12, 8))

        ttk.Label(options, text="Parse mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            options,
            textvariable=self.parse_mode_var,
            values=["hybrid", "rules", "llm"],
            state="readonly",
            width=12,
        ).grid(row=1, column=0, sticky="w", padx=(0, 16))

        ttk.Label(options, text="Backend").grid(row=0, column=1, sticky="w")
        ttk.Label(options, text="gan_ae (hard-coded)").grid(row=1, column=1, sticky="w", padx=(0, 16))

        ttk.Label(options, text="Output base path").grid(row=0, column=2, sticky="w")
        ttk.Entry(options, textvariable=self.out_var, width=35).grid(row=1, column=2, sticky="we")

        extra = ttk.Frame(container)
        extra.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(extra, text="Checkpoint routing").grid(row=0, column=0, sticky="w")
        ttk.Label(extra, text=f"Registry: {HARD_CODED_REGISTRY}").grid(row=1, column=0, sticky="w")

        actions = ttk.Frame(container)
        actions.pack(fill=tk.X, pady=(8, 10))
        ttk.Button(actions, text="Parse Preview", command=self.on_preview).pack(side=tk.LEFT)
        ttk.Button(actions, text="Generate Dataset", command=self.on_generate).pack(side=tk.LEFT, padx=(8, 0))
        self.open_btn = ttk.Button(actions, text="Open Last CSV", command=self.on_open_last, state=tk.DISABLED)
        self.open_btn.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(container, text="Result / Warnings:").pack(anchor="w", pady=(4, 4))
        self.output_text = tk.Text(container, height=12, wrap="word")
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.configure(state=tk.DISABLED)

    def on_generate(self) -> None:
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showerror("Missing Prompt", "Please enter a prompt.")
            return

        try:
            spec = parse_user_prompt(prompt, mode=self.parse_mode_var.get())
            registry = load_registry(HARD_CODED_REGISTRY)
            resolved_ckpt = resolve_checkpoint_for_profile(
                spec.target_dataset_profile,
                registry,
                prompt_text=prompt,
            )
            missing = validate_checkpoint_dir(resolved_ckpt)
            if missing:
                messagebox.showerror(
                    "Missing GAN+AE Checkpoints",
                    "Full GAN+AE artifacts not found in selected checkpoint:\n"
                    f"{resolved_ckpt}\n\n"
                    "Missing files:\n- "
                    + "\n- ".join(missing)
                    + "\n\nTrain first with:\n"
                    "python -m gan_ae_full.train --csv path/to/train.csv --out checkpoints/full_gan_ae_demo --device cpu\n\n"
                    "Then map your domain in checkpoints/registry.json",
                )
                return
            backend = build_backend(HARD_CODED_BACKEND, resolved_ckpt)
            out_path = _timestamped_csv_path(self.out_var.get())
            rows = generate_synthetic(spec, backend=backend)
            csv_path = write_csv(rows, out_path)
            self.last_csv_path = csv_path
            self.open_btn.configure(state=tk.NORMAL)
            self._write_result(spec, csv_path, len(rows), resolved_ckpt)
            self._ask_open_excel(csv_path)
        except PromptParseException as exc:
            messagebox.showerror("Generation Failed", self._format_parse_error(exc))
        except (ValueError, FileNotFoundError) as exc:
            messagebox.showerror("Generation Failed", str(exc))
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Unexpected Error", str(exc))

    def on_preview(self) -> None:
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showerror("Missing Prompt", "Please enter a prompt.")
            return
        try:
            spec = parse_user_prompt(prompt, mode=self.parse_mode_var.get())
            preview = {
                "n_rows": spec.n_rows,
                "target_dataset_profile": spec.target_dataset_profile,
                "strict_mode": spec.strict_mode,
                "seed": spec.seed,
                "filters": spec.filters,
                "distribution_hints": spec.distribution_hints,
                "priority_rules": spec.priority_rules,
                "warnings": spec.warnings,
            }
            self.output_text.configure(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", "Parsed prompt preview:\n\n" + json.dumps(preview, indent=2))
            self.output_text.configure(state=tk.DISABLED)
        except PromptParseException as exc:
            messagebox.showerror("Parse Failed", self._format_parse_error(exc))
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Parse Failed", str(exc))

    def on_open_last(self) -> None:
        if not self.last_csv_path:
            messagebox.showinfo("No File", "No generated CSV available yet.")
            return
        try:
            os.startfile(self.last_csv_path)  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showwarning("Open Failed", f"Could not open CSV automatically:\n{exc}")

    def _write_result(
        self,
        spec,
        csv_path: str,
        n_rows: int,
        ckpt_path: str,
    ) -> None:
        lines = [
            f"CSV created: {csv_path}",
            f"Rows generated: {n_rows}",
            f"Profile: {spec.target_dataset_profile}",
            f"Checkpoint used: {ckpt_path}",
            "",
            "Parsed prompt summary:",
            json.dumps(
                {
                    "n_rows": spec.n_rows,
                    "strict_mode": spec.strict_mode,
                    "seed": spec.seed,
                    "filters": spec.filters,
                    "distribution_hints": spec.distribution_hints,
                },
                indent=2,
            ),
        ]
        if spec.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {w}" for w in spec.warnings)
        else:
            lines.append("Warnings: none")
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", "\n".join(lines))
        self.output_text.configure(state=tk.DISABLED)

    def _format_parse_error(self, exc: PromptParseException) -> str:
        rep = exc.report
        parts = [rep.message]
        if rep.conflicting_fields:
            parts.append("Conflicting fields: " + ", ".join(rep.conflicting_fields))
        if rep.offending_clauses:
            parts.append("Offending clauses: " + " | ".join(rep.offending_clauses))
        if rep.suggested_prompt:
            parts.append("Suggested fix: " + rep.suggested_prompt)
        if rep.warnings:
            parts.append("Warnings: " + " | ".join(rep.warnings))
        return "\n\n".join(parts)

    def _ask_open_excel(self, csv_path: str) -> None:
        open_now = messagebox.askyesno(
            "CSV Ready",
            f"Dataset saved to:\n{csv_path}\n\nOpen in Excel now?",
        )
        if not open_now:
            return
        try:
            os.startfile(csv_path)  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showwarning("Open Failed", f"Could not open CSV automatically:\n{exc}")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    PromptGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
