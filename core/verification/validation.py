"""Validation helpers for generated testbenches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]


class TestbenchValidator:
    """Runs lightweight validation on generated testbenches."""

    def validate(self, tb_path: Path, expected_module: str) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if not tb_path.exists():
            return ValidationResult(False, [f"Testbench file {tb_path} not found"], [])

        text = tb_path.read_text()

        if "module" not in text or "endmodule" not in text:
            errors.append("Missing module/endmodule in testbench")

        if expected_module not in text:
            warnings.append(f"DUT module '{expected_module}' not referenced")

        if "TODO" in text:
            warnings.append("Testbench contains TODO markers")

        return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def summarise_validation(result: ValidationResult) -> Dict[str, List[str]]:
    return {"errors": result.errors, "warnings": result.warnings}
