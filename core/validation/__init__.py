"""
Strict validation framework for HDL designs

This module ensures all generated HDL code is:
1. Syntactically correct (strict mode)
2. Simulatable with testbenches
3. Functionally correct per specifications
4. Production-ready quality
"""

from .strict_fsm_validator import StrictFSMValidator, ValidationResult, validate_fsm_strict

__all__ = ['StrictFSMValidator', 'ValidationResult', 'validate_fsm_strict']
