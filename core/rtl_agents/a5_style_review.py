"""
A5 - Style & Review Copilot

Enforces coding standards, naming conventions, and security rules on RTL code.
Target: 0 critical violations, explainable results.
"""

import re
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .base_agent import BaseAgent, AgentOutput

logger = logging.getLogger(__name__)


@dataclass
class StyleViolation:
    """A style/security violation"""
    id: str
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'naming', 'style', 'security', 'clock', 'reset'
    rule_id: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    suggestion: Optional[str] = None


class A5_StyleReviewCopilot(BaseAgent):
    """
    Style & Review Copilot - Enforces coding standards and security rules.

    Capabilities:
    - Naming convention enforcement
    - Clock/reset domain checking
    - Security rule validation
    - Style guideline compliance
    - Annotated diff generation
    - Markdown report generation
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize A5 agent.

        Args:
            config: Configuration dict
        """
        super().__init__(
            agent_id="A5",
            agent_name="Style & Review Copilot",
            config=config
        )

        # Load style rules
        self.rules = self._load_style_rules()

        logger.info(f"A5 Style & Review Copilot initialized with {len(self.rules)} rules")

    def _load_style_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load style and security rules"""
        return {
            # Naming conventions (context-aware)
            'N003': {
                'category': 'naming',
                'severity': 'critical',
                'pattern': None,  # Custom check
                'message': 'Clock signals must be named clk or clk_*',
                'applies_to': 'clock_signals'
            },
            'N004': {
                'category': 'naming',
                'severity': 'critical',
                'pattern': None,  # Custom check
                'message': 'Active-low reset must be named rst_n or rstn',
                'applies_to': 'reset_signals'
            },

            # Security rules
            'S001': {
                'category': 'security',
                'severity': 'critical',
                'pattern': r'\b(password|key|secret)\b',
                'message': 'Sensitive data identifiers detected - ensure encryption',
                'check': 'sensitive_data'
            },
            'S002': {
                'category': 'security',
                'severity': 'warning',
                'pattern': r'//\s*(TODO|FIXME|HACK)',
                'message': 'Development comment found - resolve before production',
                'check': 'dev_comments'
            },

            # Best practices
            'BP001': {
                'category': 'best_practice',
                'severity': 'warning',
                'pattern': r'\balways\s+@',
                'message': 'Consider using always_comb, always_ff in SystemVerilog',
                'check': 'sv_practices'
            },
            'BP002': {
                'category': 'best_practice',
                'severity': 'info',
                'pattern': r'\$display',
                'message': 'Use $info/$warning/$error for better simulation messages',
                'check': 'sim_messages'
            }
        }

    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Review RTL code for style and security compliance.

        Args:
            input_data: Dict with 'rtl_code' or 'file_path'

        Returns:
            AgentOutput with violations and style report
        """
        start_time = time.time()

        if not self.validate_input(input_data):
            return self.create_output(
                success=False,
                output_data={},
                errors=["Invalid input data"]
            )

        # Get RTL code
        rtl_code = input_data.get('rtl_code', '')
        file_path = input_data.get('file_path')

        if not rtl_code and file_path:
            try:
                with open(file_path, 'r') as f:
                    rtl_code = f.read()
            except Exception as e:
                return self.create_output(
                    success=False,
                    output_data={},
                    errors=[f"Failed to read file: {e}"]
                )

        if not rtl_code:
            return self.create_output(
                success=False,
                output_data={},
                errors=["No RTL code provided"]
            )

        logger.info(f"A5 reviewing {len(rtl_code)} chars of RTL")

        # Run style checks
        violations = self._check_all_rules(rtl_code, file_path or 'inline')

        # Generate report
        report_md = self._generate_markdown_report(violations, rtl_code)

        # Calculate summary
        summary = self._calculate_summary(violations)

        execution_time = (time.time() - start_time) * 1000

        output_data = {
            'review_id': str(uuid.uuid4()),
            'violations': [self._violation_to_dict(v) for v in violations],
            'summary': summary,
            'report_markdown': report_md,
            'file_reviewed': file_path or 'inline',
            'timestamp': datetime.utcnow().isoformat()
        }

        # Success if no critical violations
        critical_count = summary['critical']
        success = (critical_count == 0)

        errors = [v.message for v in violations if v.severity == 'critical']
        warnings = [v.message for v in violations if v.severity == 'warning'][:5]  # First 5

        return self.create_output(
            success=success,
            output_data=output_data,
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time,
            metadata={'total_violations': len(violations)}
        )

    def _check_all_rules(self, rtl_code: str, file_path: str) -> List[StyleViolation]:
        """
        Check all style rules against RTL code.

        Returns:
            List of violations
        """
        violations = []
        lines = rtl_code.split('\n')

        # Context-aware rule checking
        for rule_id, rule_info in self.rules.items():
            pattern = rule_info.get('pattern')
            if not pattern:
                continue

            # Apply rule based on category
            if rule_info['category'] == 'naming':
                violations.extend(self._check_naming_rule(rule_id, rule_info, rtl_code, lines, file_path))
            elif rule_info['category'] in ['security', 'best_practice', 'style']:
                # Simple pattern matching for security and best practices
                for line_num, line in enumerate(lines, 1):
                    # Case-sensitive for security, case-insensitive for others
                    flags = 0 if rule_info['category'] == 'security' else re.IGNORECASE
                    if re.search(pattern, line, flags):
                        violation = StyleViolation(
                            id=str(uuid.uuid4()),
                            severity=rule_info['severity'],
                            category=rule_info['category'],
                            rule_id=rule_id,
                            message=rule_info['message'],
                            file=file_path,
                            line=line_num,
                            suggestion=self._generate_suggestion(rule_id, line)
                        )
                        violations.append(violation)

        return violations

    def _check_naming_rule(
        self,
        rule_id: str,
        rule_info: Dict,
        rtl_code: str,
        lines: List[str],
        file_path: str
    ) -> List[StyleViolation]:
        """Check naming convention rules with context awareness"""
        violations = []

        # N003: Check for bad clock names (not clk or clk_*)
        if rule_id == 'N003':
            for line_num, line in enumerate(lines, 1):
                # Look for clock-like names in port declarations or signals
                # Patterns: "input CLK", "input wire CLK", "wire CLK", etc.
                matches = re.findall(r'(?:input|output|wire|reg|logic)\s+(?:wire\s+)?(\w*[Cc][Ll][Kk]\w*)', line)
                for clk_name in matches:
                    # Violation if it's not exactly 'clk' or 'clk_*'
                    if not re.match(r'^clk(_\w+)?$', clk_name, re.IGNORECASE):
                        violation = StyleViolation(
                            id=str(uuid.uuid4()),
                            severity='critical',
                            category='naming',
                            rule_id=rule_id,
                            message=f'Clock signal "{clk_name}" should be named clk or clk_*',
                            file=file_path,
                            line=line_num,
                            suggestion='Rename signal to clk or clk_domain'
                        )
                        violations.append(violation)

        # N004: Check for bad reset names
        elif rule_id == 'N004':
            for line_num, line in enumerate(lines, 1):
                # Look for reset-like names in port declarations or signals
                matches = re.findall(r'(?:input|output|wire|reg|logic)\s+(?:wire\s+)?(\w*[Rr][Ee][Ss][Ee][Tt]\w*)', line)
                for rst_name in matches:
                    # Violation if it's not rst_n, rstn, or reset_n
                    if not re.match(r'^(rst_n|rstn|reset_n)$', rst_name, re.IGNORECASE):
                        violation = StyleViolation(
                            id=str(uuid.uuid4()),
                            severity='critical',
                            category='naming',
                            rule_id=rule_id,
                            message=f'Reset signal "{rst_name}" should be named rst_n or rstn',
                            file=file_path,
                            line=line_num,
                            suggestion='Rename to rst_n for active-low reset'
                        )
                        violations.append(violation)

        return violations

    def _generate_suggestion(self, rule_id: str, line: str) -> Optional[str]:
        """Generate fix suggestion for a violation"""
        suggestions = {
            'N003': 'Rename signal to clk or clk_domain',
            'N004': 'Rename to rst_n for active-low reset',
            'S001': 'Encrypt sensitive data or use secure storage',
            'S002': 'Resolve TODO/FIXME before production',
            'BP001': 'Use always_ff for sequential, always_comb for combinational',
            'ST003': 'Use = for combinational, <= for sequential'
        }
        return suggestions.get(rule_id)

    def _calculate_summary(self, violations: List[StyleViolation]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        summary = {
            'total': len(violations),
            'critical': 0,
            'warning': 0,
            'info': 0,
            'by_category': {}
        }

        for v in violations:
            # Count by severity
            if v.severity == 'critical':
                summary['critical'] += 1
            elif v.severity == 'warning':
                summary['warning'] += 1
            else:
                summary['info'] += 1

            # Count by category
            cat = v.category
            summary['by_category'][cat] = summary['by_category'].get(cat, 0) + 1

        return summary

    def _generate_markdown_report(
        self,
        violations: List[StyleViolation],
        rtl_code: str
    ) -> str:
        """
        Generate markdown report.

        Returns:
            Markdown report string
        """
        lines = []

        # Header
        lines.append("# Style & Security Review Report")
        lines.append(f"**Generated:** {datetime.utcnow().isoformat()}")
        lines.append(f"**Agent:** A5 Style & Review Copilot")
        lines.append("")

        # Summary
        summary = self._calculate_summary(violations)
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Violations:** {summary['total']}")
        lines.append(f"- **Critical:** {summary['critical']} 🔴")
        lines.append(f"- **Warning:** {summary['warning']} ⚠️")
        lines.append(f"- **Info:** {summary['info']} ℹ️")
        lines.append("")

        # By category
        if summary['by_category']:
            lines.append("### By Category")
            lines.append("")
            for cat, count in sorted(summary['by_category'].items()):
                lines.append(f"- **{cat.title()}:** {count}")
            lines.append("")

        # Violations by severity
        for severity in ['critical', 'warning', 'info']:
            sev_violations = [v for v in violations if v.severity == severity]
            if not sev_violations:
                continue

            emoji = {'critical': '🔴', 'warning': '⚠️', 'info': 'ℹ️'}[severity]
            lines.append(f"## {severity.title()} Violations {emoji}")
            lines.append("")

            for v in sev_violations:
                lines.append(f"### {v.rule_id}: {v.message}")
                lines.append(f"- **Category:** {v.category}")
                if v.file:
                    lines.append(f"- **File:** `{v.file}:{v.line}`")
                if v.suggestion:
                    lines.append(f"- **Suggestion:** {v.suggestion}")
                lines.append("")

        # Compliance Status
        lines.append("## Compliance Status")
        lines.append("")
        if summary['critical'] == 0:
            lines.append("✅ **PASS** - No critical violations")
        else:
            lines.append(f"❌ **FAIL** - {summary['critical']} critical violation(s)")
        lines.append("")

        return "\n".join(lines)

    def _violation_to_dict(self, violation: StyleViolation) -> Dict:
        """Convert violation to dict"""
        return asdict(violation)

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input"""
        has_code = 'rtl_code' in input_data and input_data['rtl_code']
        has_file = 'file_path' in input_data

        if not (has_code or has_file):
            logger.error("No RTL code or file path provided")
            return False

        return True

    def get_schema(self) -> Dict[str, Any]:
        """Return input schema for A5"""
        return {
            'type': 'object',
            'properties': {
                'rtl_code': {'type': 'string'},
                'file_path': {'type': 'string'},
                'enable_categories': {
                    'type': 'array',
                    'items': {'enum': ['naming', 'clock', 'reset', 'security', 'style', 'best_practice']}
                }
            }
        }
