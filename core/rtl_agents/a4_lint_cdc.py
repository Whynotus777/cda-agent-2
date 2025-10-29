"""
A4 - Lint & CDC Assistant

Parses tool logs, classifies violations, and proposes automatic fixes.
Target: ≥50% of lint warnings auto-fixed successfully.
"""

import re
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

from .base_agent import BaseAgent, AgentOutput

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    """Parsed issue from tool log"""
    id: str
    severity: str  # error, warning, info
    category: str  # syntax, semantic, cdc, lint, style
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    rule_violated: Optional[str] = None


class A4_LintCDCAssistant(BaseAgent):
    """
    Lint & CDC Assistant - Parses logs and suggests fixes.

    Capabilities:
    - Parse Verilator, Yosys, and other tool logs
    - Classify issues by category and severity
    - Generate automatic fix proposals
    - Calculate confidence scores
    - Prioritize fixable issues
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize A4 agent.

        Args:
            config: Configuration dict
        """
        super().__init__(
            agent_id="A4",
            agent_name="Lint & CDC Assistant",
            config=config
        )

        # Load fix patterns
        self.fix_patterns = self._load_fix_patterns()

        # Statistics
        self.stats = {
            'total_issues_parsed': 0,
            'fixes_generated': 0,
            'high_confidence_fixes': 0
        }

        logger.info("A4 Lint & CDC Assistant initialized")

    def _load_fix_patterns(self) -> Dict[str, Any]:
        """Load common lint issue patterns and their fixes"""
        return {
            # Verilator patterns
            'verilator': {
                'undeclared_signal': {
                    'pattern': r"Signal not found: '(\w+)'",
                    'fix_type': 'insertion',
                    'confidence': 0.85,
                    'template': 'wire {signal};'
                },
                'unused_signal': {
                    'pattern': r"Signal is not used: '(\w+)'",
                    'fix_type': 'deletion',
                    'confidence': 0.80,
                    'template': '// Remove unused signal: {signal}'
                },
                'width_mismatch': {
                    'pattern': r"Width mismatch.*(\d+) vs (\d+)",
                    'fix_type': 'replacement',
                    'confidence': 0.75,
                    'template': 'Add width casting'
                },
                'blocking_assignment': {
                    'pattern': r"Blocking assignment.*in sequential logic",
                    'fix_type': 'replacement',
                    'confidence': 0.90,
                    'template': 'Change = to <='
                },
                'module_elaborated': {
                    'pattern': r"Module.*elaborated successfully",
                    'fix_type': 'none',
                    'confidence': 1.0,
                    'template': 'No fix needed (info only)'
                }
            },
            # Yosys patterns
            'yosys': {
                'unknown_module': {
                    'pattern': r"Module.*not found",
                    'fix_type': 'insertion',
                    'confidence': 0.75,
                    'template': 'Add module instantiation'
                },
                'undriven_signal': {
                    'pattern': r"Wire.*is undriven",
                    'fix_type': 'insertion',
                    'confidence': 0.80,
                    'template': 'assign {signal} = 1\'b0;'
                },
                'unused_wire': {
                    'pattern': r"Unused wire",
                    'fix_type': 'deletion',
                    'confidence': 0.85,
                    'template': 'Remove unused wire'
                },
                'syntax_error': {
                    'pattern': r"Syntax error",
                    'fix_type': 'replacement',
                    'confidence': 0.40,
                    'template': 'Manual fix required'
                }
            },
            # CDC (Clock Domain Crossing) patterns
            'cdc': {
                'async_crossing': {
                    'pattern': r"[Aa]synchronous.*crossing",
                    'fix_type': 'insertion',
                    'confidence': 0.85,
                    'template': 'Insert 2-stage synchronizer'
                },
                'missing_sync': {
                    'pattern': r"[Mm]issing synchronizer",
                    'fix_type': 'insertion',
                    'confidence': 0.85,
                    'template': 'Add 2-stage synchronizer'
                },
                'reset_crossing': {
                    'pattern': r"[Rr]eset crossing",
                    'fix_type': 'insertion',
                    'confidence': 0.80,
                    'template': 'Add reset synchronization'
                }
            }
        }

    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Parse tool log and generate fix proposals.

        Args:
            input_data: Dict conforming to analysis_report schema

        Returns:
            AgentOutput with parsed issues and fix proposals
        """
        start_time = time.time()

        # Validate input
        if not self.validate_input(input_data):
            return self.create_output(
                success=False,
                output_data={},
                errors=["Invalid input data"]
            )

        tool = input_data.get('tool', 'verilator')
        log_content = input_data.get('log_content', '')

        logger.info(f"A4 parsing {tool} log ({len(log_content)} chars)")

        # Parse log to extract issues
        issues = self._parse_log(tool, log_content)

        self.stats['total_issues_parsed'] += len(issues)

        # Generate fix proposals for each issue
        fix_proposals = []
        for issue in issues:
            fixes = self._generate_fixes(issue, tool)
            fix_proposals.extend(fixes)
            if fixes:
                self.stats['fixes_generated'] += len(fixes)

        # Count high-confidence fixes
        high_conf_fixes = [f for f in fix_proposals if f.get('confidence', 0) >= 0.75]
        self.stats['high_confidence_fixes'] += len(high_conf_fixes)

        # Calculate summary statistics
        summary = self._calculate_summary(issues, fix_proposals)

        execution_time = (time.time() - start_time) * 1000

        output_data = {
            'issues': [self._issue_to_dict(issue) for issue in issues],
            'fix_proposals': fix_proposals,
            'summary': summary,
            'statistics': self.stats.copy()
        }

        # Success if we parsed issues and generated at least some fixes
        success = len(issues) > 0
        errors = [] if success else ["No issues found in log"]
        warnings = []

        if len(fix_proposals) == 0 and len(issues) > 0:
            warnings.append(f"No fixes generated for {len(issues)} issues")

        return self.create_output(
            success=success,
            output_data=output_data,
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time,
            metadata={'tool': tool, 'issues_count': len(issues)}
        )

    def _parse_log(self, tool: str, log_content: str) -> List[Issue]:
        """
        Parse tool log to extract issues.

        Args:
            tool: Tool name
            log_content: Raw log content

        Returns:
            List of Issue objects
        """
        issues = []

        if tool == 'verilator':
            issues = self._parse_verilator_log(log_content)
        elif tool == 'yosys':
            issues = self._parse_yosys_log(log_content)
        elif tool == 'spyglass':
            issues = self._parse_spyglass_log(log_content)
        else:
            logger.warning(f"Unknown tool: {tool}, using generic parser")
            issues = self._parse_generic_log(log_content)

        logger.info(f"Parsed {len(issues)} issues from {tool} log")
        return issues

    def _parse_verilator_log(self, log: str) -> List[Issue]:
        """Parse Verilator lint output"""
        issues = []

        # Verilator format: %Error: file.v:42:10: Signal not found: 'foo'
        pattern = r'%(Error|Warning|Info):\s+([^:]+):(\d+)(?::(\d+))?:\s+(.+)'

        for match in re.finditer(pattern, log):
            severity = match.group(1).lower()
            file = match.group(2)
            line = int(match.group(3))
            column = int(match.group(4)) if match.group(4) else None
            message = match.group(5)

            # Classify category
            category = self._classify_issue(message, 'verilator')

            issue = Issue(
                id=str(uuid.uuid4()),
                severity=severity,
                category=category,
                message=message,
                file=file,
                line=line,
                column=column
            )
            issues.append(issue)

        return issues

    def _parse_yosys_log(self, log: str) -> List[Issue]:
        """Parse Yosys synthesis log"""
        issues = []

        # Yosys format varies, common patterns:
        # ERROR: ... at file.v:42
        # Warning: ... in module 'foo' (file.v:42)

        patterns = [
            r'(ERROR|Warning|Info):\s+([^\(]+)\s+(?:at|in)\s+(?:\w+\s+)?["\']?([^"\']+\.\w+)["\']?:(\d+)',
            r'(ERROR|Warning):\s+(.+)',  # Generic fallback
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, log):
                severity = match.group(1).lower()
                message = match.group(2).strip()

                file = match.group(3) if len(match.groups()) >= 3 else None
                line = int(match.group(4)) if len(match.groups()) >= 4 else None

                category = self._classify_issue(message, 'yosys')

                issue = Issue(
                    id=str(uuid.uuid4()),
                    severity=severity,
                    category=category,
                    message=message,
                    file=file,
                    line=line
                )
                issues.append(issue)

        return issues

    def _parse_spyglass_log(self, log: str) -> List[Issue]:
        """Parse Spyglass CDC log (stub for now)"""
        # TODO: Implement Spyglass-specific parsing
        return self._parse_generic_log(log)

    def _parse_generic_log(self, log: str) -> List[Issue]:
        """Generic log parser for unknown tools"""
        issues = []

        # Look for common patterns
        pattern = r'(error|warning|info)[:\s]+(.+)'

        for match in re.finditer(pattern, log, re.IGNORECASE):
            severity = match.group(1).lower()
            message = match.group(2).strip()

            issue = Issue(
                id=str(uuid.uuid4()),
                severity=severity,
                category='lint',
                message=message
            )
            issues.append(issue)

        return issues

    def _classify_issue(self, message: str, tool: str) -> str:
        """Classify issue category based on message content"""
        message_lower = message.lower()

        if any(kw in message_lower for kw in ['syntax', 'parse', 'token']):
            return 'syntax'
        elif any(kw in message_lower for kw in ['clock', 'cdc', 'crossing', 'async']):
            return 'cdc'
        elif any(kw in message_lower for kw in ['blocking', 'latch', 'combinational']):
            return 'semantic'
        elif any(kw in message_lower for kw in ['unused', 'width', 'undriven']):
            return 'lint'
        elif any(kw in message_lower for kw in ['style', 'naming', 'format']):
            return 'style'
        else:
            return 'other'

    def _generate_fixes(self, issue: Issue, tool: str) -> List[Dict[str, Any]]:
        """
        Generate fix proposals for an issue.

        Args:
            issue: Issue to fix
            tool: Tool that reported the issue

        Returns:
            List of fix proposal dicts
        """
        fixes = []

        # Skip info-level issues (no fix needed)
        if issue.severity == 'info':
            return fixes

        # Get fix patterns for this tool
        tool_patterns = self.fix_patterns.get(tool, {})

        # Also check CDC patterns if issue is CDC-related
        if issue.category == 'cdc':
            tool_patterns = {**tool_patterns, **self.fix_patterns.get('cdc', {})}

        # Try to match issue message against known patterns
        for pattern_name, pattern_info in tool_patterns.items():
            pattern = pattern_info['pattern']
            match = re.search(pattern, issue.message, re.IGNORECASE)

            if match:
                # Skip if pattern indicates no fix needed
                if pattern_info.get('fix_type') == 'none':
                    continue

                # Generate fix based on pattern
                fix = self._create_fix_proposal(
                    issue=issue,
                    pattern_name=pattern_name,
                    pattern_info=pattern_info,
                    match_groups=match.groups()
                )
                fixes.append(fix)
                break

        # If no pattern matched, try generic fix
        if not fixes and issue.severity in ['error', 'warning']:
            generic_fix = self._create_generic_fix(issue)
            if generic_fix:
                fixes.append(generic_fix)

        return fixes

    def _create_fix_proposal(
        self,
        issue: Issue,
        pattern_name: str,
        pattern_info: Dict,
        match_groups: Tuple
    ) -> Dict[str, Any]:
        """Create a fix proposal from a matched pattern"""

        # Extract signal name or other captured groups
        signal_name = match_groups[0] if match_groups else 'unknown'

        # Generate fixed code from template
        template = pattern_info.get('template', '')
        fixed_code = template.format(signal=signal_name)

        # Create explanation
        explanation = f"Fix {issue.category} issue: {pattern_name.replace('_', ' ')}"

        fix_proposal = {
            'proposal_id': str(uuid.uuid4()),
            'issue_id': issue.id,
            'fix_type': pattern_info.get('fix_type', 'replacement'),
            'confidence': pattern_info.get('confidence', 0.5),
            'file': issue.file,
            'line_start': issue.line,
            'line_end': issue.line,
            'original_code': f"// Issue: {issue.message}",
            'fixed_code': fixed_code,
            'explanation': explanation,
            'category': issue.category,
            'auto_applicable': pattern_info.get('confidence', 0.5) >= 0.75,
            'test_required': True
        }

        return fix_proposal

    def _create_generic_fix(self, issue: Issue) -> Optional[Dict[str, Any]]:
        """Create a generic fix suggestion when no pattern matches"""

        # Low confidence generic suggestion
        fix_proposal = {
            'proposal_id': str(uuid.uuid4()),
            'issue_id': issue.id,
            'fix_type': 'refactor',
            'confidence': 0.30,
            'file': issue.file,
            'line_start': issue.line,
            'line_end': issue.line,
            'original_code': f"// Issue: {issue.message}",
            'fixed_code': f"// TODO: Fix {issue.category} issue manually",
            'explanation': f"Manual fix required for: {issue.message}",
            'category': issue.category,
            'auto_applicable': False,
            'test_required': True
        }

        return fix_proposal

    def _calculate_summary(
        self,
        issues: List[Issue],
        fix_proposals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate summary statistics"""

        # Count by severity
        severity_counts = {'error': 0, 'warning': 0, 'info': 0}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        # Count by category
        category_counts = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

        # Count fixable issues
        auto_fixable = sum(1 for f in fix_proposals if f.get('auto_applicable', False))
        high_confidence = sum(1 for f in fix_proposals if f.get('confidence', 0) >= 0.75)

        # Calculate success rate
        fix_rate = (len(fix_proposals) / len(issues) * 100) if issues else 0
        auto_fix_rate = (auto_fixable / len(issues) * 100) if issues else 0

        return {
            'total_issues': len(issues),
            'errors': severity_counts.get('error', 0),
            'warnings': severity_counts.get('warning', 0),
            'info': severity_counts.get('info', 0),
            'by_category': category_counts,
            'fix_proposals_generated': len(fix_proposals),
            'auto_fixable': auto_fixable,
            'high_confidence_fixes': high_confidence,
            'fix_generation_rate': f"{fix_rate:.1f}%",
            'auto_fix_rate': f"{auto_fix_rate:.1f}%"
        }

    def _issue_to_dict(self, issue: Issue) -> Dict[str, Any]:
        """Convert Issue to dictionary"""
        return {
            'id': issue.id,
            'severity': issue.severity,
            'category': issue.category,
            'message': issue.message,
            'file': issue.file,
            'line': issue.line,
            'column': issue.column,
            'code_snippet': issue.code_snippet,
            'rule_violated': issue.rule_violated
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input against analysis_report schema"""
        required_fields = ['tool', 'log_content']

        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False

        return True

    def get_schema(self) -> Dict[str, Any]:
        """Return input schema for A4"""
        schema_path = Path(__file__).parent.parent / 'schemas' / 'analysis_report.json'

        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return {}
