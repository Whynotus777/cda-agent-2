"""
Pipeline Orchestrator for 6-Agent Flow
Wraps existing agents for API/UI access
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

from api.models import (
    CodeMetrics,
    DesignSpec,
    AgentResult,
    AgentStatus,
    PipelineProgress,
    PipelineResult,
    SynthesisMetrics,
    VerificationTestbench,
)

# Import existing agents
from core.rtl_agents import (
    A1_LLMGenerator,
    A1_SpecToRTLGenerator,
    A2_BoilerplateGenerator,
    A3_ConstraintSynthesizer,
    A4_LintCDCAssistant,
    A5_StyleReviewCopilot,
    A6_EDACommandCopilot
)
from core.verification import A7_TestbenchGenerator, TestbenchGenerationResult, TestbenchValidator
# TODO: Implement EDASimulator or use simulation_engine
# from core.eda_simulator import EDASimulator


class PipelineOrchestrator:
    """Orchestrates the 6-agent pipeline with progress tracking"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.runs_dir = project_root / "data" / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Progress callbacks
        self.progress_callback: Optional[Callable] = None
        self._llm_generator: Optional[A1_LLMGenerator] = None
        self._llm_load_error: Optional[str] = None

    def _get_llm_generator(self) -> Optional[A1_LLMGenerator]:
        """Lazily load the Mixtral LLM generator if available."""
        if self._llm_generator is not None:
            return self._llm_generator

        if self._llm_load_error is not None:
            return None

        if os.getenv("USE_A1_LLM", "0").lower() not in {"1", "true", "yes", "on"}:
            self._llm_load_error = "A1 LLM generator disabled (set USE_A1_LLM=1 to enable)"
            logger.info(self._llm_load_error)
            return None

        llm_path = self.project_root / "models" / "mixtral_rtl" / "run_pure_20251030_121523" / "final_model"
        base_model_path = self.project_root / "models" / "mixtral_base" / "Mixtral-8x7B-Instruct-v0.1"
        if not llm_path.exists():
            self._llm_load_error = f"LLM model path not found: {llm_path}"
            logger.warning(self._llm_load_error)
            return None
        if not base_model_path.exists():
            self._llm_load_error = f"Base Mixtral weights not cached: {base_model_path}"
            logger.warning(self._llm_load_error)
            return None

        llm_config = {
            "model_path": str(llm_path),
            "base_model_path": str(base_model_path),
            "max_new_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        try:
            logger.info("Loading Mixtral-based A1 LLM generator...")
            self._llm_generator = A1_LLMGenerator(llm_config)
            logger.info("Mixtral A1 generator ready.")
        except Exception as exc:  # pylint: disable=broad-except
            self._llm_load_error = f"Failed to load A1 LLM generator: {exc}"
            logger.warning(self._llm_load_error)
            self._llm_generator = None

        return self._llm_generator

    def execute_pipeline(
        self,
        spec: DesignSpec,
        run_id: Optional[str] = None,
        enable_agents: Optional[Dict[str, bool]] = None
    ) -> PipelineResult:
        """
        Execute full 6-agent pipeline

        Args:
            spec: Design specification
            run_id: Optional run ID (generated if not provided)
            enable_agents: Dict of agent_name -> enabled (defaults to all enabled)

        Returns:
            PipelineResult with all agent outputs
        """
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if enable_agents is None:
            enable_agents = {
                "a1": True, "a5": True, "a4": True,
                "a3": True, "a6": True, "yosys": True
            }

        # Create run directory
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save spec
        spec_file = run_dir / "spec.json"
        with spec_file.open('w') as f:
            json.dump(spec.model_dump(), f, indent=2)

        result = PipelineResult(
            run_id=run_id,
            spec=spec,
            status=AgentStatus.RUNNING,
            duration_seconds=0.0,
            start_time=datetime.now()
        )

        start_time = time.time()

        try:
            # Stage 1: A1 - RTL Generation
            if enable_agents.get("a1", True):
                self._update_progress(run_id, "A1 - RTL Generation", 1, 6)
                result.a1_rtl_generation = self._run_a1(spec, run_dir)
                result.rtl_file = result.a1_rtl_generation.output_file

            # Stage 2: A5 - Style Review
            if enable_agents.get("a5", True) and result.rtl_file:
                self._update_progress(run_id, "A5 - Style Review", 2, 6)
                result.a5_style_review = self._run_a5(result.rtl_file, run_dir)

            # Stage 3: A4 - Lint & CDC
            if enable_agents.get("a4", True) and result.rtl_file:
                self._update_progress(run_id, "A4 - Lint & CDC", 3, 6)
                result.a4_lint_cdc = self._run_a4(result.rtl_file, run_dir)

            # Stage 4: A3 - Constraint Synthesis
            if enable_agents.get("a3", True) and result.rtl_file:
                self._update_progress(run_id, "A3 - Constraint Synthesis", 4, 6)
                result.a3_constraints = self._run_a3(spec, result.rtl_file, run_dir)
                result.sdc_file = result.a3_constraints.output_file

            # Stage 5: A6 - Synthesis Script Generation
            if enable_agents.get("a6", True) and result.rtl_file:
                self._update_progress(run_id, "A6 - Synthesis Script", 5, 6)
                result.a6_synthesis_script = self._run_a6(
                    spec,
                    result.rtl_file,
                    result.sdc_file,
                    run_dir
                )
                result.synthesis_script = result.a6_synthesis_script.output_file

            # Stage 6: Yosys Synthesis
            if enable_agents.get("yosys", True) and result.synthesis_script:
                self._update_progress(run_id, "Yosys Synthesis", 6, 6)
                result.yosys_synthesis = self._run_yosys(
                    result.synthesis_script,
                    run_dir
                )
                result.synthesis_report = result.yosys_synthesis.output_file
                result.synthesis_success = result.yosys_synthesis.status == AgentStatus.SUCCESS

            # Compute metrics
            if result.rtl_file:
                metrics = self._compute_metrics(result.rtl_file)
                result.total_lines = metrics.lines_total
                result.total_ports = metrics.ports_count

            result.status = AgentStatus.SUCCESS
            result.end_time = datetime.now()

        except Exception as e:
            result.status = AgentStatus.FAILED
            result.errors_count = 1
            result.end_time = datetime.now()
            # Log error
            error_log = run_dir / "error.log"
            error_log.write_text(str(e))

        result.duration_seconds = time.time() - start_time

        # Save result
        self._save_pipeline_result(run_dir, result)

        return result

    def _run_a1(self, spec: DesignSpec, run_dir: Path) -> AgentResult:
        """Run A1 - RTL Generation"""
        start = time.time()

        try:
            input_data = {
                "module_name": spec.module_name,
                "specification": spec.description,
                "parameters": spec.parameters or {},
                "intent_type": getattr(spec, 'intent_type', None)
            }

            generator_used = "planner_composer"
            fallback_reason = None
            llm_attempted = False
            llm_generator = self._get_llm_generator()
            agent_output = None

            if llm_generator is not None:
                llm_attempted = True
                try:
                    agent_output = llm_generator.process(input_data)
                    if agent_output.success:
                        generator_used = "llm_mixtral"
                except Exception as exc:  # pylint: disable=broad-except
                    fallback_reason = f"LLM generation failed: {exc}"
                    logger.warning(fallback_reason)
                    agent_output = None

            if agent_output is None or not agent_output.success:
                if fallback_reason is None and llm_attempted:
                    fallback_reason = "LLM output invalid; falling back to planner"
                generator = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})
                agent_output = generator.process(input_data)

            if not agent_output.success:
                return AgentResult(
                    agent_name="A1_RTL_Generation",
                    status=AgentStatus.FAILED,
                    duration_seconds=time.time() - start,
                    errors=agent_output.errors,
                    fallback_reason=fallback_reason,
                    llm_attempted=llm_attempted,
                    generator=generator_used,
                )

            rtl_code = agent_output.output_data.get('rtl_code', '')
            if not rtl_code.strip():
                return AgentResult(
                    agent_name="A1_RTL_Generation",
                    status=AgentStatus.FAILED,
                    duration_seconds=time.time() - start,
                    errors=["RTL generator returned empty code"],
                    fallback_reason=fallback_reason or "Generated RTL was empty",
                    llm_attempted=llm_attempted,
                    generator=generator_used,
                )

            rtl_file = run_dir / f"{spec.module_name}.sv"
            rtl_file.write_text(rtl_code)

            rag_chars = None
            if isinstance(agent_output.output_data, dict):
                rag_chars = agent_output.output_data.get('rag_context_chars')

            return AgentResult(
                agent_name="A1_RTL_Generation",
                status=AgentStatus.SUCCESS,
                duration_seconds=time.time() - start,
                output_file=str(rtl_file),
                metrics={
                    "lines": len(rtl_code.split('\n')),
                    "confidence": agent_output.confidence,
                    "generation_method": agent_output.output_data.get('generation_method', generator_used),
                    "generator": generator_used,
                    "language": "systemverilog"
                },
                warnings=agent_output.warnings,
                generator=generator_used,
                fallback_reason=fallback_reason,
                llm_attempted=llm_attempted,
                rag_context_chars=rag_chars,
            )

        except Exception as e:
            return AgentResult(
                agent_name="A1_RTL_Generation",
                status=AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                errors=[str(e)]
            )

    def _run_a5(self, rtl_file: str, run_dir: Path) -> AgentResult:
        """Run A5 - Style Review"""
        start = time.time()

        try:
            agent = A5_StyleReviewCopilot()
            agent_output = agent.process({
                "file_path": str(rtl_file)
            })

            report_file = run_dir / "style_review.md"
            report_markdown = agent_output.output_data.get("report_markdown", "")
            report_file.write_text(report_markdown)

            summary = agent_output.output_data.get("summary", {})
            total_violations = summary.get("total", len(agent_output.output_data.get("violations", [])))
            critical = summary.get("critical", 0)

            return AgentResult(
                agent_name="A5_Style_Review",
                status=AgentStatus.SUCCESS if agent_output.success else AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                output_file=str(report_file),
                metrics={
                    "violations": total_violations,
                    "critical": critical,
                    "confidence": agent_output.confidence
                },
                errors=agent_output.errors,
                warnings=agent_output.warnings
            )

        except Exception as e:
            return AgentResult(
                agent_name="A5_Style_Review",
                status=AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                errors=[str(e)]
            )

    def _run_a4(self, rtl_file: str, run_dir: Path) -> AgentResult:
        """Run A4 - Lint & CDC"""
        start = time.time()

        try:
            import subprocess

            verilator_cmd = ['verilator', '--lint-only', '-Wall', str(rtl_file)]
            lint_log = ""

            try:
                verilator_proc = subprocess.run(
                    verilator_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                lint_log = verilator_proc.stdout + verilator_proc.stderr

                log_file = run_dir / "verilator.log"
                log_file.write_text(lint_log)
            except FileNotFoundError:
                lint_log = "Verilator not available"
            except Exception as lint_error:
                lint_log = f"Verilator execution failed: {lint_error}"

            if not lint_log.strip():
                report_file = run_dir / "lint_report.json"
                with report_file.open('w') as f:
                    json.dump({
                        "tool": "verilator",
                        "issues": [],
                        "summary": {"total_issues": 0}
                    }, f, indent=2)

                return AgentResult(
                    agent_name="A4_Lint_CDC",
                    status=AgentStatus.SUCCESS,
                    duration_seconds=time.time() - start,
                    output_file=str(report_file),
                    metrics={
                        "issues": 0,
                        "auto_fixable": 0,
                        "confidence": 1.0
                    }
                )

            agent = A4_LintCDCAssistant()
            agent_output = agent.process({
                "tool": "verilator",
                "log_content": lint_log,
                "source_file": str(rtl_file)
            })

            report_file = run_dir / "lint_report.json"
            with report_file.open('w') as f:
                json.dump(agent_output.output_data, f, indent=2)

            summary = agent_output.output_data.get("summary", {})

            success = agent_output.success or any(
                "No issues found" in err for err in agent_output.errors
            )

            return AgentResult(
                agent_name="A4_Lint_CDC",
                status=AgentStatus.SUCCESS if success else AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                output_file=str(report_file),
                metrics={
                    "issues": summary.get("total_issues", 0),
                    "auto_fixable": summary.get("auto_fixable", 0),
                    "confidence": agent_output.confidence
                },
                errors=[] if success else agent_output.errors,
                warnings=agent_output.warnings
            )

        except Exception as e:
            return AgentResult(
                agent_name="A4_Lint_CDC",
                status=AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                errors=[str(e)]
            )

    def _run_a3(self, spec: DesignSpec, rtl_file: str, run_dir: Path) -> AgentResult:
        """Run A3 - Constraint Synthesis"""
        start = time.time()

        try:
            agent = A3_ConstraintSynthesizer()

            parameters = spec.parameters or {}

            constraints_spec = {}
            if isinstance(parameters.get("constraints"), dict):
                constraints_spec = parameters.get("constraints", {})
            else:
                for key in (
                    "clock_period_ns",
                    "target_frequency_mhz",
                    "default_input_delay_ns",
                    "default_output_delay_ns",
                ):
                    if key in parameters:
                        constraints_spec[key] = parameters[key]

            if spec.clock_freq and "target_frequency_mhz" not in constraints_spec:
                constraints_spec["target_frequency_mhz"] = spec.clock_freq
            if spec.clock_freq and "clock_period_ns" not in constraints_spec:
                constraints_spec["clock_period_ns"] = 1000.0 / spec.clock_freq

            context = {}
            clock_domains = parameters.get("clock_domains")
            if clock_domains:
                context["clock_domains"] = clock_domains
            resets = parameters.get("resets")
            if resets:
                context["resets"] = resets

            agent_output = agent.process({
                "module_name": spec.module_name,
                "constraints": constraints_spec,
                "context": context
            })

            sdc_file = run_dir / f"{Path(rtl_file).stem}.sdc"
            sdc_content = agent_output.output_data.get("constraints", "")
            sdc_file.write_text(sdc_content)

            return AgentResult(
                agent_name="A3_Constraints",
                status=AgentStatus.SUCCESS if agent_output.success else AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                output_file=str(sdc_file),
                metrics={
                    "sdc_lines": len(sdc_content.splitlines()),
                    "confidence": agent_output.confidence
                },
                errors=agent_output.errors,
                warnings=agent_output.warnings
            )

        except Exception as e:
            return AgentResult(
                agent_name="A3_Constraints",
                status=AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                errors=[str(e)]
            )

    def _run_a6(self, spec: DesignSpec, rtl_file: str, sdc_file: Optional[str], run_dir: Path) -> AgentResult:
        """Run A6 - Synthesis Script Generation"""
        start = time.time()

        try:
            agent = A6_EDACommandCopilot()

            parameters = spec.parameters or {}
            output_netlist = run_dir / f"{Path(rtl_file).stem}_synth.v"

            agent_output = agent.process({
                "tool": "yosys",
                "command_type": "synthesis",
                "input_files": [str(rtl_file)],
                "output_files": [str(output_netlist)],
                "parameters": {
                    "top_module": spec.module_name,
                    "optimization_goal": parameters.get("optimization_goal", "balanced"),
                    "tech_library": parameters.get("tech_library"),
                    "sdc_file": sdc_file
                }
            })

            script_file = run_dir / "synth_script.ys"
            script_content = agent_output.output_data.get("script_content", "")
            script_file.write_text(script_content)

            return AgentResult(
                agent_name="A6_Synthesis_Script",
                status=AgentStatus.SUCCESS if agent_output.success else AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                output_file=str(script_file),
                metrics={
                    "script_lines": len(script_content.splitlines()),
                    "confidence": agent_output.confidence
                },
                errors=agent_output.errors,
                warnings=agent_output.warnings
            )

        except Exception as e:
            return AgentResult(
                agent_name="A6_Synthesis_Script",
                status=AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                errors=[str(e)]
            )

    def _run_yosys(self, script_file: str, run_dir: Path) -> AgentResult:
        """Run Yosys Synthesis"""
        import subprocess
        start = time.time()

        try:
            # Run Yosys directly
            result = subprocess.run(
                ['yosys', '-s', str(script_file)],
                capture_output=True,
                text=True,
                timeout=60
            )

            report_file = run_dir / "synthesis_report.txt"
            report = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            report_file.write_text(report)

            return AgentResult(
                agent_name="Yosys_Synthesis",
                status=AgentStatus.SUCCESS if result.returncode == 0 else AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                output_file=str(report_file),
                metrics={
                    "exit_code": result.returncode,
                    "stdout_lines": len(result.stdout.split('\n')),
                    "stderr_lines": len(result.stderr.split('\n'))
                }
            )

        except Exception as e:
            return AgentResult(
                agent_name="Yosys_Synthesis",
                status=AgentStatus.FAILED,
                duration_seconds=time.time() - start,
                errors=[str(e)]
            )

    def _compute_metrics(self, rtl_file: str) -> CodeMetrics:
        """Compute code quality metrics"""
        code = Path(rtl_file).read_text()
        lines = code.split('\n')

        lines_code = len([l for l in lines if l.strip() and not l.strip().startswith('//')])
        lines_comment = len([l for l in lines if l.strip().startswith('//')])
        lines_blank = len([l for l in lines if not l.strip()])

        return CodeMetrics(
            lines_total=len(lines),
            lines_code=lines_code,
            lines_comment=lines_comment,
            lines_blank=lines_blank,
            modules_count=code.count('module '),
            ports_count=code.count('input ') + code.count('output '),
            parameters_count=code.count('parameter '),
            has_fsm='case' in code and 'state' in code.lower(),
            has_fifo='fifo' in code.lower(),
            complexity_score=min(10.0, lines_code / 50.0)
        )

    def _update_progress(self, run_id: str, stage: str, completed: int, total: int):
        """Update progress callback"""
        if self.progress_callback:
            progress = PipelineProgress(
                run_id=run_id,
                current_stage=stage,
                total_stages=total,
                completed_stages=completed,
                percent_complete=100.0 * completed / total,
                start_time=datetime.now()
            )
            self.progress_callback(progress)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------
    def generate_testbench_for_run(
        self,
        run_id: str,
        spec_override: Optional[DesignSpec] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> VerificationTestbench:
        options = options or {}
        run_dir = self.runs_dir / run_id
        pipeline_result = self._load_pipeline_result(run_dir)
        spec = spec_override or pipeline_result.spec

        if not pipeline_result.rtl_file:
            raise ValueError(f"Run {run_id} has no RTL artifact to generate a testbench for")

        generator = A7_TestbenchGenerator(self.project_root)
        input_data = {
            "module_name": spec.module_name,
            "rtl_path": pipeline_result.rtl_file,
            "spec": spec.model_dump(exclude_none=True),
            "clock": (spec.parameters or {}).get("clock", "clk"),
            "reset": (spec.parameters or {}).get("reset", "rst_n"),
            "ports": options.get("ports"),
        }

        agent_output = generator.process(input_data)
        if not agent_output.success:
            raise RuntimeError("Testbench generation failed", agent_output.errors)

        tb_path = Path(agent_output.output_data.get("testbench_path"))
        prompt_path = agent_output.output_data.get("prompt_path")
        context_path = agent_output.output_data.get("context_path")

        tb_id = options.get("tb_id") or f"tb_{uuid.uuid4().hex[:8]}"

        validator = TestbenchValidator()
        validation = validator.validate(tb_path, spec.module_name)

        tb_entry = VerificationTestbench(
            tb_id=tb_id,
            path=str(tb_path),
            generator=agent_output.metadata.get("generator", "template"),
            generated_at=datetime.utcnow(),
            validation={
                "errors": validation.errors,
                "warnings": validation.warnings,
            },
            metadata={
                "prompt_path": prompt_path,
                "context_path": context_path,
            },
        )

        pipeline_result.verification.testbenches.append(tb_entry)
        self._save_pipeline_result(run_dir, pipeline_result)

        return tb_entry

    def _load_pipeline_result(self, run_dir: Path) -> PipelineResult:
        result_file = run_dir / "result.json"
        if not result_file.exists():
            raise FileNotFoundError(f"Pipeline result not found in {run_dir}")
        with result_file.open("r") as f:
            data = json.load(f)
        return PipelineResult(**data)

    def _save_pipeline_result(self, run_dir: Path, pipeline_result: PipelineResult) -> None:
        result_file = run_dir / "result.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with result_file.open("w") as f:
            json.dump(pipeline_result.model_dump(exclude_none=True), f, indent=2, default=str)
