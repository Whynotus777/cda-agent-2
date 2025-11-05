"""A7 Testbench Generator agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.rtl_agents.base_agent import AgentOutput, BaseAgent

try:
    from core.rag import RAGRetriever
except Exception:  # pragma: no cover - rag optional at runtime
    RAGRetriever = None  # type: ignore


@dataclass
class TestbenchGenerationResult:
    """Structured result for generated testbenches."""

    success: bool
    code: str
    prompt: str
    context: str
    metadata: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


class A7_TestbenchGenerator(BaseAgent):
    """Generates SystemVerilog testbenches for generated RTL."""

    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id="A7", agent_name="Testbench Generator", config=config)
        self.project_root = Path(project_root)
        self.rag = None
        if RAGRetriever is not None:
            try:
                self.rag = RAGRetriever()
            except Exception as exc:  # pragma: no cover - retriever optional
                self.logger.warning("Failed to initialise RAG retriever: %s", exc)

    # ------------------------------------------------------------------
    # BaseAgent API
    # ------------------------------------------------------------------
    def get_schema(self) -> Dict[str, Any]:  # pragma: no cover - schema metadata
        return {
            "type": "object",
            "required": ["module_name", "rtl_path"],
            "properties": {
                "module_name": {"type": "string"},
                "rtl_path": {"type": "string"},
                "clock": {"type": "string"},
                "reset": {"type": "string"},
                "ports": {"type": "array", "items": {"type": "object"}},
                "spec": {"type": "object"},
            },
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return "module_name" in input_data and "rtl_path" in input_data

    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        if not self.validate_input(input_data):
            return self.create_output(
                success=False,
                output_data={},
                errors=["module_name and rtl_path are required"],
            )

        module_name: str = input_data["module_name"]
        rtl_path = Path(input_data["rtl_path"])
        clock_name = input_data.get("clock", "clk")
        reset_name = input_data.get("reset", "rst_n")
        ports = input_data.get("ports") or self._derive_ports_from_rtl(rtl_path)

        rag_context = self._retrieve_context(module_name, input_data.get("spec"))

        prompt = self._build_prompt(
            module_name=module_name,
            clock=clock_name,
            reset=reset_name,
            ports=ports,
            rag_context=rag_context,
        )

        # Placeholder deterministic generator until LLM integration complete.
        tb_code = self._generate_template_testbench(
            module_name=module_name,
            clock=clock_name,
            reset=reset_name,
            ports=ports,
        )

        output_dir = rtl_path.parent / "verification" / "testbenches"
        output_dir.mkdir(parents=True, exist_ok=True)
        tb_path = output_dir / f"{module_name}_tb.sv"
        prompt_path = output_dir / f"{module_name}_prompt.txt"
        context_path = output_dir / f"{module_name}_context.txt"

        tb_path.write_text(tb_code)
        prompt_path.write_text(prompt)
        context_path.write_text(rag_context)

        doc_hits = rag_context.count("## Source")

        metadata = {
            "module_name": module_name,
            "clock": clock_name,
            "reset": reset_name,
            "ports": ports,
            "prompt_file": str(prompt_path),
            "context_file": str(context_path),
            "generator": "template",
            "rag_docs": doc_hits,
        }

        result = TestbenchGenerationResult(
            success=True,
            code=tb_code,
            prompt=prompt,
            context=rag_context,
            metadata=metadata,
            errors=[],
            warnings=[],
        )

        return self.create_output(
            success=True,
            output_data={
                "testbench_path": str(tb_path),
                "prompt_path": str(prompt_path),
                "context_path": str(context_path),
                "result": json.loads(json.dumps(result.__dict__)),
            },
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _retrieve_context(self, module_name: str, spec: Optional[Dict[str, Any]]) -> str:
        if self.rag is None or spec is None:
            return ""
        query_parts = [module_name]
        if "description" in spec:
            query_parts.append(spec["description"])
        try:
            context = self.rag.retrieve_and_format(" | ".join(query_parts), top_k=4, max_context_length=2000)
            return context.strip()
        except Exception as exc:  # pragma: no cover - retrieval optional
            self.logger.warning("RAG retrieval failed: %s", exc)
            return ""

    def _build_prompt(self, module_name: str, clock: str, reset: str, ports: List[Dict[str, Any]], rag_context: str) -> str:
        port_lines = [f"- {p.get('direction', 'input')} {p.get('name', 'signal')}" for p in ports]
        port_list = "\n".join(port_lines) if port_lines else "- (port list unavailable)"
        context_block = f"Context from design docs:\n{rag_context}\n\n" if rag_context else ""
        return (
            f"{context_block}Generate a SystemVerilog testbench for module '{module_name}'.\n"
            f"Clock signal: {clock}, active edge positive.\n"
            f"Reset signal: {reset}, active low.\n"
            "Stimulus requirements:\n"
            "- Apply reset for 5 cycles.\n"
            "- Drive stimulus on inputs.\n"
            "- Provide a simple self-check.\n\n"
            f"Module ports:\n{port_list}\n\n"
            "Produce a synthesizable testbench with clock/reset generation and TODOs for assertions."
        )

    def _generate_template_testbench(self, module_name: str, clock: str, reset: str, ports: List[Dict[str, Any]]) -> str:
        io_decl = []
        connections = []
        for port in ports:
            direction = port.get("direction", "input")
            name = port.get("name", "signal")
            width = port.get("width", "")
            width_decl = width if isinstance(width, str) else port.get("range", "")
            if direction == "input":
                io_decl.append(f"    logic {width_decl} {name};")
            elif direction == "output":
                io_decl.append(f"    wire {width_decl} {name};")
            else:
                io_decl.append(f"    logic {width_decl} {name};")
            connections.append(f"        .{name}({name})")

        io_decl_str = "\n".join(io_decl) if io_decl else "    // TODO: declare DUT signals"
        conn_str = ",\n".join(connections) if connections else "        // TODO: connect ports"

        return f"""// Auto-generated testbench skeleton for {module_name}
`timescale 1ns/1ps

module {module_name}_tb;

    // Clock and reset
    logic {clock};
    logic {reset};

{io_decl_str}

    // DUT instance
    {module_name} dut (
{conn_str}
    );

    // Clock generation
    initial {clock} = 0;
    always #5 {clock} = ~{clock};

    // Reset task
    task automatic apply_reset();
        begin
            {reset} = 0;
            repeat (5) @(posedge {clock});
            {reset} = 1;
            repeat (2) @(posedge {clock});
        end
    endtask

    // Stimulus task
    task automatic drive_stimulus();
        begin
            // TODO: implement stimulus
            repeat (10) @(posedge {clock});
        end
    endtask

    initial begin
        apply_reset();
        drive_stimulus();
        // TODO: add self-checks/assertions
        $display("Testbench completed.");
        #50;
        $finish;
    end

endmodule
"""

    def _derive_ports_from_rtl(self, rtl_path: Path) -> List[Dict[str, Any]]:
        if not rtl_path.exists():
            return []
        ports: List[Dict[str, Any]] = []
        try:
            text = rtl_path.read_text()
            import re

            direction_regex = re.compile(r"\b(input|output|inout)\s+(?:wire|reg|logic)?\s*(\[[^]]+\])?\s*(\w+)", re.IGNORECASE)
            for match in direction_regex.finditer(text):
                direction, width, name = match.groups()
                ports.append(
                    {
                        "name": name,
                        "direction": direction.lower(),
                        "width": width.strip() if width else "",
                    }
                )
        except Exception as exc:  # pragma: no cover - best effort parsing
            self.logger.warning("Failed to derive ports from RTL: %s", exc)
        return ports
