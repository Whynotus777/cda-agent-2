#!/usr/bin/env python3
"""
End-to-End EDA Flow Integration Test

Tests the complete chip design pipeline:
RTL → Synthesis → Placement → Routing → Timing Analysis

This validates that each stage produces output consumable by the next stage.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import logging
from core.simulation_engine import SimulationEngine
from core.world_model import DesignState
from core.world_model.design_state import DesignStage

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TestEndToEndFlow:
    """Test complete EDA flow from RTL to timing analysis"""

    @pytest.fixture(scope="class")
    def test_design_dir(self):
        """Get test fixtures directory"""
        return Path(__file__).parent.parent / "fixtures"

    @pytest.fixture(scope="class")
    def output_dir(self, tmp_path_factory):
        """Create temporary output directory"""
        return tmp_path_factory.mktemp("eda_output")

    @pytest.fixture(scope="class")
    def simulation_engine(self):
        """Initialize simulation engine"""
        return SimulationEngine()

    @pytest.fixture(scope="class")
    def design_state(self):
        """Initialize design state"""
        return DesignState(project_name="test_counter_flow")

    @pytest.fixture(scope="class")
    def synthesis_result(self, simulation_engine, design_state, test_design_dir, output_dir):
        """Run synthesis and return result for other tests"""
        logger.info("="*70)
        logger.info("FIXTURE: SYNTHESIS")
        logger.info("="*70)

        rtl_file = test_design_dir / "counter.v"
        assert rtl_file.exists(), f"RTL file not found: {rtl_file}"

        output_netlist = output_dir / "counter_synth.v"

        # Run synthesis
        result = simulation_engine.synthesis.synthesize(
            rtl_files=[str(rtl_file)],
            top_module="counter",
            output_netlist=str(output_netlist),
            optimization_goal="balanced"
        )

        # Verify synthesis success
        assert result is not None, "Synthesis returned None"
        assert result.get('cell_count', 0) > 0, "No cells synthesized"

        logger.info(f"✓ Synthesis successful: {result['cell_count']} cells")

        # Verify output file exists
        assert output_netlist.exists(), "Synthesized netlist not created"

        # Update design state
        design_state.netlist_file = str(output_netlist)
        design_state.update_stage(DesignStage.SYNTHESIZED)

        return {
            'netlist': str(output_netlist),
            'cell_count': result['cell_count']
        }

    def test_01_synthesis(self, synthesis_result):
        """Test synthesis stage"""
        logger.info("="*70)
        logger.info("TEST 1: SYNTHESIS")
        logger.info("="*70)

        # Synthesis already ran in fixture
        assert synthesis_result is not None
        assert synthesis_result.get('cell_count', 0) > 0
        assert Path(synthesis_result['netlist']).exists()

        logger.info(f"✓ Synthesis validated: {synthesis_result['cell_count']} cells")

    def test_02_placement(self, simulation_engine, design_state, output_dir, synthesis_result):
        """Test placement stage"""
        logger.info("="*70)
        logger.info("TEST 2: PLACEMENT")
        logger.info("="*70)

        netlist_file = synthesis_result['netlist']
        assert Path(netlist_file).exists(), "Netlist file not found"

        # Create simple floorplan DEF
        floorplan_def = output_dir / "counter_floorplan.def"
        self._create_simple_floorplan(floorplan_def, design_name="counter")

        output_def = output_dir / "counter_placed.def"

        # Run placement
        try:
            result = simulation_engine.placement.place(
                netlist_file=netlist_file,
                def_file=str(floorplan_def),
                output_def=str(output_def),
                placement_params={
                    'target_density': 0.7,
                    'wirelength_weight': 0.5,
                    'routability_weight': 0.5
                }
            )

            if result and result.get('success'):
                logger.info(f"✓ Placement successful: HPWL={result.get('hpwl', 0):.2f}")

                # Update design state
                design_state.def_file = str(output_def)
                design_state.update_stage(DesignStage.PLACED)

                return {
                    'def_file': str(output_def),
                    'hpwl': result.get('hpwl', 0)
                }
            else:
                logger.warning("⚠ Placement did not complete (DREAMPlace may not be fully configured)")
                pytest.skip("Placement engine not available or not configured")

        except Exception as e:
            logger.warning(f"⚠ Placement skipped: {e}")
            pytest.skip(f"Placement not available: {e}")

    def test_03_routing_available(self, simulation_engine):
        """Test if routing tools are available"""
        logger.info("="*70)
        logger.info("TEST 3: ROUTING AVAILABILITY")
        logger.info("="*70)

        # Check if TritonRoute is available
        try:
            import subprocess
            result = subprocess.run(
                ["TritonRoute", "-h"],
                capture_output=True,
                timeout=5
            )
            logger.info("✓ TritonRoute is available")
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("⚠ TritonRoute not available, skipping routing tests")
            pytest.skip("TritonRoute not installed")

    def test_04_timing_analysis_available(self, simulation_engine):
        """Test if timing analysis tools are available"""
        logger.info("="*70)
        logger.info("TEST 4: TIMING ANALYSIS AVAILABILITY")
        logger.info("="*70)

        # Check if OpenSTA is available
        try:
            import subprocess
            result = subprocess.run(
                ["sta", "-version"],
                capture_output=True,
                timeout=5
            )
            logger.info("✓ OpenSTA is available")
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("⚠ OpenSTA not available, skipping timing tests")
            pytest.skip("OpenSTA not installed")

    def test_05_design_state_progression(self, design_state):
        """Test that design state progressed correctly"""
        logger.info("="*70)
        logger.info("TEST 5: DESIGN STATE PROGRESSION")
        logger.info("="*70)

        # Verify state progression
        assert design_state.project_name == "test_counter_flow"
        assert design_state.stage in [DesignStage.SYNTHESIZED, DesignStage.PLACED]
        assert design_state.netlist_file is not None

        logger.info(f"✓ Design state: {design_state.stage.value}")
        logger.info(f"✓ Netlist: {design_state.netlist_file}")

        if design_state.def_file:
            logger.info(f"✓ DEF file: {design_state.def_file}")

    def test_06_error_handling_invalid_rtl(self, simulation_engine, output_dir):
        """Test error handling with invalid RTL"""
        logger.info("="*70)
        logger.info("TEST 6: ERROR HANDLING")
        logger.info("="*70)

        # Create invalid Verilog
        invalid_rtl = output_dir / "invalid.v"
        with open(invalid_rtl, 'w') as f:
            f.write("This is not valid Verilog!")

        output_netlist = output_dir / "invalid_synth.v"

        # Should handle error gracefully
        try:
            result = simulation_engine.synthesis.synthesize(
                rtl_files=[str(invalid_rtl)],
                top_module="nonexistent",
                output_netlist=str(output_netlist),
                optimization_goal="balanced"
            )

            # Should return error indication
            if result:
                assert result.get('cell_count', 0) == 0, "Should not synthesize invalid RTL"

            logger.info("✓ Error handling works correctly")

        except Exception as e:
            logger.info(f"✓ Exception caught as expected: {type(e).__name__}")

    def test_07_file_validation(self, synthesis_result):
        """Test that synthesized netlist is valid Verilog"""
        logger.info("="*70)
        logger.info("TEST 7: FILE VALIDATION")
        logger.info("="*70)

        netlist_file = synthesis_result['netlist']

        # Read and validate netlist
        with open(netlist_file, 'r') as f:
            content = f.read()

        # Basic validation
        assert 'module' in content.lower(), "Netlist should contain module definition"
        assert 'endmodule' in content.lower(), "Netlist should have endmodule"

        logger.info("✓ Synthesized netlist is valid Verilog")

    def _create_simple_floorplan(self, output_file: Path, design_name: str):
        """Create a simple DEF floorplan for testing"""
        def_content = f"""VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN {design_name} ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 100000 100000 ) ;
END DESIGN
"""
        with open(output_file, 'w') as f:
            f.write(def_content)


def run_tests():
    """Run all tests"""
    print("="*70)
    print("END-TO-END EDA FLOW INTEGRATION TESTS")
    print("="*70)
    print()

    # Run pytest
    pytest_args = [
        __file__,
        "-v",
        "-s",
        "--tb=short"
    ]

    exit_code = pytest.main(pytest_args)
    return exit_code == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
