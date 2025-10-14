"""
Rule Engine Module

Contains Design Rule Checking (DRC) rules and physical/electrical constraints
for a given process node.

Examples:
- Minimum spacing between metal wires
- Minimum width of metal traces
- Via rules
- Antenna rules
- Electromigration rules
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of design rules"""
    SPACING = "spacing"  # Minimum spacing between features
    WIDTH = "width"  # Minimum width of features
    ENCLOSURE = "enclosure"  # Enclosure requirements
    AREA = "area"  # Minimum area requirements
    DENSITY = "density"  # Layer density requirements
    ANTENNA = "antenna"  # Antenna effect rules
    ELECTROMIGRATION = "electromigration"  # Current density limits


class Layer(Enum):
    """Metal and via layers"""
    POLY = "poly"
    CONTACT = "contact"
    METAL1 = "metal1"
    VIA1 = "via1"
    METAL2 = "metal2"
    VIA2 = "via2"
    METAL3 = "metal3"
    VIA3 = "via3"
    METAL4 = "metal4"
    # ... additional layers for advanced nodes


@dataclass
class DesignRule:
    """A single design rule"""
    rule_id: str
    rule_type: RuleType
    layer: Layer
    value: float  # Rule value (nm, um, etc.)
    description: str
    severity: str = "error"  # 'error' or 'warning'


@dataclass
class DRCViolation:
    """A design rule violation"""
    rule_id: str
    location: Tuple[float, float]  # (x, y) coordinates
    description: str
    severity: str


class RuleEngine:
    """
    Manages and checks design rules for a specific process node.

    Rules vary by process technology (7nm vs 12nm vs 28nm, etc.)
    """

    def __init__(self, process_node: str):
        """
        Initialize rule engine for a process node.

        Args:
            process_node: Process technology (e.g., '7nm', '12nm')
        """
        self.process_node = process_node
        self.rules: Dict[str, DesignRule] = {}

        # Load rules for this process
        self._load_process_rules()

        logger.info(f"Initialized RuleEngine for {process_node} with {len(self.rules)} rules")

    def _load_process_rules(self):
        """
        Load design rules for the specified process node.

        In practice, these would be loaded from technology files provided
        by the foundry. Here we define representative rules.
        """
        # Parse node size (e.g., '7nm' -> 7)
        node_size = int(self.process_node.replace('nm', ''))

        # Scale rules based on node size
        # Smaller nodes have tighter rules
        scale_factor = node_size / 7.0  # Normalize to 7nm

        # Metal 1 rules
        self.add_rule(DesignRule(
            rule_id="M1.W.1",
            rule_type=RuleType.WIDTH,
            layer=Layer.METAL1,
            value=int(30 * scale_factor),  # nm
            description=f"Minimum Metal1 width: {int(30 * scale_factor)}nm"
        ))

        self.add_rule(DesignRule(
            rule_id="M1.S.1",
            rule_type=RuleType.SPACING,
            layer=Layer.METAL1,
            value=int(30 * scale_factor),  # nm
            description=f"Minimum Metal1 spacing: {int(30 * scale_factor)}nm"
        ))

        # Metal 2 rules (typically more relaxed)
        self.add_rule(DesignRule(
            rule_id="M2.W.1",
            rule_type=RuleType.WIDTH,
            layer=Layer.METAL2,
            value=int(40 * scale_factor),
            description=f"Minimum Metal2 width: {int(40 * scale_factor)}nm"
        ))

        self.add_rule(DesignRule(
            rule_id="M2.S.1",
            rule_type=RuleType.SPACING,
            layer=Layer.METAL2,
            value=int(40 * scale_factor),
            description=f"Minimum Metal2 spacing: {int(40 * scale_factor)}nm"
        ))

        # Metal 3 rules
        self.add_rule(DesignRule(
            rule_id="M3.W.1",
            rule_type=RuleType.WIDTH,
            layer=Layer.METAL3,
            value=int(50 * scale_factor),
            description=f"Minimum Metal3 width: {int(50 * scale_factor)}nm"
        ))

        self.add_rule(DesignRule(
            rule_id="M3.S.1",
            rule_type=RuleType.SPACING,
            layer=Layer.METAL3,
            value=int(50 * scale_factor),
            description=f"Minimum Metal3 spacing: {int(50 * scale_factor)}nm"
        ))

        # Via rules
        self.add_rule(DesignRule(
            rule_id="V1.S.1",
            rule_type=RuleType.SPACING,
            layer=Layer.VIA1,
            value=int(40 * scale_factor),
            description=f"Minimum Via1 spacing: {int(40 * scale_factor)}nm"
        ))

        # Density rules (for CMP - Chemical Mechanical Polishing)
        self.add_rule(DesignRule(
            rule_id="M1.DEN.1",
            rule_type=RuleType.DENSITY,
            layer=Layer.METAL1,
            value=0.35,  # 35% minimum density
            description="Minimum Metal1 density: 35%",
            severity="warning"
        ))

        self.add_rule(DesignRule(
            rule_id="M1.DEN.2",
            rule_type=RuleType.DENSITY,
            layer=Layer.METAL1,
            value=0.75,  # 75% maximum density
            description="Maximum Metal1 density: 75%",
            severity="warning"
        ))

        # Antenna rules (charge accumulation during manufacturing)
        self.add_rule(DesignRule(
            rule_id="ANT.1",
            rule_type=RuleType.ANTENNA,
            layer=Layer.METAL1,
            value=400.0,  # Maximum antenna ratio
            description="Maximum antenna ratio: 400:1"
        ))

    def add_rule(self, rule: DesignRule):
        """Add a design rule"""
        self.rules[rule.rule_id] = rule

    def get_rule(self, rule_id: str) -> Optional[DesignRule]:
        """Get a rule by ID"""
        return self.rules.get(rule_id)

    def get_rules_by_type(self, rule_type: RuleType) -> List[DesignRule]:
        """Get all rules of a specific type"""
        return [rule for rule in self.rules.values() if rule.rule_type == rule_type]

    def get_rules_by_layer(self, layer: Layer) -> List[DesignRule]:
        """Get all rules for a specific layer"""
        return [rule for rule in self.rules.values() if rule.layer == layer]

    def check_spacing(
        self,
        layer: Layer,
        feature1_pos: Tuple[float, float],
        feature2_pos: Tuple[float, float],
        feature1_width: float,
        feature2_width: float
    ) -> Optional[DRCViolation]:
        """
        Check spacing rule between two features.

        Args:
            layer: Metal layer
            feature1_pos: (x, y) position of first feature
            feature2_pos: (x, y) position of second feature
            feature1_width: Width of first feature
            feature2_width: Width of second feature

        Returns:
            DRCViolation if rule violated, None otherwise
        """
        # Find spacing rule for this layer
        spacing_rules = [
            rule for rule in self.rules.values()
            if rule.rule_type == RuleType.SPACING and rule.layer == layer
        ]

        if not spacing_rules:
            return None

        min_spacing = spacing_rules[0].value

        # Calculate actual spacing
        x1, y1 = feature1_pos
        x2, y2 = feature2_pos

        edge_to_edge_dist = (
            ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            - feature1_width / 2
            - feature2_width / 2
        )

        # Check violation
        if edge_to_edge_dist < min_spacing:
            return DRCViolation(
                rule_id=spacing_rules[0].rule_id,
                location=((x1 + x2) / 2, (y1 + y2) / 2),
                description=(
                    f"{layer.value} spacing violation: "
                    f"{edge_to_edge_dist:.1f}nm < {min_spacing}nm minimum"
                ),
                severity=spacing_rules[0].severity
            )

        return None

    def check_width(
        self,
        layer: Layer,
        feature_width: float,
        location: Tuple[float, float]
    ) -> Optional[DRCViolation]:
        """
        Check minimum width rule.

        Args:
            layer: Metal layer
            feature_width: Width of the feature
            location: (x, y) position

        Returns:
            DRCViolation if rule violated, None otherwise
        """
        # Find width rule for this layer
        width_rules = [
            rule for rule in self.rules.values()
            if rule.rule_type == RuleType.WIDTH and rule.layer == layer
        ]

        if not width_rules:
            return None

        min_width = width_rules[0].value

        if feature_width < min_width:
            return DRCViolation(
                rule_id=width_rules[0].rule_id,
                location=location,
                description=(
                    f"{layer.value} width violation: "
                    f"{feature_width:.1f}nm < {min_width}nm minimum"
                ),
                severity=width_rules[0].severity
            )

        return None

    def check_density(
        self,
        layer: Layer,
        density: float,
        location: Tuple[float, float]
    ) -> Optional[DRCViolation]:
        """
        Check metal density rules.

        Args:
            layer: Metal layer
            density: Computed density (0.0 to 1.0)
            location: (x, y) position of density window

        Returns:
            DRCViolation if rule violated, None otherwise
        """
        density_rules = self.get_rules_by_layer(layer)
        density_rules = [r for r in density_rules if r.rule_type == RuleType.DENSITY]

        for rule in density_rules:
            # Check minimum density
            if 'Minimum' in rule.description and density < rule.value:
                return DRCViolation(
                    rule_id=rule.rule_id,
                    location=location,
                    description=(
                        f"{layer.value} density too low: "
                        f"{density:.2%} < {rule.value:.2%} minimum"
                    ),
                    severity=rule.severity
                )

            # Check maximum density
            elif 'Maximum' in rule.description and density > rule.value:
                return DRCViolation(
                    rule_id=rule.rule_id,
                    location=location,
                    description=(
                        f"{layer.value} density too high: "
                        f"{density:.2%} > {rule.value:.2%} maximum"
                    ),
                    severity=rule.severity
                )

        return None

    def get_min_spacing(self, layer: Layer) -> float:
        """Get minimum spacing for a layer"""
        spacing_rules = [
            rule for rule in self.rules.values()
            if rule.rule_type == RuleType.SPACING and rule.layer == layer
        ]
        return spacing_rules[0].value if spacing_rules else 0.0

    def get_min_width(self, layer: Layer) -> float:
        """Get minimum width for a layer"""
        width_rules = [
            rule for rule in self.rules.values()
            if rule.rule_type == RuleType.WIDTH and rule.layer == layer
        ]
        return width_rules[0].value if width_rules else 0.0

    def validate_design_rules(self, design_data: Dict) -> List[DRCViolation]:
        """
        Run comprehensive DRC on design data.

        Args:
            design_data: Physical design data with wire locations, widths, etc.

        Returns:
            List of all DRC violations found

        TODO: Implement full DRC checking
        """
        violations = []

        # Placeholder for full DRC implementation
        # Would check all features against all applicable rules

        logger.info(f"DRC complete: {len(violations)} violations found")
        return violations

    def get_rule_summary(self) -> Dict:
        """Get summary of all rules"""
        return {
            'process_node': self.process_node,
            'total_rules': len(self.rules),
            'spacing_rules': len(self.get_rules_by_type(RuleType.SPACING)),
            'width_rules': len(self.get_rules_by_type(RuleType.WIDTH)),
            'density_rules': len(self.get_rules_by_type(RuleType.DENSITY)),
            'antenna_rules': len(self.get_rules_by_type(RuleType.ANTENNA)),
        }
