# config/validator.py
"""
Topology configuration validator.

Validates YAML configuration for:
1. Syntax errors
2. Schema compliance
3. Topology-specific constraints
4. Resource limits (MAX_SWITCHES=20)
"""

import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml
from pydantic import ValidationError

from .schema import TopologyConfig, TopologyType, MAX_SWITCHES
from .loader import load_config, ConfigLoadError


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    config: Optional[TopologyConfig] = None

    def __str__(self) -> str:
        lines = []
        if self.is_valid:
            lines.append("✓ Configuration is valid")
            if self.config:
                lines.append(f"  Topology: {self.config.topology.name} ({self.config.topology.type.value})")
                lines.append(f"  Switches: {self.config.total_switches}")
                lines.append(f"  Hosts: {self.config.total_hosts}")
        else:
            lines.append("✗ Configuration is invalid")

        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  ✗ {err}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")

        return "\n".join(lines)


class TopologyValidator:
    """Validates topology configuration."""

    def __init__(self, max_switches: int = MAX_SWITCHES):
        self.max_switches = max_switches

    def validate_file(self, config_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ValidationResult with errors, warnings, and parsed config if valid
        """
        errors = []
        warnings = []
        config = None

        config_path = Path(config_path)

        # 1. Check file exists
        if not config_path.exists():
            errors.append(f"File not found: {config_path}")
            return ValidationResult(is_valid=False, errors=errors)

        # 2. Check file is readable and valid YAML
        try:
            with open(config_path, 'r') as f:
                raw_yaml = yaml.safe_load(f)
        except yaml.YAMLError as e:
            errors.append(f"YAML syntax error: {e}")
            return ValidationResult(is_valid=False, errors=errors)
        except IOError as e:
            errors.append(f"Cannot read file: {e}")
            return ValidationResult(is_valid=False, errors=errors)

        if raw_yaml is None:
            errors.append("Configuration file is empty")
            return ValidationResult(is_valid=False, errors=errors)

        # 3. Validate with Pydantic schema
        try:
            config = TopologyConfig(**raw_yaml)
        except ValidationError as e:
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error['loc'])
                msg = error['msg']
                errors.append(f"{loc}: {msg}")
            return ValidationResult(is_valid=False, errors=errors)

        # 4. Validate topology-specific constraints
        constraint_errors, constraint_warnings = self._validate_topology_constraints(config)
        errors.extend(constraint_errors)
        warnings.extend(constraint_warnings)

        # 5. Validate resource limits
        limit_errors, limit_warnings = self._validate_resource_limits(config)
        errors.extend(limit_errors)
        warnings.extend(limit_warnings)

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            config=config if is_valid else None
        )

    def validate_dict(self, raw_config: dict) -> ValidationResult:
        """
        Validate a configuration dictionary.

        Args:
            raw_config: Configuration dictionary

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        config = None

        # Validate with Pydantic schema
        try:
            config = TopologyConfig(**raw_config)
        except ValidationError as e:
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error['loc'])
                msg = error['msg']
                errors.append(f"{loc}: {msg}")
            return ValidationResult(is_valid=False, errors=errors)

        # Validate topology-specific constraints
        constraint_errors, constraint_warnings = self._validate_topology_constraints(config)
        errors.extend(constraint_errors)
        warnings.extend(constraint_warnings)

        # Validate resource limits
        limit_errors, limit_warnings = self._validate_resource_limits(config)
        errors.extend(limit_errors)
        warnings.extend(limit_warnings)

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            config=config if is_valid else None
        )

    def _validate_topology_constraints(self, config: TopologyConfig) -> tuple[List[str], List[str]]:
        """Validate topology-specific constraints."""
        errors = []
        warnings = []
        topo_type = config.topology.type

        if topo_type == TopologyType.FAT_TREE:
            ft = config.fat_tree
            # k must be even (already validated in schema, but double-check)
            if ft.k % 2 != 0:
                errors.append(f"Fat-tree k={ft.k} must be even")

            # k must be >= 2
            if ft.k < 2:
                errors.append(f"Fat-tree k={ft.k} must be at least 2")

            # Check if topology is too large for RL
            if ft.total_switches > self.max_switches:
                errors.append(
                    f"Fat-tree k={ft.k} creates {ft.total_switches} switches, "
                    f"exceeds MAX_SWITCHES={self.max_switches}"
                )

            # Warning for small topologies
            if ft.k == 2:
                warnings.append("Fat-tree k=2 is minimal (2 pods, 4 switches)")

        elif topo_type == TopologyType.LEAF_SPINE:
            ls = config.leaf_spine
            # Check switch count
            if ls.total_switches > self.max_switches:
                errors.append(
                    f"Leaf-spine has {ls.total_switches} switches, "
                    f"exceeds MAX_SWITCHES={self.max_switches}"
                )

            # Warning for unbalanced topology
            if ls.num_spines == 1:
                warnings.append("Single spine creates a single point of failure")

            if ls.num_leaves > ls.num_spines * 4:
                warnings.append(
                    f"High oversubscription: {ls.num_leaves} leaves with {ls.num_spines} spines"
                )

        elif topo_type == TopologyType.THREE_TIER:
            tt = config.three_tier
            # Check switch count
            if tt.total_switches > self.max_switches:
                errors.append(
                    f"Three-tier has {tt.total_switches} switches, "
                    f"exceeds MAX_SWITCHES={self.max_switches}"
                )

            # Validate pod structure if provided
            if tt.pods:
                all_access = set()
                all_dist = set()
                for pod in tt.pods:
                    for a in pod.access:
                        if a < 1 or a > tt.access_switches:
                            errors.append(
                                f"Pod references access switch {a}, but only {tt.access_switches} exist"
                            )
                        if a in all_access:
                            warnings.append(f"Access switch {a} appears in multiple pods")
                        all_access.add(a)

                    for d in pod.distribution:
                        if d < 1 or d > tt.distribution_switches:
                            errors.append(
                                f"Pod references distribution switch {d}, but only {tt.distribution_switches} exist"
                            )
                        all_dist.add(d)

                # Check all access switches are assigned
                expected_access = set(range(1, tt.access_switches + 1))
                missing = expected_access - all_access
                if missing:
                    warnings.append(f"Access switches not assigned to pods: {sorted(missing)}")

        return errors, warnings

    def _validate_resource_limits(self, config: TopologyConfig) -> tuple[List[str], List[str]]:
        """Validate resource limits."""
        errors = []
        warnings = []

        total_switches = config.total_switches
        total_hosts = config.total_hosts

        # Hard limit on switches
        if total_switches > self.max_switches:
            errors.append(
                f"Total switches ({total_switches}) exceeds MAX_SWITCHES={self.max_switches}"
            )

        # Warnings for large topologies
        if total_switches > 20:
            warnings.append(f"Large topology with {total_switches} switches may be slow to simulate")

        if total_hosts > 32:
            warnings.append(f"Many hosts ({total_hosts}) may strain traffic generation")

        # Check collector ratio
        topo_type = config.topology.type
        if topo_type == TopologyType.FAT_TREE:
            num_leaves = config.fat_tree.total_leaves
        elif topo_type == TopologyType.LEAF_SPINE:
            num_leaves = config.leaf_spine.num_leaves
        elif topo_type == TopologyType.THREE_TIER:
            num_leaves = config.three_tier.access_switches
        else:
            num_leaves = 0

        collectors_needed = (num_leaves + config.int_collectors.ratio - 1) // config.int_collectors.ratio
        if collectors_needed > 4:
            warnings.append(f"Will create {collectors_needed} INT collector hosts")

        return errors, warnings


def validate_config(config_path: Union[str, Path]) -> ValidationResult:
    """
    Convenience function to validate a configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ValidationResult
    """
    validator = TopologyValidator()
    return validator.validate_file(config_path)


def main():
    """CLI entry point for configuration validation."""
    parser = argparse.ArgumentParser(
        description="Validate P4 QoS topology configuration"
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--max-switches',
        type=int,
        default=MAX_SWITCHES,
        help=f'Maximum switches allowed (default: {MAX_SWITCHES})'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only output errors, no success messages'
    )

    args = parser.parse_args()

    validator = TopologyValidator(max_switches=args.max_switches)
    result = validator.validate_file(args.config)

    if not args.quiet or not result.is_valid:
        print(result)

    sys.exit(0 if result.is_valid else 1)


if __name__ == '__main__':
    main()
