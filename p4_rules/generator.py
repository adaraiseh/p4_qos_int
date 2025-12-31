# p4_rules/generator.py
"""
P4 rule generator for INT-enabled switches.

Generates P4 table entries automatically from topology configuration.
All switches get INT transit role. Leaf switches additionally get
INT source, sink, mirroring, and report encapsulation rules.
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from topology.base import TopologyBuilder, SwitchInfo, CollectorInfo
from topology.factory import create_topology
from config.loader import load_config
from config.validator import validate_config


class P4RuleGenerator:
    """
    Generates P4 table rules for all switches in a topology.

    INT Configuration:
    - ALL switches: INT transit (insert switch metadata)
    - Leaf switches: INT source, sink, mirroring, report encapsulation
    """

    # INT source port ranges for sampled traffic
    INT_SOURCE_PORT_RANGES = [
        (5888, 0xFF00),   # 0x1700–0x17FF
        (6144, 0xFF00),   # 0x1800–0x18FF
        (6400, 0xFF00),   # 0x1900–0x19FF
        (6656, 0xFF00),   # 0x1A00–0x1AFF
        (6912, 0xFF00),   # 0x1B00–0x1BFF
    ]

    # INT source parameters
    INT_MAX_HOP = 12
    INT_HOP_METADATA_LEN = 16
    INT_INS_MASK_0003 = 0xF
    INT_INS_MASK_0407 = 0xF
    INT_SAMPLE_RATE = 1  # Sample every packet

    # Mirroring session ID
    MIRROR_SESSION_ID = 500

    def __init__(self, builder: TopologyBuilder):
        """
        Initialize the rule generator.

        Args:
            builder: Built TopologyBuilder instance with topology data
        """
        self.builder = builder
        self.config = builder.config

    def generate_all(self, output_dir: Union[str, Path]) -> List[str]:
        """
        Generate P4 rules for all switches and write to files.

        Args:
            output_dir: Directory to write rule files

        Returns:
            List of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        for switch_name, switch_info in self.builder.switches.items():
            rules = self._generate_switch_rules(switch_info)
            file_path = output_dir / f"{switch_name}-commands.txt"

            with open(file_path, 'w') as f:
                f.write(rules)

            generated_files.append(str(file_path))

        return generated_files

    def _generate_switch_rules(self, switch: SwitchInfo) -> str:
        """
        Generate P4 rules for a single switch.

        Args:
            switch: SwitchInfo for the switch

        Returns:
            String containing all P4 table commands
        """
        lines = []

        # All switches get INT transit
        lines.append("//set up switch ID")
        lines.append(f"table_set_default process_int_transit.tb_int_insert init_metadata {switch.switch_id}")
        lines.append("")

        # Only leaf/access switches get additional INT rules
        if switch.is_int_source or switch.is_int_sink:
            lines.extend(self._generate_edge_switch_rules(switch))

        return '\n'.join(lines)

    def _generate_edge_switch_rules(self, switch: SwitchInfo) -> List[str]:
        """
        Generate INT source/sink rules for edge switches.

        Args:
            switch: SwitchInfo for an edge switch

        Returns:
            List of rule lines
        """
        lines = []

        # Get host-facing port numbers
        host_ports = self._get_host_ports(switch)

        if switch.is_int_source and host_ports:
            lines.extend(self._generate_int_source_rules(host_ports))
            lines.append("")

        if switch.is_int_sink and host_ports:
            lines.extend(self._generate_int_sink_rules(host_ports))
            lines.append("")

        # Mirroring setup
        collector_port = switch.collector_port
        if collector_port:
            lines.append(f"// creates INT mirroring ID {self.MIRROR_SESSION_ID} to output port {collector_port}")
            lines.append(f"mirroring_add {self.MIRROR_SESSION_ID} {collector_port}")
            lines.append("")

        # Report encapsulation
        collector_info = self._get_collector_for_switch(switch.name)
        if collector_info:
            lines.extend(self._generate_report_encap_rules(switch.name, collector_info))
            lines.append("")

        return lines

    def _get_host_ports(self, switch: SwitchInfo) -> List[int]:
        """
        Get the port numbers on a switch that connect to hosts.

        Args:
            switch: SwitchInfo for the switch

        Returns:
            List of port numbers
        """
        ports = []
        for host_name in switch.connected_hosts:
            host = self.builder.hosts.get(host_name)
            if host:
                ports.append(host.port_on_switch)
        return sorted(ports)

    def _generate_int_source_rules(self, host_ports: List[int]) -> List[str]:
        """
        Generate INT source rules.

        Args:
            host_ports: List of host-facing port numbers

        Returns:
            List of rule lines
        """
        lines = []

        # Set up source ports
        lines.append("//set up process_int_source_sink (the port number where monitored packets are coming from)")
        for port in host_ports:
            lines.append(f"table_add process_int_source_sink.tb_set_source int_set_source {port} =>")

        lines.append("")

        # Set up INT source sampling for each port range
        for port_base, mask in self.INT_SOURCE_PORT_RANGES:
            hex_range = f"0x{port_base:04X}–0x{port_base + 0xFF:04X}"
            lines.append(f"// {hex_range}  ({port_base}–{port_base + 0xFF})")
            lines.append(
                f"table_add process_int_source.tb_int_source int_source_sampled "
                f"10.0.0.0/8 {port_base}&&&0x{mask:X} => "
                f"{self.INT_MAX_HOP} {self.INT_HOP_METADATA_LEN} "
                f"0x{self.INT_INS_MASK_0003:X} 0x{self.INT_INS_MASK_0407:X} {self.INT_SAMPLE_RATE}"
            )

        return lines

    def _generate_int_sink_rules(self, host_ports: List[int]) -> List[str]:
        """
        Generate INT sink rules.

        Args:
            host_ports: List of host-facing port numbers

        Returns:
            List of rule lines
        """
        lines = []

        lines.append("// set up INT sink (the monitored packets output port)")
        for port in host_ports:
            lines.append(f"table_add process_int_source_sink.tb_set_sink int_set_sink {port} => ")

        return lines

    def _get_collector_for_switch(self, switch_name: str) -> Optional[CollectorInfo]:
        """
        Find the collector assigned to a switch.

        Args:
            switch_name: Name of the switch

        Returns:
            CollectorInfo if found, None otherwise
        """
        for collector in self.builder.collectors.values():
            if switch_name in collector.connected_switches:
                return collector
        return None

    def _generate_report_encap_rules(self, switch_name: str, collector: CollectorInfo) -> List[str]:
        """
        Generate INT report encapsulation rules.

        Args:
            switch_name: Name of the switch
            collector: CollectorInfo for the assigned collector

        Returns:
            List of rule lines
        """
        lines = []

        iface = collector.interfaces.get(switch_name, {})
        if not iface:
            return lines

        collector_mac = iface.get('collector_mac', '00:00:00:00:00:00')
        switch_mac = iface.get('switch_mac', '00:00:00:00:00:00')
        collector_ip = iface.get('collector_ip', '0.0.0.0')
        switch_ip = iface.get('switch_ip', '0.0.0.0')

        lines.append("// set up INT report encapsulation")
        lines.append(
            f"table_set_default tb_generate_report do_report_encapsulation "
            f"{collector_mac} {switch_mac} {collector_ip} {switch_ip} 1234"
        )

        return lines

    def print_summary(self) -> None:
        """Print a summary of the topology for verification."""
        print(f"\n{'='*60}")
        print(f"P4 Rule Generation Summary")
        print(f"{'='*60}")
        print(f"Topology: {self.config.topology.name} ({self.config.topology.type.value})")
        print(f"Total switches: {len(self.builder.switches)}")
        print(f"Total hosts: {len(self.builder.hosts)}")
        print(f"INT collectors: {len(self.builder.collectors)}")
        print()

        # Switch breakdown by role
        roles = {}
        for sw in self.builder.switches.values():
            role = sw.role
            if role not in roles:
                roles[role] = []
            roles[role].append(sw.name)

        for role, switches in sorted(roles.items()):
            int_info = "INT transit"
            if role in ('leaf', 'access', 'tor'):
                int_info += " + source/sink/mirror/report"
            print(f"  {role.capitalize()} ({len(switches)}): {int_info}")
            for name in sorted(switches):
                sw = self.builder.switches[name]
                print(f"    - {name} (ID={sw.switch_id}, thrift={sw.thrift_port})")

        print()

        # Collector assignments
        if self.builder.collectors:
            print("INT Collector assignments:")
            for name, collector in self.builder.collectors.items():
                print(f"  {name}: {', '.join(collector.connected_switches)}")

        print(f"{'='*60}\n")


def generate_rules(config_path: Union[str, Path],
                   output_dir: Union[str, Path] = None,
                   verbose: bool = True) -> List[str]:
    """
    Convenience function to generate P4 rules from a config file.

    Args:
        config_path: Path to YAML configuration file
        output_dir: Output directory for rules (default: rules/<topology_name>)
        verbose: Print summary if True

    Returns:
        List of generated file paths
    """
    # Validate config first
    result = validate_config(config_path)
    if not result.is_valid:
        raise ValueError(f"Configuration validation failed:\n{result}")

    config = result.config

    # Build topology
    builder = create_topology(config_path, validate=False)

    # Determine output directory
    if output_dir is None:
        topo_name = config.topology.name.replace('-', '_')
        output_dir = Path(__file__).parent.parent / 'rules' / topo_name

    # Generate rules
    generator = P4RuleGenerator(builder)

    if verbose:
        generator.print_summary()

    generated = generator.generate_all(output_dir)

    if verbose:
        print(f"Generated {len(generated)} rule files in {output_dir}")
        for f in sorted(generated):
            print(f"  - {Path(f).name}")

    return generated


def main():
    """CLI entry point for rule generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate P4 table rules from topology configuration"
    )
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to YAML topology configuration file'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory for rule files (default: rules/<topology_name>)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    try:
        generate_rules(args.config, args.output, verbose=not args.quiet)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
