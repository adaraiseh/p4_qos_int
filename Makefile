PCAP_DIR   = pcap
LOG_DIR    = log
P4SRC_DIR  = p4src

ifndef P4SRC_FILE
P4SRC_FILE = p4src/int_md.p4
endif

# =============================================
# Topology Selection
# =============================================
# Use: make run topo=fat_tree_k4
# Available topologies:
#   fat_tree_k2, fat_tree_k4
#   leaf_spine_4x2, leaf_spine_6x3, leaf_spine_8x4
#   three_tier_4, three_tier_6, three_tier_8

# Topology name to config file mapping
TOPO_DIR = config/topologies

# Default topology if none specified
ifndef topo
topo = fat_tree_k4
endif

# Map short names to config files
ifeq ($(topo),fat_tree_k2)
TOPOLOGY_CONFIG = $(TOPO_DIR)/fat_tree_k2.yaml
else ifeq ($(topo),fat_tree_k4)
TOPOLOGY_CONFIG = $(TOPO_DIR)/fat_tree_k4.yaml
else ifeq ($(topo),fat_tree_k8)
TOPOLOGY_CONFIG = $(TOPO_DIR)/fat_tree_k8.yaml
else ifeq ($(topo),leaf_spine_4x2)
TOPOLOGY_CONFIG = $(TOPO_DIR)/leaf_spine_4x2.yaml
else ifeq ($(topo),leaf_spine_6x3)
TOPOLOGY_CONFIG = $(TOPO_DIR)/leaf_spine_6x3.yaml
else ifeq ($(topo),leaf_spine_8x4)
TOPOLOGY_CONFIG = $(TOPO_DIR)/leaf_spine_8x4.yaml
else ifeq ($(topo),three_tier_4)
TOPOLOGY_CONFIG = $(TOPO_DIR)/three_tier_4.yaml
else ifeq ($(topo),three_tier_6)
TOPOLOGY_CONFIG = $(TOPO_DIR)/three_tier_6.yaml
else ifeq ($(topo),three_tier_8)
TOPOLOGY_CONFIG = $(TOPO_DIR)/three_tier_8.yaml
else ifeq ($(topo),three_tier)
TOPOLOGY_CONFIG = $(TOPO_DIR)/three_tier.yaml
else
# Allow direct path specification
TOPOLOGY_CONFIG = $(topo)
endif

# Auto-detect running topology from .active_topology file
# This file is created by 'make run' and contains the config path
DETECT_TOPOLOGY = $(shell cat .active_topology 2>/dev/null || echo "config/topologies/fat_tree_k4.yaml")

# Controller verbose output (default: suppressed)
ifdef VERBOSE
VERBOSE_FLAG = --verbose
else
VERBOSE_FLAG =
endif

# by default: start training
all: train

# =============================================
# Topology Configuration and Validation
# =============================================

# Validate YAML topology configuration
validate:
	python3 -m config.validator $(TOPOLOGY_CONFIG)

# Generate P4 rules from topology configuration
rules: validate
	python3 -m p4_rules.generator --config $(TOPOLOGY_CONFIG)

# =============================================
# Network Operations
# =============================================

# Start network with topology configuration (generates rules first)
# Usage: make run topo=fat_tree_k4
run: rules
	@echo "$(TOPOLOGY_CONFIG)" > .active_topology
	sudo python3 network.py --config $(TOPOLOGY_CONFIG) --p4 ${P4SRC_FILE}

stop:
	sudo mn -c

clean: stop
	sudo rm -f *.pcap
	sudo rm -rf $(PCAP_DIR) $(LOG_DIR) $(RULE_DIR)/rule*
	sudo rm -f topology.json .active_topology
	sudo rm -f /tmp/p4_paths.json /tmp/topology.json
	sudo rm -f $(P4SRC_DIR)/*.p4i $(P4SRC_DIR)/*.json

# =============================================
# INT Collector (Auto-detects topology)
# =============================================

# Run INT collector - auto-detects running topology
collect:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	sudo python3 report_collector/influxdb_export.py --config $(DETECT_TOPOLOGY)

monitor:
	python3 monitor_iperf_s.py --dir /tmp --window 60 --refresh 1

# =============================================
# Visualization (Auto-detects topology)
# =============================================

visualize:
	python3 visualize_routes.py --config $(DETECT_TOPOLOGY)

# =============================================
# RL Training (Auto-detects topology)
# =============================================

# Run RL agent v4 in training mode
# Auto-detects the running network topology
# Traffic weights: 5% light, 10% medium, 35% high, 50% bursty
train:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	sudo PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode train --steps 50000 \
		--config $(DETECT_TOPOLOGY) \
		--traffic-weights "light:0.05,medium:0.1,high:0.35,bursty:0.50" \
		--log-every 1 $(VERBOSE_FLAG)

# Quick training test - runs 1 episode per training profile
# Auto-detects the running network topology
train_test:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "=== Testing all training profiles (1 episode each) ==="
	@for profile in bursty_vo_1 bursty_vi_1 bursty_be_1 high_2 high_1 medium_2 medium_1 light_2 light_1; do \
		echo ""; \
		echo "=== Testing profile: $$profile ==="; \
		if ! sudo PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode train \
			--config $(DETECT_TOPOLOGY) \
			--steps 100 --max-episode-steps 100 --no-warm-start \
			--traffic-profile $$profile \
			--log-every 1 $(VERBOSE_FLAG); then \
			echo "Training interrupted or failed."; \
			ret=$$?; \
			if [ $$ret -eq 130 ]; then \
				echo "Clean interrupt."; \
				exit 0; \
			else \
				exit 1; \
			fi; \
		fi; \
	done
	@echo ""
	@echo "=== All profile tests completed ==="

# Resume RL agent v4 training from checkpoint
ifndef CHECKPOINT
CHECKPOINT = 50pct
endif
resume:
	sudo PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode train --steps 10000 \
		--config $(DETECT_TOPOLOGY) \
		--resume $(CHECKPOINT) --resume-eps 0.10 \
		--traffic-weights "light:0.05,medium:0.1,high:0.35,bursty:0.50" \
		--log-every 1 $(VERBOSE_FLAG)

# =============================================
# RL Evaluation (Auto-detects topology)
# =============================================

test:
	PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode eval \
		--config $(DETECT_TOPOLOGY) \
		--steps 1500 --weights-tag final $(VERBOSE_FLAG) --log-every 1

test_best:
	PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode eval \
		--config $(DETECT_TOPOLOGY) \
		--steps 1500 --weights-tag best $(VERBOSE_FLAG) --log-every 1

# =============================================
# Traffic Testing
# =============================================

# Test a specific traffic profile
# Usage: make test_traffic profile=high_1
# Available profiles: light_1, light_2, medium_1, medium_2, high_1, high_2,
#                     bursty_vo_1, bursty_vo_2, bursty_vi_1, bursty_vi_2,
#                     bursty_be_1, bursty_be_2
ifndef profile
profile = medium_1
endif

test_traffic:
	@echo "Testing traffic profile: $(profile)"
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	sudo PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode train \
		--config $(DETECT_TOPOLOGY) \
		--steps 200 --max-episode-steps 200 --no-warm-start \
		--traffic-profile $(profile) \
		--log-every 1 $(VERBOSE_FLAG)

# =============================================
# Production Mode (Auto-detects topology)
# =============================================

# Usage:
#   make production profile=high_2           (with best model)
#   make production profile=medium_2 BURSTY=1 BURST=bursty_be_2
ifndef profile
profile = high_1
endif

ifndef BURST
BURST = bursty_be_2
endif

ifdef BURSTY
BURSTY_FLAG = --bursty-mode --burst-profile $(BURST)
else
BURSTY_FLAG =
endif

# Production with best model (default)
production:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	sudo PYTHONUNBUFFERED=1 python3 -u rl_production.py --weights-tag best \
		--config $(DETECT_TOPOLOGY) \
		--generate-traffic --traffic-profile $(profile) \
		--log-every 1 $(VERBOSE_FLAG) $(BURSTY_FLAG)

production_best: production

production_final:
	sudo PYTHONUNBUFFERED=1 python3 -u rl_production.py --weights-tag final \
		--config $(DETECT_TOPOLOGY) \
		--generate-traffic --traffic-profile $(profile) \
		--log-every 1 $(VERBOSE_FLAG) $(BURSTY_FLAG)

production_75pct:
	sudo PYTHONUNBUFFERED=1 python3 -u rl_production.py --weights-tag 75pct \
		--config $(DETECT_TOPOLOGY) \
		--generate-traffic --traffic-profile $(profile) \
		--log-every 1 $(VERBOSE_FLAG) $(BURSTY_FLAG)

# =============================================
# Help
# =============================================

help:
	@echo "P4 QoS INT - Dynamic Multi-Topology Support"
	@echo ""
	@echo "Network Operations (use 'topo=' to select topology):"
	@echo "  make run topo=<name>      Start network with specified topology"
	@echo "  make stop                 Stop mininet"
	@echo "  make clean                Clean up all files"
	@echo ""
	@echo "Available topologies for 'topo=':"
	@echo "  fat_tree_k2               Fat-Tree k=2 (5 switches, 2 hosts)"
	@echo "  fat_tree_k4               Fat-Tree k=4 (20 switches, 16 hosts) [default]"
	@echo "  leaf_spine_4x2            Leaf-Spine 4 leaves, 2 spines (8 hosts)"
	@echo "  leaf_spine_6x3            Leaf-Spine 6 leaves, 3 spines (12 hosts)"
	@echo "  leaf_spine_8x4            Leaf-Spine 8 leaves, 4 spines (16 hosts)"
	@echo "  three_tier_4              Three-Tier 4 access switches (8 hosts)"
	@echo "  three_tier_6              Three-Tier 6 access switches (12 hosts)"
	@echo "  three_tier_8              Three-Tier 8 access switches (16 hosts)"
	@echo ""
	@echo "Training/Evaluation (auto-detect running topology):"
	@echo "  make train                Train RL agent on running network"
	@echo "  make train_test           Quick test all traffic profiles"
	@echo "  make test                 Evaluate with final model"
	@echo "  make test_best            Evaluate with best model"
	@echo "  make resume               Resume training from checkpoint"
	@echo ""
	@echo "Traffic Testing (auto-detect running topology):"
	@echo "  make test_traffic profile=<name>   Test specific traffic profile"
	@echo "  Profiles: light_1, light_2, medium_1, medium_2, high_1, high_2,"
	@echo "            bursty_vo_1, bursty_vo_2, bursty_vi_1, bursty_vi_2,"
	@echo "            bursty_be_1, bursty_be_2"
	@echo ""
	@echo "Production Mode (auto-detect running topology):"
	@echo "  make production profile=<name>              Run with best model"
	@echo "  make production_final profile=<name>        Run with final model"
	@echo "  make production profile=<name> BURSTY=1     Enable burst mode"
	@echo ""
	@echo "Collector & Visualization (auto-detect topology):"
	@echo "  make collect              Run INT collector"
	@echo "  make visualize            Run traffic visualization"
	@echo "  make monitor              Monitor iperf traffic"
	@echo ""
	@echo "Examples:"
	@echo "  make run topo=leaf_spine_6x3"
	@echo "  make train                           # auto-detects running topology"
	@echo "  make test_traffic profile=high_2     # test specific traffic profile"
	@echo "  make production profile=bursty_be_1  # production with traffic profile"
