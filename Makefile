# =============================================
# P4 QoS INT - Dynamic Multi-Topology Makefile
# =============================================

# Directory configuration
PCAP_DIR   = pcap
LOG_DIR    = log
P4SRC_DIR  = p4src
TOPO_DIR   = config/topologies

# Default values
P4SRC_FILE ?= p4src/int_md.p4
topo       ?= fat_tree_k4
profile    ?= high_1
CHECKPOINT ?= 50pct
BURST      ?= bursty_be_2
LOG_LEVEL  ?= info

# =============================================
# Dynamic Topology Mapping
# =============================================
# Supports both short names (fat_tree_k4) and direct paths
TOPOLOGY_CONFIG = $(if $(wildcard $(topo)),$(topo),$(TOPO_DIR)/$(topo).yaml)

# Auto-detect running topology from .active_topology file
DETECT_TOPOLOGY = $(shell cat .active_topology 2>/dev/null || echo "$(TOPO_DIR)/fat_tree_k4.yaml")

# =============================================
# Flag Handling
# =============================================
BURSTY_FLAG  := $(if $(BURSTY),--bursty-mode --burst-profile $(BURST),)
LOG_LEVEL_FLAG := --log-level $(LOG_LEVEL)

# =============================================
# Common Command Variables
# =============================================
PYTHON      := python3
SUDO_PYTHON := sudo -E PYTHONUNBUFFERED=1 python3 -u
RL_COMMON   := --config $(DETECT_TOPOLOGY) --log-every 1 $(LOG_LEVEL_FLAG)

# Default target
all: train

# =============================================
# Topology Configuration and Validation
# =============================================

validate:
	$(PYTHON) -m config.validator $(TOPOLOGY_CONFIG)

rules: validate
	$(PYTHON) -m p4_rules.generator --config $(TOPOLOGY_CONFIG)

# =============================================
# Network Operations
# =============================================

run: rules
	@echo "$(TOPOLOGY_CONFIG)" > .active_topology
	sudo $(PYTHON) network.py --config $(TOPOLOGY_CONFIG) --p4 $(P4SRC_FILE) $(LOG_LEVEL_FLAG)

stop:
	sudo mn -c

clean: stop
	sudo rm -f *.pcap
	sudo rm -rf $(PCAP_DIR) $(LOG_DIR) $(RULE_DIR)/rule*
	sudo rm -f topology.json .active_topology
	sudo rm -f /tmp/p4_paths.json /tmp/topology.json
	sudo rm -f $(P4SRC_DIR)/*.p4i $(P4SRC_DIR)/*.json $(P4SRC_DIR)/*.p4info.txt

# =============================================
# INT Collector & Monitoring
# =============================================

collect:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	sudo -E $(PYTHON) report_collector/influxdb_export.py --config $(DETECT_TOPOLOGY) $(LOG_LEVEL_FLAG)

monitor:
	$(PYTHON) monitor_iperf_s.py --dir /tmp --window 60 --refresh 1

visualize:
	$(PYTHON) visualize_routes.py --config $(DETECT_TOPOLOGY)

# =============================================
# RL Training
# =============================================

# Full training: 50K steps on fat_tree_k4
train:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "Log level: $(LOG_LEVEL) (use LOG_LEVEL=debug for debug output)"
	$(SUDO_PYTHON) rl_agent_4.py --mode train --steps 55000 \
		$(RL_COMMON) --traffic-weights "light:0.02,medium:0.02,high:0.50,bursty:0.46"

# Test training: 20 steps per episode, cycles through all traffic profiles
train_test:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "=== Testing all training profiles (20 steps each) ==="
	@for p in bursty_vo_1 bursty_vi_1 bursty_be_1 high_2 high_1 medium_2 medium_1 light_2 light_1; do \
		echo ""; \
		echo "=== Testing profile: $$p ==="; \
		if ! $(SUDO_PYTHON) rl_agent_4.py --mode train $(RL_COMMON) \
			--steps 20 --max-episode-steps 20 --no-warm-start \
			--traffic-profile $$p; then \
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

resume:
	$(SUDO_PYTHON) rl_agent_4.py --mode train --steps 10000 \
		$(RL_COMMON) --resume $(CHECKPOINT) --resume-eps 0.10 \
		--traffic-weights "light:0.05,medium:0.05,high:0.35,bursty:0.55"

# =============================================
# Traffic Testing
# =============================================

# Load specific traffic profile and run indefinitely (Ctrl+C to stop)
test_traffic:
	@echo "Testing traffic profile: $(profile)"
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "Running indefinitely (Ctrl+C to stop)..."
	$(SUDO_PYTHON) rl_agent_4.py --mode train $(RL_COMMON) \
		--steps 999999 --max-episode-steps 999999 --no-warm-start \
		--traffic-profile $(profile)

# =============================================
# Traffic Stress Testing
# =============================================

# Quick stress test: 100 cycles, 5s each (~10 min)
traffic_stress_test_quick:
	@echo "Running quick traffic stress test (100 cycles, 5s each)"
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	$(SUDO_PYTHON) test/stress_test_traffic.py --quick --config $(DETECT_TOPOLOGY)

# Full stress test: 3600 cycles, 10s each (~1 hour)
traffic_stress_test:
	@echo "Running full traffic stress test (3600 cycles, 10s each = ~1 hour)"
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	$(SUDO_PYTHON) test/stress_test_traffic.py --config $(DETECT_TOPOLOGY)

# Custom stress test: use CYCLES and DURATION variables
# Example: make traffic_stress_test_custom CYCLES=500 DURATION=5
CYCLES   ?= 3600
DURATION ?= 10
traffic_stress_test_custom:
	@echo "Running custom traffic stress test ($(CYCLES) cycles, $(DURATION)s each)"
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	$(SUDO_PYTHON) test/stress_test_traffic.py --cycles $(CYCLES) --duration $(DURATION) --config $(DETECT_TOPOLOGY)


# =============================================
# Production Mode
# =============================================

PROD_COMMON = $(SUDO_PYTHON) rl_production.py $(RL_COMMON) \
	--generate-traffic --traffic-profile $(profile) $(BURSTY_FLAG)

# Run production with specific traffic profile
production:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "Traffic profile: $(profile)"
	$(PROD_COMMON) --weights-tag best

production_best: production

production_final:
	$(PROD_COMMON) --weights-tag final

production_75pct:
	$(PROD_COMMON) --weights-tag 75pct

# =============================================
# Help
# =============================================

help:
	@echo "P4 QoS INT - Dynamic Multi-Topology Support"
	@echo ""
	@echo "Network Operations:"
	@echo "  make run topo=<name>      Start network with specified topology"
	@echo "  make stop                 Stop mininet"
	@echo "  make clean                Clean up all files"
	@echo ""
	@echo "Available topologies for 'topo=':"
	@echo "  Fat-Tree:    fat_tree_k2, fat_tree_k4 [default], fat_tree_k8"
	@echo "  Leaf-Spine:  leaf_spine_{4,6,8,10,12,14,16}x{2,3,4}"
	@echo "  Three-Tier:  three_tier_4, three_tier_6, three_tier_8"
	@echo ""
	@echo "Training:"
	@echo "  make train                Full training (50K steps, fat_tree_k4)"
	@echo "  make train_test           Test all profiles (20 steps each)"
	@echo "  make resume               Resume training from checkpoint"
	@echo ""
	@echo "Traffic Testing:"
	@echo "  make test_traffic profile=<name>   Run specific profile indefinitely"
	@echo "  Profiles: light_{1,2}, medium_{1,2}, high_{1,2},"
	@echo "            bursty_{vo,vi,be}_{1,2}"
	@echo ""
	@echo "Traffic Stress Testing:"
	@echo "  make traffic_stress_test_quick     Quick test (100 cycles, ~10 min)"
	@echo "  make traffic_stress_test           Full test (3600 cycles, ~1 hour)"
	@echo "  make traffic_stress_test_custom CYCLES=500 DURATION=5"
	@echo ""
	@echo "Production Mode:"
	@echo "  make production profile=<name>     Run with best model"
	@echo "  make production_final profile=<name>"
	@echo "  make production profile=<name> BURSTY=1  Enable burst mode"
	@echo ""
	@echo "Monitoring:"
	@echo "  make collect              Run INT collector"
	@echo "  make visualize            Run traffic visualization"
	@echo "  make monitor              Monitor iperf traffic"
	@echo ""
	@echo "Logging Options:"
	@echo "  LOG_LEVEL=debug           Log debug output to file (default)"
	@echo "  LOG_LEVEL=info            Log info output to file only"
	@echo "  (Console always shows INFO level)"
	@echo ""
	@echo "Examples:"
	@echo "  make run topo=fat_tree_k4"
	@echo "  make train LOG_LEVEL=debug"
	@echo "  make test_traffic profile=high_2"
	@echo "  make production profile=bursty_be_1"

.PHONY: all validate rules run stop clean collect monitor visualize \
        train train_test resume test test_best test_traffic \
        traffic_stress_test traffic_stress_test_quick traffic_stress_test_custom \
        production production_best production_final production_75pct help
