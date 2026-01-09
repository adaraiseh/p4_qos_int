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
VERBOSE_FLAG := $(if $(VERBOSE),--verbose,)
BURSTY_FLAG  := $(if $(BURSTY),--bursty-mode --burst-profile $(BURST),)

# =============================================
# Common Command Variables
# =============================================
PYTHON      := python3
SUDO_PYTHON := sudo -E PYTHONUNBUFFERED=1 python3 -u
RL_COMMON   := --config $(DETECT_TOPOLOGY) --log-every 1 $(VERBOSE_FLAG)

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
	sudo $(PYTHON) network.py --config $(TOPOLOGY_CONFIG) --p4 $(P4SRC_FILE)

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
	sudo -E $(PYTHON) report_collector/influxdb_export.py --config $(DETECT_TOPOLOGY)

monitor:
	$(PYTHON) monitor_iperf_s.py --dir /tmp --window 60 --refresh 1

visualize:
	$(PYTHON) visualize_routes.py --config $(DETECT_TOPOLOGY)

# =============================================
# RL Training
# =============================================

train:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	$(SUDO_PYTHON) rl_agent_4.py --mode train --steps 50000 \
		$(RL_COMMON) --traffic-weights "light:0.1,medium:0.1,high:0.45,bursty:0.35"

train_test:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "=== Testing all training profiles (1 episode each) ==="
	@for profile in bursty_vo_1 bursty_vi_1 bursty_be_1 high_2 high_1 medium_2 medium_1 light_2 light_1; do \
		echo ""; \
		echo "=== Testing profile: $$profile ==="; \
		if ! $(SUDO_PYTHON) rl_agent_4.py --mode train $(RL_COMMON) \
			--steps 100 --max-episode-steps 100 --no-warm-start \
			--traffic-profile $$profile; then \
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
# RL Evaluation
# =============================================

TEST_COMMON = PYTHONUNBUFFERED=1 $(PYTHON) -u rl_agent_4.py --mode eval $(RL_COMMON) --steps 1500

test:
	$(TEST_COMMON) --weights-tag final

test_best:
	$(TEST_COMMON) --weights-tag best

# =============================================
# Traffic Testing
# =============================================

test_traffic:
	@echo "Testing traffic profile: $(profile)"
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	$(SUDO_PYTHON) rl_agent_4.py --mode train $(RL_COMMON) \
		--steps 200 --max-episode-steps 200 --no-warm-start \
		--traffic-profile $(profile)

# =============================================
# Production Mode
# =============================================

PROD_COMMON = $(SUDO_PYTHON) rl_production.py $(RL_COMMON) \
	--generate-traffic --traffic-profile $(profile) $(BURSTY_FLAG)

production:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
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
	@echo "Network Operations (use 'topo=' to select topology):"
	@echo "  make run topo=<name>      Start network with specified topology"
	@echo "  make stop                 Stop mininet"
	@echo "  make clean                Clean up all files"
	@echo ""
	@echo "Available topologies for 'topo=':"
	@echo "  Fat-Tree:    fat_tree_k2, fat_tree_k4 [default], fat_tree_k8"
	@echo "  Leaf-Spine:  leaf_spine_{4,6,8,10,12,14,16}x{2,3,4}"
	@echo "               (e.g., leaf_spine_6x3 = 6 leaves x 3 spines)"
	@echo "  Three-Tier:  three_tier_4, three_tier_6, three_tier_8"
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
	@echo "  Profiles: light_{1,2}, medium_{1,2}, high_{1,2},"
	@echo "            bursty_{vo,vi,be}_{1,2}"
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

.PHONY: all validate rules run stop clean collect monitor visualize \
        train train_test resume test test_best test_traffic \
        production production_best production_final production_75pct help
