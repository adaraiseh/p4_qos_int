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
LOG_LEVEL  ?= info
STEPS      ?= 300
TRAFFIC_SEED ?= 42
SUDO_KEEPALIVE_INTERVAL ?= 60
TRAIN_REPLAY_CAPACITY ?= 80000
TRAIN_FOUNDATION_STEPS ?= 35000
TRAIN_BURST_STEPS ?= 25000
TRAIN_POLISH_STEPS ?= 20000
TRAIN_FOUNDATION_EPS_DECAY_STEPS ?= 45000
TRAIN_BURST_EPS_DECAY_STEPS ?= 25000
TRAIN_POLISH_EPS_DECAY_STEPS ?= 15000
TRAIN_BURST_RESUME_EPS ?= 0.45
TRAIN_POLISH_RESUME_EPS ?= 0.15
TRAIN_FOUNDATION_RESET_PROB_START ?= 0.95
TRAIN_FOUNDATION_RESET_PROB_END ?= 0.70
TRAIN_BURST_RESET_PROB_START ?= 0.85
TRAIN_BURST_RESET_PROB_END ?= 0.60
TRAIN_POLISH_RESET_PROB_START ?= 0.65
TRAIN_POLISH_RESET_PROB_END ?= 0.45
TRAIN_BASELINE_COOLDOWN ?= 5
TRAIN_WARM_COOLDOWN ?= 3
TRAIN_LOG_FLUSH_EVERY ?= 100
TRAIN_INFLUX_DETAIL ?= off
TRAFFIC_TEST_INFLUX_DETAIL ?= off
TELEMETRY_BACKEND ?= cache
TELEMETRY_CACHE_SOCKET ?= /tmp/p4_qos_int_telemetry.sock
TELEMETRY_CACHE_TIMEOUT ?= 1.0
EXTERNAL_ARTIFACT_ROOT ?= /media/sf_amjad/p4_qos_int/training_runs
TRAINING_STATE_FILE ?= /tmp/p4_qos_int_training_state.json
COLLECTOR_LOCAL_SPOOL_SPLIT_STEPS ?= 10000
COLLECTOR_INFLUX_WRITE ?= off
COLLECTOR_LOCAL_SPOOL ?= on
COLLECTOR_LOCAL_SPOOL_DIR ?= $(EXTERNAL_ARTIFACT_ROOT)/_collector_pending
COLLECTOR_PROCESS_LOG_DIR ?= $(EXTERNAL_ARTIFACT_ROOT)/_collector_logs
PRODUCTION_INFLUX_WRITE ?= off
TRAIN_LOG_DIR ?=
ALL_TRAFFIC_PROFILES ?= light_1 light_2 medium_1 medium_2 high_1 high_2 bursty_vo_1 bursty_vo_2 bursty_vo_3 bursty_vi_1 bursty_vi_2 bursty_vi_3 bursty_be_1 bursty_be_2 bursty_be_3
ALL_TRAFFIC_PROFILES_CSV ?= light_1,light_2,medium_1,medium_2,high_1,high_2,bursty_vo_1,bursty_vo_2,bursty_vo_3,bursty_vi_1,bursty_vi_2,bursty_vi_3,bursty_be_1,bursty_be_2,bursty_be_3
PRODUCTION_PROFILES ?= $(ALL_TRAFFIC_PROFILES)
TRAIN_FOUNDATION_PROFILE_WEIGHTS ?= light_1:2,light_2:2,medium_1:7,medium_2:9,high_1:24,high_2:12,bursty_vo_1:2,bursty_vo_2:3,bursty_vo_3:9,bursty_vi_1:2,bursty_vi_2:3,bursty_vi_3:10,bursty_be_1:2,bursty_be_2:3,bursty_be_3:10
TRAIN_BURST_PROFILE_WEIGHTS ?= light_1:1,light_2:1,medium_1:4,medium_2:4,high_1:22,high_2:8,bursty_vo_1:1,bursty_vo_2:2,bursty_vo_3:16,bursty_vi_1:1,bursty_vi_2:2,bursty_vi_3:18,bursty_be_1:1,bursty_be_2:2,bursty_be_3:17
TRAIN_POLISH_PROFILE_WEIGHTS ?= light_1:2.5,light_2:2.5,medium_1:7,medium_2:8,high_1:24,high_2:11,bursty_vo_1:4,bursty_vo_2:4,bursty_vo_3:7,bursty_vi_1:4,bursty_vi_2:4,bursty_vi_3:7,bursty_be_1:4,bursty_be_2:4,bursty_be_3:7
BENCH_PROFILES ?= $(ALL_TRAFFIC_PROFILES_CSV)
BENCH_REPETITIONS ?= 6
BENCH_BASE_SEED ?= 42
BENCH_WARMUP ?= 5
BENCH_COOLDOWN ?= 30
BENCH_MIN_VALID ?= 0.80
BENCH_RETRIES ?= 1
BENCH_WEIGHTS_TAG ?= final
BENCH_METHODS ?= rl,ecmp,ospf
BENCH_OUTPUT ?=
BENCHMARK_DIR ?= benchmark_results
CALIBRATION_DIR ?= calibration_results

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
LOG_LEVEL_FLAG := --log-level $(LOG_LEVEL)
BENCH_OUTPUT_FLAG := $(if $(BENCH_OUTPUT),--output-dir $(BENCH_OUTPUT),)
TRAIN_LOG_DIR_FLAG = $(if $(TRAIN_LOG_DIR),--training-log-dir $(TRAIN_LOG_DIR),)
TRAIN_ARTIFACT_FLAGS = --artifact-root $(EXTERNAL_ARTIFACT_ROOT) --training-state-file $(TRAINING_STATE_FILE) --collector-spool-split-steps $(COLLECTOR_LOCAL_SPOOL_SPLIT_STEPS)
TRAIN_LOGGING_FLAGS = --training-log-flush-every $(TRAIN_LOG_FLUSH_EVERY) $(TRAIN_LOG_DIR_FLAG) $(TRAIN_ARTIFACT_FLAGS)
TRAIN_REPLAY_FLAGS = --buffer-capacity $(TRAIN_REPLAY_CAPACITY)
TELEMETRY_FLAGS = --telemetry-backend $(TELEMETRY_BACKEND) --telemetry-cache-socket $(TELEMETRY_CACHE_SOCKET) --telemetry-cache-timeout $(TELEMETRY_CACHE_TIMEOUT)
COLLECTOR_ARTIFACT_FLAGS = --artifact-root $(EXTERNAL_ARTIFACT_ROOT) --training-state-file $(TRAINING_STATE_FILE) --local-spool-split-steps $(COLLECTOR_LOCAL_SPOOL_SPLIT_STEPS)
COLLECTOR_TELEMETRY_FLAGS = --influx-write $(COLLECTOR_INFLUX_WRITE) --telemetry-cache-socket $(TELEMETRY_CACHE_SOCKET) --local-spool $(COLLECTOR_LOCAL_SPOOL) --local-spool-dir $(COLLECTOR_LOCAL_SPOOL_DIR) $(COLLECTOR_ARTIFACT_FLAGS)

# =============================================
# Common Command Variables
# =============================================
PYTHON      := python3
SUDO_PYTHON := sudo -E PYTHONUNBUFFERED=1 python3 -u
RL_COMMON   := --config $(DETECT_TOPOLOGY) --log-every 1 $(LOG_LEVEL_FLAG) $(TELEMETRY_FLAGS)

# Default target
all: train_paper

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

clean:
ifneq ($(filter bench,$(MAKECMDGOALS)),)
	$(MAKE) --no-print-directory clean_bench
else
	$(MAKE) --no-print-directory stop
	sudo rm -f *.pcap
	sudo rm -rf $(PCAP_DIR) $(LOG_DIR) $(RULE_DIR)/rule*
	sudo rm -f topology.json .active_topology
	sudo rm -f /tmp/p4_paths.json /tmp/topology.json
	sudo rm -f $(P4SRC_DIR)/*.p4i $(P4SRC_DIR)/*.json $(P4SRC_DIR)/*.p4info.txt
endif

# Remove benchmark/calibration outputs generated by paper comparison runs.
# This is intentionally scoped to result artifacts, not model checkpoints,
# source files, topology config, or training data.
clean_bench clean-benchmark:
	@echo "Removing benchmark result directories and benchmark run artifacts..."
	sudo rm -rf $(BENCHMARK_DIR) $(CALIBRATION_DIR)
	sudo rm -f data/ecmp_log_*.csv data/ospf_log_*.csv data/production_log_*.csv
	sudo rm -f log/traffic_log.csv
	@echo "Benchmark statistics and logs cleaned."

# Remove durable training report artifacts while preserving model checkpoints
# and the tracked training_logs/.gitkeep placeholder.
clean_training_logs clean-training-logs:
	@echo "Removing legacy local durable training logs from training_files/training_logs..."
	sudo rm -rf training_files/training_logs/*
	@echo "Legacy local training logs cleaned; media-backed runs are preserved."

bench:
ifeq ($(filter clean,$(MAKECMDGOALS)),)
	$(MAKE) --no-print-directory clean_bench
else
	@:
endif

# =============================================
# INT Collector & Monitoring
# =============================================

collect:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "Telemetry cache socket: $(TELEMETRY_CACHE_SOCKET), Influx writes: $(COLLECTOR_INFLUX_WRITE)"
	@echo "External artifact root: $(EXTERNAL_ARTIFACT_ROOT)"
	@echo "Training state file: $(TRAINING_STATE_FILE)"
	@echo "Local telemetry spool: $(COLLECTOR_LOCAL_SPOOL) ($(COLLECTOR_LOCAL_SPOOL_DIR)), split steps=$(COLLECTOR_LOCAL_SPOOL_SPLIT_STEPS)"
	sudo -E $(PYTHON) report_collector/influxdb_export.py --config $(DETECT_TOPOLOGY) $(LOG_LEVEL_FLAG) --log-dir $(COLLECTOR_PROCESS_LOG_DIR) $(COLLECTOR_TELEMETRY_FLAGS)

monitor:
	$(PYTHON) monitor_iperf_s.py --dir /tmp --window 60 --refresh 1

visualize:
	$(PYTHON) visualize_routes.py --config $(DETECT_TOPOLOGY)

# =============================================
# RL Training
# =============================================

# Recovery/burst/stability curriculum for the 14-action K1/K2 agent:
# 1) reset-heavy recovery foundation, 2) bursty_3 specialization,
# 3) stability/no-op polishing.
train: train_paper

train_paper:
	@set -e; \
	sudo -v; \
	while true; do sudo -n -v || exit; sleep $(SUDO_KEEPALIVE_INTERVAL); done & \
	SUDO_KEEPALIVE_PID=$$!; \
	trap 'kill $$SUDO_KEEPALIVE_PID 2>/dev/null || true' EXIT INT TERM; \
	echo "Using topology config: $(DETECT_TOPOLOGY)"; \
	echo "Log level: $(LOG_LEVEL) (use LOG_LEVEL=debug for debug output)"; \
	echo "Influx training detail: $(TRAIN_INFLUX_DETAIL)"; \
	echo "External artifact root: $(EXTERNAL_ARTIFACT_ROOT)"; \
	echo "Training state file: $(TRAINING_STATE_FILE), collector split steps=$(COLLECTOR_LOCAL_SPOOL_SPLIT_STEPS)"; \
	echo "Replay capacity: $(TRAIN_REPLAY_CAPACITY)"; \
	echo "RL telemetry backend: $(TELEMETRY_BACKEND) ($(TELEMETRY_CACHE_SOCKET))"; \
	echo "=== Stage 1/3: reset-heavy recovery foundation ($(TRAIN_FOUNDATION_STEPS) steps, eps_decay=$(TRAIN_FOUNDATION_EPS_DECAY_STEPS)) ==="; \
	echo "Profile weights: $(TRAIN_FOUNDATION_PROFILE_WEIGHTS)"; \
	$(SUDO_PYTHON) rl_agent_4.py --mode train --steps $(TRAIN_FOUNDATION_STEPS) \
		$(RL_COMMON) $(TRAIN_LOGGING_FLAGS) $(TRAIN_REPLAY_FLAGS) \
		--reset-prob-start $(TRAIN_FOUNDATION_RESET_PROB_START) \
		--reset-prob-end $(TRAIN_FOUNDATION_RESET_PROB_END) \
		--baseline-cooldown-seconds $(TRAIN_BASELINE_COOLDOWN) \
		--warm-cooldown-seconds $(TRAIN_WARM_COOLDOWN) \
		--training-influx-detail $(TRAIN_INFLUX_DETAIL) \
		--eps-decay-steps $(TRAIN_FOUNDATION_EPS_DECAY_STEPS) \
		--traffic-profile-weights "$(TRAIN_FOUNDATION_PROFILE_WEIGHTS)"; \
	echo "=== Stage 2/3: burst specialization ($(TRAIN_BURST_STEPS) steps, resume final, eps=$(TRAIN_BURST_RESUME_EPS), eps_decay=$(TRAIN_BURST_EPS_DECAY_STEPS)) ==="; \
	echo "Profile weights: $(TRAIN_BURST_PROFILE_WEIGHTS)"; \
	$(SUDO_PYTHON) rl_agent_4.py --mode train --steps $(TRAIN_BURST_STEPS) \
		$(RL_COMMON) $(TRAIN_LOGGING_FLAGS) $(TRAIN_REPLAY_FLAGS) \
		--reset-prob-start $(TRAIN_BURST_RESET_PROB_START) \
		--reset-prob-end $(TRAIN_BURST_RESET_PROB_END) \
		--baseline-cooldown-seconds $(TRAIN_BASELINE_COOLDOWN) \
		--warm-cooldown-seconds $(TRAIN_WARM_COOLDOWN) \
		--training-influx-detail $(TRAIN_INFLUX_DETAIL) \
		--eps-decay-steps $(TRAIN_BURST_EPS_DECAY_STEPS) \
		--resume final --resume-eps $(TRAIN_BURST_RESUME_EPS) \
		--traffic-profile-weights "$(TRAIN_BURST_PROFILE_WEIGHTS)"; \
	echo "=== Stage 3/3: stability/no-op polishing ($(TRAIN_POLISH_STEPS) steps, resume final, eps=$(TRAIN_POLISH_RESUME_EPS), eps_decay=$(TRAIN_POLISH_EPS_DECAY_STEPS)) ==="; \
	echo "Profile weights: $(TRAIN_POLISH_PROFILE_WEIGHTS)"; \
	$(SUDO_PYTHON) rl_agent_4.py --mode train --steps $(TRAIN_POLISH_STEPS) \
		$(RL_COMMON) $(TRAIN_LOGGING_FLAGS) $(TRAIN_REPLAY_FLAGS) \
		--reset-prob-start $(TRAIN_POLISH_RESET_PROB_START) \
		--reset-prob-end $(TRAIN_POLISH_RESET_PROB_END) \
		--baseline-cooldown-seconds $(TRAIN_BASELINE_COOLDOWN) \
		--warm-cooldown-seconds $(TRAIN_WARM_COOLDOWN) \
		--training-influx-detail $(TRAIN_INFLUX_DETAIL) \
		--eps-decay-steps $(TRAIN_POLISH_EPS_DECAY_STEPS) \
		--resume final --resume-eps $(TRAIN_POLISH_RESUME_EPS) \
		--traffic-profile-weights "$(TRAIN_POLISH_PROFILE_WEIGHTS)"

train_stage2:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "Log level: $(LOG_LEVEL) (use LOG_LEVEL=debug for debug output)"
	@echo "Influx training detail: $(TRAIN_INFLUX_DETAIL)"
	@echo "External artifact root: $(EXTERNAL_ARTIFACT_ROOT)"
	@echo "Training state file: $(TRAINING_STATE_FILE), collector split steps=$(COLLECTOR_LOCAL_SPOOL_SPLIT_STEPS)"
	@echo "Replay capacity: $(TRAIN_REPLAY_CAPACITY)"
	@echo "RL telemetry backend: $(TELEMETRY_BACKEND) ($(TELEMETRY_CACHE_SOCKET))"
	@echo "=== Stage 2/3: burst specialization ($(TRAIN_BURST_STEPS) steps, resume final, eps=$(TRAIN_BURST_RESUME_EPS), eps_decay=$(TRAIN_BURST_EPS_DECAY_STEPS)) ==="
	@echo "Profile weights: $(TRAIN_BURST_PROFILE_WEIGHTS)"
	$(SUDO_PYTHON) rl_agent_4.py --mode train --steps $(TRAIN_BURST_STEPS) \
		$(RL_COMMON) $(TRAIN_LOGGING_FLAGS) $(TRAIN_REPLAY_FLAGS) \
		--reset-prob-start $(TRAIN_BURST_RESET_PROB_START) \
		--reset-prob-end $(TRAIN_BURST_RESET_PROB_END) \
		--baseline-cooldown-seconds $(TRAIN_BASELINE_COOLDOWN) \
		--warm-cooldown-seconds $(TRAIN_WARM_COOLDOWN) \
		--training-influx-detail $(TRAIN_INFLUX_DETAIL) \
		--eps-decay-steps $(TRAIN_BURST_EPS_DECAY_STEPS) \
		--resume final --resume-eps $(TRAIN_BURST_RESUME_EPS) \
		--traffic-profile-weights "$(TRAIN_BURST_PROFILE_WEIGHTS)"

train_stage3:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "Log level: $(LOG_LEVEL) (use LOG_LEVEL=debug for debug output)"
	@echo "Influx training detail: $(TRAIN_INFLUX_DETAIL)"
	@echo "External artifact root: $(EXTERNAL_ARTIFACT_ROOT)"
	@echo "Training state file: $(TRAINING_STATE_FILE), collector split steps=$(COLLECTOR_LOCAL_SPOOL_SPLIT_STEPS)"
	@echo "Replay capacity: $(TRAIN_REPLAY_CAPACITY)"
	@echo "RL telemetry backend: $(TELEMETRY_BACKEND) ($(TELEMETRY_CACHE_SOCKET))"
	@echo "=== Stage 3/3: stability/no-op polishing ($(TRAIN_POLISH_STEPS) steps, resume final, eps=$(TRAIN_POLISH_RESUME_EPS), eps_decay=$(TRAIN_POLISH_EPS_DECAY_STEPS)) ==="
	@echo "Profile weights: $(TRAIN_POLISH_PROFILE_WEIGHTS)"
	$(SUDO_PYTHON) rl_agent_4.py --mode train --steps $(TRAIN_POLISH_STEPS) \
		$(RL_COMMON) $(TRAIN_LOGGING_FLAGS) $(TRAIN_REPLAY_FLAGS) \
		--reset-prob-start $(TRAIN_POLISH_RESET_PROB_START) \
		--reset-prob-end $(TRAIN_POLISH_RESET_PROB_END) \
		--baseline-cooldown-seconds $(TRAIN_BASELINE_COOLDOWN) \
		--warm-cooldown-seconds $(TRAIN_WARM_COOLDOWN) \
		--training-influx-detail $(TRAIN_INFLUX_DETAIL) \
		--eps-decay-steps $(TRAIN_POLISH_EPS_DECAY_STEPS) \
		--resume final --resume-eps $(TRAIN_POLISH_RESUME_EPS) \
		--traffic-profile-weights "$(TRAIN_POLISH_PROFILE_WEIGHTS)"

# =============================================
# Traffic Testing
# =============================================

# Load specific traffic profile and run indefinitely (Ctrl+C to stop)
test_traffic:
	@echo "Testing traffic profile: $(profile)"
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "Running indefinitely (Ctrl+C to stop)..."
	@echo "Influx training detail: $(TRAFFIC_TEST_INFLUX_DETAIL)"
	$(SUDO_PYTHON) rl_agent_4.py --mode train $(RL_COMMON) $(TRAIN_LOGGING_FLAGS) \
		--training-influx-detail $(TRAFFIC_TEST_INFLUX_DETAIL) \
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
	--generate-traffic --traffic-profile $(profile) \
	--traffic-seed $(TRAFFIC_SEED) \
	--production-influx-write $(PRODUCTION_INFLUX_WRITE)

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

production_all:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "=== Running finite production smoke for all profiles ($(STEPS) steps each) ==="
	@for p in $(PRODUCTION_PROFILES); do \
		echo ""; \
		echo "=== Production profile: $$p ==="; \
		if ! $(SUDO_PYTHON) rl_production.py $(RL_COMMON) \
			--generate-traffic --traffic-profile $$p \
			--traffic-seed $(TRAFFIC_SEED) \
			--production-influx-write $(PRODUCTION_INFLUX_WRITE) \
			--weights-tag best --steps $(STEPS); then \
			echo "Production smoke failed for $$p"; \
			exit 1; \
		fi; \
	done

# Finite RL run with the same step count/traffic seed used by ECMP
rl_compare:
	$(PROD_COMMON) --weights-tag best --steps $(STEPS)

# Queue-independent ECMP baseline for RL comparison
ecmp:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "Traffic profile: $(profile), seed: $(TRAFFIC_SEED)"
	$(SUDO_PYTHON) ecmp_baseline.py --config $(DETECT_TOPOLOGY) \
		--traffic-profile $(profile) --traffic-seed $(TRAFFIC_SEED) \
		--steps $(STEPS) $(LOG_LEVEL_FLAG) $(TELEMETRY_FLAGS)

ecmp_plan:
	$(PYTHON) ecmp_baseline.py --config $(TOPOLOGY_CONFIG) --plan-only

# Deterministic single-shortest-path OSPF/SPF baseline
ospf:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "Traffic profile: $(profile), seed: $(TRAFFIC_SEED)"
	$(SUDO_PYTHON) ospf_baseline.py --config $(DETECT_TOPOLOGY) \
		--traffic-profile $(profile) --traffic-seed $(TRAFFIC_SEED) \
		--steps $(STEPS) $(LOG_LEVEL_FLAG) $(TELEMETRY_FLAGS)

# Paper-oriented paired benchmark: RL vs ECMP and RL vs OSPF
benchmark:
	@echo "Using topology config: $(DETECT_TOPOLOGY)"
	@echo "Profiles: $(BENCH_PROFILES)"
	@echo "Methods: $(BENCH_METHODS)"
	@echo "Repetitions: $(BENCH_REPETITIONS), steps/run: $(STEPS)"
	$(SUDO_PYTHON) benchmark.py --config $(DETECT_TOPOLOGY) \
		--profiles "$(BENCH_PROFILES)" \
		--methods "$(BENCH_METHODS)" \
		--repetitions $(BENCH_REPETITIONS) --steps $(STEPS) \
		--base-seed $(BENCH_BASE_SEED) \
		--warmup-seconds $(BENCH_WARMUP) \
		--cooldown-seconds $(BENCH_COOLDOWN) \
		--min-valid-fraction $(BENCH_MIN_VALID) \
		--max-retries $(BENCH_RETRIES) \
		--weights-tag $(BENCH_WEIGHTS_TAG) \
		--production-influx-write $(PRODUCTION_INFLUX_WRITE) \
		$(BENCH_OUTPUT_FLAG) $(LOG_LEVEL_FLAG) $(TELEMETRY_FLAGS)

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
	@echo "  make clean_bench          Clean benchmark statistics/log artifacts"
	@echo "  make clean_training_logs  Clean durable training logs only"
	@echo ""
	@echo "Available topologies for 'topo=':"
	@echo "  Fat-Tree:    fat_tree_k2, fat_tree_k4 [default], fat_tree_k8"
	@echo "  Leaf-Spine:  leaf_spine_{4,6,8,10,12,14,16}x{2,3,4}"
	@echo "  Three-Tier:  three_tier_4, three_tier_6, three_tier_8"
	@echo ""
	@echo "Training:"
	@echo "  make train_paper          3-stage curriculum: recovery, burst, stability"
	@echo "  make train_stage2         Resume final checkpoint and run stage 2 only"
	@echo "  make train_stage3         Resume final checkpoint and run stage 3 only"
	@echo "  Training artifacts: $(EXTERNAL_ARTIFACT_ROOT)/<run_id>/{logs,checkpoints,collector}/"
	@echo "  TRAIN_INFLUX_DETAIL=minimal|off"
	@echo "  TRAIN_LOG_FLUSH_EVERY=100 EXTERNAL_ARTIFACT_ROOT=/media/sf_amjad/p4_qos_int/training_runs"
	@echo "  TRAINING_STATE_FILE=$(TRAINING_STATE_FILE) COLLECTOR_LOCAL_SPOOL_SPLIT_STEPS=10000"
	@echo "  TRAIN_REPLAY_CAPACITY=$(TRAIN_REPLAY_CAPACITY)"
	@echo "  TRAIN_BASELINE_COOLDOWN=5 TRAIN_WARM_COOLDOWN=3"
	@echo "  Paper plan defaults: $(TRAIN_FOUNDATION_STEPS)+$(TRAIN_BURST_STEPS)+$(TRAIN_POLISH_STEPS) steps"
	@echo "  Epsilon decay defaults: foundation=$(TRAIN_FOUNDATION_EPS_DECAY_STEPS), burst=$(TRAIN_BURST_EPS_DECAY_STEPS), polish=$(TRAIN_POLISH_EPS_DECAY_STEPS)"
	@echo "  Resume epsilon defaults: burst=$(TRAIN_BURST_RESUME_EPS), polish=$(TRAIN_POLISH_RESUME_EPS)"
	@echo "  Override stage weights with TRAIN_FOUNDATION_PROFILE_WEIGHTS,"
	@echo "    TRAIN_BURST_PROFILE_WEIGHTS, or TRAIN_POLISH_PROFILE_WEIGHTS"
	@echo ""
	@echo "Traffic Testing:"
	@echo "  make test_traffic profile=<name>   Run specific profile indefinitely"
	@echo "  Profiles: light_{1,2}, medium_{1,2}, high_{1,2},"
	@echo "            bursty_{vo,vi,be}_{1,2,3}"
	@echo ""
	@echo "Traffic Stress Testing:"
	@echo "  make traffic_stress_test_quick     Quick test (100 cycles, ~10 min)"
	@echo "  make traffic_stress_test           Full test (3600 cycles, ~1 hour)"
	@echo "  make traffic_stress_test_custom CYCLES=500 DURATION=5"
	@echo ""
	@echo "Production Mode:"
	@echo "  make production profile=<name>     Run with best model"
	@echo "  make production_final profile=<name>"
	@echo "  make production_all STEPS=30       Finite smoke over all profiles"
	@echo "  make rl_compare profile=<name> STEPS=300 TRAFFIC_SEED=42"
	@echo "  make ecmp profile=<name> STEPS=300 TRAFFIC_SEED=42"
	@echo "  make ospf profile=<name> STEPS=300 TRAFFIC_SEED=42"
	@echo "  make ecmp_plan topo=<name>  Validate ECMP groups offline"
	@echo "  make benchmark             Paired routing benchmark"
	@echo "    BENCH_PROFILES=$(ALL_TRAFFIC_PROFILES_CSV)"
	@echo "    BENCH_METHODS=rl,ecmp,ospf BENCH_REPETITIONS=6 STEPS=300 BENCH_COOLDOWN=30"
	@echo "  make clean_bench           Remove benchmark_results/ and run logs"
	@echo "  make clean_training_logs   Remove legacy training_files/training_logs contents"
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
	@echo "  make train_stage2 LOG_LEVEL=debug"
	@echo "  make test_traffic profile=high_2"
	@echo "  make production profile=bursty_be_1"

.PHONY: all validate rules run stop clean collect monitor visualize \
        train train_paper train_stage2 train_stage3 test test_best test_traffic \
        traffic_stress_test traffic_stress_test_quick traffic_stress_test_custom \
        production production_best production_final production_75pct production_all \
        rl_compare ecmp ecmp_plan ospf benchmark clean_bench clean-benchmark \
        clean_training_logs clean-training-logs bench help
