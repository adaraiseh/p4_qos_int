PCAP_DIR   = pcap
LOG_DIR    = log
P4SRC_DIR  = p4src

ifndef P4SRC_FILE
P4SRC_FILE = p4src/int_md.p4
endif

# Controller verbose output (default: suppressed)
# Use VERBOSE=1 to show P4 table add/delete messages
ifdef VERBOSE
VERBOSE_FLAG = --verbose
else
VERBOSE_FLAG =
endif

# by default: start training
all: train

# start network
run:
	sudo python3 network.py --p4 ${P4SRC_FILE}

stop:
	sudo mn -c

clean: stop
	sudo rm -f *.pcap
	sudo rm -rf $(PCAP_DIR) $(LOG_DIR) $(RULE_DIR)/rule*
	sudo rm -f topology.json
	sudo rm -f /tmp/p4_paths.json
	sudo rm -f $(P4SRC_DIR)/*.p4i $(P4SRC_DIR)/*.json

collect:
	sudo python3 report_collector/influxdb_export.py

monitor:
	python3 monitor_iperf_s.py --dir /tmp --window 60 --refresh 1

# run RL agent v4 (recommended) in training mode
# Traffic weights: 5% light, 20% medium, 30% high, 45% bursty
train:
	sudo PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode train --steps 50000 \
		--traffic-weights "light:0.05,medium:0.1,high:0.35,bursty:0.50" \
		--log-every 1 $(VERBOSE_FLAG)

# quick training test - runs 1 episode per training profile (light, medium, high, bursty)
# Tests that the agent can handle all traffic types used in training
train_test:
	@echo "=== Testing all training profiles (1 episode each) ==="
	@for profile in bursty_vo_1 bursty_vi_1 bursty_be_1 high_2 high_1 medium_2 medium_1 light_2 light_1; do \
		echo ""; \
		echo "=== Testing profile: $$profile ==="; \
		if ! sudo PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode train \
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

# resume RL agent v4 training from checkpoint (default: 50pct)
# Usage: make resume or make resume CHECKPOINT=best
# Traffic weights: 5% light, 20% medium, 30% high, 45% bursty
ifndef CHECKPOINT
CHECKPOINT = 50pct
endif
resume:
	sudo PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode train --steps 10000 \
		--resume $(CHECKPOINT) --resume-eps 0.10 \
		--traffic-weights "light:0.05,medium:0.1,high:0.35,bursty:0.50" \
		--log-every 1 $(VERBOSE_FLAG)

# run RL agent v4 in evaluation mode
test:
	PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode eval --steps 1500 --weights-tag final $(VERBOSE_FLAG) --log-every 1

# run RL agent v4 with best model
test_best:
	PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode eval --steps 1500 --weights-tag best $(VERBOSE_FLAG) --log-every 1

# run RL agent v4 in production mode (inference only, comprehensive metrics logging)
# Usage: make production or make production PROFILE=high_2
ifndef PROFILE
PROFILE = high_1
endif
production:
	sudo PYTHONUNBUFFERED=1 python3 -u rl_production.py --weights-tag best \
		--generate-traffic --traffic-profile $(PROFILE) \
		--log-every 1 $(VERBOSE_FLAG)

# run production with final model
production_final:
	sudo PYTHONUNBUFFERED=1 python3 -u rl_production.py --weights-tag final \
		--generate-traffic --traffic-profile $(PROFILE) \
		--log-every 1 $(VERBOSE_FLAG)

# run production with periodic BE bursts (every 60s, 10s-5min duration)
production_bursty:
	sudo PYTHONUNBUFFERED=1 python3 -u rl_production.py --weights-tag final \
		--generate-traffic --traffic-profile medium_1 --bursty-mode \
		--log-every 1 $(VERBOSE_FLAG)
