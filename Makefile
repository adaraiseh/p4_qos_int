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
	sudo rm -f $(P4SRC_DIR)/*.p4i $(P4SRC_DIR)/*.json

collect:
	sudo python3 report_collector/influxdb_export.py

monitor:
	python3 monitor_iperf_s.py --dir /tmp --window 60 --refresh 1

# run RL agent v3 in training mode (saves weights to training_files)
train3:
	python3 rl_agent_3.py --mode train --steps 25000

# run RL agent v3 in evaluation mode (loads *_final.pth from training_files)
test3:
	python3 rl_agent_3.py --mode test --weights_dir training_files --tag final --steps 600

# run simplified RL agent in training mode
train_simple:
	python3 rl_agent_simplifed.py --mode train --steps 10000

# run RL agent v4 (recommended) in training mode
train:
	PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode train --steps 30000 --log-every 1 $(VERBOSE_FLAG)

# resume RL agent v4 training from checkpoint (default: 50pct)
# Usage: make resume or make resume CHECKPOINT=best
ifndef CHECKPOINT
CHECKPOINT = 50pct
endif
resume:
	PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode train --steps 30000 --resume $(CHECKPOINT) --log-every 1 $(VERBOSE_FLAG)

# run RL agent v4 in evaluation mode
test:
	PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode eval --steps 1500 --weights-tag final $(VERBOSE_FLAG) --log-every 1

# run RL agent v4 with best model
test_best:
	PYTHONUNBUFFERED=1 python3 -u rl_agent_4.py --mode eval --steps 1500 --weights-tag best $(VERBOSE_FLAG) --log-every 1
