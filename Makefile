PCAP_DIR   = pcap
LOG_DIR    = log
P4SRC_DIR  = p4src

ifndef P4SRC_FILE
P4SRC_FILE = p4src/int_md.p4
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
	python3 monitor_iperf_s.py --dir /tmp --window 60 --refresh 2

# run RL agent in training mode (saves weights to training_files)
train:
	python3 rl_agent_3.py --mode train --steps 10000

# run RL agent in evaluation mode (loads *_final.pth from training_files)
test:
	python3 rl_agent_3.py --mode test --weights_dir training_files --tag final --steps 600
