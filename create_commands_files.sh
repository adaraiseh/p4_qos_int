#!/bin/bash

# Define the directory for the command files
CONFIG_DIR="rules/test/"

# Create the directory if it doesn't exist
mkdir -p "$CONFIG_DIR"

# Define the switches
TOR_SWITCHES=("t1" "t2" "t3" "t4" "t5" "t6" "t7" "t8")
AGG_SWITCHES=("a1" "a2" "a3" "a4" "a5" "a6" "a7" "a8")
CORE_SWITCHES=("c1" "c2" "c3" "c4")

# Function to create empty command files
create_command_files() {
  local switches=("$@")
  for switch in "${switches[@]}"; do
    touch "$CONFIG_DIR/${switch}-commands.txt"
    echo "Created $CONFIG_DIR/${switch}-commands.txt"
  done
}

# Create empty command files for all switches
create_command_files "${TOR_SWITCHES[@]}"
create_command_files "${AGG_SWITCHES[@]}"
create_command_files "${CORE_SWITCHES[@]}"

echo "All configuration command files have been created in $CONFIG_DIR."