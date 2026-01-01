/* -*- P4_16 -*- */
control process_int_source_sink (
    inout headers hdr,
    inout local_metadata_t local_metadata,
    inout standard_metadata_t standard_metadata) {

    action int_set_source () {
        local_metadata.int_meta.source = true;
    }

    action int_set_sink () {
        local_metadata.int_meta.sink = true;
    }

    table tb_set_source {
        key = {
            standard_metadata.ingress_port: exact;
        }
        actions = {
            int_set_source;
            NoAction();
        }
        const default_action = NoAction();
        size = MAX_PORTS;
    }

    table tb_set_sink {
        key = {
            standard_metadata.egress_port: exact;
        }
        actions = {
            int_set_sink;
            NoAction();
        }
        const default_action = NoAction();
        size = MAX_PORTS;
    }

    apply {
        tb_set_source.apply();
        tb_set_sink.apply();
    }
}

// Insert INT header to the packet (with per-flow, per-queue time-based sampling)
control process_int_source (
    inout headers hdr,
    inout local_metadata_t local_metadata,
    inout standard_metadata_t standard_metadata) {

    // Sampling configuration - 300ms per flow per queue
    const bit<64> TIME_THRESHOLD_US = 200000;  // 200ms = 200,000 microseconds

    // Per-flow, per-queue sampling register
    // Index = flow_id * 8 + queue_idx
    // flow_id range: 10-99 (90 flows max), queue_idx: 0-7
    // Size: 100 * 8 = 800 entries (using 1024 for safety)
    register<bit<64>>(1024) int_flow_queue_sample_time;

    // Metadata for sampling
    bit<32> reg_index;
    bit<64> last_time;
    bit<64> elapsed;

    // Store action parameters for use in apply block
    bit<5> stored_hop_metadata_len;
    bit<8> stored_remaining_hop_cnt;
    bit<4> stored_ins_mask0003;
    bit<4> stored_ins_mask0407;

    // Action to store parameters and mark that table matched
    action int_source_sampled(bit<5> hop_metadata_len, bit<8> remaining_hop_cnt, bit<4> ins_mask0003, bit<4> ins_mask0407) {
        // Store parameters for later use in apply block
        stored_hop_metadata_len = hop_metadata_len;
        stored_remaining_hop_cnt = remaining_hop_cnt;
        stored_ins_mask0003 = ins_mask0003;
        stored_ins_mask0407 = ins_mask0407;
    }

    // Original action (kept for backwards compatibility - no sampling)
    action int_source(bit<5> hop_metadata_len, bit<8> remaining_hop_cnt, bit<4> ins_mask0003, bit<4> ins_mask0407) {
        // insert INT shim header
        hdr.intl4_shim.setValid();
        hdr.intl4_shim.int_type = 1;                            // int_type: Hop-by-hop type (1) , destination type (2), MX-type (3)
        hdr.intl4_shim.npt = 0;                                 // next protocol type: 0
        hdr.intl4_shim.len = INT_HEADER_WORD;                   // This is 3 from 0xC (INT_TOTAL_HEADER_SIZE >> 2)
        hdr.intl4_shim.udp_ip_dscp = hdr.ipv4.dscp;             // Store original DSCP value
        hdr.intl4_shim.udp_ip_ecn = hdr.ipv4.ecn;               // Store original ECN bits
        hdr.ipv4.ecn = hdr.ipv4.ecn | INT_ECN_BIT;              // Set INT bit in DSCP field
        hdr.intl4_shim.rsvd2 = 0;

        // insert INT header
        hdr.int_header.setValid();
        hdr.int_header.ver = 2;
        hdr.int_header.d = 0;
        hdr.int_header.e = 0;
        hdr.int_header.m = 0;
        hdr.int_header.rsvd = 0;
        hdr.int_header.hop_metadata_len = hop_metadata_len;
        hdr.int_header.remaining_hop_cnt = remaining_hop_cnt;
        hdr.int_header.instruction_mask_0003 = ins_mask0003;
        hdr.int_header.instruction_mask_0407 = ins_mask0407;
        hdr.int_header.instruction_mask_0811 = 0; // not supported
        hdr.int_header.instruction_mask_1215 = 0; // not supported

        hdr.int_header.domain_specific_id = 0;                  // Unique INT Domain ID
        hdr.int_header.ds_instruction = 0;                      // Instruction bitmap specific to the INT Domain identified by the Domain specific ID
        hdr.int_header.ds_flags = 0;                            // Domain specific flags

        // add the header len (3 words) to total len
        hdr.ipv4.len = hdr.ipv4.len + INT_TOTAL_HEADER_SIZE;

        if(hdr.udp.isValid()) {
            hdr.udp.length_ = hdr.udp.length_ + INT_TOTAL_HEADER_SIZE;
        }
    }

    // Action to actually insert INT headers (called from apply block)
    action do_insert_int() {
        // Insert INT shim header
        hdr.intl4_shim.setValid();
        hdr.intl4_shim.int_type = 1;
        hdr.intl4_shim.npt = 0;
        hdr.intl4_shim.len = INT_HEADER_WORD;
        hdr.intl4_shim.udp_ip_dscp = hdr.ipv4.dscp;
        hdr.intl4_shim.udp_ip_ecn = hdr.ipv4.ecn;
        hdr.ipv4.ecn = hdr.ipv4.ecn | INT_ECN_BIT;
        hdr.intl4_shim.rsvd2 = 0;

        // Insert INT header
        hdr.int_header.setValid();
        hdr.int_header.ver = 2;
        hdr.int_header.d = 0;
        hdr.int_header.e = 0;
        hdr.int_header.m = 0;
        hdr.int_header.rsvd = 0;
        hdr.int_header.hop_metadata_len = stored_hop_metadata_len;
        hdr.int_header.remaining_hop_cnt = stored_remaining_hop_cnt;
        hdr.int_header.instruction_mask_0003 = stored_ins_mask0003;
        hdr.int_header.instruction_mask_0407 = stored_ins_mask0407;
        hdr.int_header.instruction_mask_0811 = 0;
        hdr.int_header.instruction_mask_1215 = 0;
        hdr.int_header.domain_specific_id = 0;
        hdr.int_header.ds_instruction = 0;
        hdr.int_header.ds_flags = 0;

        // Update lengths
        hdr.ipv4.len = hdr.ipv4.len + INT_TOTAL_HEADER_SIZE;

        if(hdr.udp.isValid()) {
            hdr.udp.length_ = hdr.udp.length_ + INT_TOTAL_HEADER_SIZE;
        }
    }

    table tb_int_source {
        key = {
            // configure for each flow to be monitored
            // 4 fields identifying flow
            hdr.ipv4.src_ipv4_addr: lpm;
            local_metadata.l4_dst_port: ternary;
        }
        actions = {
            int_source;
            int_source_sampled;
            NoAction;
        }
        const default_action = NoAction();
    }

    apply {
        // Initialize
        stored_hop_metadata_len = 0;
        stored_remaining_hop_cnt = 0;
        stored_ins_mask0003 = 0;
        stored_ins_mask0407 = 0;

        // Apply table - if int_source_sampled is called, it stores params
        // if int_source is called, it directly inserts INT headers (no sampling)
        if (tb_int_source.apply().hit) {
            // Check if we're using the sampled action (params were stored)
            if (stored_hop_metadata_len != 0) {
                bit<64> now = standard_metadata.ingress_global_timestamp;

                // Use dst_port directly as register index for per-flow, per-queue sampling
                // dst_port encoding: 6000 + flow_id * 10 + queue_id
                // Each unique (flow_id, queue_id) combination has a unique port
                // Use lower 10 bits to map to register index 0-1023
                reg_index = (bit<32>)(local_metadata.l4_dst_port & 0x3FF);

                // Read last sample time for this flow+queue combination
                int_flow_queue_sample_time.read(last_time, reg_index);
                elapsed = now - last_time;

                // Sample if first packet (last_time == 0) or threshold elapsed
                if (last_time == 0 || elapsed >= TIME_THRESHOLD_US) {
                    int_flow_queue_sample_time.write(reg_index, now);
                    do_insert_int();
                }
            }
        }
    }
}
