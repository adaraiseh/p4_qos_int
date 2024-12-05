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

// Insert INT header to the packet
control process_int_source (
    inout headers hdr,
    inout local_metadata_t local_metadata) {

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

    table tb_int_source {
        // AMJ INT source match keys
        key = {
            // configure for each flow to be monitored
            // 4 fields identifying flow
            hdr.ipv4.src_ipv4_addr: lpm;
            //hdr.ipv4.dst_ipv4_addr: ternary;
            //local_metadata.l4_src_port: ternary;
            //local_metadata.l4_dst_port: ternary;
        }
        actions = {
            int_source;
            NoAction;
        }
        const default_action = NoAction();
    }

    apply {
        tb_int_source.apply();
    }
}