#include "defines.p4"
#include "headers.p4"

control l3_forward(inout headers hdr,
                       inout local_metadata_t local_metadata,
                       inout standard_metadata_t standard_metadata) {

    action drop(){
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(ip_address_t nextHop, port_t port) {
        standard_metadata.egress_spec = port;
        standard_metadata.egress_port = port;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
        local_metadata.routing.nhop_ipv4 = nextHop;
    }

    action set_ecmp_group(bit<16> group_id, bit<16> path_count) {
        local_metadata.routing.ecmp_group = group_id;

        // Deliberately exclude DSCP and L4 ports from the hash. The traffic
        // generator encodes the QoS queue in both fields, so including either
        // would make ECMP queue-dependent. A per-group salt gives each switch
        // an independent choice while keeping all queues of the same demand
        // on the same path.
        hash(
            local_metadata.routing.ecmp_select,
            HashAlgorithm.crc16,
            (bit<16>)0,
            {
                hdr.ipv4.src_ipv4_addr,
                hdr.ipv4.dst_ipv4_addr,
                group_id
            },
            path_count
        );
    }

    table ecmp_group {
        key = {
            hdr.ipv4.dst_ipv4_addr : lpm;
        }
        actions = {
            set_ecmp_group;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    table ecmp_nhop {
        key = {
            local_metadata.routing.ecmp_group : exact;
            local_metadata.routing.ecmp_select : exact;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 2048;
        default_action = drop();
    }

    table ipv4_lpm {
        key = {
            hdr.ipv4.dst_ipv4_addr : lpm;
            hdr.ipv4.dscp: exact;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = drop();
    }

    apply {
        if(hdr.ipv4.isValid()) {
            if (hdr.ipv4.dscp == 0x2E) { // EF
                standard_metadata.priority = (bit<3>)7;
            } else if (hdr.ipv4.dscp == 0x18) { // CS3
                standard_metadata.priority = (bit<3>)6;
            } else if (hdr.ipv4.dscp == 0x12 || hdr.ipv4.dscp == 0x14 || hdr.ipv4.dscp == 0x16) { // AF21, AF22, AF23
                standard_metadata.priority = (bit<3>)5;
            } else {
                // Best Effort (default priority)
                standard_metadata.priority = (bit<3>)0;
            }

            // ECMP is an optional queue-agnostic overlay. If no ECMP group is
            // installed, preserve the existing QoS-aware LPM behavior used by
            // the RL controller.
            if (ecmp_group.apply().hit) {
                ecmp_nhop.apply();
            } else {
                ipv4_lpm.apply();
            }
        }
            
    }
}

control port_forward(inout headers hdr,
                       inout local_metadata_t local_metadata,
                       inout standard_metadata_t standard_metadata) {

    action drop(){
        mark_to_drop(standard_metadata);
    }

    action set_dmac(mac_t mac) {
        hdr.ethernet.dst_addr = mac;
    }

    action set_smac(mac_t mac) {
        hdr.ethernet.src_addr = mac;
    }

    table switching_table {
        key = {
            local_metadata.routing.nhop_ipv4 : exact;
        }
        actions = {
            set_dmac;
            drop;
            NoAction;
        }
        default_action = NoAction();
    }

    table mac_rewriting_table {
        key = {
            standard_metadata.egress_spec: exact;
        }
        actions = {
            set_smac;
            drop;
            NoAction;
        }
        default_action = NoAction();
    }

    apply {
        switching_table.apply();
        mac_rewriting_table.apply();
    }
}
