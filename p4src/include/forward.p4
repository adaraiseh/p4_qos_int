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
            ipv4_lpm.apply();
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
