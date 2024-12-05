#include "defines.p4"
#include "headers.p4"

control l3_forward(inout headers hdr,
                       inout local_metadata_t local_metadata,
                       inout standard_metadata_t standard_metadata) {

    action drop(){
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(mac_t dstAddr, port_t port) {
        standard_metadata.egress_spec = port;
        standard_metadata.egress_port = port;
        hdr.ethernet.src_addr = hdr.ethernet.dst_addr;
        hdr.ethernet.dst_addr = dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
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
                standard_metadata.priority = (bit<3>)3;
            } else if (hdr.ipv4.dscp == 0x18) { // CS3
                standard_metadata.priority = (bit<3>)2;
            } else if (hdr.ipv4.dscp == 0x12 || hdr.ipv4.dscp == 0x14 || hdr.ipv4.dscp == 0x16) { // AF21, AF22, AF23
                standard_metadata.priority = (bit<3>)1;
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

    action send_to_cpu() {
        standard_metadata.egress_port = CPU_PORT;
        standard_metadata.egress_spec = CPU_PORT;
    }

    action set_egress_port(port_t port) {
        standard_metadata.egress_port = port;
        standard_metadata.egress_spec = port;
    }

    action drop(){
        mark_to_drop(standard_metadata);
    }

    table tb_port_forward {
        key = {
            hdr.ipv4.dst_ipv4_addr: lpm;
            hdr.ipv4.dscp: exact;
        }
        actions = {
            set_egress_port;
            send_to_cpu;
            drop;
        }
        const default_action = drop();
    }

    apply {
        tb_port_forward.apply();
     }
}
