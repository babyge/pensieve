import sys
import os

mahi_trace_name = sys.argv[1]
output_trace_name = mahi_trace_name + ".w_bw"
merge_interval = 0.3
granularity = 1000.0
pkt_size = 1400 * 8 / 1e6

pkt_timestamp = []
with open(mahi_trace_name, 'r') as fp:
    for line in fp:
        pkt_timestamp.append(int(line.strip()))

output_throughput = []
cur_timestamp = 0
pkt_counter = 0
for ts in pkt_timestamp:
    if ts < cur_timestamp + merge_interval * granularity:
        pkt_counter += 1
    else:
        output_throughput.append((cur_timestamp / granularity,
                                 pkt_counter * pkt_size / merge_interval))
        cur_timestamp = ts
        pkt_counter = 0

with open(output_trace_name, 'w') as fp:
    for ts, throughput in output_throughput:
        fp.write("%s\t%s\n" % (ts, throughput))
