import sys, os

trace_file_name = sys.argv[1]
pkt_size = 1424

pkt_timestamps = []

with open(trace_file_name, 'r') as fp:
    for line in fp:
        pkt_timestamps.append(int(line.strip()))
