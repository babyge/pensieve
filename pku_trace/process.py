import sys, os
import subprocess
import dateutil
import dateutil.parser

THRESH = 1400

def decompress_pcap(pcap_file_name):
    output_file_name = pcap_file_name + ".txt"
    if not os.path.isfile(output_file_name):
        subprocess.check_call(['sh', '-c', 'tshark -r %s -T fields -E separator=@ -e frame.time -e frame.len -e ip.src -e ip.dst > %s' % (pcap_file_name, output_file_name)])
    return output_file_name

def load_text_log(output_file_name):
    send_ip, recv_ip = None, None
    send_log, recv_log = [], []
    with open(output_file_name, 'r') as fp:
        for line in fp:
            timestamp, length, src_ip, dst_ip = line.split('@')
            timestamp = dateutil.parser.parse(timestamp)
            length = int(length)
            if send_ip == None:
                send_ip = src_ip
                recv_ip = dst_ip
            if src_ip == send_ip:
                send_log.append((timestamp, length))
            elif src_ip == recv_ip:
                recv_log.append((timestamp, length))

    return send_log, recv_log

def write_large_log(send_log, recv_log, write_file_name):
    send_pkts = sum([i[1] for i in send_log])
    recv_pkts = sum([i[1] for i in recv_log])
    if send_pkts > recv_pkts:
        write_log = send_log
    else:
        write_log = recv_log

    start_time = min([i[0] for i in write_log])
    total_pkts = 0
    with open(write_file_name, 'w') as fp:
        for ts, length in sorted(write_log, key=lambda x : x[0]):
            total_pkts += length
            if total_pkts > THRESH:
                fp.write("%d\n" % ((ts - start_time).total_seconds() * 1000))
                total_pkts = 0


pcap_file_name = sys.argv[1]

output_file_name = decompress_pcap(pcap_file_name)

send_log, recv_log = load_text_log(output_file_name)

write_file_name = output_file_name + ".mahimahi"
write_large_log(send_log, recv_log, write_file_name)
