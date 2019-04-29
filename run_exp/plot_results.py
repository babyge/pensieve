import os
import numpy as np
import matplotlib.pyplot as plt


RESULTS_FOLDER = './results/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 64
VIDEO_BIT_RATE = [200, 400, 600, 800, 1000, 1500, 2500, 4000, 8000, 12000] # Kbps
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired 
SIM_DP = 'sim_dp'
SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', "robustMPC", "fastMPC"]#'RL', ]# 'sim_rl', SIM_DP]

PACKET_SIZE = 1500.0  # bytes
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0
N = 100

trace_folder = '../cooked_traces/'
def get_bandwidth(trace_name, base_timestamp):
    time_all = []
    packet_sent_all = []
    last_time_stamp = 0
    packet_sent = 0
    file_name = os.path.join(trace_folder, trace_name)
    with open(file_name, 'rb') as f:
            for line in f:
                    time_stamp = int(line.split()[0])
                    if time_stamp == last_time_stamp:
                            packet_sent += 1
                            continue
                    else:
                            time_all.append(last_time_stamp)
                            packet_sent_all.append(packet_sent)
                            packet_sent = 1
                            last_time_stamp = time_stamp

    time_window = np.array(time_all[1:]) - np.array(time_all[:-1])
    throuput_all = PACKET_SIZE * \
                               BITS_IN_BYTE * \
                               np.array(packet_sent_all[1:]) / \
                               time_window * \
                               MILLISECONDS_IN_SECONDS / \
                               MBITS_IN_BITS

    x = np.array(time_all[1:]) / MILLISECONDS_IN_SECONDS
    y = np.convolve(throuput_all, np.ones(N,)/N, mode='same')
    return x, y

def main():
    time_all = {}
    bit_rate_all = {}
    buff_all = {}
    bw_all = {}
    raw_reward_all = {}

    for scheme in SCHEMES:
        time_all[scheme] = {}
        raw_reward_all[scheme] = {}
        bit_rate_all[scheme] = {}
        buff_all[scheme] = {}
        bw_all[scheme] = {}

    log_files = os.listdir(RESULTS_FOLDER)
    for log_file in log_files:

        time_ms = []
        bit_rate = []
        buff = []
        bw = []
        reward = []

        print log_file

        with open(RESULTS_FOLDER + log_file, 'rb') as f:
            if SIM_DP in log_file:
                for line in f:
                    parse = line.split()
                    if len(parse) == 1:
                        reward = float(parse[0])
                    elif len(parse) >= 6:
                        time_ms.append(float(parse[3]))
                        bit_rate.append(VIDEO_BIT_RATE[int(parse[6])])
                        buff.append(float(parse[4]))
                        bw.append(float(parse[5]))

            else:
                for line in f:
                    parse = line.split()
                    if len(parse) <= 1:
                        break
                    time_ms.append(float(parse[0]))
                    bit_rate.append(int(parse[1]))
                    buff.append(float(parse[2]))
                    bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                    reward.append(float(parse[6]))

        if SIM_DP in log_file:
            time_ms = time_ms[::-1]
            bit_rate = bit_rate[::-1]
            buff = buff[::-1]
            bw = bw[::-1]
        
        time_ms = np.array(time_ms)
        time_ms -= time_ms[0]
        
        # print log_file

        for scheme in SCHEMES:
            if scheme in log_file:
                time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
                bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
                buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
                bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
                raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
                break

    # ---- ---- ---- ----
    # Reward records
    # ---- ---- ---- ----
        
    log_file_all = []
    reward_all = {}
    for scheme in SCHEMES:
        reward_all[scheme] = []

    for l in time_all[SCHEMES[0]]:
        schemes_check = True
        for scheme in SCHEMES:
            if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
                schemes_check = False
                break
        if schemes_check:
            log_file_all.append(l)
            for scheme in SCHEMES:
                if scheme == SIM_DP:
                    reward_all[scheme].append(raw_reward_all[scheme][l])
                else:
                    reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]))

    mean_rewards = {}
    for scheme in SCHEMES:
        mean_rewards[scheme] = np.mean(reward_all[scheme])

    SCHEMES_REW = []
    for scheme in SCHEMES:
        SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme]))

    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(SCHEMES))]

    fig = plt.figure(figsize=(15,10))

    for idx, scheme in enumerate(SCHEMES):
        plt.plot(reward_all[scheme], label=SCHEMES_REW[idx], color=colors[idx])

    plt.legend(bbox_to_anchor=(0, 1), loc='lower left', ncol=4)
    
    plt.ylabel('total reward')
    plt.xlabel('trace index')
    plt.savefig('reward.png')

    # ---- ---- ---- ----
    # CDF 
    # ---- ---- ---- ----

    fig = plt.figure(figsize=(15,10))

    for idx, scheme in enumerate(SCHEMES):
        values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
        cumulative = np.cumsum(values)
        plt.plot(base[:-1], cumulative, label=SCHEMES_REW[idx], color=colors[idx])

    plt.legend(bbox_to_anchor=(0, 1), loc='lower left', ncol=4)
    
    plt.ylabel('CDF')
    plt.xlabel('total reward')
    plt.savefig("cdf.png")

    # ---- ---- ---- ----
    # check each trace
    # ---- ---- ---- ----

    print time_all.keys()

    for l in time_all[SCHEMES[0]]:
        print "l", l
        for scheme in SCHEMES:
            #print scheme
            #print time_all[scheme].keys()
            #print bit_rate_all[scheme].keys()
            if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
                schemes_check = False
        schemes_check = True
        if schemes_check:

            SCHEMES_REW = []
            for scheme in SCHEMES:
                if scheme == SIM_DP:
                    SCHEMES_REW.append(scheme + ': ' + str(raw_reward_all[scheme][l]))
                else:
                    SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))

            max_x = max(time_all[scheme][l][:VIDEO_LEN])

            fig = plt.figure(figsize=(15, 15))

            ax = fig.add_subplot(411)
            ax.set_xlim(0, max_x)
            for scheme in SCHEMES:
                ax.plot(time_all[scheme][l][:VIDEO_LEN], bit_rate_all[scheme][l][:VIDEO_LEN])
            colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
            for i,j in enumerate(ax.lines):
                j.set_color(colors[i])    
            plt.ylabel('bit rate selection (kbps)')
            plt.title(l, y=1.2)

            ax.legend(SCHEMES_REW, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=int(np.ceil(len(SCHEMES) / 2.0)))

            ax = fig.add_subplot(412)
            ax.set_xlim(0, max_x)
            for scheme in SCHEMES:
                ax.plot(time_all[scheme][l][:VIDEO_LEN], buff_all[scheme][l][:VIDEO_LEN])
            colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
            for i,j in enumerate(ax.lines):
                j.set_color(colors[i])    
            plt.ylabel('buffer size (sec)')

            ax = fig.add_subplot(413)
            ax.set_xlim(0, max_x)
            ax.set_ylim(0, 50)
            for scheme in SCHEMES:
                ax.plot(time_all[scheme][l][:VIDEO_LEN], bw_all[scheme][l][:VIDEO_LEN])
            colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
            for i,j in enumerate(ax.lines):
                j.set_color(colors[i])    
            plt.ylabel('bandwidth (mbps)')
            plt.xlabel('time (sec)')

            ax = fig.add_subplot(414)

            ax.set_xlim(0, max_x)
            x, y = get_bandwidth(l)
            #print("bandwidth:", min(x), max(x))
            ax.plot(x, y)

            plt.ylabel('Real available bandwidth (Mbit/s)')

            print(l + '.png')
            plt.savefig(l + '.png', bbox_inches='tight')
            #plt.show()


if __name__ == '__main__':
    main()
