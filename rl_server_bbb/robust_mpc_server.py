#!/usr/bin/env python
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import base64
import urllib
import sys
import os
import json
import time
os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
import time
import itertools

################## ROBUST MPC ###################

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
MPC_FUTURE_CHUNK_COUNT = 5
#VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
VIDEO_BIT_RATE = [200, 400, 600, 800, 1000, 1500, 2500, 4000, 8000, 12000] # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 300: 1, 750: 2, 1200: 3, 1850: 12, 2850: 15, 4300: 20}
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 159.0
TOTAL_VIDEO_CHUNKS = 154
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 12  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size download_time reward
NN_MODEL = None

CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

# video chunk sizes
video_sizes = {
    0: [92461, 119747, 99133, 97299, 102502, 82826, 113074, 104896, 102374, 104125, 98982, 89729, 103425, 108546, 104538, 84329, 103707, 121129, 89484, 115167, 100075, 83674, 96049, 107019, 91377, 100505, 114267, 106891, 103440, 96331, 109365, 102073, 86917, 95830, 110176, 108139, 108358, 81581, 109412, 105978, 85890, 123835, 86685, 97890, 100606, 116122, 104386, 102961, 77211, 116999, 89322, 98644, 119848, 107809, 83122, 96791, 97037, 124495, 79539, 102253, 98126, 122416, 101413, 92149, 100567, 103866, 98998, 97144, 115063, 95714, 106225, 87036, 97255, 100493, 127228, 92072, 100129, 92975, 110659, 110954, 94702, 91201, 106815, 108321, 86911, 96291, 112962, 112029, 103684, 96928, 103431, 79962, 116638, 99488, 93817, 101390, 93381, 98256, 119456, 93164, 114823, 94453, 98628, 109994, 88586, 107532, 107834, 98368, 102090, 96625, 92493, 125610, 74523, 97850, 101075, 110553, 105372, 109363, 91042, 101347, 102564, 89995, 107127, 105982, 93902, 105306, 118429, 101519, 100814, 102617, 99166, 99550, 103881, 101606, 102809, 98885, 101094, 103547, 99950, 105237, 94151, 90447, 99744, 102078, 118552, 101530, 100162, 103592, 99864, 103050, 98187, 88476, 84090, 75223, 93021, 107469, 105660, 104187, 65393],
    1: [161338, 234489, 197437, 191951, 205759, 159352, 226463, 208317, 202974, 203214, 202576, 174524, 193219, 226918, 190963, 179455, 202753, 243383, 173447, 233051, 199188, 175049, 176919, 207060, 188652, 197414, 226183, 214596, 206276, 189272, 219237, 198443, 171084, 191008, 221526, 215892, 210735, 158593, 224223, 213045, 163302, 247515, 173290, 193342, 197642, 233519, 207059, 203571, 154257, 235770, 172021, 200004, 236450, 213379, 165157, 191497, 191165, 250603, 159873, 200726, 193410, 249667, 193643, 179944, 198956, 205790, 196838, 195305, 231651, 188447, 211800, 170083, 192954, 199830, 253691, 183264, 203621, 176996, 217301, 223079, 192922, 175503, 217917, 206787, 172651, 190726, 230384, 218347, 205379, 189868, 207051, 158022, 241362, 188735, 188704, 193922, 188510, 194064, 229028, 193953, 228663, 194464, 185685, 222482, 187200, 195203, 222441, 194103, 203576, 180509, 184661, 252323, 149279, 194157, 198949, 211603, 229190, 205193, 182648, 185485, 226771, 167261, 211992, 216208, 180614, 213904, 231420, 200067, 199981, 203661, 197726, 198733, 203430, 198256, 210231, 196856, 202346, 202696, 199758, 204947, 191304, 174361, 199097, 201320, 229099, 210047, 202224, 204320, 200398, 201584, 198297, 167748, 140697, 102284, 145750, 203359, 216545, 218315, 124361],
    2: [252871, 346282, 287350, 295737, 310129, 237276, 338940, 311240, 298461, 312184, 293338, 274685, 282156, 337659, 292418, 264130, 299985, 367604, 266031, 337889, 303335, 256285, 268918, 321607, 269991, 295473, 339001, 317043, 305771, 290091, 326733, 298225, 257697, 287992, 327349, 322080, 314060, 249604, 317858, 320444, 247580, 369293, 255614, 288391, 299028, 350040, 309934, 304944, 228649, 353860, 257054, 298685, 351466, 318364, 247293, 287910, 286288, 370933, 242083, 297255, 291925, 370424, 295706, 265539, 298364, 316654, 288023, 290102, 340718, 287379, 317882, 254375, 287682, 298791, 378352, 274865, 302386, 276693, 314275, 331947, 290274, 264812, 317651, 314376, 261767, 286813, 345526, 322376, 307994, 285599, 311544, 233901, 357344, 288061, 276594, 292538, 282587, 290109, 339329, 303317, 332537, 285818, 281254, 335651, 285174, 284572, 330086, 293242, 301104, 275210, 274557, 379967, 221076, 293446, 295708, 315652, 344985, 301835, 277574, 282255, 330542, 253739, 318192, 319487, 277018, 317882, 344514, 299070, 303035, 304032, 297749, 302582, 296297, 284242, 323633, 295133, 304974, 295437, 304355, 308233, 288272, 256948, 297133, 300413, 315876, 349731, 296170, 302856, 301433, 299618, 297121, 252923, 223765, 165263, 224761, 303526, 325705, 325986, 187198],
    3: [318111, 461860, 395188, 381747, 413292, 317504, 440766, 422987, 402727, 400275, 366532, 385140, 390928, 446971, 401319, 344642, 403377, 486606, 351233, 450199, 408234, 331991, 370874, 445145, 343690, 392870, 451043, 413535, 390910, 410067, 435944, 393896, 349543, 383764, 434909, 428555, 416432, 344348, 394970, 435487, 337916, 487917, 343948, 384474, 400229, 460333, 414705, 405697, 307933, 468033, 346622, 395141, 462858, 429015, 325232, 388137, 381606, 493220, 317165, 399675, 390723, 487380, 399341, 351053, 395848, 423740, 382270, 382411, 454470, 385357, 417730, 343087, 384886, 396341, 502204, 364093, 396205, 373689, 411717, 452554, 383232, 352443, 410185, 428688, 354722, 382067, 451685, 435185, 413194, 374631, 419270, 312820, 472635, 382304, 372971, 386970, 377347, 383964, 446988, 397857, 444307, 365001, 396925, 446981, 382561, 376165, 435262, 398339, 400926, 361424, 368189, 506723, 293017, 390960, 394247, 418019, 459420, 401241, 368470, 378823, 417001, 357210, 429203, 418318, 369944, 411605, 467835, 402044, 400380, 405607, 393657, 404632, 395700, 386398, 421185, 391237, 406619, 395720, 397501, 417615, 381740, 345245, 398829, 400282, 412851, 467297, 393788, 401833, 396885, 405679, 394051, 337574, 299354, 213017, 284809, 401980, 434224, 440123, 247188],
    4: [371770, 578175, 489175, 481847, 509579, 397416, 553036, 524136, 502724, 507026, 455876, 475794, 487920, 562204, 484975, 438170, 503236, 605180, 437800, 565514, 510312, 420980, 454593, 553335, 430217, 489769, 563042, 513277, 492775, 510211, 547210, 492225, 435594, 478831, 544623, 534600, 518509, 422621, 500788, 548123, 417369, 612439, 427918, 476459, 503294, 573643, 519500, 507152, 389342, 588089, 429166, 495935, 575993, 536613, 407293, 484337, 478824, 614270, 396929, 501207, 489859, 611197, 497402, 436191, 492274, 546861, 464455, 478164, 570092, 481531, 518664, 431412, 477510, 495033, 627447, 460466, 499043, 452296, 513152, 573165, 481045, 437176, 520637, 524889, 444105, 474097, 566970, 538433, 517944, 473704, 518418, 390059, 596458, 473742, 464606, 477646, 477103, 480946, 554201, 511033, 550238, 470730, 481185, 553256, 484838, 458390, 552751, 491279, 504334, 444832, 465133, 625622, 370555, 487956, 493100, 522369, 571251, 505866, 459819, 459386, 532084, 443297, 530221, 531660, 455796, 524831, 574850, 498098, 501240, 508323, 493555, 504543, 490506, 479703, 533460, 490283, 504218, 493731, 501111, 517871, 482347, 426816, 497563, 498227, 513093, 590751, 493078, 504427, 495540, 505891, 491384, 415355, 355652, 235473, 334769, 502966, 537065, 553572, 308403],
    5: [528009, 869475, 721563, 738226, 764975, 588907, 821979, 800143, 755377, 755174, 646516, 731817, 739285, 861911, 719819, 647487, 749932, 909114, 655349, 850775, 765252, 644926, 670246, 839789, 641171, 734162, 838516, 761827, 718493, 794586, 823936, 732825, 657224, 721687, 816888, 799085, 771216, 647574, 722208, 836595, 625751, 918248, 640855, 714992, 754904, 848023, 788764, 758711, 574110, 885497, 637063, 746002, 861732, 802645, 613978, 725071, 706347, 920476, 596193, 749166, 733818, 916647, 748629, 649653, 734682, 833161, 681339, 717261, 857736, 724399, 770048, 654612, 707402, 743048, 941918, 687851, 743702, 678241, 755265, 872357, 724306, 651710, 780428, 779080, 678176, 708829, 853776, 806230, 776292, 712628, 774104, 583065, 898696, 701515, 705897, 708760, 720865, 721572, 816922, 775026, 826083, 690673, 739902, 829668, 734861, 668395, 844891, 737455, 745798, 665942, 698572, 935204, 557350, 726555, 737854, 785162, 847778, 760241, 694871, 668307, 810705, 660863, 788955, 799725, 688353, 809976, 828378, 749180, 754304, 755329, 744201, 750534, 738556, 725648, 803420, 733262, 742719, 763468, 738015, 770365, 735608, 628297, 739190, 751216, 760767, 897203, 736517, 752040, 750057, 755225, 741105, 606007, 513066, 324335, 476582, 753983, 805733, 828726, 464560],
    6: [758241, 1468268, 1203120, 1226400, 1286481, 973752, 1386324, 1325789, 1241566, 1266442, 1025877, 1250377, 1239566, 1453047, 1232886, 1029670, 1256767, 1512459, 1113195, 1373360, 1299577, 1053509, 1118270, 1431477, 1030166, 1216644, 1412962, 1244112, 1207232, 1316038, 1379299, 1207359, 1105861, 1206761, 1358236, 1329507, 1281938, 1098983, 1157321, 1411190, 1046946, 1525523, 1073100, 1181075, 1269955, 1401704, 1312136, 1268643, 966188, 1482698, 1056394, 1237209, 1423879, 1342312, 1035484, 1204763, 1165082, 1526738, 996069, 1246797, 1227994, 1527958, 1247841, 1081671, 1212206, 1395933, 1136105, 1190129, 1448994, 1194230, 1260112, 1119598, 1164394, 1231324, 1564978, 1162437, 1234518, 1113860, 1251203, 1470616, 1200805, 1081755, 1299325, 1298989, 1135039, 1178497, 1431746, 1344290, 1282356, 1206588, 1275900, 967374, 1488710, 1181513, 1162729, 1177724, 1202257, 1196270, 1341327, 1327655, 1351718, 1134260, 1253612, 1379234, 1247910, 1080057, 1421103, 1266085, 1202675, 1107432, 1158615, 1567312, 927047, 1194107, 1228274, 1314212, 1429887, 1236444, 1158610, 1094550, 1329695, 1133912, 1303960, 1348042, 1182369, 1386194, 1290429, 1257634, 1249344, 1262395, 1241188, 1247485, 1229112, 1209775, 1314371, 1235580, 1256795, 1242390, 1253437, 1273602, 1222551, 1061846, 1228485, 1239822, 1274911, 1474812, 1240756, 1257852, 1242922, 1254112, 1218096, 1035089, 818910, 466441, 743488, 1274537, 1372091, 1336735, 765544],
    7: [1019520, 2426613, 1919213, 1958125, 2042805, 1572125, 2231830, 2111286, 1994552, 2014269, 1630557, 2012773, 1971522, 2344271, 2011627, 1619338, 2027218, 2397062, 1777140, 2172544, 2110283, 1687689, 1774112, 2309151, 1635706, 1915032, 2276466, 1961563, 1927531, 2116479, 2212939, 1910043, 1794105, 1951990, 2159834, 2117505, 2041468, 1781053, 1803513, 2265333, 1675798, 2420400, 1754894, 1863762, 2052129, 2242638, 2084588, 2028544, 1542050, 2393989, 1669074, 1980959, 2246885, 2167973, 1669870, 1939338, 1846948, 2436399, 1608242, 1979431, 1973020, 2429251, 2008712, 1733825, 1932881, 2279956, 1797294, 1897422, 2319871, 1920067, 1982152, 1901666, 1768970, 1913663, 2476514, 1840367, 2036503, 1740273, 1996239, 2380426, 1922819, 1722341, 2079337, 2058991, 1851755, 1861742, 2314956, 2123165, 2073577, 1885090, 2068433, 1559836, 2365005, 1896587, 1865649, 1854291, 1953483, 1917874, 2129103, 2004642, 2204330, 1749625, 2175968, 2193322, 2008778, 1685225, 2302605, 2033060, 1901142, 1771003, 1866657, 2461675, 1502754, 1908764, 1968289, 2139155, 2243872, 1984124, 1848939, 1713599, 2105632, 1864828, 2081017, 2146688, 1948082, 2203491, 2029064, 2006577, 1999297, 2019644, 1978607, 1961234, 1986367, 2023889, 2051050, 1974292, 2006834, 2007209, 1997677, 2013238, 1985218, 1730213, 1913060, 1979020, 2042051, 2348564, 1990554, 2012146, 1997647, 2006069, 1946035, 1632298, 1239475, 673481, 1122801, 2037121, 2204911, 2144214, 1217352],
    8: [1578944, 4854465, 3907443, 3853580, 4055949, 3168596, 4426736, 4265430, 3977665, 4017937, 3299587, 3980167, 3971747, 4629777, 4182838, 3212187, 4086500, 4725508, 3532301, 4324390, 4238013, 3270718, 3613013, 4438813, 3469791, 3686962, 4591181, 3939203, 3921777, 4090699, 4454412, 3770487, 3690375, 3887700, 4321935, 4198105, 4084529, 3587003, 3521341, 4532146, 3389194, 4852100, 3548455, 3668526, 4218797, 4362754, 4179001, 4058854, 3079188, 4825544, 3308523, 3957048, 4413021, 4436182, 3346445, 3916455, 3659969, 4869199, 3192784, 4017622, 3966119, 4816564, 4027725, 3499428, 3830901, 4584341, 3629815, 3752324, 4671830, 3821683, 3860194, 4033428, 3354815, 3909761, 4957345, 3757432, 4009023, 3488125, 3990390, 4748263, 3828391, 3456155, 4128122, 4066084, 3807626, 3699475, 4649336, 4234418, 4139129, 3761380, 4140477, 3179556, 4663186, 3768582, 3732155, 3769747, 3891641, 3904629, 4274723, 4111360, 4266531, 3513463, 4443367, 4310479, 4046696, 3265835, 4646693, 4083618, 3788793, 3589491, 3719906, 4915838, 2988602, 3848391, 3980620, 4554621, 4315396, 4012025, 3704855, 3402173, 4167011, 3776083, 4161794, 4216903, 3674292, 4629441, 4060192, 4036471, 3988953, 4042680, 3965279, 3970667, 3920391, 4078257, 4018265, 3936205, 4015714, 4014289, 4037445, 4025296, 3984207, 3729942, 3548455, 3975158, 4033429, 4678844, 4048516, 4003257, 3975064, 4021712, 3715950, 3298670, 2120434, 1024053, 2033353, 4324160, 4332903, 4160150, 2387001],
    9: [2290965, 7207392, 5776146, 5718214, 6003330, 4979100, 6396325, 6508557, 5988100, 6080318, 4902068, 5782301, 5937922, 6905273, 6436591, 4731698, 6211771, 7036669, 5264474, 6500236, 6397140, 4795602, 5730230, 6310109, 5595001, 5711551, 6698814, 5528918, 5978234, 6508782, 6714765, 5418881, 5480024, 6084777, 6393825, 6355030, 6275950, 5550130, 5051457, 6595841, 5315356, 7151902, 5251933, 5652121, 6231072, 6576712, 6361248, 6060224, 4616624, 7209216, 4882537, 5940247, 6480716, 6921597, 4916601, 5861466, 5609477, 7273990, 4577928, 6011795, 6017825, 7114498, 6089267, 5282980, 5813687, 6874935, 5304184, 5612639, 7004981, 5725287, 5650612, 6125860, 5182502, 5855418, 7465837, 5418334, 6153043, 5396846, 5904905, 7049225, 5607655, 5285793, 6246694, 5997784, 5806446, 5567993, 6846997, 6402443, 6250104, 5555049, 6283235, 4821835, 6988008, 5628232, 5419079, 5735440, 5775981, 5850956, 6334707, 6499198, 6096200, 5193981, 6532660, 6625896, 6045843, 4959737, 6888906, 6264377, 5613857, 5563867, 5407121, 7427439, 4305852, 5819915, 5897050, 6890339, 6383010, 6072199, 5502399, 5126650, 5987721, 5776544, 6105126, 6303807, 5693330, 6669443, 6081488, 6098728, 6038050, 6095673, 5955530, 6014093, 5808992, 6163154, 5975938, 5405382, 5216806, 5890377, 6294620, 6988692, 5950577, 6044983, 5332743, 5598518, 6095029, 6947127, 6059552, 6003071, 5951504, 5881798, 5039011, 5236144, 2791383, 1400470, 2924068, 6858359, 6211922, 6110854, 3628981],
}

def get_chunk_size(quality, index):
    if ( index < 0 or index > TOTAL_VIDEO_CHUNKS ):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    return video_sizes[quality][index]
    # sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0: size_video6[index]}
    # return sizes[quality]

def make_request_handler(input_dict):

    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.log_file = input_dict['log_file']
            #self.saver = input_dict['saver']
            self.s_batch = input_dict['s_batch']
            #self.a_batch = input_dict['a_batch']
            #self.r_batch = input_dict['r_batch']
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            print post_data

            if ( 'pastThroughput' in post_data ):
                # @Hongzi: this is just the summary of throughput/quality at the end of the load
                # so we don't want to use this information to send back a new quality
                print "Summary: ", post_data
            else:
                # option 1. reward for just quality
                # reward = post_data['lastquality']
                # option 2. combine reward for quality and rebuffer time
                #           tune up the knob on rebuf to prevent it more
                # reward = post_data['lastquality'] - 0.1 * (post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])
                # option 3. give a fixed penalty if video is stalled
                #           this can reduce the variance in reward signal
                # reward = post_data['lastquality'] - 10 * ((post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) > 0)

                # option 4. use the metric in SIGCOMM MPC paper
                rebuffer_time = float(post_data['RebufferTime'] -self.input_dict['last_total_rebuf'])

                # --linear reward--
                reward = VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                        - REBUF_PENALTY * rebuffer_time / M_IN_K \
                        - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                                                  self.input_dict['last_bit_rate']) / M_IN_K

                # --log reward--
                # log_bit_rate = np.log(VIDEO_BIT_RATE[post_data['lastquality']] / float(VIDEO_BIT_RATE[0]))   
                # log_last_bit_rate = np.log(self.input_dict['last_bit_rate'] / float(VIDEO_BIT_RATE[0]))

                # reward = log_bit_rate \
                #          - 4.3 * rebuffer_time / M_IN_K \
                #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

                # --hd reward--
                # reward = BITRATE_REWARD[post_data['lastquality']] \
                #         - 8 * rebuffer_time / M_IN_K - np.abs(BITRATE_REWARD[post_data['lastquality']] - BITRATE_REWARD_MAP[self.input_dict['last_bit_rate']])

                self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]
                self.input_dict['last_total_rebuf'] = post_data['RebufferTime']

                # retrieve previous state
                if len(self.s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(self.s_batch[-1], copy=True)

                # compute bandwidth measurement
                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                if video_chunk_fetch_time == 0:
                    video_chunk_fetch_time = 1
                video_chunk_size = post_data['lastChunkSize']

                # compute number of video chunks left
                video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.input_dict['video_chunk_coount']
                self.input_dict['video_chunk_coount'] += 1

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                try:
                    state[0, -1] = VIDEO_BIT_RATE[post_data['lastquality']] / float(np.max(VIDEO_BIT_RATE))
                    state[1, -1] = post_data['buffer'] / BUFFER_NORM_FACTOR
                    state[2, -1] = rebuffer_time / M_IN_K
                    state[3, -1] = float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K  # kilo byte / ms
                    state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
                    curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
                    if ( len(past_bandwidth_ests) > 0 ):
                        curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
                    past_errors.append(curr_error)
                except ZeroDivisionError:
                    # this should occur VERY rarely (1 out of 3000), should be a dash issue
                    # in this case we ignore the observation and roll back to an eariler one
                    past_errors.append(0)
                    if len(self.s_batch) == 0:
                        state = [np.zeros((S_INFO, S_LEN))]
                    else:
                        state = np.array(self.s_batch[-1], copy=True)

                # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
                self.log_file.write(str(time.time()) + '\t' +
                                    str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                                    str(post_data['buffer']) + '\t' +
                                    str(rebuffer_time / M_IN_K) + '\t' +
                                    str(video_chunk_size) + '\t' +
                                    str(video_chunk_fetch_time) + '\t' +
                                    str(reward) + '\n')
                self.log_file.flush()

                # pick bitrate according to MPC           
                # first get harmonic mean of last 5 bandwidths
                past_bandwidths = state[3,-5:]
                print "past_bandwidths", past_bandwidths
                while past_bandwidths[0] == 0.0:
                    past_bandwidths = past_bandwidths[1:]
                #if ( len(state) < 5 ):
                #    past_bandwidths = state[3,-len(state):]
                #else:
                #    past_bandwidths = state[3,-5:]
                bandwidth_sum = 0
                for past_val in past_bandwidths:
                    bandwidth_sum += (1/float(past_val))
                harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

                # future bandwidth prediction
                # divide by 1 + max of last 5 (or up to 5) errors
                max_error = 0
                error_pos = -5
                if ( len(past_errors) < 5 ):
                    error_pos = -len(past_errors)
                max_error = float(max(past_errors[error_pos:]))
                future_bandwidth = harmonic_bandwidth/(1+max_error)
                past_bandwidth_ests.append(harmonic_bandwidth)


                # future chunks length (try 4 if that many remaining)
                last_index = int(post_data['lastRequest'])
                future_chunk_length = MPC_FUTURE_CHUNK_COUNT
                if ( TOTAL_VIDEO_CHUNKS - last_index < 5 ):
                    future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

                # all possible combinations of 5 chunk bitrates (9^5 options)
                # iterate over list and for each, compute reward and store max reward combination
                max_reward = -100000000
                best_combo = ()
                start_buffer = float(post_data['buffer'])
                #start = time.time()
                for full_combo in CHUNK_COMBO_OPTIONS:
                    combo = full_combo[0:future_chunk_length]
                    # calculate total rebuffer time for this combination (start with start_buffer and subtract
                    # each download time and add 2 seconds in that order)
                    curr_rebuffer_time = 0
                    curr_buffer = start_buffer
                    bitrate_sum = 0
                    smoothness_diffs = 0
                    last_quality = int(post_data['lastquality'])
                    for position in range(0, len(combo)):
                        chunk_quality = combo[position]
                        index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
                        download_time = (get_chunk_size(chunk_quality, index)/1000000.)/future_bandwidth # this is MB/MB/s --> seconds
                        if ( curr_buffer < download_time ):
                            curr_rebuffer_time += (download_time - curr_buffer)
                            curr_buffer = 0
                        else:
                            curr_buffer -= download_time
                        curr_buffer += 4
                        
                        # linear reward
                        bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                        smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])

                        # log reward
                        # log_bit_rate = np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                        # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0]))
                        # bitrate_sum += log_bit_rate
                        # smoothness_diffs += abs(log_bit_rate - log_last_bit_rate)

                        # hd reward
                        #bitrate_sum += BITRATE_REWARD[chunk_quality]
                        #smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])

                        last_quality = chunk_quality
                    # compute reward for this combination (one reward per 5-chunk combo)
                    # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
                    
                    # linear reward 
                    reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time) - (smoothness_diffs/1000.)

                    # log reward
                    # reward = (bitrate_sum) - (REBUG_PENALTY*curr_rebuffer_time) - (smoothness_diffs)

                    # hd reward
                    #reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)

                    if ( reward > max_reward ):
                        max_reward = reward
                        best_combo = combo
                # send data to html side (first chunk of best combo)
                send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
                if ( best_combo != () ): # some combo was good
                    send_data = str(best_combo[0])

                end = time.time()
                #print "TOOK: " + str(end-start)
                print send_data

                end_of_video = False
                if ( post_data['lastRequest'] == TOTAL_VIDEO_CHUNKS ):
                    send_data = "REFRESH"
                    end_of_video = True
                    self.input_dict['last_total_rebuf'] = 0
                    self.input_dict['last_bit_rate'] = DEFAULT_QUALITY
                    self.input_dict['video_chunk_coount'] = 0
                    self.log_file.write('\n')  # so that in the log we know where video ends

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', len(send_data))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data)

                # record [state, action, reward]
                # put it here after training, notice there is a shift in reward storage

                if end_of_video:
                    self.s_batch = [np.zeros((S_INFO, S_LEN))]
                else:
                    self.s_batch.append(state)

        def do_GET(self):
            print >> sys.stderr, 'GOT REQ'
            self.send_response(200)
            #self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', 20)
            self.end_headers()
            self.wfile.write("console.log('here');")

        def log_message(self, format, *args):
            return

    return Request_Handler


def run(server_class=HTTPServer, port=8333, log_file_path=LOG_FILE):

    np.random.seed(RANDOM_SEED)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # make chunk combination options
    for combo in itertools.product(range(len(VIDEO_BIT_RATE)), repeat=MPC_FUTURE_CHUNK_COUNT):
        CHUNK_COMBO_OPTIONS.append(combo)

    with open(log_file_path, 'wb') as log_file:

        s_batch = [np.zeros((S_INFO, S_LEN))]

        last_bit_rate = DEFAULT_QUALITY
        last_total_rebuf = 0
        # need this storage, because observation only contains total rebuffering time
        # we compute the difference to get

        video_chunk_count = 0

        input_dict = {'log_file': log_file,
                      'last_bit_rate': last_bit_rate,
                      'last_total_rebuf': last_total_rebuf,
                      'video_chunk_coount': video_chunk_count,
                      's_batch': s_batch}

        # interface to abr_rl server
        handler_class = make_request_handler(input_dict=input_dict)

        server_address = ('localhost', port)
        httpd = server_class(server_address, handler_class)
        print 'Listening on port ' + str(port)
        httpd.serve_forever()


def main():
    if len(sys.argv) == 2:
        trace_file = sys.argv[1]
        run(log_file_path=LOG_FILE + '_robustMPC_' + trace_file)
    else:
        run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print "Keyboard interrupted."
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
