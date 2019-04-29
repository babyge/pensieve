import sys, os

bbb_home_folder = "/var/www/html/"
representations = [
    "bbb_30fps_1024x576_2500k",
    "bbb_30fps_1280x720_4000k",
    "bbb_30fps_1920x1080_8000k",
    "bbb_30fps_320x180_200k",
    "bbb_30fps_320x180_400k",
    "bbb_30fps_480x270_600k",
    "bbb_30fps_640x360_1000k",
    "bbb_30fps_640x360_800k",
    "bbb_30fps_768x432_1500k",
    "bbb_30fps_3840x2160_12000k"
]

if len(sys.argv) > 1:
    bbb_folder = sys.argv[1]


for r in representations:
    bbb_folder = bbb_home_folder + r
    file_sizes = [os.path.getsize(os.path.join(bbb_folder, f)) for f in sorted(os.listdir(bbb_folder), key = lambda x: int(x.split('.')[0].split('_')[-1]))]
    print(file_sizes)
