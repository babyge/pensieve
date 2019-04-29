import sys, os
import subprocess

base_website = "https://dash.akamaized.net/akamai/bbb_30fps/"

num_slices = 160

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

file_name_template = "{representation}/{representation}_{number}.m4v"

ps = []
for r in representations:
    for i in range(num_slices):
        file_name = file_name_template.format(representation=r, number=(i))
        download_link = base_website + file_name
        folder, fn = file_name.split('/')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        if os.path.isfile(fn):
            os.rename(fn, file_name)
            continue
        if os.path.isfile(file_name):
            continue
        p = subprocess.Popen(['sh', '-c', 'wget ' + download_link])
        ps.append(p)

for p in ps:
    p.wait()
