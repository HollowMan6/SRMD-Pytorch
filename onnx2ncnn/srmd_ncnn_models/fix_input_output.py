#!/usr/bin/python3
# -*- coding:utf-8 -*-
# by 'hollowman6' from Lanzhou University(兰州大学)

filelist = ["srmd_x2.param", "srmd_x3.param", "srmd_x4.param", "srmdnf_x2.param", "srmdnf_x2.param", "srmdnf_x2.param"]

for file in filelist:
    lines = []
    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines[-1] = lines[-1].replace("Reshape_27","output    ").replace("52","output")
        lines[2] = lines[2].replace("input.1","input  ")
        lines[3] = lines[3].replace("input.1","input")
    with open(file, 'w') as fp:
        fp.writelines(lines)
    print(file + "Complete!")