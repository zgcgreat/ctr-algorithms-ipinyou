# _*_ coding: utf-8 _*_
import subprocess

# linux 下可执行，windows不能执行

cmd = 'python3 data2fm.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 libfm.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 evaluate.py'
subprocess.call(cmd, shell=True)

