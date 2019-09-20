#####################################################################
# 
# Author: Xiongtao Ruan
# 
# Copyright (C) 2012-2018 Murphy Lab
# Computational Biology Department
# School of Computer Science
# Carnegie Mellon University
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.
# 
# For additional information visit http://murphylab.web.cmu.edu or
# send email to murphy@cmu.edu
# 
#####################################################################


import os, time, re, random 

'''clean old temporary files with extension .tmp'''

def Chunk_lock_clean(path, duration=120, initial_wait=False):
    
    if not path:
        print("A path must be specified!")
        return 0

    if initial_wait:
        if os.environ.get('SLURM_JOB_ID'):
            job_id = re.sub('[\r\n]', '',  os.getenv('SLURM_JOB_ID'))
        else:
            job_id = ''
        wait_time_str = job_id[::-1]
        if len(wait_time_str) > 0 and wait_time_str.isdigit():
            wait_time_str = wait_time_str[:1] + '.' + wait_time_str[1:]
            wait_time = float(wait_time_str)
        else:
            wait_time = random.uniform(0, 1) * 5;
        time.sleep(wait_time)

    hard_duration = 360
    out_stat = False
    keep_flag = True
    file_info = os.listdir(path)
    temp_filenames = [f for f in file_info if f.endswith('.tmp')]
    # print temp_filenames
    deleted_files = []
    if len(temp_filenames):
        for cur_file in temp_filenames:
            if time.time() - os.path.getmtime(path + '/' + cur_file) > duration * 60:
                deleted_files.append(cur_file)
                os.remove(path + '/' + cur_file)
    
    out_stat = True
    return (out_stat, deleted_files)

