"""
Not real tests just scratch work to make sure everything calls and doesn't
panic!
TODO: BENCHMARKS
"""

import numpy as np
import corrosiffpy
import siffreadermodule

filename = '/Users/stephen/Desktop/Data/imaging/2024-04/2024-04-17/21Dhh_GCaFLITS/Fly1/Flashes_1.siff'
sr = corrosiffpy.open_file(filename)

print(sr)
print("opened")

print(f"File header: {sr.get_file_header()}")
import time
now = time.time()
#fr = corrosiffpy.read_frames(sr, list(range(100000)))
fr = sr.get_frames(frames=list(range(100000)))
print("took ", time.time() - now, " seconds to read 100000 frames")

print(sr.get_num_frames(), "frames in file including flyback")

print(f"Default shape is {sr.get_frames().shape}")

now = time.time()
hist = sr.get_histogram()

print(f"Default hist shape {hist.shape} computed in {time.time() - now} seconds")

now = time.time()
rust_meta = sr.get_frame_metadata()
print(f"Default metadata computed in {time.time() - now} seconds")

now = time.time()
time_exp = sr.get_experiment_timestamps(frames = list(range(10000)))
time_laser = sr.get_epoch_timestamps_laser(frames = list(range(10000)))
time_system = sr.get_epoch_timestamps_system(frames = list(range(10000)))
time_both = sr.get_epoch_both(frames = list(range(10000)))
print(f"10k timestamps for all four functions computed in {time.time() - now} seconds.",
      f"First 10 system: {time_exp[:10]}",
      f"First 10 laser: {time_laser[:10]}",
        f"First 10 system: {time_system[:10]}",
        f"First 10 columns for the both-in-one-call: {time_both[...,:10]}",
)

appended = sr.get_appended_text()
print(f"Any appended text? {appended}")

sr_text = corrosiffpy.open_file('/Users/stephen/Desktop/Data/imaging/2024-05/2024-05-27/L2Split_GCaFLITS_KCL/Fly1/KClApplication_1.siff')
appended = sr_text.get_appended_text()
print(f"Any appended text in a second file? {appended}")

sio = siffreadermodule.SiffIO()
sio.open(filename)

now = time.time()
fr2 = sio.get_frames(frames=list(range(100000)), registration = {})
print("took ", time.time() - now, " seconds to read 100000 frames with C++")

now = time.time()
sio.get_histogram()
print(f"C++ default histogram took {time.time() - now} seconds")

print(f"The two implementations agree? {(fr == fr2 ).all()}")

rdict = {
    k : (k % 10, k % 364) for k in range(100000)
}

fr = sr.get_frames(frames=list(range(100000)), registration = rdict)

print(f"The two agree with only one registered? {(fr == fr2 ).all()}")

fr2 = sio.get_frames(frames=list(range(100000)), registration = rdict)

print(f"The two implementations agree with registration? {(fr == fr2 ).all()}")