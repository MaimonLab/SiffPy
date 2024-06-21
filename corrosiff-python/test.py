"""
Not real tests just scratch work to make sure everything calls and doesn't
panic!
TODO: BENCHMARKS
"""

import time
import numpy as np
import corrosiffpy
import siffreadermodule

from siffpy.core.flim import FLIMParams, Exp, Irf

#filename = '/Users/stephen/Desktop/Data/imaging/2024-04/2024-04-17/21Dhh_GCaFLITS/Fly1/Flashes_1.siff'
filename = '/Users/stephen/Desktop/Data/imaging/2024-05/2024-05-27/R60D05_TqCaFLITS/Fly1/EBAgain_1.siff'
sr = corrosiffpy.open_file(filename)

test_params = FLIMParams(
  Exp(tau = 0.5, frac = 0.5, units = 'nanoseconds'),
  Exp(tau = 2.5, frac = 0.5, units = 'nanoseconds'),
  Irf(offset = 1.1, sigma = 0.2, units = 'nanoseconds'),
)

now = time.time()

fr_fl = sr.flim_map(test_params, frames = list(range(10000)))

print("took ", time.time() - now, " seconds to read 10000 frames of FLIM")

roi = np.random.rand(*sr.frame_shape()) > 0.3

sr_text = corrosiffpy.open_file('/Users/stephen/Desktop/Data/imaging/2024-05/2024-05-27/L2Split_GCaFLITS_KCL/Fly1/KClApplication_1.siff')
appended = sr_text.get_appended_text()
assert len(appended) > 0

sio = siffreadermodule.SiffIO()
sio.open(filename)

now = time.time()
with test_params.as_units('countbins'):
  fr_fl2 = sio.flim_map(test_params, frames = list(range(10000)), registration = None)
print("took ", time.time() - now, " seconds to read 10000 frames of FLIM with C++")

assert np.allclose(fr_fl[0], fr_fl2[0], equal_nan = True)
assert (fr_fl[1] == fr_fl2[1]).all()

# now = time.time()
# masked_c = sio.sum_roi(roi, frames = list(range(10000)), registration = None)
# print(f"Masked sum with C++ took {time.time() - now} seconds")

# assert (masked == masked_c).all()

for m in range(1,15):
  threed_roi = np.random.rand(m, *sr.frame_shape()) > 0.3

  now = time.time()
  threed_mask = sr.sum_roi(threed_roi, frames = list(range(10000)), registration = None)
  print("took ", time.time() - now, " seconds to read 100000 frames with a 3D mask")

  now = time.time()
  threed_mask_c = sio.sum_roi(threed_roi, frames = list(range(10000)), registration = None)
  print(f"3D Masked sum with C++ took {time.time() - now} seconds")

  print(f"For m = {m}, they agree? {(threed_mask == threed_mask_c).all()}")