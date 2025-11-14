"""
Contains benchmarking for the opening time of files.
"""

import siffpy

from local_consts import temp_img_reg_test


def test_suite2p_registration_vol():
    sr = siffpy.SiffReader(temp_img_reg_test)
    rdict = sr.register(
        registration_method='suite2p',
    )

def test_suite2p_registration_planewise():
    sr = siffpy.SiffReader(temp_img_reg_test)
    rdict = sr.register(
        registration_method='suite2p',
        planewise_registration=True
    )

if __name__ == '__main__':
    import timeit
    print(
        "Suite2p registration, volume registration:\n",
        timeit.timeit("test()",
            setup="from __main__ import test_suite2p_registration_vol as test",
            number = 10,
        )/10 , "sec per iter"
    )

    print(
        "Suite2p registration, planewise registration:\n",
        timeit.timeit("test()",
            setup="from __main__ import test_suite2p_registration_planewise as test",
            number = 10,
        )/10 , "sec per iter"
    )