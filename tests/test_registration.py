from siffpy import SiffReader

TEST_SUITE2P = False
try:
    import suite2p
    TEST_SUITE2P = True
except ImportError:
    pass

if TEST_SUITE2P:
    def test_suite2p_registration_info_init():
        filename = '/Volumes/May 2025_/imaging/2025-10/2025-10-25/R45C10_GFlamp2_sytRCaMP3/Fly2/Bar_1.siff'
        reader = SiffReader(filename) 

        rdict_full = reader.register(
            registration_method='suite2p',
        )

        rdict_vbound = reader.register(
            registration_method='suite2p',
            volume_bounds = (0, 100)
        )

        rdict_full_planewise = reader.register(
            registration_method='suite2p',
            planewise_registration = True
        )

        assert( all (rdict_full_planewise[k] == rdict_full[k] for k in rdict_full_planewise.keys()) )

        # actually these won't be the same
        #assert( all ( rdict_full[k] == rdict_vbound[k] for k in rdict_vbound.keys() ) )

        print(rdict_vbound)

if __name__ == "__main__":
    if TEST_SUITE2P:
        test_suite2p_registration_info_init()