from siffpy import SiffReader

TEST_SUITE2P = False
try:
    import suite2p
    TEST_SUITE2P = True
except ImportError:
    pass

if TEST_SUITE2P:
    def test_suite2p_registration_info_init():
        # filename = '/Volumes/May 2025_/imaging/2025-10/2025-10-25/R45C10_GFlamp2_sytRCaMP3/Fly2/Bar_1.siff'
        filename = '/Volumes/May 2025_/imaging/2025-10/2025-10-23/R65D06_FLIM_DA/Fly1/AngledPlate_1.siff'
        reader = SiffReader(filename) 

        rdict_full = reader.register(
            registration_method='suite2p',
        )

        full_frames = reader.get_frames(
            frames = reader.im_params.flatten_by_timepoints(timepoint_end = 100),
            registration_dict=rdict_full
        )

        rdict_vbound = reader.register(
            registration_method='suite2p',
            volume_bounds = (0, 100)
        )

        rdict_full_planewise = reader.register(
            registration_method='suite2p',
            planewise_registration = True
        )

        planewise_frames = reader.get_frames(
            frames = reader.im_params.flatten_by_timepoints(timepoint_end = 100),
            registration_dict=rdict_full_planewise
        )

        assert( all (rdict_full_planewise[k] == rdict_full[k] for k in rdict_full_planewise.keys()) )
        assert( (full_frames == planewise_frames).all() )
        # actually these won't be the same
        #assert( all ( rdict_full[k] == rdict_vbound[k] for k in rdict_vbound.keys() ) )

        print(rdict_vbound)

if __name__ == "__main__":
    if TEST_SUITE2P:
        test_suite2p_registration_info_init()