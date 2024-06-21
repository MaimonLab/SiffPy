from bench import files, run_test_on_file

def test_read_unregistered(reader):
    reader.get_frames(registration_dict={})

def test_read_registered(reader):
    reader.get_frames(frames = reader.im_params.flatten_by_timepoints())

for file in files:
    print(
        run_test_on_file(file, test_read_unregistered, 5)
    )

    print(
        run_test_on_file(file, test_read_registered, 5)
    )