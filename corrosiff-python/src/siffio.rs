//! The `SiffIO` Python class is used to call
//! the corrosiff library from Rust.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::conversion::ToPyObject;
use corrosiff::{metadata, SiffReader, CorrosiffError};

use std::collections::HashMap;

/// The SiffIO class wraps a `SiffReader` object
/// in rust and provides methods to read from the
/// file
#[pyclass(name = "RustSiffIO", dict, module = "corrosiff_python")]
pub struct SiffIO {
    reader: SiffReader,
}

impl SiffIO {
    /// Can be called in `Rust` packages using `corrosiff`
    /// to produce a `SiffIO` object if they interface with
    /// `Python` as well.
    pub fn new_from_reader(reader: SiffReader) -> Self {
        SiffIO { reader }
    }
}

/// The default value of frames is a Vec<u64> containing
/// all of the frames in the file. `frames_default!(frames, siffio)`
macro_rules! frames_default(
    ($frames : expr, $siffio : expr) => {
        $frames.or_else(
            || Some((0 as u64..$siffio.reader.num_frames() as u64).collect())
        ).unwrap()
    };
);

#[pymethods]
impl SiffIO {

    /// Returns the name of the open file
    #[getter]
    pub fn filename(&self) -> PyResult<String> {
        Ok(self.reader.filename())
    }

    /// Returns the metadata of the file
    #[pyo3(name = "get_file_header")]
    pub fn get_file_header_py<'py>(&self, py : Python<'py>) -> PyResult<Bound<'py, PyDict>> {

        let ret_dict = PyDict::new_bound(py);

        ret_dict.set_item("Filename", self.reader.filename())?;
        ret_dict.set_item("BigTiff", self.reader.is_bigtiff())?;
        ret_dict.set_item("IsSiff", self.reader.is_siff())?;
        ret_dict.set_item("Number of frames", self.reader.num_frames())?;
        ret_dict.set_item("Non-varying frame data", self.reader.nvfd())?;
        ret_dict.set_item("ROI string", self.reader.roi_string())?;
        
        Ok(ret_dict)
    }

    /// Returns the number of frames in the file
    /// include flyback (basically the number of
    /// IFDs).
    pub fn get_num_frames(&self) -> u64 {
        self.reader.num_frames() as u64
    }

    pub fn __repr__(&self) -> String {
        format!(
            "RustSiffIO(filename={})\n
            The `SiffIO` object is implemented in Rust 
            for fast parallelizable file reading operations 
            that Python is not well-suited to perform. Its job 
            is to return `Python` objects, especially `numpy` arrays, 
            for visualization and further analysis in `Python`.
            For more information, consult `siffpy.readthedocs.io` or
            the `corrosiff` repository on Github.",
            self.reader.filename()
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__() 
    }

    /// Returns a list of dictionaries, each containing the metadata
    /// corresponding to the frames requested (in the same order as 
    /// `frames`).
    #[pyo3(signature = (frames=None))]
    pub fn get_frame_metadata<'py>(&self, py : Python<'py>, frames : Option<Vec<u64>>)
    -> PyResult<Bound<'py, PyList>> {
        
        let frames = frames_default!(frames, self);
        let metadatas = self.reader.get_frame_metadata(&frames)
            .map_err(|e| 
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("{:?}", e)
                )
            )?;
        
        let ret_list = PyList::empty_bound(py);

        for metadata in metadatas {
            let py_dict = PyDict::new_bound(py);
            // Ugly and brute force
            py_dict.set_item("width", metadata.width)?;
            py_dict.set_item("height", metadata.height)?;
            py_dict.set_item("bits_per_sample", metadata.bits_per_sample)?;
            py_dict.set_item("compression", metadata.compression)?;
            py_dict.set_item("photometric_interpretation", metadata.photometric_interpretation)?;
            py_dict.set_item("end_of_ifd", metadata.end_of_ifd)?;
            py_dict.set_item("data_offset", metadata.data_offset)?;
            py_dict.set_item("samples_per_pixel", metadata.samples_per_pixel)?;
            py_dict.set_item("rows_per_strip", metadata.rows_per_strip)?;
            py_dict.set_item("strip_byte_counts", metadata.strip_byte_counts)?;
            py_dict.set_item("x_resolution", metadata.x_resolution)?;
            py_dict.set_item("y_resolution", metadata.y_resolution)?;
            py_dict.set_item("resolution_unit", metadata.resolution_unit)?;
            py_dict.set_item("sample_format", metadata.sample_format)?;
            py_dict.set_item("siff_tag", metadata.siff_compress)?;
            py_dict.set_item("Frame metadata", metadata.metadata_string)?;
        
            ret_list.append(py_dict)?;
        }
        Ok(ret_list)
    }

    /// Returns the timestamps of each frame in seconds since the start of the experiment
    #[pyo3(name = "get_experiment_timestamps", signature = (frames=None))]
    pub fn get_experiment_timestamps_py<'py>(&self, py : Python<'py>, frames : Option<Vec<u64>>)
    -> PyResult<Bound<'py, PyArray1<f64>>> {
        let frames = frames_default!(frames, self);

        Ok(
            self.reader
            .get_experiment_timestamps(&frames)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{:?}", e)))?
            .into_pyarray_bound(py)
        )
    }

    #[pyo3(name = "get_epoch_timestamps_laser", signature = (frames=None))]
    pub fn get_epoch_timestamps_laser_py<'py>(&self, py : Python<'py>, frames : Option<Vec<u64>>)
    -> PyResult<Bound<'py, PyArray1<u64>>> {
        let frames = frames_default!(frames, self);

        Ok(
            self.reader
            .get_epoch_timestamps_laser(&frames)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{:?}", e)))?
            .into_pyarray_bound(py)
        )
    }

    #[pyo3(name = "get_epoch_timestamps_system", signature = (frames=None))]
    pub fn get_epoch_timestamps_system_py<'py>(&self, py : Python<'py>, frames : Option<Vec<u64>>)
    -> PyResult<Bound<'py, PyArray1<u64>>>
    {
        let frames = frames_default!(frames, self);

        Ok(
            self.reader
            .get_epoch_timestamps_system(&frames)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{:?}", e)))?
            .into_pyarray_bound(py)
        )
    }

    #[pyo3(name = "get_epoch_both", signature = (frames=None))]
    pub fn get_epoch_both_py<'py>(&self, py : Python<'py>, frames : Option<Vec<u64>>)
    -> PyResult<Bound<'py, PyArray2<u64>>> {
        let frames = frames_default!(frames, self);
        Ok(
            self.reader
            .get_epoch_timestamps_both(&frames)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{:?}", e)))?
            .into_pyarray_bound(py)
        )
    }

    /// THIS one was a nightmare! Hard to work with `PyTuple` of different types of objects...
    #[pyo3(name = "get_appended_text", signature = (frames=None))]
    pub fn get_appended_text_py<'py>(&self, py : Python<'py>, frames : Option<Vec<u64>>)
    -> PyResult<Bound<'py, PyList>> {
        let frames = frames_default!(frames, self);
        let ret_list = PyList::empty_bound(py);
        self.reader
        .get_appended_text(&frames)
        .iter().for_each(|(frame, text, option_timestamp)| {
            match option_timestamp {
                Some(timestamp) => {
                    let tuple : Py<PyTuple> = (
                        frame.into_py(py),
                        text.into_py(py),
                        timestamp.into_py(py),
                    ).into_py(py);
                    ret_list.append(tuple).unwrap();
                
                },
                None => {
                    let tuple : Py<PyTuple> = (
                        frame.into_py(py),
                        text.into_py(py),
                    ).into_py(py);
                    ret_list.append(tuple).unwrap();
                }
            }
        });
        Ok(ret_list)
    }

    /// Returns an array of the frame data
    #[pyo3(signature = (frames=None, registration= None))]
    pub fn get_frames<'py>(
        &self, 
        py : Python<'py>,
        frames: Option<Vec<u64>>,
        registration : Option<HashMap<u64, (i32, i32)>>,
    ) -> PyResult<Bound<'py, PyArray3<u16>>> {
        // If frames is None, then we will read all frames
        let frames = frames_default!(frames, self);

        Ok(
            self.reader
            .get_frames_intensity(&frames, registration.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{:?}", e)))?
            .into_pyarray_bound(py)
        )
    }

    /// Returns a 2d histogram of the arrival time data.
    /// The first dimension is the frame number and the second
    /// dimension is the arrival time.
    #[pyo3(signature = (frames=None))]
    pub fn get_histogram<'py>(&self, py : Python<'py>, frames : Option<Vec<u64>>)
    -> PyResult<Bound<'py, PyArray2<u64>>> {
        let frames = frames_default!(frames, self);

        Ok(
            self.reader
            .get_histogram(&frames)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{:?}", e)))?
            .into_pyarray_bound(py)
        )
    }
}