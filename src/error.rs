use onnxruntime::OrtError;

use onnxruntime::ndarray::ShapeError;

#[derive(Debug)]
pub enum PyannoteError {
    OrtError(OrtError),
    ShapeError(ShapeError),
}

impl From<OrtError> for PyannoteError {
    fn from(err: OrtError) -> Self {
        PyannoteError::OrtError(err)
    }
}

impl From<ShapeError> for PyannoteError {
    fn from(err: ShapeError) -> Self {
        PyannoteError::ShapeError(err)
    }
}
