use onnxruntime::ndarray;
use onnxruntime::session::Session as OnnxSession;

use ndarray::Array1;
use ndarray::Array3;

use onnxruntime::error::NonMatchingDimensionsError;

use onnxruntime::session::Input;

use onnxruntime::TensorElementDataType;

use crate::error::PyannoteError;

use onnxruntime::error::OrtError;
use onnxruntime::ndarray::ShapeError;

pub struct Pyannote<'a> {
    session: OnnxSession<'a>,
}

pub struct FusedSizeArray1<T, const N: usize> {
    pub data: Array1<T>,
}

impl<T, const N: usize> FusedSizeArray1<T, N> {
    pub fn try_from_array1(data: Array1<T>) -> Result<Self, ShapeError> {
        let arr = data.into_shape(N)?;

        Ok(Self { data: arr })
    }
}

pub type PyannoteInput = FusedSizeArray1<f32, 80000>;

impl Pyannote<'_> {
    pub fn run(&mut self, input: PyannoteInput) -> Result<Array3<f32>, PyannoteError> {
        let input_tensor = input
            .data
            .into_shape((1, 1, 80000))
            .expect("shape error, it should not happen!");

        let outputs: Vec<onnxruntime::tensor::OrtOwnedTensor<f32, _>> =
            self.session.run(vec![input_tensor])?;

        assert_eq!(outputs.len(), 1);

        let output = outputs
            .into_iter()
            .nth(0)
            .unwrap()
            .to_owned() //copy!
            .into_shape((1, 293, 3))
            .unwrap();

        Ok(output)
    }

    pub fn from_session<'a>(session: OnnxSession<'a>) -> Result<Pyannote<'a>, PyannoteError> {
        let pyannote =
            Pyannote::try_from(session).map_err(|err| OrtError::NonMatchingDimensions(err))?;

        Ok(pyannote)
    }
}

fn is_pyannote_format(inputs: &Vec<Input>) -> Result<(), NonMatchingDimensionsError> {
    if let [input] = inputs.as_slice() {
        if let Input {
            input_type: TensorElementDataType::Float,
            dimensions,
            ..
        } = input
        {
            if let [Some(1), Some(1), Some(80000)] = dimensions.as_slice() {
                return Ok(());
            }
        }
    }

    let err = NonMatchingDimensionsError::InputsCount {
        inference_input_count: inputs.len(),
        model_input_count: 1,
        inference_input: vec![vec![1, 1, 80000]],
        model_input: inputs
            .iter()
            .map(|input| input.dimensions.clone())
            .collect(),
    };

    Err(err)
}

impl<'a> TryFrom<OnnxSession<'a>> for Pyannote<'a> {
    type Error = NonMatchingDimensionsError;
    fn try_from(session: OnnxSession<'a>) -> Result<Self, Self::Error> {
        is_pyannote_format(&session.inputs)?;
        Ok(Self { session })
    }
}
