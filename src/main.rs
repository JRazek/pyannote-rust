#![feature(iter_array_chunks)]

use onnxruntime::environment::Environment;
use onnxruntime::ndarray::Array3;
use onnxruntime::session::Session;
use onnxruntime::{ndarray, tensor::OrtOwnedTensor};
use std::io::prelude::*;

use plotters::prelude::*;

use std::fs::File;

mod plot;

use pyannote_rust::pyannote::Pyannote;

//(1, 293, 3)

fn ort_tensors_to_ndarray(
    tensors: &Vec<OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>>,
) -> ndarray::Array3<f32> {
    // For simplicity, we're assuming the tensors vector has only one tensor.
    let tensor = &tensors[0];

    let vec = tensor.iter().copied().collect::<Vec<f32>>();

    let tensor = ndarray::Array3::from_shape_vec((1, 293, 3), vec);

    let tensor = tensor.unwrap();

    tensor
}

fn run_sample(session: &mut Session, sample: Vec<f32>) -> Array3<f32> {
    let tensor = ndarray::Array::from_shape_vec((1, 1, 80000), sample).unwrap();

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![tensor]).unwrap();

    let tensor = ort_tensors_to_ndarray(&outputs);

    tensor
}

fn main() {
    let env = Environment::builder().with_name("test").build().unwrap();

    let mut session = env
        .new_session_builder()
        .unwrap()
        .with_model_from_file("model.onnx")
        .unwrap();

    //(1, 1, 8000)

    //InputsLength { inference_input: [[1, 1, 8000]], model_input: [[Some(1), Some(1), Some(80000)]] }

    let files = [("pcm_s16le_speech.raw", "speech_plot.png")];

    files.iter().for_each(|(file_in, file_out)| {
        let mut file = File::open(file_in).unwrap();
        let mut buffer = vec![0u8; 80000 * 2];
        file.read(&mut buffer).unwrap();

        let pcm = buffer
            .iter()
            .cloned()
            .array_chunks::<2>()
            .map(|[x, y]| i16::from_le_bytes([x, y]))
            .map(|x| x as f32 / 32768.0)
            .collect::<Vec<f32>>();

        assert_eq!(pcm.len(), 80000);

        let tensor = run_sample(&mut session, pcm);
        let array1: Vec<_> = tensor.lanes(ndarray::Axis(1)).into_iter().collect();

        let root = BitMapBackend::new(file_out, (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        plot::plot_charts(array1.into_iter(), &root);
    });

    let mut pyannote = Pyannote::try_from(session).unwrap();

    use ndarray::Array1;

    let array = Array1::from_iter((0..80000).into_iter().map(|x| x as f32));

    let output = pyannote.run(array).unwrap();

    println!("{:?}", output);
}
