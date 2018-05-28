extern crate rulinalg;
extern crate rand;

use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrixMut;

pub struct NeuralNetwork {
    inputLayerSize : i32,
    outputLayerSize : i32,
    hiddenLayerSize : i32,
    w1: Matrix<f32>,
    w2: Matrix<f32>
}

pub fn sigmoid(z : f32) -> f32 {
    1_f32 / ( 1_f32 + (-1_f32 * z).exp())
}

pub fn sigmoid_mat(matrix: Matrix<f32>) -> Matrix<f32> {
    let matrix_b = matrix.apply(&sigmoid);
    matrix_b
}

impl NeuralNetwork {
    pub fn forward(&mut self, matrix : Matrix<f32>)
    {
        
    }
}

fn main() {
    let mut network = NeuralNetwork {
        inputLayerSize: 2,
        outputLayerSize: 1,
        hiddenLayerSize: 3,
        w1: Matrix::new(2, 3, vec![
            rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>(), 
            rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>() ]),
        w2: Matrix::new(1, 3, vec![
            rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>()                
        ]),
    };
}
