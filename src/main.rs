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

impl NeuralNetwork {
    pub fn forward(&mut self, matrix : Matrix<f32>) -> Matrix<f32>
    {
        let z2 = matrix * &self.w1;
        let a2 = z2.apply(&sigmoid);
        let z3 = a2 * &self.w2;
        z3.apply(&sigmoid)
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
        w2: Matrix::new(3, 1, vec![
            rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>()                
        ]),
    };
    
    let X = Matrix::new(3, 2, vec![
        3_f32, 5_f32, 
        5_f32, 1_f32, 
        10_f32, 2_f32]);

    let yhat = network.forward(X);
    println!("{}", yhat);
}
