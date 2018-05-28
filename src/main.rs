extern crate rulinalg;
extern crate rand;

use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrixMut;

// Primary neural network structure
pub struct NeuralNetwork {
    inputLayerSize : i32,
    outputLayerSize : i32,
    hiddenLayerSize : i32,
    w1: Matrix<f32>,
    w2: Matrix<f32>,
    z3: Matrix<f32>,
    z2: Matrix<f32>,
    a2: Matrix<f32>
}

// Sigmoid activation function
pub fn sigmoid(z : f32) -> f32 {
    1_f32 / ( 1_f32 + (-1_f32 * z).exp())
}

pub fn negative_elem(x: f32) -> f32 {
    x * -1_f32
}

// Sigmoid prime function
pub fn sigmoid_prime(z: f32) -> f32 {
    (-1_f32 * z).exp() / ((1_f32 + (-1_f32 * z).exp()).powi(2))
}

impl NeuralNetwork {
    // Forward propagation function
    pub fn forward(&mut self, matrix : Matrix<f32>) -> Matrix<f32>
    {
        self.z2 = matrix * &self.w1;
        self.a2 = self.z2.apply(&sigmoid) as Matrix<f32>;
        self.z3 = self.a2 * &self.w2;
        self.z3.apply(&sigmoid)
    }

    // Calculate cost function prime
    pub fn costFunctionPrime(&mut self, X : Matrix<f32>, y : Matrix<f32>) -> (Matrix<f32>, Matrix<f32>) {
        let yHat = self.forward(X);

        let delta3 = self.z3.apply(&sigmoid_prime);
        let delta33 = delta3 * ((y - yHat).apply(&negative_elem));
        let djdw2 = self.a2.transpose() * delta33;

        let delta2 = delta33 * self.w2.transpose() * self.z2.apply(&sigmoid_prime);
        let djdw1 = X.transpose() * delta2;

        (djdw2, djdw1)
    }
}

fn main() {

    // New neural network with random weights
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
        a2: Matrix::new(1,0, vec![1_f32]),
        z2: Matrix::new(1,0, vec![1_f32]),
        z3: Matrix::new(1,0, vec![1_f32])
    };
    
    // Starting stuff
    let X = Matrix::new(3, 2, vec![
        3_f32, 5_f32, 
        5_f32, 1_f32, 
        10_f32, 2_f32]);

    // Get and print predictions
    let yhat = network.forward(X);
    println!("{}", yhat);
}

#[test]
fn test_sigmoid() {
    println!("{}", sigmoid_prime(0_f32));
    println!("{}", sigmoid_prime(-2_f32));
    println!("{}", sigmoid_prime(2_f32));
    println!("{}", sigmoid_prime(-5_f32));
    println!("{}", sigmoid_prime(5_f32));
}