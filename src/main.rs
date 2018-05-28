
pub struct NeuralNetwork {
    inputLayerSize : i32,
    outputLayerSize : i32,
    hiddenLayerSize : i32
}

impl NeuralNetwork {
    pub fn forward(&mut self, matrix: Mat<i32>)
    {
        
    }
}

fn main() {
    let mut network = NeuralNetwork {
        inputLayerSize: 2,
        outputLayerSize: 1,
        hiddenLayerSize: 3
    };
}