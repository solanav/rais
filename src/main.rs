use ndarray::{Array1, Array2, Array};
use rand::Rng;
use rand::prelude::ThreadRng;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 16;
const OUTPUT_SIZE: usize = 10;

#[inline]
fn relu(x: f64) -> f64 {
    match x {
        x if x < 0f64 => 0f64,
        x => x,
    }
}

// Given a layer, weights and biases it returns the value of the next layer
fn transition(l: Array1<f64>, w: Array2<f64>, b: Array1<f64>) -> Array1<f64> {
    let a1: Array1<f64> = w.dot(&l) + b;
    a1.map(|&e| relu(e))
}

fn main() {
    let dist = Uniform::new(0f64, 1f64);

    // Layers with a random input for now
    let l0 = Array1::<f64>::random(INPUT_SIZE, dist);
    let l1;
    let l2;
    let l3;

    // Random weights (connections between layers)
    let w0 = Array2::<f64>::random((HIDDEN_SIZE, INPUT_SIZE), dist);
    let w1 = Array2::<f64>::random((HIDDEN_SIZE, HIDDEN_SIZE), dist);
    let w2 = Array2::<f64>::random((OUTPUT_SIZE, HIDDEN_SIZE), dist);

    // Random bias
    let b0 = Array1::<f64>::random(HIDDEN_SIZE, dist);
    let b1 = Array1::<f64>::random(HIDDEN_SIZE, dist);
    let b2 = Array1::<f64>::random(OUTPUT_SIZE, dist);

    l1 = transition(l0.clone(), w0, b0);
    l2 = transition(l1.clone(), w1, b1);
    l3 = transition(l2.clone(), w2, b2);

    println!("{}", l0);
    println!("{}", l1);
    println!("{}", l2);
}
