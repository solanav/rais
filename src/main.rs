use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rayon::prelude::*;
use idx_decoder::IDXDecoder;
use std::f64::EPSILON;

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 16;
const OUTPUT_SIZE: usize = 10;

#[inline]
fn sigmoid(x: f64) -> f64 { 1f64 / (1f64 + (EPSILON.powf(-x))) }

#[inline]
fn clamp(x: u8) -> f64 { x as f64 / 255f64 }

// Given a layer, weights and biases it returns the value of the next layer
fn transition(l: Array1<f64>, w: Array2<f64>, b: Array1<f64>) -> Array1<f64> {
    let a1: Array1<f64> = w.dot(&l) + b;
    a1.map(|&e| sigmoid(e))
}

fn draw_vec(v: Vec<u8>) {
    let mut c = -1;
    for x in v {
        c += 1;

        let printable_x = match x {
            x if x > 170 => "\u{2593}\u{2593}",
            x if x > 85 => "\u{2592}\u{2592}",
            x if x > 0 => "\u{2591}\u{2591}",
            _ => " ",
        };

        print!("{} ", printable_x);

        if c == 27 {
            c = -1;
            println!();
        }
    }
}

fn main() {
    let f = std::fs::File::open("./train-images.idx3-ubyte").unwrap();
    let mut data = IDXDecoder::<_, idx_decoder::types::U8, nalgebra::U3>::new(f).unwrap();
    let f = std::fs::File::open("./train-labels.idx1-ubyte").unwrap();
    let mut labels = IDXDecoder::<_, idx_decoder::types::U8, nalgebra::U1>::new(f).unwrap();

    let dist = Uniform::new(0f64, 1f64);

    // Random weights (connections between layers)
    let w0 = Array2::<f64>::random((HIDDEN_SIZE, INPUT_SIZE), dist);
    let w1 = Array2::<f64>::random((HIDDEN_SIZE, HIDDEN_SIZE), dist);
    let w2 = Array2::<f64>::random((OUTPUT_SIZE, HIDDEN_SIZE), dist);

    // Random bias
    let b0 = Array1::<f64>::random(HIDDEN_SIZE, dist);
    let b1 = Array1::<f64>::random(HIDDEN_SIZE, dist);
    let b2 = Array1::<f64>::random(OUTPUT_SIZE, dist);

    let image: Vec<f64> = data.next().unwrap()
        .iter()
        .map(|&e| sigmoid(e as f64))
        .collect();

    let answer = data.next().unwrap();

    let l0 = Array1::<f64>::from_shape_vec(INPUT_SIZE, image).unwrap();
    let l1 = transition(l0.clone(), w0.clone(), b0.clone());
    let l2 = transition(l1.clone(), w1.clone(), b1.clone());
    let l3 = transition(l2.clone(), w2.clone(), b2.clone());
}
