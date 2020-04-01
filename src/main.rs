use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use idx_decoder::IDXDecoder;
use std::f64::EPSILON;

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 15;
const OUTPUT_SIZE: usize = 10;

#[inline]
fn sigmoid(x: f64) -> f64 { 1f64 / (1f64 + (EPSILON.powf(-x))) }

#[inline]
fn clamp(x: u8) -> f64 { x as f64 / 255f64 }

fn cost(l: Array1<f64>, result: usize) -> f64 {
    l.to_vec().iter()
        .enumerate()
        .map(|(i, &e)| {
            if i == result { (e - 1f64) * (e - 1f64) }
            else { e * e }
        }).sum()
}

fn feed_forward(l: Array1<f64>, w: Array2<f64>, b: Array1<f64>) -> Array1<f64> {
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
    // Training data images
    let f = std::fs::File::open("./train-images.idx3-ubyte").unwrap();
    let mut data = IDXDecoder::<_, idx_decoder::types::U8, nalgebra::U3>::new(f).unwrap();

    // Training data labels
    let f = std::fs::File::open("./train-labels.idx1-ubyte").unwrap();
    let mut labels = IDXDecoder::<_, idx_decoder::types::U8, nalgebra::U1>::new(f).unwrap();

    let dist = Uniform::new(0f64, 1f64);

    // Random weights (connections between layers)
    let w0 = Array2::<f64>::random((HIDDEN_SIZE, INPUT_SIZE), dist);
    let w1 = Array2::<f64>::random((OUTPUT_SIZE, HIDDEN_SIZE), dist);

    // Random bias
    let b0 = Array1::<f64>::random(HIDDEN_SIZE, dist);
    let b1 = Array1::<f64>::random(OUTPUT_SIZE, dist);

    let image: Vec<f64> = data.next().unwrap()
        .iter()
        .map(|&e| e as f64 / 255f64)
        .collect();

    let answer = labels.next().unwrap();

    let input = Array1::<f64>::from_shape_vec(INPUT_SIZE, image.clone()).unwrap();
    let hidden = feed_forward(input.clone(), w0.clone(), b0.clone());
    let output = feed_forward(hidden.clone(), w1.clone(), b1.clone());

    let mut res = -1;
    let mut res_i = -1;
    for (i, &e) in output.indexed_iter() {
        if e > res as f64 {
            res = e as i32;
            res_i = i as i32;

            println!("[{}] - {}", res_i, res);
        }
    }

    println!("RES > {} with {:.2}%", res_i, output.to_vec()[res_i as usize] * 100f64);

    println!("{}", cost(output, answer as usize));
}
