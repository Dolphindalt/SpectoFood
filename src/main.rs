extern crate portaudio;
extern crate gnuplot;
extern crate argparse_rs;
extern crate num_complex;

mod common;

use portaudio as pa;
use common::*;
use std::io;
use gnuplot::*;
use std::f64::consts::PI;
use num_complex::Complex;

const SAMPLE_RATE: f64 = 44_100.0;
const FRAMES: u32 = 256;
const CHANNELS: i32 = 2;
const INTERLEAVED: bool = true;

const I: Complex<f64> = Complex { re: 0.0, im: 1.0 };

fn main() {
    match run() {
        Ok(_) => {},
        e => {
            eprintln!("Failed with the following: {:?}", e);
        }
    };
}

fn run() -> Result<(), pa::Error> {
    let c = Common::new().unwrap();

    let pa = try!(pa::PortAudio::new());

    let default_host = try!(pa.default_host_api());

    let def_input = try!(pa.default_input_device());
    let input_info = try!(pa.device_info(def_input));

    let latency = input_info.default_low_input_latency;
    let input_params = pa::StreamParameters::<f32>::new(def_input, CHANNELS, INTERLEAVED, latency);
    try!(pa.is_input_format_supported(input_params, SAMPLE_RATE));
    let input_settings = pa::InputStreamSettings::new(input_params, SAMPLE_RATE, FRAMES);

    let mut fg1 = Figure::new();
    let mut fg2 = Figure::new();
    let mut x: Vec<i32> = Vec::new();
    let mut logx: Vec<f64> = Vec::new();
    for i in 0..(FRAMES as i32 * CHANNELS) {
        x.push(i);
        logx.push((i as f64).log(10.0));
    }
    // A callback to pass to the non-blocking input stream.
    let input_callback = move |pa::InputStreamCallbackArgs { buffer, frames, .. }| {
        assert!(frames == FRAMES as usize);

        let vec_buffer = Vec::from(buffer);
        assert!(vec_buffer.len() == FRAMES as usize * CHANNELS as usize);
        
        fg1.clear_axes();
        fg1.axes2d()
            .set_title("Input Waves", &[])
            .lines(&x, &vec_buffer, &[LineWidth(1.5), Color("blue"), LineStyle(DotDash)])
            .set_x_ticks(Some((Fix((x.len() as f64)/10.0), 1)), &[Mirror(false)], &[])
            .set_y_ticks(Some((Fix(0.1), 1)), &[Mirror(false)], &[]);

        let com_buf = vec_buffer.into_iter().map(|r| Complex { re: (r as f64), im: 0.0 }).collect::<Vec<Complex<f64>>>();
        let fftres: Vec<f64> = fft(&com_buf).into_iter().map(|c| (c.norm_sqr() * 2.0) / (FRAMES as f64)).collect();
        fg2.clear_axes();
        fg2.axes2d()
            .set_title("Fourier Transform", &[])
            .lines(&logx, &fftres[0..(FRAMES as usize)], &[LineWidth(1.5), Color("blue"), LineStyle(DotDash)])
            .set_x_ticks(Some((Fix(1.0), 1)), &[Mirror(false)], &[])
            .set_y_ticks(Some((Fix(1.0), 1)), &[Mirror(false)], &[]);

        c.show(&mut fg1);
        c.show(&mut fg2);
        pa::Continue
    };

    let mut input_stream = try!(pa.open_non_blocking_stream(input_settings, input_callback));

    try!(input_stream.start());

    println!("Recording has started. Press Enter to stop.");

    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).ok();

    try!(input_stream.stop());

    Ok(())
}

fn fft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    fn fft_inner(
        buf_a: &mut [Complex<f64>],
        buf_b: &mut [Complex<f64>],
        n: usize,
        step: usize,
    ) {
        if step >= n {
            return;
        }

        fft_inner(buf_b, buf_a, n, step * 2);
        fft_inner(&mut buf_b[step..], &mut buf_a[step..], n, step * 2);
        let (left, right) = buf_a.split_at_mut(n / 2);
        for i in (0..n).step_by(step * 2) {
            let t = (-I * PI * (i as f64) / (n as f64)).exp() * buf_b[i + step];
            left[i / 2] = buf_b[i] + t;
            right[i / 2] = buf_b[i] - t;
        }
    }

    let n_orig = input.len();
    let n = n_orig.next_power_of_two();
    let mut buf_a = input.to_vec();
    buf_a.append(&mut vec![Complex {re: 0.0, im: 0.0}; n - n_orig]);
    let mut buf_b = buf_a.clone();
    fft_inner(&mut buf_a, &mut buf_b, n, 1);
    buf_a
}