use ndarray::{array, Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::{Rng};
use rand_distr;
use std::time;
use rayon::prelude::*;

fn main() {

    // Fake data

    const N_DRAW:u32 = 1000;
    let market_num = 40;
    let market_ids = Vec::from_iter(0..market_num);
    let n_firm = 5;
    let n_market_traits = 8;
    let n_firm_traits = 6;

    let market_specific_data: Array2<f64> = Array::random((market_num, n_market_traits), rand_distr::StandardNormal);

    // Parameters
    let A: Array1<f64> = Array::random(n_firm_traits, rand_distr::StandardNormal);
    let B: Array1<f64> = Array::random(n_market_traits, rand_distr::StandardNormal);
    let rho:f64 = 0.5;
    let sigma = (1. - rho.powi(2)).sqrt();
    let delta:f64 = 1.9;

    let start_time = time::Instant::now();
    // For each market
    for market in market_ids{

        let market_specific_X = market_specific_data.row(market);
        // println!("{:?}", market_specific_data.shape());

        //Simulation closure
        let draw_simulation = || {
            //placeholder for previous prediction for this draw
            let mut prediction_prev = vec![false;n_firm];
            let UM: f64 = rand::thread_rng().sample(rand_distr::StandardNormal);
            let U_firms:Vec<f64> = rand::thread_rng().sample_iter(rand_distr::StandardNormal).take(n_firm).collect();

            let mut n_try = 0;
            // keep adding up n.try in this market until it exceeds n.pred
            // indicating an consistent num
            // n.try must be less than total num of airline firms
            while n_try <= n_firm {
                let mut n_pred = 0;

                // prediction for this draw
                let mut prediction = vec![false;n_firm];

                // for each firm, calc the profit
                for firm in 0..n_firm{
                    // market specific data
                    let X = &market_specific_X;     // borrow from ArrayView

                    // firm-market specific data
                    let Z:Array1<f64> = Array::random(n_firm_traits, rand_distr::StandardNormal);


                    let U: f64 = U_firms[firm];

                    // X: N_market, B: N_market
                    // Z: N_frim,   A: N_firm
                    let profit = X.dot(&B) + Z.dot(&A) + UM * rho + U * sigma + (n_try as f64).ln() * delta;

                    let enter:bool = profit > 0.;

                    prediction[firm] = enter;
                    n_pred += 1;
                }

                if n_pred < n_try{
                    break;
                }

                n_try+=1;
                prediction_prev = prediction;
            }   //end while n_try

            // println!("{:?}", prediction_prev);
            return prediction_prev;
        }; // end draw function closure

        let simulations_this_market = Array2::from_shape_vec(
                    (N_DRAW as usize, n_firm),
                    (0..N_DRAW)
                    .into_par_iter()
                    .map(|_| draw_simulation())
                    .flatten().collect()
                    )
                    .unwrap()
                    .mapv(|e| f64::from(e))
                    .mean_axis(Axis(0)).unwrap();
        // println!("{:?}", simulations_this_market)

    } // end this market
    let duration = start_time.elapsed();
    println!("Total elapsed time: {}ms", duration.as_millis());
    println!("{} ms per run", (duration.as_millis() as f64) / (N_DRAW as f64) );
}

