// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Prover subroutines for a ZeroCheck protocol.

use super::ZeroCheckProver;
use crate::{
    barycentric_weights, extrapolate,
    poly_iop::{
        errors::PolyIOPErrors,
        structs::{IOPProverMessage, IOPProverState},
        sum_check::SumCheckProver,
    },
};
use arithmetic::{build_eq_x_r_vec, fix_variables, VirtualPolynomial};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::{cfg_into_iter, end_timer, start_timer, vec::Vec};
use itertools::Itertools;
use rayon::{
    iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator},
    prelude::IntoParallelIterator,
};
use std::{mem::take, sync::Arc};

pub struct ZeroCheckProverState<F: PrimeField> {
    pub iop: IOPProverState<F>,
    pub(crate) eq_evals_L: DenseMultilinearExtension<F>,
    pub(crate) eq_evals_R: DenseMultilinearExtension<F>,
}

impl<F: PrimeField> ZeroCheckProver<F> for ZeroCheckProverState<F> {
    type VirtualPolynomial = VirtualPolynomial<F>;
    type ProverMessage = IOPProverMessage<F>;

    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    fn prover_init(
        polynomial: &Self::VirtualPolynomial,
        zerocheck_r: &[F],
    ) -> Result<Self, PolyIOPErrors> {
        let start = start_timer!(|| "sum check prover init");
        if polynomial.aux_info.num_variables == 0 {
            return Err(PolyIOPErrors::InvalidParameters(
                "Attempt to prove a constant.".to_string(),
            ));
        }
        end_timer!(start);

        let max_degree = polynomial.aux_info.max_degree + 1;
        let half_len = zerocheck_r.len() / 2;
        let (extrapolation_aux, (eq_evals_L, eq_evals_R)) = rayon::join(
            || {
                (1..max_degree)
                    .map(|degree| {
                        let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
                        let weights = barycentric_weights(&points);
                        (points, weights)
                    })
                    .collect()
            },
            || {
                rayon::join(
                    || {
                        DenseMultilinearExtension::from_evaluations_vec(
                            half_len,
                            build_eq_x_r_vec(&zerocheck_r[..half_len]).unwrap(),
                        )
                    },
                    || {
                        DenseMultilinearExtension::from_evaluations_vec(
                            zerocheck_r.len() - half_len,
                            build_eq_x_r_vec(&zerocheck_r[half_len..]).unwrap(),
                        )
                    },
                )
            },
        );

        Ok(Self {
            iop: IOPProverState {
                challenges: Vec::with_capacity(polynomial.aux_info.num_variables),
                round: 0,
                poly: polynomial.clone(),
                extrapolation_aux,
            },
            eq_evals_L,
            eq_evals_R,
        })
    }

    /// Receive message from verifier, generate prover message, and proceed to
    /// next round.
    ///
    /// Main algorithm used is from section 3.2 of [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
    fn prove_round_and_update_state(
        &mut self,
        challenge: &Option<F>,
    ) -> Result<Self::ProverMessage, PolyIOPErrors> {
        let start = start_timer!(|| format!(
            "sum check prove {}-th round and update state",
            self.iop.round
        ));

        if self.iop.round >= self.iop.poly.aux_info.num_variables {
            return Err(PolyIOPErrors::InvalidProver(
                "Prover is not active".to_string(),
            ));
        }

        let eq_len_L = self.iop.poly.aux_info.num_variables / 2;
        let eq_len_R = self.iop.poly.aux_info.num_variables - eq_len_L;

        if self.iop.round > eq_len_L {
            end_timer!(start);
            return SumCheckProver::<F>::prove_round_and_update_state(&mut self.iop, challenge);
        }

        let fix_argument = start_timer!(|| "fix argument");

        // Step 1:
        // fix argument and evaluate f(x) over x_m = r; where r is the challenge
        // for the current round, and m is the round number, indexed from 1
        //
        // i.e.:
        // at round m <= n, for each mle g(x_1, ... x_n) within the flattened_mle
        // which has already been evaluated to
        //
        //    g(r_1, ..., r_{m-1}, x_m ... x_n)
        //
        // eval g over r_m, and mutate g to g(r_1, ... r_m,, x_{m+1}... x_n)

        if let Some(chal) = challenge {
            if self.iop.round == 0 {
                return Err(PolyIOPErrors::InvalidProver(
                    "first round should be prover first.".to_string(),
                ));
            }
            self.iop.challenges.push(*chal);

            let r = self.iop.challenges[self.iop.round - 1];
            rayon::join(
                || {
                    self.iop
                        .poly
                        .flattened_ml_extensions
                        .par_iter_mut()
                        .for_each(|mle| *mle = Arc::new(fix_variables(mle, &[r])))
                },
                || self.eq_evals_L = fix_variables(&self.eq_evals_L, &[r]),
            );
        } else if self.iop.round > 0 {
            return Err(PolyIOPErrors::InvalidProver(
                "verifier message is empty".to_string(),
            ));
        }
        end_timer!(fix_argument);

        if self.iop.round == eq_len_L {
            // End of accelerated zero-check
            assert_eq!(self.eq_evals_L.evaluations.len(), 1);
            self.iop
                .poly
                .mul_by_mle(Arc::new(take(&mut self.eq_evals_R)), self.eq_evals_L[0])?;
            end_timer!(start);
            return SumCheckProver::<F>::prove_round_and_update_state(&mut self.iop, &None);
        }

        self.iop.round += 1;

        let products_list = self.iop.poly.products.clone();

        // Step 2: generate sum for the partial evaluated polynomial:
        // f(r_1, ... r_m,, x_{m+1}... x_n)

        let products_sum = products_list
            .par_iter()
            .map(|(coefficient, products)| {
                let mut sum = cfg_into_iter!(0..1 << (eq_len_L - self.iop.round))
                    .map(|x1| {
                        let (mut partial, eq_evals) = rayon::join(
                            || {
                                cfg_into_iter!(0..1 << eq_len_R)
                                    .fold(
                                        || {
                                            (
                                                vec![(F::zero(), F::zero()); products.len()],
                                                vec![F::zero(); products.len() + 2],
                                            )
                                        },
                                        |(mut buf, mut acc), x2| {
                                            let b = x2 << (eq_len_L - self.iop.round) | x1;
                                            buf.iter_mut().zip_eq(products.iter()).for_each(
                                                |((eval, step), f)| {
                                                    let table =
                                                        &self.iop.poly.flattened_ml_extensions[*f];
                                                    *eval = table[b << 1];
                                                    *step = table[(b << 1) + 1] - table[b << 1];
                                                },
                                            );
                                            acc[0] +=
                                                buf.iter().map(|(eval, _)| eval).product::<F>()
                                                    * self.eq_evals_R[x2];
                                            acc[1..].iter_mut().for_each(|acc| {
                                                buf.iter_mut()
                                                    .for_each(|(eval, step)| *eval += step as &_);
                                                *acc +=
                                                    buf.iter().map(|(eval, _)| eval).product::<F>()
                                                        * self.eq_evals_R[x2];
                                            });
                                            (buf, acc)
                                        },
                                    )
                                    .map(|(_, acc)| acc)
                                    .reduce(
                                        || vec![F::zero(); products.len() + 2],
                                        |mut sum, partial| {
                                            sum.iter_mut()
                                                .zip_eq(partial.iter())
                                                .for_each(|(sum, partial)| *sum += partial);
                                            sum
                                        },
                                    )
                            },
                            || {
                                let table = &self.eq_evals_L;
                                let eval = table[x1 << 1];
                                let step = table[(x1 << 1) + 1] - table[x1 << 1];
                                let mut acc = vec![F::zero(); products.len() + 2];
                                acc[0] = eval;
                                for i in 1..acc.len() {
                                    acc[i] = acc[i - 1] + step;
                                }
                                acc
                            },
                        );
                        partial
                            .iter_mut()
                            .zip_eq(&eq_evals)
                            .for_each(|(val, eq_eval)| {
                                *val *= eq_eval;
                            });
                        partial
                    })
                    .reduce(
                        || vec![F::zero(); products.len() + 2],
                        |mut sum, partial| {
                            sum.iter_mut()
                                .zip_eq(partial.iter())
                                .for_each(|(sum, partial)| *sum += partial);
                            sum
                        },
                    );
                sum.iter_mut().for_each(|sum| *sum *= coefficient);
                sum.extend(
                    cfg_into_iter!(0..self.iop.poly.aux_info.max_degree - products.len())
                        .map(|i| {
                            let (points, weights) = &self.iop.extrapolation_aux[products.len()];
                            let at = F::from((products.len() + 2 + i) as u64);
                            extrapolate(points, weights, &sum, &at)
                        })
                        .collect::<Vec<_>>(),
                );
                sum
            })
            .reduce(
                || vec![F::zero(); self.iop.poly.aux_info.max_degree + 2],
                |acc, x| acc.iter().zip(&x).map(|(x, y)| *x + y).collect(),
            );

        end_timer!(start);

        Ok(IOPProverMessage {
            evaluations: products_sum,
        })
    }

    fn get_final_mle_evaluations(&mut self, challenge: F) -> Result<Vec<F>, PolyIOPErrors> {
        self.iop.get_final_mle_evaluations(challenge)
    }
}
