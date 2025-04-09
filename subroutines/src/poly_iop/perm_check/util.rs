// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements useful functions for the permutation check protocol.

use crate::poly_iop::errors::PolyIOPErrors;
use arithmetic::{identity_permutation_mle, math::Math};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use rayon::iter::{
    IntoParallelIterator, ParallelIterator,
};
use std::{mem::take, sync::Arc};

use deNetwork::{DeMultiNet as Net, DeNet};

/// Returns the evaluations of two list of MLEs:
/// - numerators = (a1, ..., ak)
/// - denominators = (b1, ..., bk)
///
///  where
///  - beta and gamma are challenges
///  - (f1, ..., fk), (g1, ..., gk),
///  - (s_id1, ..., s_idk), (perm1, ..., permk) are mle-s
///
/// - ai(x) is the MLE for `fi(x) + \beta s_id_i(x) + \gamma`
/// - bi(x) is the MLE for `gi(x) + \beta perm_i(x) + \gamma`
///
/// The caller is responsible for sanity-check
#[allow(clippy::type_complexity)]
pub(super) fn compute_leaves<F: PrimeField, const DISTRIBUTED: bool>(
    beta: &F,
    gamma: &F,
    fxs: &[Arc<DenseMultilinearExtension<F>>],
    gxs: &[Arc<DenseMultilinearExtension<F>>],
    perms: &[Arc<DenseMultilinearExtension<F>>],
) -> Result<Vec<Vec<Vec<F>>>, PolyIOPErrors> {
    let start = start_timer!(|| "compute numerators and denominators");

    let mut numerators = vec![];
    let mut denominators = vec![];
    let mut leaves = vec![];

    let mut shift = 0;
    let mut last_num_var = fxs[0].num_vars;
    for l in 0..fxs.len() {
        let num_vars = fxs[l].num_vars;
        if num_vars != last_num_var {
            numerators.append(&mut denominators);
            leaves.push(take(&mut numerators));
        }
        last_num_var = num_vars;

        let eval_len = num_vars.pow2() as u64;

        let s_id = if DISTRIBUTED {
            identity_permutation_mle::<F>(shift + eval_len * (Net::party_id() as u64), num_vars)
        } else {
            identity_permutation_mle::<F>(shift, num_vars)
        };
        if DISTRIBUTED {
            shift += eval_len * (Net::n_parties() as u64);
        } else {
            shift += eval_len;
        }

        let (numerator_evals, denominator_evals) = (
            &fxs[l].evaluations,
            &gxs[l].evaluations,
            &s_id.evaluations,
            &perms[l].evaluations,
        )
            .into_par_iter()
            .map(|(&f_ev, &g_ev, &s_id_ev, &perm_ev)| {
                let numerator = f_ev + *beta * s_id_ev + gamma;
                let denominator = g_ev + *beta * perm_ev + gamma;
                (numerator, denominator)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        numerators.push(numerator_evals);
        denominators.push(denominator_evals);
    }

    numerators.append(&mut denominators);
    leaves.push(take(&mut numerators));

    end_timer!(start);
    Ok(leaves)
}
