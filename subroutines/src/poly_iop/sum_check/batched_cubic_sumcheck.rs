// Copyright (c) Jolt Project
// Copyright (c) 2023 HyperPlonk Project
// Copyright (c) 2024 HyperPianist Project

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements batched cubic sumchecks

use crate::poly_iop::sum_check::generic_sumcheck::SumcheckInstanceProof;
use arithmetic::{
    bind_poly_var_bot, bit_decompose, eq_eval,
    eq_poly::EqPolynomial,
    math::Math,
    unipoly::{CompressedUniPoly, UniPoly},
};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

// A cubic sumcheck instance that is not represented as virtual polynomials.
// Instead the struct itself can hold arbitrary state as long as it can bind
// varaibles and produce a cubic polynomial on demand.
// Used by the layered circuit implementation for rational sumcheck
pub trait BatchedCubicSumcheckInstance<F: PrimeField>: Sync {
    fn num_rounds(&self) -> usize;
    fn bind(&mut self, eq_poly: &mut DenseMultilinearExtension<F>, r: &F);
    // Returns evals at 0, 2, 3
    fn compute_cubic(
        &self,
        coeffs: &[F],
        eq_poly: &DenseMultilinearExtension<F>,
        lambda: &F,
    ) -> (F, F, F);
    fn final_claims(&self) -> Vec<Vec<F>>;
    fn compute_cubic_direct(
        &self,
        coeffs: &[F],
        evaluations: &[Vec<DenseMultilinearExtension<F>>],
        lambda: &F,
    ) -> (F, F, F);

    // #[tracing::instrument(skip_all, name =
    // "BatchedCubicSumcheck::prove_sumcheck")]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        coeffs: &[F],
        eq_poly: &mut DenseMultilinearExtension<F>,
        transcript: &mut IOPTranscript<F>,
        lambda: &F,
    ) -> (SumcheckInstanceProof<F>, Vec<F>, Vec<Vec<F>>) {
        debug_assert_eq!(eq_poly.num_vars, self.num_rounds());

        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _round in 0..self.num_rounds() {
            let evals = self.compute_cubic(coeffs, eq_poly, lambda);
            let cubic_poly =
                UniPoly::from_evals(&[evals.0, previous_claim - evals.0, evals.1, evals.2]);
            // append the prover's message to the transcript
            transcript
                .append_serializable_element(b"poly", &cubic_poly)
                .unwrap();
            // derive the verifier's challenge for the next round
            let r_j = transcript
                .get_and_append_challenge(b"challenge_nextround")
                .unwrap();

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(eq_poly, &r_j);

            previous_claim = cubic_poly.evaluate(&r_j);
            cubic_polys.push(cubic_poly.compress());
        }

        debug_assert_eq!(eq_poly.evaluations.len(), 1);

        (
            SumcheckInstanceProof::new(cubic_polys),
            r,
            self.final_claims(),
        )
    }

    fn d_prove_sumcheck(
        &mut self,
        claim: &F,
        coeffs: &[F],
        eq_poly: &mut DenseMultilinearExtension<F>,
        r_grand_product_party: &[F],
        transcript: &mut IOPTranscript<F>,
        lambda: &F,
    ) -> Option<(SumcheckInstanceProof<F>, Vec<F>, Vec<Vec<Vec<F>>>)> {
        let num_party_vars = Net::n_parties().log_2();
        debug_assert_eq!(eq_poly.num_vars, self.num_rounds());

        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        // rev() because this is big endian
        let index_vec: Vec<F> = bit_decompose(Net::party_id() as u64, num_party_vars)
            .into_iter()
            .rev()
            .map(|x| F::from(x))
            .collect();
        let eq_coeff = eq_eval(&index_vec, r_grand_product_party).unwrap();

        let eq_coeffs = Net::send_to_master(&eq_coeff);

        for _round in 0..self.num_rounds() {
            let evals = self.compute_cubic(coeffs, eq_poly, lambda);
            let messages = Net::send_to_master(&evals);

            let r_j = if Net::am_master() {
                let evals = messages
                    .unwrap()
                    .iter()
                    .zip(eq_coeffs.as_ref().unwrap())
                    .fold((F::zero(), F::zero(), F::zero()), |acc, (x, eq_coeff)| {
                        (
                            acc.0 + x.0 * eq_coeff,
                            acc.1 + x.1 * eq_coeff,
                            acc.2 + x.2 * eq_coeff,
                        )
                    });
                let cubic_poly =
                    UniPoly::from_evals(&[evals.0, previous_claim - evals.0, evals.1, evals.2]);

                // append the prover's message to the transcript
                transcript
                    .append_serializable_element(b"poly", &cubic_poly)
                    .unwrap();
                let r_j = transcript
                    .get_and_append_challenge(b"challenge_nextround")
                    .unwrap();

                previous_claim = cubic_poly.evaluate(&r_j);
                cubic_polys.push(cubic_poly.compress());

                Net::recv_from_master_uniform(Some(r_j))
            } else {
                Net::recv_from_master_uniform(None)
            };

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(eq_poly, &r_j);
        }

        debug_assert_eq!(eq_poly.evaluations.len(), 1);

        // Dimensions: (party, poly_id, batch_index)
        let final_claims = Net::send_to_master(&self.final_claims());
        if !Net::am_master() {
            return None;
        }

        let final_claims = final_claims.unwrap();
        let mut polys = (0..final_claims[0].len())
            .into_par_iter()
            .map(|poly_id| {
                (0..final_claims[0][poly_id].len())
                    .map(|batch_index| {
                        DenseMultilinearExtension::from_evaluations_vec(
                            num_party_vars,
                            final_claims
                                .iter()
                                .map(|claim| claim[poly_id][batch_index])
                                .collect(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let eq_non_party = eq_poly.evaluations[0];
        let eq_poly = DenseMultilinearExtension::from_evaluations_vec(
            num_party_vars,
            EqPolynomial::evals(&r_grand_product_party),
        );
        polys.push(vec![eq_poly]);

        for _round in 0..num_party_vars {
            let evals = self.compute_cubic_direct(coeffs, &polys, lambda);
            let evals = (evals.0 * eq_non_party, evals.1 * eq_non_party, evals.2 * eq_non_party);
            let cubic_poly =
                UniPoly::from_evals(&[evals.0, previous_claim - evals.0, evals.1, evals.2]);

            transcript
                .append_serializable_element(b"poly", &cubic_poly)
                .unwrap();
            let r_j = transcript
                .get_and_append_challenge(b"challenge_nextround")
                .unwrap();
            r.push(r_j);

            polys.par_iter_mut().for_each(|polys| {
                for poly in polys {
                    bind_poly_var_bot(poly, &r_j);
                }
            });

            previous_claim = cubic_poly.evaluate(&r_j);
            cubic_polys.push(cubic_poly.compress());
        }

        Some((SumcheckInstanceProof::new(cubic_polys), r, final_claims))
    }
}
