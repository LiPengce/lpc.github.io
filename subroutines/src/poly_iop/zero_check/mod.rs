// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the ZeroCheck protocol.

use std::fmt::Debug;

use crate::{
    poly_iop::{
        errors::PolyIOPErrors,
        structs::{IOPProverMessage, IOPProverState},
        sum_check::{SumCheck, SumCheckProver},
        PolyIOP,
    },
    IOPProof,
};
use arithmetic::{bit_decompose, eq_eval, math::Math};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer};
use prover::ZeroCheckProverState;
use std::sync::Arc;
use transcript::IOPTranscript;
use itertools::Itertools;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

mod prover;

/// Trait for sum check protocol prover side APIs.
pub trait ZeroCheckProver<F: PrimeField>
where
    Self: Sized,
{
    type VirtualPolynomial;
    type ProverMessage;

    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    fn prover_init(
        polynomial: &Self::VirtualPolynomial,
        zerocheck_r: &[F],
    ) -> Result<Self, PolyIOPErrors>;

    /// Receive message from verifier, generate prover message, and proceed to
    /// next round.
    ///
    /// Main algorithm used is from section 3.2 of [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
    fn prove_round_and_update_state(
        &mut self,
        challenge: &Option<F>,
    ) -> Result<Self::ProverMessage, PolyIOPErrors>;

    /// Bind the final r_n to get the final evaluation.
    /// This should be called immediately after prove_round_and_update_state, as
    /// it assumes that the prover state has unique ownership of its polynomials
    fn get_final_mle_evaluations(&mut self, challenge: F) -> Result<Vec<F>, PolyIOPErrors>;
}

/// A zero check IOP subclaim for `f(x)` consists of the following:
///   - the initial challenge vector r which is used to build eq(x, r) in
///     SumCheck
///   - the random vector `v` to be evaluated
///   - the claimed evaluation of `f(v)`
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ZeroCheckSubClaim<F: PrimeField> {
    // the evaluation point
    pub point: Vec<F>,
    /// the expected evaluation
    pub expected_evaluation: F,
    // the initial challenge r which is used to build eq(x, r)
    pub init_challenge: Vec<F>,
}

/// A ZeroCheck for `f(x)` proves that `f(x) = 0` for all `x \in {0,1}^num_vars`
/// It is derived from SumCheck.
pub trait ZeroCheck<F: PrimeField>: SumCheck<F> {
    type ZeroCheckSubClaim: Clone + Debug + Default + PartialEq;
    type ZeroCheckProof: Clone + Debug + Default + PartialEq + CanonicalSerialize + CanonicalDeserialize;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a ZeroCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// ZeroCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    /// initialize the prover to argue for the sum of polynomial over
    /// {0,1}^`num_vars` is zero.
    fn prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckProof, PolyIOPErrors>;

    fn d_prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Option<Self::ZeroCheckProof>, PolyIOPErrors>;

    /// verify the claimed sum using the proof
    fn verify(
        proof: &Self::ZeroCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckSubClaim, PolyIOPErrors>;
}

impl<F: PrimeField> ZeroCheck<F> for PolyIOP<F> {
    type ZeroCheckSubClaim = ZeroCheckSubClaim<F>;
    type ZeroCheckProof = Self::SumCheckProof;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<F>::new(b"Initializing ZeroCheck transcript")
    }

    fn prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckProof, PolyIOPErrors> {
        let start = start_timer!(|| "sum check prove");

        let length = poly.aux_info.num_variables;
        let r = transcript.get_and_append_challenge_vectors(b"0check r", length)?;

        {
            let mut aux_info = poly.aux_info.clone();
            aux_info.max_degree += 1;
            transcript.append_serializable_element(b"aux info", &aux_info)?;
        }

        let mut prover_state: ZeroCheckProverState<F> =
            ZeroCheckProverState::prover_init(poly, &r)?;
        let mut challenge = None;
        let mut prover_msgs = Vec::with_capacity(poly.aux_info.num_variables);
        for _ in 0..poly.aux_info.num_variables {
            let prover_msg =
                ZeroCheckProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
            transcript.append_serializable_element(b"prover msg", &prover_msg)?;
            prover_msgs.push(prover_msg);
            challenge = Some(transcript.get_and_append_challenge(b"Internal round")?);
        }
        // pushing the last challenge point to the state
        if let Some(p) = challenge {
            prover_state.iop.challenges.push(p)
        };

        end_timer!(start);
        Ok(IOPProof {
            point: prover_state.iop.challenges,
            proofs: prover_msgs,
        })
    }

    fn d_prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Option<Self::SumCheckProof>, PolyIOPErrors> {
        let start = start_timer!(|| "zero check prove");

        let length = poly.aux_info.num_variables;
        let num_party_vars = Net::n_parties().log_2();

        let r = if Net::am_master() {
            let r = transcript
                .get_and_append_challenge_vectors(b"0check r", length + num_party_vars)?;
            Net::recv_from_master_uniform(Some(r))
        } else {
            Net::recv_from_master_uniform(None)
        };

        let index_vec: Vec<F> = bit_decompose(Net::party_id() as u64, num_party_vars)
            .into_iter()
            .map(|x| F::from(x))
            .collect();

        let coeff = eq_eval(&r[length..], &index_vec)?;
        let old_poly = poly;
        let mut poly = poly.clone();
        poly.products.iter_mut().for_each(|(item, _)| {
            *item *= coeff;
        });

        if Net::am_master() {
            let mut aux_info = poly.aux_info.clone();
            aux_info.num_variables += num_party_vars;
            aux_info.max_degree += 1;
            transcript.append_serializable_element(b"aux info", &aux_info)?;
        }

        let mut prover_state = ZeroCheckProverState::prover_init(&poly, &r[..length])?;
        let mut challenge = None;
        let mut prover_msgs = Vec::with_capacity(poly.aux_info.num_variables + num_party_vars);
        let max_degree = poly.aux_info.max_degree + 1;
        for _ in 0..poly.aux_info.num_variables {
            let mut prover_msg =
                ZeroCheckProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
            let messages = Net::send_to_master(&prover_msg);
            if Net::am_master() {
                // Sum up the subprovers' messages
                prover_msg =
                messages.unwrap().iter().fold(
                    IOPProverMessage {
                        evaluations: vec![F::zero(); max_degree + 1],
                    },
                    |acc, x| IOPProverMessage {
                        evaluations: acc
                            .evaluations
                            .iter()
                            .zip_eq(x.evaluations.iter())
                            .map(|(x, y)| *x + y)
                            .collect(),
                    },
                );
                transcript.append_serializable_element(b"prover msg", &prover_msg)?;
            }
            prover_msgs.push(prover_msg);
            if Net::am_master() {
                let challenge_value = transcript.get_and_append_challenge(b"Internal round")?;
                Net::recv_from_master_uniform(Some(challenge_value));
                challenge = Some(challenge_value);
            } else {
                challenge = Some(Net::recv_from_master_uniform::<F>(None));
            }
        }

        let step = start_timer!(|| "Compute final mle");

        let mut final_mle_evals =
            ZeroCheckProverState::get_final_mle_evaluations(&mut prover_state, challenge.unwrap())?;
        *final_mle_evals.last_mut().unwrap() *= coeff * prover_state.eq_evals_L[0];
        let final_mle_evals = Net::send_to_master(&final_mle_evals);

        if !Net::am_master() {
            end_timer!(step);
            end_timer!(start);
            return Ok(None);
        }

        let final_mle_evals = final_mle_evals.unwrap();
        let new_mles = (0..final_mle_evals[0].len())
            .map(|poly_index| {
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_party_vars,
                    final_mle_evals
                        .iter()
                        .map(|mle_evals| mle_evals[poly_index])
                        .collect(),
                ))
            })
            .collect();

        let mut poly = prover_state.iop.poly.clone();
        // Restore the coefficients
        poly.products
            .iter_mut()
            .zip_eq(old_poly.products.iter())
            .for_each(|((coeff, _), (old_coeff, _))| *coeff = *old_coeff);
        poly.aux_info.num_variables = num_party_vars;
        poly.replace_mles(new_mles);

        end_timer!(step);

        let mut old_challenges = prover_state.iop.challenges.clone();
        let mut prover_state = IOPProverState::prover_init(&poly)?;
        challenge = None;
        for _ in 0..poly.aux_info.num_variables {
            let prover_msg =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
            transcript.append_serializable_element(b"prover msg", &prover_msg)?;
            prover_msgs.push(prover_msg);
            challenge = Some(transcript.get_and_append_challenge(b"Internal round")?);
        }
        // pushing the last challenge point to the state
        if let Some(p) = challenge {
            prover_state.challenges.push(p)
        };

        old_challenges.append(&mut prover_state.challenges);

        end_timer!(start);
        Ok(Some(IOPProof {
            point: old_challenges,
            proofs: prover_msgs,
        }))
    }

    fn verify(
        proof: &Self::ZeroCheckProof,
        fx_aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "zero check verify");

        // check that the sum is zero
        if proof.proofs[0].evaluations[0] + proof.proofs[0].evaluations[1] != F::zero() {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "zero check: sum {} is not zero",
                proof.proofs[0].evaluations[0] + proof.proofs[0].evaluations[1]
            )));
        }

        // generate `r` and pass it to the caller for correctness check
        let length = fx_aux_info.num_variables;
        let r = transcript.get_and_append_challenge_vectors(b"0check r", length)?;

        // hat_fx's max degree is increased by eq(x, r).degree() which is 1
        let mut hat_fx_aux_info = fx_aux_info.clone();
        hat_fx_aux_info.max_degree += 1;
        let sum_subclaim =
            <Self as SumCheck<F>>::verify(F::zero(), proof, &hat_fx_aux_info, transcript)?;

        // expected_eval = sumcheck.expect_eval/eq(v, r)
        // where v = sum_check_sub_claim.point
        let eq_x_r_eval = eq_eval(&sum_subclaim.point, &r)?;
        let expected_evaluation = sum_subclaim.expected_evaluation / eq_x_r_eval;

        end_timer!(start);
        Ok(ZeroCheckSubClaim {
            point: sum_subclaim.point,
            expected_evaluation,
            init_challenge: r,
        })
    }
}

#[cfg(test)]
mod test {

    use super::ZeroCheck;
    use crate::poly_iop::{errors::PolyIOPErrors, PolyIOP};
    use arithmetic::VirtualPolynomial;
    use ark_bls12_381::Fr;
    use ark_std::test_rng;

    fn test_zerocheck(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        {
            // good path: zero virtual poly
            let poly =
                VirtualPolynomial::rand_zero(nv, num_multiplicands_range, num_products, &mut rng)?;

            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::prove(&poly, &mut transcript)?;

            let poly_info = poly.aux_info.clone();
            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let zero_subclaim =
                <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly_info, &mut transcript)?;
            assert!(
                poly.evaluate(&zero_subclaim.point)? == zero_subclaim.expected_evaluation,
                "wrong subclaim"
            );
        }

        {
            // bad path: random virtual poly whose sum is not zero
            let (poly, _sum) =
                VirtualPolynomial::<Fr>::rand(nv, num_multiplicands_range, num_products, &mut rng)?;

            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::prove(&poly, &mut transcript)?;

            let poly_info = poly.aux_info.clone();
            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;

            assert!(
                <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly_info, &mut transcript)
                    .is_err()
            );
        }

        Ok(())
    }

    // #[test]
    // fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
    //     let nv = 1;
    //     let num_multiplicands_range = (4, 5);
    //     let num_products = 1;

    //     test_zerocheck(nv, num_multiplicands_range, num_products)
    // }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        // let nv = 7;
        // let num_multiplicands_range = (9, 17);
        // let num_products = 21;

        let nv = 13;
        let num_multiplicands_range = (6, 7);
        let num_products = 5;

        test_zerocheck(nv, num_multiplicands_range, num_products)
    }

    // #[test]
    // fn zero_polynomial_should_error() -> Result<(), PolyIOPErrors> {
    //     let nv = 0;
    //     let num_multiplicands_range = (4, 13);
    //     let num_products = 5;

    //     assert!(test_zerocheck(nv, num_multiplicands_range,
    // num_products).is_err());     Ok(())
    // }
}
