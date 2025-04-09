// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements the sum check protocol.

use crate::poly_iop::{
    errors::PolyIOPErrors,
    structs::{IOPProof, IOPProverMessage, IOPProverState, IOPVerifierState},
    PolyIOP,
};
use arithmetic::{math::Math, VPAuxInfo, VirtualPolynomial};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer};
use std::{fmt::Debug, sync::Arc};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

pub mod batched_cubic_sumcheck;
pub mod generic_sumcheck;
mod prover;
mod verifier;

/// Trait for doing sum check protocols.
pub trait SumCheck<F: PrimeField> {
    type VirtualPolynomial;
    type VPAuxInfo;
    type MultilinearExtension;
    type SumCheckProof: Clone + Debug + Default + PartialEq + CanonicalSerialize + CanonicalDeserialize;
    type Transcript;
    type SumCheckSubClaim: Clone + Debug + Default + PartialEq;

    /// Extract sum from the proof
    fn extract_sum(proof: &Self::SumCheckProof) -> F;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a SumCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// SumCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    fn prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckProof, PolyIOPErrors>;

    fn prove_sumfold(
        poly0: &Self::VirtualPolynomial,
        poly1: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<(Self::VirtualPolynomial, Self::SumCheckProof), PolyIOPErrors>;

    /// Generate proof of the sum of polynomial over {0,1}^`num_vars`
    ///
    /// The polynomial is represented in the form of a VirtualPolynomial.
    fn d_prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Option<Self::SumCheckProof>, PolyIOPErrors>;

    /// Verify the claimed sum using the proof
    fn verify(
        sum: F,
        proof: &Self::SumCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckSubClaim, PolyIOPErrors>;
}

/// Trait for sum check protocol prover side APIs.
pub trait SumCheckProver<F: PrimeField>
where
    Self: Sized,
{
    type VirtualPolynomial;
    type ProverMessage;

    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    fn prover_init(polynomial: &Self::VirtualPolynomial) -> Result<Self, PolyIOPErrors>;

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

/// Trait for sum check protocol verifier side APIs.
pub trait SumCheckVerifier<F: PrimeField> {
    type VPAuxInfo;
    type ProverMessage;
    type Challenge;
    type Transcript;
    type SumCheckSubClaim;

    /// Initialize the verifier's state.
    fn verifier_init(index_info: &Self::VPAuxInfo) -> Self;

    /// Run verifier for the current round, given a prover message.
    ///
    /// Note that `verify_round_and_update_state` only samples and stores
    /// challenges; and update the verifier's state accordingly. The actual
    /// verifications are deferred (in batch) to `check_and_generate_subclaim`
    /// at the last step.
    fn verify_round_and_update_state(
        &mut self,
        prover_msg: &Self::ProverMessage,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::Challenge, PolyIOPErrors>;

    /// This function verifies the deferred checks in the interactive version of
    /// the protocol; and generate the subclaim. Returns an error if the
    /// proof failed to verify.
    ///
    /// If the asserted sum is correct, then the multilinear polynomial
    /// evaluated at `subclaim.point` will be `subclaim.expected_evaluation`.
    /// Otherwise, it is highly unlikely that those two will be equal.
    /// Larger field size guarantees smaller soundness error.
    fn check_and_generate_subclaim(
        &self,
        asserted_sum: &F,
    ) -> Result<Self::SumCheckSubClaim, PolyIOPErrors>;
}

/// A SumCheckSubClaim is a claim generated by the verifier at the end of
/// verification when it is convinced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SumCheckSubClaim<F: PrimeField> {
    /// the multi-dimensional point that this multilinear extension is evaluated
    /// to
    pub point: Vec<F>,
    /// the expected evaluation
    pub expected_evaluation: F,
}

impl<F: PrimeField> SumCheck<F> for PolyIOP<F> {
    type SumCheckProof = IOPProof<F>;
    type VirtualPolynomial = VirtualPolynomial<F>;
    type VPAuxInfo = VPAuxInfo<F>;
    type MultilinearExtension = Arc<DenseMultilinearExtension<F>>;
    type SumCheckSubClaim = SumCheckSubClaim<F>;
    type Transcript = IOPTranscript<F>;

    fn extract_sum(proof: &Self::SumCheckProof) -> F {
        let start = start_timer!(|| "extract sum");
        let res = proof.proofs[0].evaluations[0] + proof.proofs[0].evaluations[1];
        end_timer!(start);

        res
    }

    fn init_transcript() -> Self::Transcript {
        let start = start_timer!(|| "init transcript");
        let res = IOPTranscript::<F>::new(b"Initializing SumCheck transcript");
        end_timer!(start);
        res
    }

    fn prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckProof, PolyIOPErrors> {
        let start = start_timer!(|| "sum check prove");

        transcript.append_serializable_element(b"aux info", &poly.aux_info)?;

        let mut prover_state = IOPProverState::prover_init(poly)?;
        let mut challenge = None;
        let mut prover_msgs = Vec::with_capacity(poly.aux_info.num_variables);
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

        end_timer!(start);
        Ok(IOPProof {
            point: prover_state.challenges,
            proofs: prover_msgs,
        })
    }
    
    fn prove_sumfold(
        poly0: &Self::VirtualPolynomial,
        poly1: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<(Self::VirtualPolynomial, Self::SumCheckProof), PolyIOPErrors> {
        let start = start_timer!(|| "sum fold prove");

        // transcript.append_serializable_element(b"poly0 aux info", &poly0.aux_info)?;
        // transcript.append_serializable_element(b"poly1 aux info", &poly1.aux_info)?;

        assert!(poly0.products.len() == 1); // only support 1 product for now
        assert!(poly1.products.len() == 1); // only support 1 product for now

        // construct a new polynomial = eq(X, b) * f_0(b, x) * f_1(b, x) * ... * f_{t-1}(b, x)
        // this polynomial sums to T_0 + T_1
        let nv = poly0.aux_info.num_variables;
        assert!(nv == poly1.aux_info.num_variables);
        let degree = poly0.aux_info.max_degree;
        assert!(degree == poly1.aux_info.max_degree); // for now

        let mut multiplicands = Vec::with_capacity(degree + 1); // +1 for eq(X, b)
        /*
          the new polynomial is constructed like this:
          g g g g 1
          h h h h 0
          g g g g 0
          h h h h 1
         */

        for (g, h) in poly0.flattened_ml_extensions.iter().zip(poly1.flattened_ml_extensions.iter()) {
            let new_mle = [
                g.evaluations.clone(),
                h.evaluations.clone(),
                g.evaluations.clone(),
                h.evaluations.clone(),
            ].concat();
            multiplicands.push(new_mle);
        }

        // add eq(X, b)
        let eq_mle = [
            vec![F::one(); 1 << nv],
            vec![F::zero(); 1 << nv],
            vec![F::zero(); 1 << nv],
            vec![F::one(); 1 << nv],
        ].concat();
        multiplicands.push(eq_mle);

        let mle_list = multiplicands
            .into_iter()
            .map(|mle| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv+2, mle)))
            .collect::<Vec<_>>();

        let mut poly = VirtualPolynomial::new(nv+2);
        poly.add_mle_list(mle_list, F::one())?;

        // // fix X to \beta
        // let mut prover_state = IOPProverState::prover_init(poly)?; // just for a single round
        // let mut beta = None;

        // let prover_msg_X = IOPProverState::prove_round_and_update_state(&mut prover_state, &beta)?;

        // let mut prover_state = IOPProverState::prover_init(poly)?;
        // let mut challenge = None;
        // let mut prover_msgs = Vec::with_capacity(poly.aux_info.num_variables);
        // for _ in 0..poly.aux_info.num_variables {
        //     let prover_msg =
        //         IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
        //     transcript.append_serializable_element(b"prover msg", &prover_msg)?;
        //     prover_msgs.push(prover_msg);
        //     challenge = Some(transcript.get_and_append_challenge(b"Internal round")?);
        // }
        // // pushing the last challenge point to the state
        // if let Some(p) = challenge {
        //     prover_state.challenges.push(p)
        // };

        // end_timer!(start);
        // Ok(IOPProof {
        //     point: prover_state.challenges,
        //     proofs: prover_msgs,
        // })

        // seems like we only need to call Self::prove to do SumCheck?
        let result = Self::prove(&poly, transcript);

        end_timer!(start);
        if let Ok(proof) = result {
            Ok((poly, proof))
        } else {
            Err(result.err().unwrap())
        }
    }

    fn d_prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Option<Self::SumCheckProof>, PolyIOPErrors> {
        let start = start_timer!(|| "sum check prove");

        let num_party_vars = Net::n_parties().log_2() as usize;
        if Net::am_master() {
            let mut aux_info = poly.aux_info.clone();
            aux_info.num_variables += num_party_vars;
            transcript.append_serializable_element(b"aux info", &aux_info)?;
        }

        let mut prover_state = IOPProverState::prover_init(poly)?;
        let mut challenge = None;
        let mut prover_msgs = Vec::with_capacity(poly.aux_info.num_variables + num_party_vars);
        for _ in 0..poly.aux_info.num_variables {
            let mut prover_msg =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
            let messages = Net::send_to_master(&prover_msg);
            if Net::am_master() {
                // Sum up the subprovers' messages
                prover_msg = messages.unwrap().iter().fold(
                    IOPProverMessage {
                        evaluations: vec![F::zero(); poly.aux_info.max_degree + 1],
                    },
                    |acc, x| 
                        IOPProverMessage {
                        evaluations: acc
                            .evaluations
                            .iter()
                            .zip(x.evaluations.iter())
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

        let final_mle_evals = IOPProverState::get_final_mle_evaluations(&mut prover_state, challenge.unwrap())?;
        let final_mle_evals = Net::send_to_master(&final_mle_evals);

        if !Net::am_master() {
            end_timer!(start);
            return Ok(None);
        }

        let final_mle_evals = final_mle_evals.unwrap();
        let new_mles = (0..final_mle_evals[0].len())
            .map(|poly_index| Arc::new(DenseMultilinearExtension::from_evaluations_vec(num_party_vars,
                final_mle_evals.iter().map(
                    |mle_evals| mle_evals[poly_index]
                ).collect()
            )))
            .collect();

        let mut poly = prover_state.poly.clone();
        poly.aux_info.num_variables = num_party_vars;
        poly.replace_mles(new_mles);

        end_timer!(step);

        let mut old_challenges = prover_state.challenges.clone();
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
        claimed_sum: F,
        proof: &Self::SumCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "sum check verify");

        transcript.append_serializable_element(b"aux info", aux_info)?;
        let mut verifier_state = IOPVerifierState::verifier_init(aux_info);
        for i in 0..aux_info.num_variables {
            let prover_msg = proof.proofs.get(i).expect("proof is incomplete");
            transcript.append_serializable_element(b"prover msg", prover_msg)?;
            IOPVerifierState::verify_round_and_update_state(
                &mut verifier_state,
                prover_msg,
                transcript,
            )?;
        }

        let res = IOPVerifierState::check_and_generate_subclaim(&verifier_state, &claimed_sum);

        end_timer!(start);
        res
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::test_rng;
    use std::sync::Arc;

    fn test_sumfold(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();

        assert!(num_products == 1); // only support 1 product for now

        let (poly0, asserted_sum0) =
            VirtualPolynomial::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
        let (poly1, asserted_sum1) =
            VirtualPolynomial::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
        
        println!("T0: {}", asserted_sum0);
        println!("T1: {}", asserted_sum1);

        let (poly, proof) = <PolyIOP<Fr> as SumCheck<Fr>>::prove_sumfold(&poly0, &poly1, &mut transcript)?;
        let poly_info = poly.aux_info.clone();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            asserted_sum0 + asserted_sum1,
            &proof,
            &poly_info,
            &mut transcript,
        )?;
        assert!(
            poly.evaluate(&subclaim.point).unwrap() == subclaim.expected_evaluation,
            "wrong subclaim"
        );
        Ok(())
    }

    fn test_sumcheck(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();

        let (poly, asserted_sum) =
            VirtualPolynomial::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
        let proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(&poly, &mut transcript)?;
        let poly_info = poly.aux_info.clone();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            asserted_sum,
            &proof,
            &poly_info,
            &mut transcript,
        )?;
        assert!(
            poly.evaluate(&subclaim.point).unwrap() == subclaim.expected_evaluation,
            "wrong subclaim"
        );
        Ok(())
    }

    fn test_sumcheck_internal(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let (poly, asserted_sum) =
            VirtualPolynomial::<Fr>::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
        let poly_info = poly.aux_info.clone();
        let mut prover_state = IOPProverState::prover_init(&poly)?;
        let mut verifier_state = IOPVerifierState::verifier_init(&poly_info);
        let mut challenge = None;
        let mut transcript = IOPTranscript::new(b"a test transcript");
        transcript
            .append_message(b"testing", b"initializing transcript for testing")
            .unwrap();
        for _ in 0..poly.aux_info.num_variables {
            let prover_message =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)
                    .unwrap();

            challenge = Some(
                IOPVerifierState::verify_round_and_update_state(
                    &mut verifier_state,
                    &prover_message,
                    &mut transcript,
                )
                .unwrap(),
            );
        }
        let subclaim =
            IOPVerifierState::check_and_generate_subclaim(&verifier_state, &asserted_sum)
                .expect("fail to generate subclaim");
        assert!(
            poly.evaluate(&subclaim.point).unwrap() == subclaim.expected_evaluation,
            "wrong subclaim"
        );
        Ok(())
    }

    #[test]
    fn test_sumfold_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 2;
        let num_multiplicands_range = (3, 4);
        let num_products = 1;

        test_sumfold(nv, num_multiplicands_range, num_products)
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 1;
        let num_multiplicands_range = (4, 13);
        let num_products = 5;

        test_sumcheck(nv, num_multiplicands_range, num_products)?;
        test_sumcheck_internal(nv, num_multiplicands_range, num_products)
    }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 12;
        let num_multiplicands_range = (4, 9);
        let num_products = 5;

        test_sumcheck(nv, num_multiplicands_range, num_products)?;
        test_sumcheck_internal(nv, num_multiplicands_range, num_products)
    }
    #[test]
    fn zero_polynomial_should_error() {
        let nv = 0;
        let num_multiplicands_range = (4, 13);
        let num_products = 5;

        assert!(test_sumcheck(nv, num_multiplicands_range, num_products).is_err());
        assert!(test_sumcheck_internal(nv, num_multiplicands_range, num_products).is_err());
    }

    #[test]
    fn test_extract_sum() -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (poly, asserted_sum) = VirtualPolynomial::<Fr>::rand(8, (3, 4), 3, &mut rng)?;

        let proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(&poly, &mut transcript)?;
        assert_eq!(
            <PolyIOP<Fr> as SumCheck<Fr>>::extract_sum(&proof),
            asserted_sum
        );
        Ok(())
    }

    #[test]
    /// Test that the memory usage of shared-reference is linear to number of
    /// unique MLExtensions instead of total number of multiplicands.
    fn test_shared_reference() -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let ml_extensions: Vec<_> = (0..5)
            .map(|_| Arc::new(DenseMultilinearExtension::<Fr>::rand(8, &mut rng)))
            .collect();
        let mut poly = VirtualPolynomial::new(8);
        poly.add_mle_list(
            vec![
                ml_extensions[2].clone(),
                ml_extensions[3].clone(),
                ml_extensions[0].clone(),
            ],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(
            vec![
                ml_extensions[1].clone(),
                ml_extensions[4].clone(),
                ml_extensions[4].clone(),
            ],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(
            vec![
                ml_extensions[3].clone(),
                ml_extensions[2].clone(),
                ml_extensions[1].clone(),
            ],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(
            vec![ml_extensions[0].clone(), ml_extensions[0].clone()],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(vec![ml_extensions[4].clone()], Fr::rand(&mut rng))?;

        assert_eq!(poly.flattened_ml_extensions.len(), 5);

        // test memory usage for prover
        let prover = IOPProverState::<Fr>::prover_init(&poly).unwrap();
        assert_eq!(prover.poly.flattened_ml_extensions.len(), 5);
        drop(prover);

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let poly_info = poly.aux_info.clone();
        let proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(&poly, &mut transcript)?;
        let asserted_sum = <PolyIOP<Fr> as SumCheck<Fr>>::extract_sum(&proof);

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            asserted_sum,
            &proof,
            &poly_info,
            &mut transcript,
        )?;
        assert!(
            poly.evaluate(&subclaim.point)? == subclaim.expected_evaluation,
            "wrong subclaim"
        );
        Ok(())
    }
}
