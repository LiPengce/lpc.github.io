mod common;

use arithmetic::{math::Math, VirtualPolynomial};
use ark_bls12_381::Fr;
use common::test_rng;
use subroutines::{PolyIOP, PolyIOPErrors, ZeroCheck};

use deNetwork::{DeMultiNet as Net, DeNet};

fn test_zerocheck(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();

    // good path: zero virtual poly
    let poly = VirtualPolynomial::rand_zero_fixed_coeff(nv, num_multiplicands_range, num_products, &mut rng)?;

    let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
    transcript.append_message(b"testing", b"initializing transcript for testing")?;
    let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::d_prove(&poly, &mut transcript)?;

    if Net::am_master() {
        let proof = proof.unwrap();
        let mut poly_info = poly.aux_info.clone();
        poly_info.num_variables += Net::n_parties().log_2();
        let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let zero_subclaim =
            <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly_info, &mut transcript)?;
        assert_eq!(&proof.point, &zero_subclaim.point);
        assert!(
            common::d_evaluate(&poly, Some(&zero_subclaim.point)).unwrap()
                == zero_subclaim.expected_evaluation,
            "wrong subclaim"
        );
    } else {
        common::d_evaluate(&poly, None);
    }

    Ok(())
}

fn test_small_polynomial() -> Result<(), PolyIOPErrors> {
    let nv = 2;
    let num_multiplicands_range = (7, 8);
    let num_products = 5;

    test_zerocheck(nv, num_multiplicands_range, num_products)
}

fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
    let nv = 12;
    let num_multiplicands_range = (6, 7);
    let num_products = 5;

    test_zerocheck(nv, num_multiplicands_range, num_products)
}

fn main() {
    common::network_run(|| {
        test_small_polynomial().unwrap();
        test_normal_polynomial().unwrap();
    });
}
