use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::poly_iop::utils::split_bits;

#[derive(Default)]
pub struct OrSubtable<F: PrimeField> {
    _field: PhantomData<F>,
}

impl<F: PrimeField> OrSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: PrimeField> LassoSubtable<F> for OrSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        let mut entries: Vec<F> = Vec::with_capacity(M);
        let bits_per_operand = (log2(M) / 2) as usize;

        // Materialize table entries in order where (x | y) ranges 0..M
        for idx in 0..M {
            let (x, y) = split_bits(idx, bits_per_operand);
            let row = F::from_u64((x | y) as u64).unwrap();
            entries.push(row);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // x + y - x * y
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, y) = point.split_at(b);

        let mut result = F::zero();
        for i in 0..b {
            let x = x[b - i - 1];
            let y = y[b - i - 1];
            result += F::from_u64(1u64 << i).unwrap() * (x + y - x * y);
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bls12_381::Fr;

    use crate::{
        poly_iop::lookup::subtable::{or::OrSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(or_ark_materialize_mle_parity, OrSubtable<Fr>, Fr, 256);
}
