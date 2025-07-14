use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

#[derive(Clone)]
pub struct BatchesJoin<B1, B2, Op, T1, T2, TOut, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  TOut: StaticTupleTrait,
  Prov: Provenance,
  Op: BatchBinaryOp<B1::Batch, B2::Batch>,
  Op::IOut: Batch<TOut, Prov>,
  B1: Batches<T1, Prov>,
  B2: Batches<T2, Prov>,
{
  b1: B1,
  b1_curr: Option<B1::Batch>,
  b2: B2,
  b2_source: B2,
  op: Op,
  phantom: PhantomData<(TOut, T2)>,
}

impl<B1, B2, Op, T1, T2, TOut, Prov> BatchesJoin<B1, B2, Op, T1, T2, TOut, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  TOut: StaticTupleTrait,
  Prov: Provenance,
  Op: BatchBinaryOp<B1::Batch, B2::Batch>,
  Op::IOut: Batch<TOut, Prov>,
  B1: Batches<T1, Prov>,
  B2: Batches<T2, Prov>,
{
  pub fn join(mut b1: B1, b2: B2, op: Op) -> Self {
    let b1_curr = b1.next();
    let b2_source = b2.clone();
    Self {
      b1,
      b1_curr,
      b2,
      b2_source,
      op,
      phantom: PhantomData,
    }
  }
}

impl<B1, B2, Op, T1, T2, TOut, Prov> Iterator for BatchesJoin<B1, B2, Op, T1, T2, TOut, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  TOut: StaticTupleTrait,
  Prov: Provenance,
  Op: BatchBinaryOp<B1::Batch, B2::Batch>,
  Op::IOut: Batch<TOut, Prov>,
  B1: Batches<T1, Prov>,
  B2: Batches<T2, Prov>,
{
  type Item = Op::IOut;

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      match &self.b1_curr {
        Some(b1_curr) => match self.b2.next() {
          Some(b2_curr) => {
            let result = self.op.apply(b1_curr.clone(), b2_curr);
            return Some(result);
          }
          None => {
            self.b1_curr = self.b1.next();
            self.b2 = self.b2_source.clone();
          }
        },
        None => return None,
      }
    }
  }
}

impl<B1, B2, Op, T1, T2, TOut, Prov> Batches<TOut, Prov> for BatchesJoin<B1, B2, Op, T1, T2, TOut, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  TOut: StaticTupleTrait,
  Prov: Provenance,
  Op: BatchBinaryOp<B1::Batch, B2::Batch>,
  Op::IOut: Batch<TOut, Prov>,
  B1: Batches<T1, Prov>,
  B2: Batches<T2, Prov>,
{
  type Batch = Op::IOut;
}
