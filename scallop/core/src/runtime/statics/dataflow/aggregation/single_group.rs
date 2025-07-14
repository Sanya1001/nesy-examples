use std::marker::PhantomData;

use crate::runtime::env::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use super::super::*;

pub struct AggregationSingleGroup<'a, A, D, T1, Prov>
where
  T1: StaticTupleTrait,
  D: Dataflow<T1, Prov>,
  A: Aggregator<T1, Prov>,
  Prov: Provenance,
{
  agg: A,
  d: D,
  rt: &'a RuntimeEnvironment,
  ctx: &'a Prov,
  phantom: PhantomData<T1>,
}

impl<'a, A, D, T1, Prov> AggregationSingleGroup<'a, A, D, T1, Prov>
where
  T1: StaticTupleTrait,
  D: Dataflow<T1, Prov>,
  A: Aggregator<T1, Prov>,
  Prov: Provenance,
{
  pub fn new(agg: A, d: D, rt: &'a RuntimeEnvironment, ctx: &'a Prov) -> Self {
    Self {
      agg,
      d,
      rt,
      ctx,
      phantom: PhantomData,
    }
  }
}

impl<'a, A, D, T1, Prov> Dataflow<A::Output, Prov> for AggregationSingleGroup<'a, A, D, T1, Prov>
where
  T1: StaticTupleTrait,
  D: Dataflow<T1, Prov>,
  A: Aggregator<T1, Prov>,
  Prov: Provenance,
{
  type Stable = EmptyBatches<std::iter::Empty<StaticElement<A::Output, Prov>>>;

  type Recent = SingleBatch<std::vec::IntoIter<StaticElement<A::Output, Prov>>>;

  fn iter_stable(&self) -> Self::Stable {
    Self::Stable::default()
  }

  fn iter_recent(self) -> Self::Recent {
    // Sanitize input relation
    let batch = if let Some(b) = self.d.iter_recent().next() {
      b.collect::<Vec<_>>()
    } else {
      return Self::Recent::empty();
    };

    // Aggregate the result using aggregator
    let result = self.agg.aggregate(batch, self.rt, self.ctx);
    Self::Recent::singleton(result.into_iter())
  }
}

impl<'a, A, D, T1, Prov> Clone for AggregationSingleGroup<'a, A, D, T1, Prov>
where
  T1: StaticTupleTrait,
  D: Dataflow<T1, Prov>,
  A: Aggregator<T1, Prov>,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d: self.d.clone(),
      rt: self.rt,
      ctx: self.ctx,
      phantom: PhantomData,
    }
  }
}
