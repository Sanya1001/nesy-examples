use itertools::iproduct;

use super::super::*;

#[derive(Clone, PartialEq, PartialOrd, Eq)]
pub struct DNFFormula {
  pub clauses: Vec<Clause>,
}

impl DNFFormula {
  pub fn new(clauses: Vec<Clause>) -> Self {
    Self { clauses }
  }

  pub fn get_singleton_id(&self) -> Option<usize> {
    if self.clauses.len() == 1 {
      if self.clauses[0].literals.len() == 1 {
        match &self.clauses[0].literals[0] {
          Literal::Pos(id) => Some(*id),
          _ => None,
        }
      } else {
        None
      }
    } else {
      None
    }
  }

  pub fn is_empty(&self) -> bool {
    self.clauses.is_empty()
  }

  pub fn iter(&self) -> impl Iterator<Item = &Clause> + '_ {
    self.clauses.iter()
  }

  pub fn zero() -> Self {
    Self { clauses: vec![] }
  }

  pub fn one() -> Self {
    Self {
      clauses: vec![Clause::empty()],
    }
  }

  pub fn singleton(f: usize) -> Self {
    Self {
      clauses: vec![Clause::singleton(Literal::Pos(f))],
    }
  }

  pub fn or(&self, t2: &Self) -> Self {
    Self {
      clauses: self.clauses.iter().chain(t2.clauses.iter()).cloned().collect(),
    }
  }

  pub fn and(&self, t2: &Self) -> Self {
    Self {
      clauses: iproduct!(&self.clauses, &t2.clauses)
        .into_iter()
        .filter_map(|(c1, c2)| c1.merge(c2))
        .collect(),
    }
  }

  pub fn dedup(&mut self) {
    let mut result_clauses = vec![];
    for clause in &self.clauses {
      let mut is_duplicated = false;
      for result_clause in &result_clauses {
        if clause == result_clause {
          is_duplicated = true;
          break;
        }
      }
      if !is_duplicated {
        result_clauses.push(clause.clone());
      }
    }
    self.clauses = result_clauses;
  }
}

impl std::fmt::Debug for DNFFormula {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_set().entries(&self.clauses).finish()
  }
}

impl std::fmt::Display for DNFFormula {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, clause) in self.clauses.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{}", clause))?;
    }
    f.write_str("}")
  }
}

impl AsBooleanFormula for DNFFormula {
  fn as_boolean_formula(&self) -> sdd::BooleanFormula {
    sdd::bf_disjunction(
      self
        .clauses
        .iter()
        .map(|c| sdd::bf_conjunction(c.literals.iter().map(|l| l.as_boolean_formula()))),
    )
  }
}

impl Tag for DNFFormula {}
