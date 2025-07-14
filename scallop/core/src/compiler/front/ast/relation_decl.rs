use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstantTuple {
  pub elems: Vec<ConstantOrVariable>,
}

impl ConstantTuple {
  pub fn arity(&self) -> usize {
    self.elems().len()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstantSetTuple {
  pub tag: Option<Constant>,
  pub tuple: ConstantTuple,
}

impl ConstantSetTuple {
  pub fn arity(&self) -> usize {
    self.tuple().arity()
  }

  pub fn iter_constants(&self) -> impl Iterator<Item = &ConstantOrVariable> {
    self.tuple().elems().iter()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstantSet {
  pub tuples: Vec<ConstantSetTuple>,
  pub is_disjunction: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstantSetDecl {
  pub attrs: Attributes,
  pub name: Identifier,
  pub set: ConstantSet,
}

impl ConstantSetDecl {
  pub fn predicate_name(&self) -> &String {
    self.name().name()
  }

  pub fn is_disjunction(&self) -> bool {
    self.set().is_disjunction().clone()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _FactDecl {
  pub attrs: Attributes,
  pub tag: Option<Expr>,
  pub atom: Atom,
}

impl FactDecl {
  pub fn predicate_name(&self) -> &String {
    self.atom().predicate().name()
  }

  pub fn arity(&self) -> usize {
    self.atom().arity()
  }

  pub fn iter_args(&self) -> impl Iterator<Item = &Expr> {
    self.atom().iter_args()
  }

  pub fn iter_constants(&self) -> impl Iterator<Item = &Constant> {
    self.iter_args().filter_map(|expr| expr.as_constant())
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _RuleDecl {
  pub attrs: Attributes,
  pub tag: Option<Expr>,
  pub rule: Rule,
}

impl RuleDecl {
  pub fn rule_tag_predicate(&self) -> String {
    if let Some(head_atom) = self.rule().head().as_atom() {
      format!(
        "rt#{}#{}",
        head_atom.predicate(),
        self.location_id().expect("location id has not been tagged yet")
      )
    } else {
      unimplemented!("Rule head is not an atom")
    }
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ReduceRuleDecl {
  pub attrs: Attributes,
  pub rule: ReduceRule,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum RelationDecl {
  Set(ConstantSetDecl),
  Fact(FactDecl),
  Rule(RuleDecl),
  ReduceRule(ReduceRuleDecl),
}

impl RelationDecl {
  pub fn attrs(&self) -> &Attributes {
    match self {
      RelationDecl::Set(s) => s.attrs(),
      RelationDecl::Fact(f) => f.attrs(),
      RelationDecl::Rule(r) => r.attrs(),
      RelationDecl::ReduceRule(r) => r.attrs(),
    }
  }

  pub fn attrs_mut(&mut self) -> &mut Attributes {
    match self {
      RelationDecl::Set(s) => s.attrs_mut(),
      RelationDecl::Fact(f) => f.attrs_mut(),
      RelationDecl::Rule(r) => r.attrs_mut(),
      RelationDecl::ReduceRule(r) => r.attrs_mut(),
    }
  }

  pub fn head_predicates(&self) -> Vec<String> {
    match self {
      RelationDecl::Set(s) => vec![s.predicate_name().clone()],
      RelationDecl::Fact(f) => vec![f.predicate_name().clone()],
      RelationDecl::Rule(r) => r.rule().head().iter_predicates(),
      RelationDecl::ReduceRule(r) => vec![r.rule().head().name().clone()],
    }
  }
}

impl From<RelationDecl> for Item {
  fn from(q: RelationDecl) -> Self {
    Self::RelationDecl(q)
  }
}
