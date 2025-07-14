use std::collections::*;

use crate::common::adt_variant_registry::*;
use crate::common::foreign_aggregate::*;
use crate::common::foreign_function::*;
use crate::common::foreign_predicate::*;
use crate::common::input_tag::*;
use crate::common::output_option::*;
use crate::common::value_type::*;

use crate::compiler::front;

use super::attributes::DemandAttribute;
use super::attributes::MagicSetAttribute;
use super::Attributes;

pub type Type = crate::common::value_type::ValueType;

pub type Constant = crate::common::value::Value;

#[derive(Clone, Debug)]
pub struct Program {
  pub relations: Vec<Relation>,
  pub outputs: HashMap<String, OutputOption>,
  pub facts: Vec<Fact>,
  pub disjunctive_facts: Vec<Vec<Fact>>,
  pub rules: Vec<Rule>,
  pub function_registry: ForeignFunctionRegistry,
  pub predicate_registry: ForeignPredicateRegistry,
  pub aggregate_registry: AggregateRegistry,
  pub adt_variant_registry: ADTVariantRegistry,
}

impl Program {
  pub fn relation_of_predicate(&self, pred: &String) -> Option<&Relation> {
    self.relations.iter().find(|r| &r.predicate == pred)
  }

  pub fn rules_of_predicate(&self, pred: String) -> impl Iterator<Item = &Rule> {
    self.rules.iter().filter(move |r| r.head_predicate() == &pred)
  }

  pub fn is_demand_predicate(&self, pred: &String) -> Option<bool> {
    self
      .relation_of_predicate(pred)
      .map(|r| r.attributes.get::<DemandAttribute>().is_some())
  }

  pub fn is_magic_set_predicate(&self, pred: &String) -> Option<bool> {
    self
      .relation_of_predicate(pred)
      .map(|r| r.attributes.get::<MagicSetAttribute>().is_some())
  }

  pub fn set_output_relations(&mut self, outputs: Vec<&str>) {
    self.outputs.iter_mut().for_each(|(rela, rela_output_options)| {
      *rela_output_options = if outputs.contains(&rela.as_str()) {
        OutputOption::Default
      } else {
        OutputOption::Hidden
      };
    });
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fact {
  pub tag: DynamicInputTag,
  pub predicate: String,
  pub args: Vec<Constant>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Relation {
  pub attributes: Attributes,
  pub predicate: String,
  pub arg_types: Vec<Type>,
}

impl Relation {
  pub fn new(predicate: String, arg_types: Vec<Type>) -> Self {
    Self::new_with_attrs(Attributes::new(), predicate, arg_types)
  }

  pub fn new_with_attrs(attributes: Attributes, predicate: String, arg_types: Vec<Type>) -> Self {
    Self {
      attributes,
      predicate,
      arg_types,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
  pub attributes: Attributes,
  pub head: Head,
  pub body: Conjunction,
}

impl Rule {
  pub fn head_predicate(&self) -> &String {
    &self.head.predicate()
  }

  pub fn body_literals(&self) -> impl Iterator<Item = &Literal> {
    self.body.args.iter()
  }

  pub fn body_literals_mut(&mut self) -> impl Iterator<Item = &mut Literal> {
    self.body.args.iter_mut()
  }

  pub fn collect_new_expr_functors(&self) -> impl Iterator<Item = &String> {
    self.body_literals().filter_map(|l| match l {
      Literal::Assign(assign) => match &assign.right {
        AssignExpr::New(new_expr) => Some(&new_expr.functor),
        _ => None,
      },
      _ => None,
    })
  }

  pub fn needs_dynamically_parse_entity(
    &self,
    foreign_function_registry: &ForeignFunctionRegistry,
    foreign_predicate_registry: &ForeignPredicateRegistry,
  ) -> bool {
    self.body_literals().any(|l| match l {
      Literal::Atom(a) => {
        if let Some(p) = foreign_predicate_registry.get(&a.predicate) {
          p.free_argument_types().iter().any(|t| t == &ValueType::Entity)
        } else {
          false
        }
      }
      Literal::Assign(assign) => match &assign.right {
        AssignExpr::Call(c) => {
          if let Some(f) = foreign_function_registry.get(&c.function) {
            match f.return_type() {
              ForeignFunctionParameterType::BaseType(ValueType::Entity) => true,
              _ => false,
            }
          } else {
            false
          }
        }
        _ => false,
      },
      _ => false,
    })
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Head {
  /// A simple atom as the head
  Atom(Atom),

  /// A disjunction of atoms as the head; all atoms should have the same predicate
  Disjunction(Vec<Atom>),
}

impl Head {
  pub fn atom(predicate: String, args: Vec<Term>) -> Self {
    Self::Atom(Atom::new(predicate, args))
  }

  pub fn predicate(&self) -> &String {
    match self {
      Self::Atom(a) => &a.predicate,
      Self::Disjunction(disj) => &disj[0].predicate,
    }
  }

  pub fn get_atom(&self) -> Option<&Atom> {
    match self {
      Self::Atom(a) => Some(a),
      _ => None,
    }
  }

  pub fn variable_args(&self) -> Vec<&Variable> {
    match self {
      Self::Atom(a) => a.variable_args().collect(),
      Self::Disjunction(disj) => disj.iter().flat_map(|a| a.variable_args()).collect(),
    }
  }

  /// Substitute the atom's arguments with the given term rewrite function
  pub fn substitute<F: Fn(&Term) -> Term + Copy>(&self, f: F) -> Self {
    match self {
      Self::Atom(a) => Self::Atom(a.substitute(f)),
      Self::Disjunction(d) => Self::Disjunction(d.iter().map(|a| a.substitute(f)).collect()),
    }
  }

  /// Get the variable patterns of the head
  ///
  /// Atomic head has only one pattern;
  /// Disjunctive head could have multiple patterns
  pub fn has_multiple_patterns(&self) -> bool {
    match self {
      Self::Atom(_) => {
        // Atomic head has only one pattern
        false
      }
      Self::Disjunction(disj) => {
        // Extract the pattern of the first atom in the disjunction
        let first_pattern = disj[0]
          .args
          .iter()
          .map(|t| match t {
            Term::Variable(v) => v.name.clone(),
            Term::Constant(_) => String::new(),
          })
          .collect::<Vec<_>>();

        // Check if the first pattern is satisfied by all other atoms
        for a in disj.iter().skip(1) {
          let satisfies_pattern = a.args.iter().enumerate().all(|(i, t)| {
            if let Some(p) = first_pattern.get(i) {
              match t {
                Term::Variable(v) => p == &v.name,
                Term::Constant(_) => p.is_empty(),
              }
            } else {
              false
            }
          });

          // If not satisfied, then the head has multiple patterns
          if !satisfies_pattern {
            return true;
          }
        }

        // If all atoms satisfy the first pattern, then the head has only one pattern
        false
      }
    }
  }
}

/// A conjunction of literals
#[derive(Clone, Debug, PartialEq)]
pub struct Conjunction {
  pub args: Vec<Literal>,
}

/// A term is the argument of a literal
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Term {
  Variable(Variable),
  Constant(Constant),
}

impl Term {
  /// Create a new variable term using the given name and type
  pub fn variable(name: String, ty: Type) -> Self {
    Self::Variable(Variable { name, ty })
  }

  /// Check if the term is a variable
  pub fn is_variable(&self) -> bool {
    match self {
      Self::Variable(_) => true,
      _ => false,
    }
  }

  /// Check if the term is a constant
  pub fn is_constant(&self) -> bool {
    match self {
      Self::Constant(_) => true,
      _ => false,
    }
  }

  /// Get the variable if the term is a variable
  pub fn as_variable(&self) -> Option<&Variable> {
    match self {
      Self::Variable(v) => Some(v),
      _ => None,
    }
  }

  /// Get the constant if the term is a constant
  pub fn as_constant(&self) -> Option<&Constant> {
    match self {
      Self::Constant(c) => Some(c),
      _ => None,
    }
  }
}

#[derive(Clone, Debug, Hash, PartialEq, PartialOrd, Eq, Ord)]
pub struct Variable {
  pub name: String,
  pub ty: Type,
}

impl Variable {
  pub fn new(name: String, ty: Type) -> Self {
    Self { name, ty }
  }
}

#[derive(Clone, PartialEq)]
pub enum Literal {
  Atom(Atom),
  NegAtom(NegAtom),
  Assign(Assign),
  Constraint(Constraint),
  Reduce(Reduce),
  True,
  False,
}

impl std::fmt::Debug for Literal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Atom(a) => std::fmt::Debug::fmt(a, f),
      Self::NegAtom(a) => std::fmt::Debug::fmt(a, f),
      Self::Assign(a) => std::fmt::Debug::fmt(a, f),
      Self::Constraint(a) => std::fmt::Debug::fmt(a, f),
      Self::Reduce(a) => std::fmt::Debug::fmt(a, f),
      Self::True => f.write_str("true"),
      Self::False => f.write_str("false"),
    }
  }
}

impl Literal {
  /// Create a new assignment of binary expression
  pub fn binary_expr(left: Variable, op: BinaryExprOp, op1: Term, op2: Term) -> Self {
    Self::Assign(Assign {
      left,
      right: AssignExpr::Binary(BinaryAssignExpr { op, op1, op2 }),
    })
  }

  /// Create a new assignment of unary expression
  pub fn unary_expr(left: Variable, op: UnaryExprOp, op1: Term) -> Self {
    Self::Assign(Assign {
      left,
      right: AssignExpr::Unary(UnaryAssignExpr { op, op1 }),
    })
  }

  /// Create a new assignment of if-then-else expression
  pub fn if_then_else_expr(left: Variable, cond: Term, then_br: Term, else_br: Term) -> Self {
    Self::Assign(Assign {
      left,
      right: AssignExpr::IfThenElse(IfThenElseAssignExpr { cond, then_br, else_br }),
    })
  }

  /// Create a new assignment of call expression
  pub fn call_expr(left: Variable, function: String, args: Vec<Term>) -> Self {
    Self::Assign(Assign {
      left,
      right: AssignExpr::Call(CallExpr { function, args }),
    })
  }

  /// Create a new assignment of call expression
  pub fn new_expr(left: Variable, functor: String, args: Vec<Term>) -> Self {
    Self::Assign(Assign {
      left,
      right: AssignExpr::New(NewExpr { functor, args }),
    })
  }

  /// Create a new assignment of if-then-else expression
  pub fn binary_constraint(op: BinaryConstraintOp, op1: Term, op2: Term) -> Self {
    Self::Constraint(Constraint::Binary(BinaryConstraint { op, op1, op2 }))
  }

  pub fn unary_constraint(op: UnaryConstraintOp, op1: Term) -> Self {
    Self::Constraint(Constraint::Unary(UnaryConstraint { op, op1 }))
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Atom {
  pub predicate: String,
  pub args: Vec<Term>,
}

impl Atom {
  /// Create a new atom
  pub fn new(predicate: String, args: Vec<Term>) -> Self {
    Self { predicate, args }
  }

  /// An atom is pure means it contains only variable arguments and
  /// all variables are distinct.
  ///
  /// This implies this atom can be used directly as a dataflow.
  /// Otherwise, additional filter and find may apply.
  pub fn is_pure(&self) -> bool {
    let mut existed_args = HashSet::new();
    for a in &self.args {
      match a {
        Term::Variable(v) => {
          if existed_args.contains(v) {
            return false;
          } else {
            existed_args.insert(v.clone());
          }
        }
        _ => {
          return false;
        }
      }
    }
    return true;
  }

  /// Get the atom's arguments which are variables
  pub fn variable_args(&self) -> impl Iterator<Item = &Variable> {
    self.args.iter().filter_map(|a| match a {
      Term::Variable(v) => Some(v),
      _ => None,
    })
  }

  /// Get a set of unique variables in the atom's arguments
  pub fn unique_variable_args(&self) -> impl Iterator<Item = Variable> {
    self.variable_args().cloned().collect::<BTreeSet<_>>().into_iter()
  }

  /// Check if the atom has constant arguments
  pub fn has_constant_arg(&self) -> bool {
    self.args.iter().any(|a| a.is_constant())
  }

  /// Check if the constants in the atom's arguments are all in the front
  pub fn constant_args_are_upfront(&self) -> bool {
    let mut encountered_variable = false;
    for arg in &self.args {
      match arg {
        Term::Variable(_) => {
          encountered_variable = true;
        }
        Term::Constant(_) => {
          if encountered_variable {
            return false;
          }
        }
      }
    }
    return true;
  }

  /// Get the constant arguments
  pub fn constant_args(&self) -> impl Iterator<Item = &Constant> {
    self.args.iter().filter_map(|a| match a {
      Term::Constant(c) => Some(c),
      _ => None,
    })
  }

  /// Create a partition of the atom's arguments into constant and variable
  pub fn const_var_partition(&self) -> (Vec<(usize, &Constant)>, Vec<(usize, &Variable)>) {
    let (constants, variables): (Vec<_>, Vec<_>) = self.args.iter().enumerate().partition(|(_, t)| t.is_constant());
    let constants = constants
      .into_iter()
      .map(|(i, c)| match c {
        Term::Constant(c) => (i, c),
        _ => panic!("Should not happen"),
      })
      .collect();
    let variables = variables
      .into_iter()
      .map(|(i, v)| match v {
        Term::Variable(v) => (i, v),
        _ => panic!("Should not happen"),
      })
      .collect();
    (constants, variables)
  }

  /// Substitute the atom's arguments with the given term rewrite function
  pub fn substitute<F: Fn(&Term) -> Term>(&self, f: F) -> Self {
    Self {
      predicate: self.predicate.clone(),
      args: self.args.iter().map(|a| f(a)).collect(),
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NegAtom {
  pub atom: Atom,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Assign {
  pub left: Variable,
  pub right: AssignExpr,
}

impl Assign {
  pub fn variable_args(&self) -> Vec<&Variable> {
    let mut args = vec![];
    match &self.right {
      AssignExpr::Binary(b) => {
        args.extend(b.op1.as_variable().iter());
        args.extend(b.op2.as_variable().iter());
      }
      AssignExpr::Unary(u) => {
        args.extend(u.op1.as_variable().iter());
      }
      AssignExpr::IfThenElse(i) => {
        args.extend(i.cond.as_variable().iter());
        args.extend(i.then_br.as_variable().iter());
        args.extend(i.else_br.as_variable().iter());
      }
      AssignExpr::Call(c) => {
        for a in &c.args {
          args.extend(a.as_variable().iter());
        }
      }
      AssignExpr::New(n) => {
        for a in &n.args {
          args.extend(a.as_variable().iter());
        }
      }
    }
    args
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssignExpr {
  Binary(BinaryAssignExpr),
  Unary(UnaryAssignExpr),
  IfThenElse(IfThenElseAssignExpr),
  Call(CallExpr),
  New(NewExpr),
}

impl AssignExpr {
  pub fn is_new_expr(&self) -> bool {
    match self {
      Self::New(_) => true,
      _ => false,
    }
  }
}

pub type BinaryExprOp = crate::common::binary_op::BinaryOp;

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryAssignExpr {
  pub op: BinaryExprOp,
  pub op1: Term,
  pub op2: Term,
}

pub type UnaryExprOp = crate::common::unary_op::UnaryOp;

#[derive(Clone, Debug, PartialEq)]
pub struct UnaryAssignExpr {
  pub op: UnaryExprOp,
  pub op1: Term,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IfThenElseAssignExpr {
  pub cond: Term,
  pub then_br: Term,
  pub else_br: Term,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallExpr {
  pub function: String,
  pub args: Vec<Term>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NewExpr {
  pub functor: String,
  pub args: Vec<Term>,
}

/// A constraint literal which is either a binary or unary constraint
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
  Binary(BinaryConstraint),
  Unary(UnaryConstraint),
}

impl Constraint {
  /// Create a new equality constraint using two terms
  pub fn eq(op1: Term, op2: Term) -> Self {
    Self::Binary(BinaryConstraint {
      op: BinaryConstraintOp::Eq,
      op1,
      op2,
    })
  }

  /// Create a new inequality constraint using two terms
  pub fn neq(op1: Term, op2: Term) -> Self {
    Self::Binary(BinaryConstraint {
      op: BinaryConstraintOp::Neq,
      op1,
      op2,
    })
  }

  /// Create a new binary constraint using an operator and two terms
  pub fn binary(op: BinaryConstraintOp, op1: Term, op2: Term) -> Self {
    Self::Binary(BinaryConstraint { op, op1, op2 })
  }

  /// Find the variable arguments occurred in this constraint
  pub fn variable_args(&self) -> Vec<&Variable> {
    let mut args = vec![];
    match self {
      Self::Binary(b) => {
        args.extend(b.op1.as_variable().iter());
        args.extend(b.op2.as_variable().iter());
      }
      Self::Unary(u) => {
        args.extend(u.op1.as_variable().iter());
      }
    }
    args
  }

  pub fn unique_variable_args(&self) -> impl Iterator<Item = &Variable> {
    self.variable_args().into_iter().collect::<BTreeSet<_>>().into_iter()
  }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryConstraintOp {
  Eq,
  Neq,
  Lt,
  Leq,
  Gt,
  Geq,
}

impl From<&BinaryExprOp> for Option<BinaryConstraintOp> {
  fn from(op: &BinaryExprOp) -> Self {
    match op {
      BinaryExprOp::Eq => Some(BinaryConstraintOp::Eq),
      BinaryExprOp::Neq => Some(BinaryConstraintOp::Neq),
      BinaryExprOp::Gt => Some(BinaryConstraintOp::Gt),
      BinaryExprOp::Geq => Some(BinaryConstraintOp::Geq),
      BinaryExprOp::Lt => Some(BinaryConstraintOp::Lt),
      BinaryExprOp::Leq => Some(BinaryConstraintOp::Leq),
      _ => None,
    }
  }
}

impl From<&BinaryConstraintOp> for BinaryExprOp {
  fn from(op: &BinaryConstraintOp) -> Self {
    match op {
      BinaryConstraintOp::Eq => Self::Eq,
      BinaryConstraintOp::Neq => Self::Neq,
      BinaryConstraintOp::Gt => Self::Gt,
      BinaryConstraintOp::Geq => Self::Geq,
      BinaryConstraintOp::Lt => Self::Lt,
      BinaryConstraintOp::Leq => Self::Leq,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryConstraint {
  pub op: BinaryConstraintOp,
  pub op1: Term,
  pub op2: Term,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnaryConstraintOp {
  Not,
}

impl From<&front::ast::_UnaryOp> for Option<UnaryConstraintOp> {
  fn from(op: &front::ast::_UnaryOp) -> Self {
    match op {
      front::ast::_UnaryOp::Not => Some(UnaryConstraintOp::Not),
      _ => None,
    }
  }
}

impl From<&UnaryConstraintOp> for UnaryExprOp {
  fn from(op: &UnaryConstraintOp) -> Self {
    match op {
      UnaryConstraintOp::Not => Self::Not,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnaryConstraint {
  pub op: UnaryConstraintOp,
  pub op1: Term,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reduce {
  // Aggregator
  pub aggregator: String,
  pub params: Vec<Constant>,
  pub named_params: BTreeMap<String, Constant>,
  pub has_exclamation_mark: bool,

  // Concretized types of reduce arguments
  pub left_var_types: Vec<ValueType>,
  pub arg_var_types: Vec<ValueType>,
  pub input_var_types: Vec<ValueType>,

  // Variables
  pub left_vars: Vec<Variable>,
  pub group_by_vars: Vec<Variable>,
  pub other_group_by_vars: Vec<Variable>,
  pub arg_vars: Vec<Variable>,
  pub to_aggregate_vars: Vec<Variable>,

  // Bodies of reduce
  pub body_formula: Atom,
  pub group_by_formula: Option<Atom>,
}

impl Reduce {
  pub fn new(
    aggregator: String,
    params: Vec<Constant>,
    named_params: BTreeMap<String, Constant>,
    has_exclamation_mark: bool,
    left_var_types: Vec<ValueType>,
    arg_var_types: Vec<ValueType>,
    input_var_types: Vec<ValueType>,
    left_vars: Vec<Variable>,
    group_by_vars: Vec<Variable>,
    other_group_by_vars: Vec<Variable>,
    arg_vars: Vec<Variable>,
    to_aggregate_vars: Vec<Variable>,
    body_formula: Atom,
    group_by_formula: Option<Atom>,
  ) -> Self {
    Self {
      aggregator,
      params,
      named_params,
      has_exclamation_mark,
      left_var_types,
      arg_var_types,
      input_var_types,
      left_vars,
      group_by_vars,
      other_group_by_vars,
      arg_vars,
      to_aggregate_vars,
      body_formula,
      group_by_formula,
    }
  }

  pub fn variable_args(&self) -> impl Iterator<Item = &Variable> {
    self
      .left_vars
      .iter()
      .chain(self.arg_vars.iter())
      .chain(self.group_by_vars.iter())
  }
}
