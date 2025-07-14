use crate::compiler::front::*;

/// Transforming forall into not_exists
///
/// For example
///
/// ``` scl
/// b = forall(o: A(o) -> B(o))
/// ```
///
/// will be transformed into
///
/// ``` scl
/// temp_b = exists(o: A(o) and not B(o)), b = temp_b
/// ```
#[derive(Clone, Debug)]
pub struct TransformForall;

impl<'a> Transformation<'a> for TransformForall {}

impl TransformForall {
  pub fn new() -> Self {
    Self
  }

  fn transform_forall(&mut self, r: &Reduce) -> Option<Formula> {
    // First check if this reduce is a forall aggregation
    if r.operator().name().name() == "forall" && r.num_left() == 1 {
      // Unwrap ok since num_left is 1
      if let VariableOrWildcard::Variable(left_var) = &r.left().get(0).unwrap() {
        // Do the transformation
        match r.body() {
          Formula::Implies(i) => {
            // Create b = !b_temp constraint
            let temp_var_name = format!("{}#forall#temp", left_var.name());
            let temp_var = Variable::new(Identifier::new(temp_var_name));
            let not_temp_var = Expr::unary(UnaryExpr::new(UnaryOp::not(), Expr::Variable(temp_var.clone())));
            let left_var_expr = Expr::Variable(left_var.clone());
            let left_var_eq_not_temp_var =
              Expr::binary(BinaryExpr::new(BinaryOp::new_eq(), left_var_expr, not_temp_var));
            let constraint = Constraint::new(left_var_eq_not_temp_var);

            // Create exists aggregation literal
            let left_and_not_right = Formula::Conjunction(Conjunction::new_with_loc(
              vec![i.left().clone(), i.right().negate()],
              i.location().clone_without_id(),
            ));
            let reduce = Reduce::new_with_loc(
              vec![VariableOrWildcard::Variable(temp_var)], // Left
              ReduceOp::new_with_loc(
                Identifier::new_with_loc("exists".to_string(), r.operator().name().location().clone_without_id()),
                vec![],
                r.operator().has_exclaimation_mark().clone(),
                r.operator().location().clone_without_id(),
              ), // Reduce op
              r.args().clone(),                             // args
              r.bindings().clone(),                         // bindings
              left_and_not_right,
              r.group_by().clone(),
              i.location().clone_without_id(),
            );

            // Conjunction of both
            let result = Formula::Conjunction(Conjunction::new_with_loc(
              vec![Formula::Constraint(constraint), Formula::Reduce(reduce)],
              r.location().clone_without_id(),
            ));
            Some(result)
          }
          _ => None,
        }
      } else {
        None
      }
    } else {
      None
    }
  }
}

impl NodeVisitor<Formula> for TransformForall {
  fn visit_mut(&mut self, formula: &mut Formula) {
    match formula {
      Formula::Reduce(r) => {
        if let Some(f) = self.transform_forall(r) {
          *formula = f;
        }
      }
      _ => {}
    }
  }
}
