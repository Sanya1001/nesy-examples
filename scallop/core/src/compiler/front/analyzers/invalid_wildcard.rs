use super::super::*;

#[derive(Clone, Debug)]
pub struct InvalidWildcardAnalyzer {
  pub errors: Vec<InvalidWildcardError>,
}

impl InvalidWildcardAnalyzer {
  pub fn new() -> Self {
    Self { errors: Vec::new() }
  }

  pub fn check_expr(&mut self, expr: &Expr, position: &'static str) {
    if expr.is_wildcard() {
      self.errors.push(InvalidWildcardError::InvalidWildcard {
        wildcard_loc: expr.location().clone(),
        position,
      });
    }
  }
}

impl NodeVisitor<Rule> for InvalidWildcardAnalyzer {
  fn visit(&mut self, rule: &Rule) {
    for arg in rule.head().iter_args() {
      self.check_expr(arg, "head of rule");
    }
  }
}

impl NodeVisitor<BinaryExpr> for InvalidWildcardAnalyzer {
  fn visit(&mut self, binary_expr: &BinaryExpr) {
    self.check_expr(binary_expr.op1(), "binary expression");
    self.check_expr(binary_expr.op2(), "binary expression");
  }
}

impl NodeVisitor<UnaryExpr> for InvalidWildcardAnalyzer {
  fn visit(&mut self, unary_expr: &UnaryExpr) {
    self.check_expr(unary_expr.op1(), "unary expression");
  }
}

impl NodeVisitor<IfThenElseExpr> for InvalidWildcardAnalyzer {
  fn visit(&mut self, if_then_else_expr: &IfThenElseExpr) {
    self.check_expr(if_then_else_expr.cond(), "if-then-else expression");
    self.check_expr(if_then_else_expr.then_br(), "if-then-else expression");
    self.check_expr(if_then_else_expr.else_br(), "if-then-else expression");
  }
}

#[derive(Clone, Debug)]
pub enum InvalidWildcardError {
  InvalidWildcard {
    wildcard_loc: NodeLocation,
    position: &'static str,
  },
}

impl FrontCompileErrorTrait for InvalidWildcardError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::InvalidWildcard { wildcard_loc, position } => {
        format!("Invalid wildcard in the {}:\n{}", position, wildcard_loc.report(src))
      }
    }
  }
}
