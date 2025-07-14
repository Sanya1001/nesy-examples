use crate::common::binary_op::BinaryOp;
use crate::common::entity;
use crate::common::expr::{BinaryExpr, Expr, UnaryExpr};
use crate::common::foreign_function::*;
use crate::common::unary_op::UnaryOp;
use crate::common::value::Value;
use crate::runtime::env::*;

use super::super::*;

pub fn constant_fold(rule: &mut Rule, function_registry: &ForeignFunctionRegistry) {
  let runtime = RuntimeEnvironment::new_with_function_registry(function_registry.clone());
  for lit in rule.body_literals_mut() {
    match lit {
      Literal::Assign(a) => match &a.right {
        AssignExpr::Binary(b) => match (&b.op1, &b.op2) {
          (Term::Constant(c1), Term::Constant(c2)) => {
            let expr = BinaryExpr {
              op: b.op.clone().into(),
              op1: Box::new(Expr::Constant(c1.clone())),
              op2: Box::new(Expr::Constant(c2.clone())),
            };
            let maybe_result = runtime.eval_binary(&expr, &().into());
            if let Some(result) = maybe_result {
              *lit = Literal::Constraint(Constraint::Binary(BinaryConstraint {
                op: BinaryConstraintOp::Eq,
                op1: Term::Variable(a.left.clone()),
                op2: Term::Constant(result.as_value()),
              }));
            } else {
              *lit = Literal::False
            }
          }
          _ => {}
        },
        AssignExpr::Unary(u) => match &u.op1 {
          Term::Constant(c1) => {
            let expr = UnaryExpr {
              op: u.op.clone().into(),
              op1: Box::new(Expr::Constant(c1.clone())),
            };
            let maybe_result = runtime.eval_unary(&expr, &().into());
            if let Some(result) = maybe_result {
              *lit = Literal::Constraint(Constraint::Binary(BinaryConstraint {
                op: BinaryConstraintOp::Eq,
                op1: Term::Variable(a.left.clone()),
                op2: Term::Constant(result.as_value()),
              }));
            } else {
              *lit = Literal::False
            }
          }
          _ => {}
        },
        AssignExpr::IfThenElse(i) => match &i.cond {
          Term::Constant(Value::Bool(true)) => {
            *lit = Literal::Constraint(Constraint::Binary(BinaryConstraint {
              op: BinaryConstraintOp::Eq,
              op1: Term::Variable(a.left.clone()),
              op2: i.then_br.clone(),
            }))
          }
          Term::Constant(Value::Bool(false)) => {
            *lit = Literal::Constraint(Constraint::Binary(BinaryConstraint {
              op: BinaryConstraintOp::Eq,
              op1: Term::Variable(a.left.clone()),
              op2: i.else_br.clone(),
            }))
          }
          _ => {}
        },
        // AssignExpr::Call(c) => {
        //   let all_constant = c.args.iter().all(|a| a.is_constant());
        //   if all_constant {
        //     if let Some(f) = runtime.function_registry.get(&c.function) {
        //       let args = c.args.iter().map(|a| a.as_constant().unwrap().clone()).collect();
        //       let maybe_value = f.execute(args);
        //       if let Some(value) = maybe_value {
        //         *lit = Literal::Constraint(Constraint::Binary(BinaryConstraint {
        //           op: BinaryConstraintOp::Eq,
        //           op1: Term::Variable(a.left.clone()),
        //           op2: Term::Constant(value),
        //         }))
        //       } else {
        //         *lit = Literal::False
        //       }
        //     }
        //   }
        // }
        AssignExpr::New(n) => {
          let all_constant = n.args.iter().all(|a| a.is_constant());
          if all_constant {
            let raw_id = entity::encode_entity(&n.functor, n.args.iter().map(|a| a.as_constant().unwrap()));
            let id = Term::Constant(Constant::Entity(raw_id));
            *lit = Literal::Constraint(Constraint::Binary(BinaryConstraint {
              op: BinaryConstraintOp::Eq,
              op1: Term::Variable(a.left.clone()),
              op2: id,
            }))
          }
        }
        _ => {}
      },
      Literal::Constraint(c) => match c {
        Constraint::Binary(b) => match (&b.op, &b.op1, &b.op2) {
          (op, Term::Constant(c1), Term::Constant(c2)) => {
            let expr = BinaryExpr {
              op: BinaryOp::from(op),
              op1: Box::new(Expr::Constant(c1.clone())),
              op2: Box::new(Expr::Constant(c2.clone())),
            };
            let maybe_result = runtime.eval_binary(&expr, &().into());
            if let Some(result) = maybe_result {
              if result.as_bool() {
                *lit = Literal::True;
              } else {
                *lit = Literal::False;
              }
            } else {
              *lit = Literal::False;
            }
          }
          (BinaryConstraintOp::Neq, Term::Variable(v1), Term::Variable(v2)) if v1 == v2 => {
            *lit = Literal::False;
          }
          (BinaryConstraintOp::Eq, Term::Variable(v1), Term::Variable(v2)) if v1 == v2 => {
            *lit = Literal::True;
          }
          _ => {}
        },
        Constraint::Unary(u) => match &u.op1 {
          Term::Constant(c1) => {
            let expr = UnaryExpr {
              op: UnaryOp::from(&u.op),
              op1: Box::new(Expr::Constant(c1.clone())),
            };
            let maybe_result = runtime.eval_unary(&expr, &().into());
            if let Some(result) = maybe_result {
              if result.as_bool() {
                *lit = Literal::True;
              } else {
                *lit = Literal::False;
              }
            } else {
              *lit = Literal::False;
            }
          }
          _ => {}
        },
      },
      _ => {}
    }
  }
}
