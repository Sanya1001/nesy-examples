use std::collections::*;

use crate::common::constants::*;
use crate::common::entity;
use crate::common::expr::*;
use crate::common::foreign_aggregate::*;
use crate::common::foreign_function::*;
use crate::common::foreign_predicate::*;
use crate::common::foreign_tensor;
use crate::common::tuple::*;
use crate::common::value::*;
use crate::common::value_type::*;
use crate::compiler::ram;
use crate::utils::*;

use super::*;

#[derive(Clone)]
pub struct RuntimeEnvironment {
  /// Random seed for reference
  pub random_seed: u64,

  /// Random number generater initialized from the random seed
  pub random: Random,

  /// Whether we want to early discard 0-tagged facts
  pub early_discard: bool,

  /// Stopping Criteria
  pub stopping_criteria: StoppingCriteria,

  /// Scheduling
  pub scheduler_manager: SchedulerManager,

  /// Foreign function registry
  pub function_registry: ForeignFunctionRegistry,

  /// Foreign predicate registry
  pub predicate_registry: ForeignPredicateRegistry,

  /// Foreign aggregate registry
  pub aggregate_registry: AggregateRegistry,

  /// Mutual exclusion ID allocator
  pub exclusion_id_allocator: IdAllocator2,

  /// Symbol registry
  pub symbol_registry: SymbolRegistry2,

  /// Dynamic entity storage
  pub dynamic_entity_store: DynamicEntityStorage2,

  /// Tensor registry
  pub tensor_registry: TensorRegistry2,
}

impl Default for RuntimeEnvironment {
  fn default() -> Self {
    Self::new_std()
  }
}

impl RuntimeEnvironment {
  pub fn new_std() -> Self {
    Self {
      random_seed: DEFAULT_RANDOM_SEED,
      random: Random::new(DEFAULT_RANDOM_SEED),
      early_discard: true,
      stopping_criteria: StoppingCriteria::default(),
      scheduler_manager: SchedulerManager::default(),
      function_registry: ForeignFunctionRegistry::std(),
      predicate_registry: ForeignPredicateRegistry::std(),
      aggregate_registry: AggregateRegistry::std(),
      exclusion_id_allocator: IdAllocator2::new(),
      symbol_registry: SymbolRegistry2::new(),
      dynamic_entity_store: DynamicEntityStorage2::new(),
      tensor_registry: TensorRegistry2::new(),
    }
  }

  pub fn new_with_random_seed(seed: u64) -> Self {
    Self {
      random_seed: seed,
      random: Random::new(seed),
      early_discard: true,
      stopping_criteria: StoppingCriteria::default(),
      scheduler_manager: SchedulerManager::default(),
      function_registry: ForeignFunctionRegistry::std(),
      predicate_registry: ForeignPredicateRegistry::std(),
      aggregate_registry: AggregateRegistry::std(),
      exclusion_id_allocator: IdAllocator2::new(),
      symbol_registry: SymbolRegistry2::new(),
      dynamic_entity_store: DynamicEntityStorage2::new(),
      tensor_registry: TensorRegistry2::new(),
    }
  }

  pub fn new(ffr: ForeignFunctionRegistry, fpr: ForeignPredicateRegistry, far: AggregateRegistry) -> Self {
    Self {
      random_seed: DEFAULT_RANDOM_SEED,
      random: Random::new(DEFAULT_RANDOM_SEED),
      early_discard: true,
      stopping_criteria: StoppingCriteria::default(),
      scheduler_manager: SchedulerManager::default(),
      function_registry: ffr,
      predicate_registry: fpr,
      aggregate_registry: far,
      exclusion_id_allocator: IdAllocator2::new(),
      symbol_registry: SymbolRegistry2::new(),
      dynamic_entity_store: DynamicEntityStorage2::new(),
      tensor_registry: TensorRegistry2::new(),
    }
  }

  pub fn new_with_function_registry(ffr: ForeignFunctionRegistry) -> Self {
    Self {
      random_seed: DEFAULT_RANDOM_SEED,
      random: Random::new(DEFAULT_RANDOM_SEED),
      early_discard: true,
      stopping_criteria: StoppingCriteria::default(),
      scheduler_manager: SchedulerManager::default(),
      function_registry: ffr,
      predicate_registry: ForeignPredicateRegistry::std(),
      aggregate_registry: AggregateRegistry::std(),
      exclusion_id_allocator: IdAllocator2::new(),
      symbol_registry: SymbolRegistry2::new(),
      dynamic_entity_store: DynamicEntityStorage2::new(),
      tensor_registry: TensorRegistry2::new(),
    }
  }

  pub fn new_from_options(options: RuntimeEnvironmentOptions) -> Self {
    let stopping_criteria = StoppingCriteria::default()
      .with_iter_limit(&options.iter_limit)
      .with_stop_when_goal_non_empty(options.stop_when_goal_non_empty);
    let scheduler_manager = if let Some(sche) = &options.default_scheduler {
      SchedulerManager::new_with_default_scheduler(sche.clone())
    } else {
      SchedulerManager::default()
    };
    Self {
      random_seed: options.random_seed,
      random: Random::new(options.random_seed),
      early_discard: options.early_discard,
      stopping_criteria,
      scheduler_manager,
      function_registry: ForeignFunctionRegistry::std(),
      predicate_registry: ForeignPredicateRegistry::std(),
      aggregate_registry: AggregateRegistry::std(),
      exclusion_id_allocator: IdAllocator2::new(),
      symbol_registry: SymbolRegistry2::new(),
      dynamic_entity_store: DynamicEntityStorage2::new(),
      tensor_registry: TensorRegistry2::new(),
    }
  }

  pub fn set_early_discard(&mut self, early_discard: bool) {
    self.early_discard = early_discard
  }

  pub fn set_iter_limit(&mut self, k: usize) {
    self.stopping_criteria.set_iter_limit(k);
  }

  pub fn remove_iter_limit(&mut self) {
    self.stopping_criteria.remove_iter_limit();
  }

  pub fn get_default_scheduler(&self) -> &Scheduler {
    self.scheduler_manager.get_default_scheduler()
  }

  pub fn get_scheduler(&self, name: &String) -> &Scheduler {
    self.scheduler_manager.get_scheduler(name)
  }

  pub fn allocate_new_exclusion_id(&self) -> usize {
    self.exclusion_id_allocator.alloc()
  }

  pub fn load_from_ram_program(&mut self, ram_program: &ram::Program) {
    self.function_registry = ram_program.function_registry.clone();
    self.predicate_registry = ram_program.predicate_registry.clone();
    self.aggregate_registry = ram_program.aggregate_registry.clone();
    self
      .dynamic_entity_store
      .update_variant_registry(ram_program.adt_variant_registry.clone());
  }

  pub fn internalize_tuple(&self, tup: &Tuple) -> Option<Tuple> {
    match tup {
      Tuple::Tuple(ts) => ts
        .iter()
        .map(|t| self.internalize_tuple(t))
        .collect::<Option<_>>()
        .map(Tuple::Tuple),
      Tuple::Value(v) => self.internalize_value(v).map(Tuple::Value),
    }
  }

  pub fn internalize_value(&self, val: &Value) -> Option<Value> {
    match val {
      Value::SymbolString(s) => {
        let symbol_id = self.symbol_registry.register(s.clone());
        Some(Value::Symbol(symbol_id))
      }
      Value::Tensor(t) => {
        let tensor_symbol = self.tensor_registry.register(t.clone());
        tensor_symbol.map(|s| Value::TensorValue(s.into()))
      }
      Value::EntityString(s) => self.dynamic_entity_store.compile_and_add_entity_string(s).ok(),
      other => Some(other.clone()),
    }
  }

  pub fn internalize_expr(&self, expr: &Expr) -> Option<Expr> {
    match expr {
      Expr::Access(a) => Some(Expr::Access(a.clone())),
      Expr::Tuple(t) => t
        .iter()
        .map(|e| self.internalize_expr(e))
        .collect::<Option<_>>()
        .map(Expr::Tuple),
      Expr::Binary(b) => Some(Expr::binary(
        b.op.clone(),
        self.internalize_expr(&b.op1)?,
        self.internalize_expr(&b.op2)?,
      )),
      Expr::Unary(u) => Some(Expr::unary(u.op.clone(), self.internalize_expr(&u.op1)?)),
      Expr::Call(c) => Some(Expr::call(
        c.function.clone(),
        c.args.iter().map(|e| self.internalize_expr(e)).collect::<Option<_>>()?,
      )),
      Expr::Constant(c) => self.internalize_value(c).map(Expr::Constant),
      Expr::IfThenElse(ite) => Some(Expr::ite(
        self.internalize_expr(&ite.cond)?,
        self.internalize_expr(&ite.then_br)?,
        self.internalize_expr(&ite.else_br)?,
      )),
      Expr::New(n) => Some(Expr::new(
        n.functor.clone(),
        n.args.iter().map(|e| self.internalize_expr(e)).collect::<Option<_>>()?,
      )),
    }
  }

  pub fn externalize_tuple(&self, tup: &Tuple) -> Option<Tuple> {
    match tup {
      Tuple::Tuple(ts) => Some(Tuple::Tuple(
        ts.iter()
          .map(|t| self.externalize_tuple(t))
          .collect::<Option<Box<[_]>>>()?,
      )),
      Tuple::Value(v) => self.externalize_value(v).map(Tuple::Value),
    }
  }

  pub fn externalize_value(&self, val: &Value) -> Option<Value> {
    match val {
      Value::Symbol(s) => {
        let symbol = self.symbol_registry.get_symbol(*s).expect("Cannot find symbol");
        Some(Value::SymbolString(symbol))
      }
      Value::TensorValue(t) => {
        let tensor = self.tensor_registry.eval(t);
        tensor.map(Value::Tensor)
      }
      other => Some(other.clone()),
    }
  }

  pub fn drain_new_entities<F: Fn(&str) -> bool>(&self, f: F) -> HashMap<String, Vec<Tuple>> {
    self.dynamic_entity_store.drain_entities(f)
  }

  pub fn eval(&self, expr: &Expr, tuple: &Tuple) -> Option<Tuple> {
    match expr {
      Expr::Tuple(t) => Some(Tuple::Tuple(
        t.iter().map(|e| self.eval(e, tuple)).collect::<Option<_>>()?,
      )),
      Expr::Access(a) => Some(tuple[a].clone()),
      Expr::Constant(c) => Some(Tuple::Value(c.clone())),
      Expr::Binary(b) => self.eval_binary(b, tuple),
      Expr::Unary(u) => self.eval_unary(u, tuple),
      Expr::IfThenElse(i) => self.eval_if_then_else(i, tuple),
      Expr::Call(c) => self.eval_call(c, tuple),
      Expr::New(n) => self.eval_new(n, tuple),
    }
  }

  pub fn eval_binary(&self, expr: &BinaryExpr, v: &Tuple) -> Option<Tuple> {
    use crate::common::binary_op::BinaryOp::*;
    use Value::*;

    // Recursively evaluate sub-expressions
    let lhs_v = self.eval(&expr.op1, v)?;
    let rhs_v = self.eval(&expr.op2, v)?;

    // Compute result
    let result = match (&expr.op, lhs_v, rhs_v) {
      // Addition
      (Add, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(I8(i1.saturating_add(i2))),
      (Add, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(I16(i1.saturating_add(i2))),
      (Add, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(I32(i1.saturating_add(i2))),
      (Add, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(I64(i1.saturating_add(i2))),
      (Add, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(I128(i1.saturating_add(i2))),
      (Add, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(ISize(i1.saturating_add(i2))),
      (Add, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(U8(i1.saturating_add(i2))),
      (Add, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(U16(i1.saturating_add(i2))),
      (Add, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(U32(i1.saturating_add(i2))),
      (Add, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(U64(i1.saturating_add(i2))),
      (Add, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(U128(i1.saturating_add(i2))),
      (Add, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(USize(i1.saturating_add(i2))),
      (Add, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(F32(i1 + i2)),
      (Add, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(F64(i1 + i2)),
      (Add, Tuple::Value(String(s1)), Tuple::Value(String(s2))) => Tuple::Value(String(format!("{}{}", s1, s2))),
      (Add, Tuple::Value(DateTime(t1)), Tuple::Value(Duration(d2))) => Tuple::Value(DateTime(t1 + d2)),
      (Add, Tuple::Value(Duration(d1)), Tuple::Value(DateTime(t2))) => Tuple::Value(DateTime(t2 + d1)),
      (Add, Tuple::Value(Duration(d1)), Tuple::Value(Duration(d2))) => Tuple::Value(Duration(d1 + d2)),
      (Add, Tuple::Value(TensorValue(v1)), Tuple::Value(TensorValue(v2))) => {
        v1.add(v2).map(TensorValue).map(Tuple::Value)?
      }
      (Add, Tuple::Value(TensorValue(v1)), Tuple::Value(F64(f2))) => {
        v1.add(f2.into()).map(TensorValue).map(Tuple::Value)?
      }
      (Add, Tuple::Value(F64(f1)), Tuple::Value(TensorValue(v2))) => {
        v2.add(f1.into()).map(TensorValue).map(Tuple::Value)?
      }
      (Add, b1, b2) => panic!("Cannot perform ADD on {:?} and {:?}", b1, b2),

      // Subtraction
      (Sub, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(I8(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(I16(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(I32(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(I64(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(I128(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(ISize(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(U8(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(U16(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(U32(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(U64(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(U128(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(USize(i1.saturating_sub(i2))),
      (Sub, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(F32(i1 - i2)),
      (Sub, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(F64(i1 - i2)),
      (Sub, Tuple::Value(DateTime(i1)), Tuple::Value(Duration(i2))) => Tuple::Value(DateTime(i1 - i2)),
      (Sub, Tuple::Value(DateTime(i1)), Tuple::Value(DateTime(i2))) => Tuple::Value(Duration((i1 - i2).into())),
      (Sub, Tuple::Value(Duration(i1)), Tuple::Value(Duration(i2))) => Tuple::Value(Duration(i1 - i2)),
      (Sub, Tuple::Value(TensorValue(v1)), Tuple::Value(TensorValue(v2))) => {
        v1.sub(v2).map(TensorValue).map(Tuple::Value)?
      }
      (Sub, Tuple::Value(TensorValue(v1)), Tuple::Value(F64(f2))) => {
        v1.sub(f2.into()).map(TensorValue).map(Tuple::Value)?
      }
      (Sub, Tuple::Value(F64(f1)), Tuple::Value(TensorValue(v2))) => foreign_tensor::TensorValue::from(f1)
        .sub(v2)
        .map(TensorValue)
        .map(Tuple::Value)?,
      (Sub, b1, b2) => panic!("Cannot perform SUB on {:?} and {:?}", b1, b2),

      // Multiplication
      (Mul, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(I8(i1 * i2)),
      (Mul, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(I16(i1 * i2)),
      (Mul, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(I32(i1 * i2)),
      (Mul, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(I64(i1 * i2)),
      (Mul, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(I128(i1 * i2)),
      (Mul, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(ISize(i1 * i2)),
      (Mul, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(U8(i1 * i2)),
      (Mul, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(U16(i1 * i2)),
      (Mul, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(U32(i1 * i2)),
      (Mul, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(U64(i1 * i2)),
      (Mul, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(U128(i1 * i2)),
      (Mul, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(USize(i1 * i2)),
      (Mul, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(F32(i1 * i2)),
      (Mul, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(F64(i1 * i2)),
      (Mul, Tuple::Value(Duration(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Duration(i1 * i2)),
      (Mul, Tuple::Value(I32(i1)), Tuple::Value(Duration(i2))) => Tuple::Value(Duration(i2 * i1)),
      (Mul, Tuple::Value(TensorValue(v1)), Tuple::Value(TensorValue(v2))) => {
        v1.mul(v2).map(TensorValue).map(Tuple::Value)?
      }
      (Mul, Tuple::Value(TensorValue(v1)), Tuple::Value(F64(f2))) => {
        v1.mul(f2.into()).map(TensorValue).map(Tuple::Value)?
      }
      (Mul, Tuple::Value(F64(f1)), Tuple::Value(TensorValue(v2))) => foreign_tensor::TensorValue::from(f1)
        .mul(v2)
        .map(TensorValue)
        .map(Tuple::Value)?,
      (Mul, b1, b2) => panic!("Cannot perform MUL on {:?} and {:?}", b1, b2),

      // Division
      (Div, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(I8(i1 / i2)),
      (Div, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(I16(i1 / i2)),
      (Div, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(I32(i1 / i2)),
      (Div, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(I64(i1 / i2)),
      (Div, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(I128(i1 / i2)),
      (Div, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(ISize(i1 / i2)),
      (Div, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(U8(i1 / i2)),
      (Div, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(U16(i1 / i2)),
      (Div, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(U32(i1 / i2)),
      (Div, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(U64(i1 / i2)),
      (Div, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(U128(i1 / i2)),
      (Div, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(USize(i1 / i2)),
      (Div, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => {
        let r = i1 / i2;
        if r.is_nan() {
          return None;
        } else {
          Tuple::Value(F32(r))
        }
      }
      (Div, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => {
        let r = i1 / i2;
        if r.is_nan() {
          return None;
        } else {
          Tuple::Value(F64(r))
        }
      }
      (Div, Tuple::Value(Duration(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Duration(i1 / i2)),
      (Div, b1, b2) => panic!("Cannot perform DIV on {:?} and {:?}", b1, b2),

      // Mod
      (Mod, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(I8(i1 % i2)),
      (Mod, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(I16(i1 % i2)),
      (Mod, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(I32(i1 % i2)),
      (Mod, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(I64(i1 % i2)),
      (Mod, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(I128(i1 % i2)),
      (Mod, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(ISize(i1 % i2)),
      (Mod, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(U8(i1 % i2)),
      (Mod, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(U16(i1 % i2)),
      (Mod, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(U32(i1 % i2)),
      (Mod, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(U64(i1 % i2)),
      (Mod, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(U128(i1 % i2)),
      (Mod, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(USize(i1 % i2)),
      (Mod, b1, b2) => panic!("Cannot perform MOD on {:?} and {:?}", b1, b2),

      // Boolean
      (And, Tuple::Value(Bool(b1)), Tuple::Value(Bool(b2))) => Tuple::Value(Bool(b1 && b2)),
      (And, b1, b2) => panic!("Cannot perform AND on {:?} and {:?}", b1, b2),
      (Or, Tuple::Value(Bool(b1)), Tuple::Value(Bool(b2))) => Tuple::Value(Bool(b1 || b2)),
      (Or, b1, b2) => panic!("Cannot perform OR on {:?} and {:?}", b1, b2),
      (Xor, Tuple::Value(Bool(b1)), Tuple::Value(Bool(b2))) => Tuple::Value(Bool(b1 ^ b2)),
      (Xor, b1, b2) => panic!("Cannot perform XOR on {:?} and {:?}", b1, b2),

      // Equal to
      (Eq, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(Char(i1)), Tuple::Value(Char(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(Bool(i1)), Tuple::Value(Bool(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(Str(i1)), Tuple::Value(Str(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(String(i1)), Tuple::Value(String(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(Symbol(i1)), Tuple::Value(Symbol(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(DateTime(i1)), Tuple::Value(DateTime(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(Duration(i1)), Tuple::Value(Duration(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(Entity(i1)), Tuple::Value(Entity(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, b1, b2) => panic!("Cannot perform EQ on {:?} and {:?}", b1, b2),

      // Not equal to
      (Neq, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(Char(i1)), Tuple::Value(Char(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(Bool(i1)), Tuple::Value(Bool(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(Str(i1)), Tuple::Value(Str(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(String(i1)), Tuple::Value(String(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(Symbol(i1)), Tuple::Value(Symbol(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(DateTime(i1)), Tuple::Value(DateTime(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(Duration(i1)), Tuple::Value(Duration(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(Entity(i1)), Tuple::Value(Entity(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, b1, b2) => panic!("Cannot perform NEQ on {:?} and {:?}", b1, b2),

      // Greater than
      (Gt, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(DateTime(i1)), Tuple::Value(DateTime(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(Duration(i1)), Tuple::Value(Duration(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, b1, b2) => panic!("Cannot perform GT on {:?} and {:?}", b1, b2),

      // Greater than or equal to
      (Geq, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(DateTime(i1)), Tuple::Value(DateTime(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(Duration(i1)), Tuple::Value(Duration(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, b1, b2) => panic!("Cannot perform GEQ on {:?} and {:?}", b1, b2),

      // Less than
      (Lt, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(DateTime(i1)), Tuple::Value(DateTime(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(Duration(i1)), Tuple::Value(Duration(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, b1, b2) => panic!("Cannot perform LT on {:?} and {:?}", b1, b2),

      // Less than or equal to
      (Leq, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(DateTime(i1)), Tuple::Value(DateTime(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(Duration(i1)), Tuple::Value(Duration(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, b1, b2) => panic!("Cannot perform LEQ on {:?} and {:?}", b1, b2),
    };
    Some(result)
  }

  pub fn eval_unary(&self, expr: &UnaryExpr, v: &Tuple) -> Option<Tuple> {
    use crate::common::unary_op::UnaryOp::*;
    use crate::common::value::Value::*;

    let arg_v = self.eval(&expr.op1, v)?;
    match (&expr.op, arg_v) {
      // Negative
      (Neg, Tuple::Value(I8(i))) => Some(Tuple::Value(I8(-i))),
      (Neg, Tuple::Value(I16(i))) => Some(Tuple::Value(I16(-i))),
      (Neg, Tuple::Value(I32(i))) => Some(Tuple::Value(I32(-i))),
      (Neg, Tuple::Value(I64(i))) => Some(Tuple::Value(I64(-i))),
      (Neg, Tuple::Value(I128(i))) => Some(Tuple::Value(I128(-i))),
      (Neg, Tuple::Value(ISize(i))) => Some(Tuple::Value(ISize(-i))),
      (Neg, Tuple::Value(F32(i))) => Some(Tuple::Value(F32(-i))),
      (Neg, Tuple::Value(F64(i))) => Some(Tuple::Value(F64(-i))),
      (Neg, v) => panic!("Negate operation cannot be operating on value of type {:?}", v),

      // Positive
      (Pos, x) => Some(x),

      // Not
      (Not, Tuple::Value(Bool(b))) => Some(Tuple::Value(Bool(!b))),
      (Not, v) => panic!("Not operation cannot be operating on value of type {:?}", v),

      // Type cast
      (TypeCast(dst), arg) => {
        use ValueType as T;
        match (arg, dst) {
          (Tuple::Value(I8(i)), T::I8) => Some(Tuple::Value(I8(i))),
          (Tuple::Value(I8(i)), T::I16) => Some(Tuple::Value(I16(i as i16))),
          (Tuple::Value(I8(i)), T::I32) => Some(Tuple::Value(I32(i as i32))),
          (Tuple::Value(I8(i)), T::I64) => Some(Tuple::Value(I64(i as i64))),
          (Tuple::Value(I8(i)), T::I128) => Some(Tuple::Value(I128(i as i128))),
          (Tuple::Value(I8(i)), T::ISize) => Some(Tuple::Value(ISize(i as isize))),
          (Tuple::Value(I8(i)), T::U8) => Some(Tuple::Value(U8(i as u8))),
          (Tuple::Value(I8(i)), T::U16) => Some(Tuple::Value(U16(i as u16))),
          (Tuple::Value(I8(i)), T::U32) => Some(Tuple::Value(U32(i as u32))),
          (Tuple::Value(I8(i)), T::U64) => Some(Tuple::Value(U64(i as u64))),
          (Tuple::Value(I8(i)), T::U128) => Some(Tuple::Value(U128(i as u128))),
          (Tuple::Value(I8(i)), T::USize) => Some(Tuple::Value(USize(i as usize))),
          (Tuple::Value(I8(i)), T::F32) => Some(Tuple::Value(F32(i as f32))),
          (Tuple::Value(I8(i)), T::F64) => Some(Tuple::Value(F64(i as f64))),

          (Tuple::Value(I32(i)), T::I8) => Some(Tuple::Value(I8(i as i8))),
          (Tuple::Value(I32(i)), T::I16) => Some(Tuple::Value(I16(i as i16))),
          (Tuple::Value(I32(i)), T::I32) => Some(Tuple::Value(I32(i))),
          (Tuple::Value(I32(i)), T::I64) => Some(Tuple::Value(I64(i as i64))),
          (Tuple::Value(I32(i)), T::I128) => Some(Tuple::Value(I128(i as i128))),
          (Tuple::Value(I32(i)), T::ISize) => Some(Tuple::Value(ISize(i as isize))),
          (Tuple::Value(I32(i)), T::U8) => Some(Tuple::Value(U8(i as u8))),
          (Tuple::Value(I32(i)), T::U16) => Some(Tuple::Value(U16(i as u16))),
          (Tuple::Value(I32(i)), T::U32) => Some(Tuple::Value(U32(i as u32))),
          (Tuple::Value(I32(i)), T::U64) => Some(Tuple::Value(U64(i as u64))),
          (Tuple::Value(I32(i)), T::U128) => Some(Tuple::Value(U128(i as u128))),
          (Tuple::Value(I32(i)), T::USize) => Some(Tuple::Value(USize(i as usize))),
          (Tuple::Value(I32(i)), T::F32) => Some(Tuple::Value(F32(i as f32))),
          (Tuple::Value(I32(i)), T::F64) => Some(Tuple::Value(F64(i as f64))),

          (Tuple::Value(USize(i)), T::I8) => Some(Tuple::Value(I8(i as i8))),
          (Tuple::Value(USize(i)), T::I16) => Some(Tuple::Value(I16(i as i16))),
          (Tuple::Value(USize(i)), T::I32) => Some(Tuple::Value(I32(i as i32))),
          (Tuple::Value(USize(i)), T::I64) => Some(Tuple::Value(I64(i as i64))),
          (Tuple::Value(USize(i)), T::I128) => Some(Tuple::Value(I128(i as i128))),
          (Tuple::Value(USize(i)), T::ISize) => Some(Tuple::Value(ISize(i as isize))),
          (Tuple::Value(USize(i)), T::U8) => Some(Tuple::Value(U8(i as u8))),
          (Tuple::Value(USize(i)), T::U16) => Some(Tuple::Value(U16(i as u16))),
          (Tuple::Value(USize(i)), T::U32) => Some(Tuple::Value(U32(i as u32))),
          (Tuple::Value(USize(i)), T::U64) => Some(Tuple::Value(U64(i as u64))),
          (Tuple::Value(USize(i)), T::U128) => Some(Tuple::Value(U128(i as u128))),
          (Tuple::Value(USize(i)), T::USize) => Some(Tuple::Value(USize(i))),
          (Tuple::Value(USize(i)), T::F32) => Some(Tuple::Value(F32(i as f32))),
          (Tuple::Value(USize(i)), T::F64) => Some(Tuple::Value(F64(i as f64))),

          (Tuple::Value(Char(s)), T::I8) => s.to_digit(10).map(|i| Tuple::Value(I8(i as i8))),
          (Tuple::Value(Char(s)), T::I16) => s.to_digit(10).map(|i| Tuple::Value(I16(i as i16))),
          (Tuple::Value(Char(s)), T::I32) => s.to_digit(10).map(|i| Tuple::Value(I32(i as i32))),
          (Tuple::Value(Char(s)), T::I64) => s.to_digit(10).map(|i| Tuple::Value(I64(i as i64))),
          (Tuple::Value(Char(s)), T::I128) => s.to_digit(10).map(|i| Tuple::Value(I128(i as i128))),
          (Tuple::Value(Char(s)), T::ISize) => s.to_digit(10).map(|i| Tuple::Value(ISize(i as isize))),
          (Tuple::Value(Char(s)), T::U8) => s.to_digit(10).map(|i| Tuple::Value(U8(i as u8))),
          (Tuple::Value(Char(s)), T::U16) => s.to_digit(10).map(|i| Tuple::Value(U16(i as u16))),
          (Tuple::Value(Char(s)), T::U32) => s.to_digit(10).map(|i| Tuple::Value(U32(i as u32))),
          (Tuple::Value(Char(s)), T::U64) => s.to_digit(10).map(|i| Tuple::Value(U64(i as u64))),
          (Tuple::Value(Char(s)), T::U128) => s.to_digit(10).map(|i| Tuple::Value(U128(i as u128))),
          (Tuple::Value(Char(s)), T::USize) => s.to_digit(10).map(|i| Tuple::Value(USize(i as usize))),
          (Tuple::Value(Char(s)), T::F32) => s.to_string().parse().ok().map(|f| Tuple::Value(F32(f))),
          (Tuple::Value(Char(s)), T::F64) => s.to_string().parse().ok().map(|f| Tuple::Value(F64(f))),

          (Tuple::Value(String(s)), T::I8) => s.parse().ok().map(|i| Tuple::Value(I8(i))),
          (Tuple::Value(String(s)), T::I16) => s.parse().ok().map(|i| Tuple::Value(I16(i))),
          (Tuple::Value(String(s)), T::I32) => s.parse().ok().map(|i| Tuple::Value(I32(i))),
          (Tuple::Value(String(s)), T::I64) => s.parse().ok().map(|i| Tuple::Value(I64(i))),
          (Tuple::Value(String(s)), T::I128) => s.parse().ok().map(|i| Tuple::Value(I128(i))),
          (Tuple::Value(String(s)), T::ISize) => s.parse().ok().map(|i| Tuple::Value(ISize(i))),
          (Tuple::Value(String(s)), T::U8) => s.parse().ok().map(|i| Tuple::Value(U8(i))),
          (Tuple::Value(String(s)), T::U16) => s.parse().ok().map(|i| Tuple::Value(U16(i))),
          (Tuple::Value(String(s)), T::U32) => s.parse().ok().map(|i| Tuple::Value(U32(i))),
          (Tuple::Value(String(s)), T::U64) => s.parse().ok().map(|i| Tuple::Value(U64(i))),
          (Tuple::Value(String(s)), T::U128) => s.parse().ok().map(|i| Tuple::Value(U128(i))),
          (Tuple::Value(String(s)), T::USize) => s.parse().ok().map(|i| Tuple::Value(USize(i))),
          (Tuple::Value(String(s)), T::F32) => s.parse().ok().map(|i| Tuple::Value(F32(i))),
          (Tuple::Value(String(s)), T::F64) => s.parse().ok().map(|i| Tuple::Value(F64(i))),

          (Tuple::Value(F32(f)), T::F64) => Some(Tuple::Value(F64(f as f64))),
          (Tuple::Value(F64(f)), T::F32) => Some(Tuple::Value(F32(f as f32))),

          (Tuple::Value(I8(i)), T::String) => Some(Tuple::Value(String(i.to_string()))),
          (Tuple::Value(I16(i)), T::String) => Some(Tuple::Value(String(i.to_string()))),
          (Tuple::Value(I32(i)), T::String) => Some(Tuple::Value(String(i.to_string()))),
          (Tuple::Value(I64(i)), T::String) => Some(Tuple::Value(String(i.to_string()))),
          (Tuple::Value(I128(i)), T::String) => Some(Tuple::Value(String(i.to_string()))),
          (Tuple::Value(ISize(i)), T::String) => Some(Tuple::Value(String(i.to_string()))),
          (Tuple::Value(U8(u)), T::String) => Some(Tuple::Value(String(u.to_string()))),
          (Tuple::Value(U16(u)), T::String) => Some(Tuple::Value(String(u.to_string()))),
          (Tuple::Value(U32(u)), T::String) => Some(Tuple::Value(String(u.to_string()))),
          (Tuple::Value(U64(u)), T::String) => Some(Tuple::Value(String(u.to_string()))),
          (Tuple::Value(U128(u)), T::String) => Some(Tuple::Value(String(u.to_string()))),
          (Tuple::Value(USize(u)), T::String) => Some(Tuple::Value(String(u.to_string()))),
          (Tuple::Value(F32(f)), T::String) => Some(Tuple::Value(String(f.to_string()))),
          (Tuple::Value(F64(f)), T::String) => Some(Tuple::Value(String(f.to_string()))),
          (Tuple::Value(Bool(b)), T::String) => Some(Tuple::Value(String(b.to_string()))),
          (Tuple::Value(Char(c)), T::String) => Some(Tuple::Value(String(c.to_string()))),
          (Tuple::Value(Str(s)), T::String) => Some(Tuple::Value(String(s.to_string()))),
          (Tuple::Value(String(s)), T::String) => Some(Tuple::Value(String(s.clone()))),
          (Tuple::Value(Symbol(id)), T::String) => Some(Tuple::Value(String(
            self
              .symbol_registry
              .get_symbol(id)
              .expect("[Internal Error] Cannot find symbol"),
          ))),

          // Not implemented
          (v, t) => unimplemented!("Unimplemented type cast from `{:?}` to `{}`", v.tuple_type(), t),
        }
      }
    }
  }

  pub fn eval_if_then_else(&self, expr: &IfThenElseExpr, v: &Tuple) -> Option<Tuple> {
    if self.eval(&expr.cond, v)?.as_bool() {
      self.eval(&expr.then_br, v)
    } else {
      self.eval(&expr.else_br, v)
    }
  }

  pub fn eval_call(&self, expr: &CallExpr, v: &Tuple) -> Option<Tuple> {
    // Get a function
    self.function_registry.get(&expr.function).and_then(|f| {
      // Arguments
      let args = expr
        .args
        .iter()
        .map(|a| self.eval(a, v).map(|t| t.as_value()))
        .collect::<Option<Vec<_>>>()?;

      // Run the function
      let result = f.execute_with_env(self, args)?;
      let internal_result = self.internalize_value(&result)?;

      // Turn result into tuple
      Some(Tuple::Value(internal_result))
    })
  }

  pub fn eval_new(&self, expr: &NewExpr, v: &Tuple) -> Option<Tuple> {
    // Evaluate the arguments
    let args = expr
      .args
      .iter()
      .map(|a| self.eval(a, v).map(|t| t.as_value()))
      .collect::<Option<Vec<_>>>()?;

    // Hash all the arguments to get a new entity
    let raw_id = entity::encode_entity(&expr.functor, args.iter());
    let id = Value::Entity(raw_id);

    // Combine them to form a tuple for later insertion to new entities list
    let tuple = Tuple::from_values(args.into_iter());
    self
      .dynamic_entity_store
      .add_entity_fact(&expr.functor, id.clone(), tuple);

    // Return the value
    Some(Tuple::Value(id))
  }
}
