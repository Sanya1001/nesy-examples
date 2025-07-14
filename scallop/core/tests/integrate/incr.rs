use scallop_core::integrate;
use scallop_core::runtime::provenance;
use scallop_core::testing::*;
use scallop_core::utils::RcFamily;

#[test]
fn incr_edge_path_left_recursion() {
  let prov_ctx = provenance::unit::UnitProvenance::default();
  let mut ctx = integrate::IntegrateContext::<_, RcFamily>::new(prov_ctx);

  // Source
  ctx.add_relation("edge(usize, usize)").unwrap();
  ctx
    .add_rule(r#"path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)"#)
    .unwrap();

  // Facts
  ctx
    .add_facts(
      "edge",
      vec![(None, (0usize, 1usize).into()), (None, (1usize, 2usize).into())],
      false,
    )
    .unwrap();

  // Execution
  ctx.run().unwrap();

  // Result
  expect_output_collection(
    "path",
    ctx.computed_relation_ref("path").unwrap(),
    vec![(0usize, 1usize), (0, 2), (1, 2)],
  );
}

#[test]
fn incr_edge_path_left_branching_1() {
  let prov_ctx = provenance::unit::UnitProvenance::default();
  let mut ctx = integrate::IntegrateContext::<_>::new_incremental(prov_ctx);

  // Base context
  ctx.add_relation("edge(usize, usize)").unwrap();
  ctx
    .add_facts(
      "edge",
      vec![(None, (0usize, 1usize).into()), (None, (1usize, 2usize).into())],
      false,
    )
    .unwrap();
  ctx.run().unwrap();
  expect_output_collection(
    "edge",
    ctx.computed_relation_ref("edge").unwrap(),
    vec![(0usize, 1usize), (1, 2)],
  );

  // First branch
  let mut first_branch = ctx.clone();
  first_branch
    .add_rule(r#"path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)"#)
    .unwrap();
  first_branch.run().unwrap();
  expect_output_collection(
    "path",
    first_branch.computed_relation_ref("path").unwrap(),
    vec![(0usize, 1usize), (0, 2), (1, 2)],
  );

  // Second branch
  let mut second_branch = ctx.clone();
  second_branch
    .add_rule(r#"path(a, c) = edge(a, c) \/ edge(a, b) /\ path(b, c)"#)
    .unwrap();
  second_branch.run().unwrap();
  expect_output_collection(
    "path",
    second_branch.computed_relation_ref("path").unwrap(),
    vec![(0usize, 1usize), (0, 2), (1, 2)],
  );

  // Second branch, continuation
  second_branch
    .add_rule(r#"result(x, y) = path(x, y) and x == 1"#)
    .unwrap();
  second_branch.run().unwrap();
  expect_output_collection(
    "result",
    second_branch.computed_relation_ref("result").unwrap(),
    vec![(1usize, 2usize)],
  );
}

#[test]
fn incr_fib_test_0() {
  let prov_ctx = provenance::unit::UnitProvenance::default();
  let mut ctx = integrate::IntegrateContext::<_>::new_incremental(prov_ctx);

  ctx.add_relation("fib(i32, i32)").expect("Compile error");
  ctx
    .add_rule("fib(x, a + b) :- fib(x - 1, a), fib(x - 2, b), x <= 5")
    .expect("Compile error");
  ctx
    .edb()
    .add_facts("fib", vec![(0i32, 1i32), (1, 1)])
    .expect("Cannot add facts");

  ctx.run().expect("Runtime error");

  expect_output_collection(
    "fib",
    ctx.computed_relation_ref("fib").unwrap(),
    vec![(0i32, 1i32), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8)],
  );
}

#[test]
fn incr_fib_test_1() {
  let prov_ctx = provenance::unit::UnitProvenance::default();
  let mut ctx = integrate::IntegrateContext::<_>::new_incremental(prov_ctx);

  ctx
    .add_program(
      r#"
      rel fib = {(0, 1), (1, 1)}
      rel fib(x, a + b) = fib(x - 1, a) and fib(x - 2, b) and x <= 5
    "#,
    )
    .expect("Compile error");

  ctx.run().expect("Runtime error");

  expect_output_collection(
    "fib",
    ctx.computed_relation_ref("fib").unwrap(),
    vec![(0i32, 1i32), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8)],
  );
}
