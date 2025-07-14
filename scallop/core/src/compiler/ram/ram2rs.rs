use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::*;

use super::*;
use crate::common::binary_op::*;
use crate::common::expr::*;
use crate::common::input_tag::*;
use crate::common::output_option::OutputOption;
use crate::common::tuple::*;
use crate::common::tuple_type::*;
use crate::common::unary_op::UnaryOp;
use crate::common::value::*;
use crate::common::value_type::*;
use crate::compiler::options::CompileOptions;

impl ast::Program {
  pub fn to_rs_module(&self, opt: &CompileOptions) -> TokenStream {
    // Precomputed cache
    let rel_to_strat_map = compute_relation_to_stratum_map(self);
    let strat_dep = compute_stratum_dependency(self, &rel_to_strat_map);
    let inters_dep = compute_interstratum_dependency(self);

    // Final output struct
    let final_result_relations = self
      .relations()
      .filter(|r| r.output == OutputOption::Default)
      .collect::<Vec<_>>();
    let final_result_fields = final_result_relations
      .iter()
      .map(|r| {
        let field = r.to_rs_output_struct_field(opt);
        quote! { pub #field }
      })
      .collect::<Vec<_>>();
    let final_result_struct_decl = quote! {
      #[derive(Clone)]
      pub struct OutputRelations<C: Provenance> { #(#final_result_fields)* }
    };

    // Strata Struct/run code
    let strata_rs = self
      .strata
      .iter()
      .enumerate()
      .map(|(i, stratum)| {
        let result_struct_decl = stratum.to_rs_result_struct_decl(i, &inters_dep, opt);
        let run_fn = stratum.to_rs_run_fn(i, &rel_to_strat_map, &strat_dep, &inters_dep);
        quote! { #result_struct_decl #run_fn }
      })
      .collect::<Vec<_>>();

    // Execute all strata
    let exec_strata = self
      .strata
      .iter()
      .enumerate()
      .map(|(i, _)| {
        let curr_strat_name = format_ident!("stratum_{}_result", i);
        let curr_strat_run_name = format_ident!("run_stratum_{}", i);
        let args = strat_dep[&i]
          .iter()
          .map(|dep_id| {
            let dep_strat_name = format_ident!("stratum_{}_result", dep_id);
            quote! { &#dep_strat_name, }
          })
          .collect::<Vec<_>>();
        quote! { let #curr_strat_name = #curr_strat_run_name(&rt, ctx, &mut edb, #(#args)*); }
      })
      .collect::<Vec<_>>();

    // Ensemble the final output
    let ensemble_output_relations = final_result_relations
      .iter()
      .map(|r| {
        let field_name = relation_name_to_rs_field_name(&r.predicate);
        let strat_id = rel_to_strat_map[&r.predicate];
        let strat_result_ident = format_ident!("stratum_{}_result", strat_id);
        quote! { #field_name: #strat_result_ident.#field_name.recover(ctx), }
      })
      .collect::<Vec<_>>();
    let output_relations = quote! { OutputRelations { #(#ensemble_output_relations)* } };

    // Composite
    quote! {
      use scallop_core::common::input_tag::*;
      use scallop_core::runtime::provenance::*;
      use scallop_core::runtime::env::*;
      use scallop_core::runtime::statics::*;
      use scallop_core::runtime::database::extensional::*;
      #(#strata_rs)*
      #final_result_struct_decl
      pub fn run<C: Provenance>(ctx: &mut C) -> OutputRelations<C> {
        run_with_edb(ctx, ExtensionalDatabase::new())
      }
      pub fn run_with_edb<C: Provenance>(ctx: &mut C, mut edb: ExtensionalDatabase<C>) -> OutputRelations<C> {
        let rt = RuntimeEnvironment::default();
        edb.internalize(&rt, ctx);
        #(#exec_strata)*
        #output_relations
      }
    }
  }

  pub fn to_rs_create_edb_fn(&self) -> TokenStream {
    // Generate relation types
    let relation_types = self
      .relations()
      .map(|rel| {
        let name = rel.predicate.clone();
        let ty = tuple_type_to_rs_type(&rel.tuple_type);
        quote! { (#name.to_string(), <TupleType as FromType<#ty>>::from_type()) }
      })
      .collect::<Vec<_>>();

    // Generate a `create_edb` function
    quote! {
      mod create_edb_fn {
        use scallop_core::common::tuple_type::TupleType;
        use scallop_core::common::value_type::FromType;
        use scallop_core::runtime::database::extensional::*;
        use scallop_core::runtime::provenance::*;
        pub fn create_edb<C: Provenance>() -> ExtensionalDatabase<C> {
          ExtensionalDatabase::with_relation_types(vec![ #(#relation_types),* ].into_iter())
        }
      }
      pub use create_edb_fn::create_edb;
    }
  }

  pub fn to_rs_output(&self, result_name: &str) -> TokenStream {
    let result_ident = format_ident!("{}", result_name);
    let outputs = self
      .strata
      .iter()
      .map(|stratum| {
        stratum
          .relations
          .iter()
          .filter_map(|(_, relation)| match &relation.output {
            OutputOption::Default => {
              let field_name = relation_name_to_rs_field_name(&relation.predicate);
              let relation_name = relation.predicate.clone();
              Some(quote! {
                println!("{}: {}", #relation_name, #result_ident.#field_name);
              })
            }
            _ => None,
          })
      })
      .flatten()
      .collect::<Vec<_>>();
    quote! { #(#outputs)* }
  }
}

impl ast::Stratum {
  pub fn to_rs_result_struct_decl(
    &self,
    id: usize,
    inters_dep: &InterStratumDependency,
    opt: &CompileOptions,
  ) -> TokenStream {
    let struct_name = format_ident!("Stratum{}Result", id);
    let fields = self
      .relations
      .iter()
      .filter_map(|(_, r)| {
        if r.output.is_not_hidden() || inters_dep.contains(&r.predicate) {
          Some(r.to_rs_result_struct_field(opt))
        } else {
          None
        }
      })
      .collect::<Vec<_>>();
    quote! { struct #struct_name<C: Provenance> { #(#fields)* } }
  }

  pub fn to_rs_run_fn(
    &self,
    id: usize,
    rel_to_strat_map: &RelationToStratumMap,
    strat_dep: &StratumDependency,
    inters_dep: &InterStratumDependency,
  ) -> TokenStream {
    // Signature
    let fn_name = format_ident!("run_stratum_{}", id);
    let args = strat_dep[&id].iter().map(|dep_id| {
      let stratum_input_name = format_ident!("stratum_{}_result", dep_id);
      let stratum_input_type = format_ident!("Stratum{}Result", dep_id);
      quote! { #stratum_input_name: &#stratum_input_type<C>, }
    });
    let ret_ty = format_ident!("Stratum{}Result", id);

    // 1. Create relations
    let create_relation_stmts = self
      .relations
      .iter()
      .map(|(predicate, relation)| {
        // 1.1. Create relation
        let rs_rel_name = relation_name_to_rs_field_name(&predicate);
        let rs_ty = tuple_type_to_rs_type(&relation.tuple_type);
        let create_stmt = quote! { let #rs_rel_name = iter.create_relation::<#rs_ty>(); };

        // 1.2. Add facts
        let tuples = relation
          .facts
          .iter()
          .map(|f| {
            let tag = input_tag_to_rs_input_tag(&f.tag);
            let tup = tuple_to_rs_tuple(&f.tuple);
            quote! { (#tag, #tup) }
          })
          .collect::<Vec<_>>();
        let add_fact_stmt = if tuples.is_empty() {
          quote! {}
        } else {
          quote! { #rs_rel_name.insert_dynamically_tagged(iter.provenance_context, vec![#(#tuples),*]); }
        };

        // 1.3. Load from edb
        let load_from_edb_stmt =
          quote! { edb.load_into_static_relation(#predicate, iter.provenance_context, &#rs_rel_name); };

        // Ensemble statements
        quote! { #create_stmt #add_fact_stmt #load_from_edb_stmt }
      })
      .collect::<Vec<_>>();

    // 2. Iteration
    let updates = self
      .updates
      .iter()
      .map(|update| update.to_rs_insert(id, rel_to_strat_map))
      .collect::<Vec<_>>();

    // 2.5. complete statements
    let complete_stmts = self
      .relations
      .iter()
      .map(|(_, r)| {
        let field_name = relation_name_to_rs_field_name(&r.predicate);
        quote! {
          let #field_name = iter.complete(&#field_name);
        }
        // let relation_name = r.predicate.clone();
        // quote! { println!("{}: {:?}", #relation_name, #field_name); }
      })
      .collect::<Vec<_>>();
    let complete_stmts = quote! { #(#complete_stmts)* };

    // 3. Ensemble final result
    let ensemble_result_fields = self
      .relations
      .iter()
      .filter_map(|(_, r)| {
        if r.output.is_not_hidden() || inters_dep.contains(&r.predicate) {
          let rs_name = relation_name_to_rs_field_name(&r.predicate);
          Some(quote! { #rs_name, })
        } else {
          None
        }
      })
      .collect::<Vec<_>>();
    let ensemble_result = quote! { #ret_ty { #(#ensemble_result_fields)* } };

    // Final function
    quote! {
      fn #fn_name<C: Provenance>(rt: &RuntimeEnvironment, ctx: &mut C, edb: &mut ExtensionalDatabase<C>, #(#args)*) -> #ret_ty<C> {
        let mut iter = StaticIteration::<C>::new(rt, ctx);
        #(#create_relation_stmts)*
        while iter.changed() || iter.is_first_iteration() {
          #(#updates)*
          iter.step();
        }
        #complete_stmts
        #ensemble_result
      }
    }
  }
}

impl ast::Relation {
  pub fn to_rs_field_name(&self) -> TokenStream {
    relation_name_to_rs_field_name(&self.predicate)
  }

  pub fn to_rs_result_struct_field(&self, _: &CompileOptions) -> TokenStream {
    let field_name = relation_name_to_rs_field_name(&self.predicate);
    let ty = tuple_type_to_rs_type(&self.tuple_type);
    quote! { #field_name: StaticCollection<#ty, C>, }
  }

  pub fn to_rs_output_struct_field(&self, _: &CompileOptions) -> TokenStream {
    let field_name = relation_name_to_rs_field_name(&self.predicate);
    let ty = tuple_type_to_rs_type(&self.tuple_type);
    quote! { #field_name: StaticOutputCollection<#ty, C>, }
  }
}

impl ast::Update {
  pub fn to_rs_insert(&self, curr_strat_id: usize, rel_to_strat_map: &RelationToStratumMap) -> TokenStream {
    let rs_rel_name = relation_name_to_rs_field_name(&self.target);
    let rs_dataflow = self.dataflow.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
    quote! { iter.insert_dataflow(&#rs_rel_name, #rs_dataflow); }
  }
}

impl ast::Dataflow {
  pub fn to_rs_dataflow(&self, curr_strat_id: usize, rel_to_strat_map: &RelationToStratumMap) -> TokenStream {
    match self {
      Self::Unit(tuple_type) => {
        let ty = tuple_type_to_rs_type(tuple_type);
        quote! { iter.unit::<#ty>(iter.is_first_iteration()) }
      }
      Self::UntaggedVec(_) => unimplemented!(),
      Self::Relation(r) => {
        let rel_ident = relation_name_to_rs_field_name(r);
        let stratum_id = rel_to_strat_map[r];
        if stratum_id == curr_strat_id {
          quote! { &#rel_ident }
        } else {
          let stratum_result = format_ident!("stratum_{}_result", stratum_id);
          quote! { dataflow::collection(&#stratum_result.#rel_ident, iter.is_first_iteration()) }
        }
      }
      Self::Project(d1, expr) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        let rs_expr = expr_to_rs_expr(expr);
        quote! { dataflow::project(#rs_d1, |t| #rs_expr) }
      }
      Self::Filter(d1, expr) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        let rs_expr = expr_to_rs_expr(expr);
        quote! { dataflow::filter(#rs_d1, |t| #rs_expr) }
      }
      Self::Find(d1, tuple) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        let rs_tuple = tuple_to_rs_tuple(tuple);
        quote! { dataflow::find(#rs_d1, #rs_tuple) }
      }
      Self::Sorted(d1) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        quote! { dataflow::sorted(#rs_d1) }
      }
      Self::Union(d1, d2) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        let rs_d2 = d2.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        quote! { iter.union(#rs_d1, #rs_d2) }
      }
      Self::Join(d1, d2) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        let rs_d2 = d2.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        quote! { iter.join(#rs_d1, #rs_d2) }
      }
      Self::JoinIndexedVec(d1, r) => {
        // TODO: implement this
        // NOTE: current implementation falls back to join

        // Get left
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);

        // Get right relation
        let rel_ident = relation_name_to_rs_field_name(r);
        let stratum_id = rel_to_strat_map[r];
        let stratum_result = format_ident!("stratum_{}_result", stratum_id);
        let rs_d2 = quote! { dataflow::collection(&#stratum_result.#rel_ident, iter.is_first_iteration()) };

        // Join the two relations
        quote! { iter.join(#rs_d1, #rs_d2) }
      }
      Self::Intersect(d1, d2) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        let rs_d2 = d2.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        quote! { iter.intersect(#rs_d1, #rs_d2) }
      }
      Self::Product(d1, d2) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        let rs_d2 = d2.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        quote! { iter.product(#rs_d1, #rs_d2) }
      }
      Self::Antijoin(d1, d2) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        let rs_d2 = d2.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        quote! { iter.antijoin(#rs_d1, #rs_d2) }
      }
      Self::Difference(d1, d2) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        let rs_d2 = d2.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        quote! { iter.difference(#rs_d1, #rs_d2) }
      }
      Self::Reduce(r) => {
        let get_col = |r| {
          let rel_ident = relation_name_to_rs_field_name(r);
          let stratum_id = rel_to_strat_map[r];
          let stratum_result = format_ident!("stratum_{}_result", stratum_id);
          quote! { dataflow::collection(&#stratum_result.#rel_ident, iter.is_first_iteration()) }
        };

        // Get the to_aggregate collection
        let to_agg_col = get_col(&r.predicate);

        // Get the aggregator
        let non_multi_world = r.aggregate_info.has_exclamation_mark;
        let agg = match r.aggregator.as_str() {
          "count" => quote! { CountAggregator::new(#non_multi_world) },
          "sum" => quote! { SumAggregator::new() },
          "prod" => quote! { ProdAggregator::new() },
          "max" => {
            if r.aggregate_info.arg_var_types.is_empty() {
              quote! { MaxAggregator::new() }
            } else {
              unimplemented!("Implicit max with argument")
            }
          }
          "min" => {
            if r.aggregate_info.arg_var_types.is_empty() {
              quote! { MinAggregator::new() }
            } else {
              unimplemented!("Implicit max with argument")
            }
          }
          "argmax" => {
            quote! { ArgmaxAggregator::new() }
          }
          "argmin" => {
            quote! { ArgminAggregator::new() }
          }
          "exists" => quote! { ExistsAggregator::new(#non_multi_world) },
          "top" => {
            let k = r.aggregate_info.pos_params.get(0).map(|v| v.as_usize()).unwrap_or(1);
            quote! { TopKAggregator::new(#k) }
          }
          _ => unimplemented!(),
        };

        // Get the dataflow
        match &r.group_by {
          ReduceGroupByType::None => {
            quote! { iter.aggregate(#agg, #to_agg_col) }
          }
          ReduceGroupByType::Implicit => {
            quote! { iter.aggregate_implicit_group(#agg, #to_agg_col) }
          }
          ReduceGroupByType::Join(group_by_rel) => {
            let group_by_col = get_col(&group_by_rel);
            quote! { iter.aggregate_join_group(#agg, #group_by_col, #to_agg_col) }
          }
        }
      }
      Self::OverwriteOne(d1) => {
        let rs_d1 = d1.to_rs_dataflow(curr_strat_id, rel_to_strat_map);
        quote! { dataflow::overwrite_one(#rs_d1) }
      }
      Self::Exclusion(_, _) => unimplemented!(),
      Self::ForeignPredicateGround(_, _) => unimplemented!(),
      Self::ForeignPredicateConstraint(_, _, _) => unimplemented!(),
      Self::ForeignPredicateJoin(_, _, _) => unimplemented!(),
    }
  }
}

fn relation_name_to_rs_field_name(n: &str) -> TokenStream {
  let name = if n.contains("#") {
    format!(
      "_{}",
      n.replace("#", "_p")
        .replace("(", "_l")
        .replace(")", "_r")
        .replace(",", "_c")
    )
  } else {
    n.to_string()
  };
  let ident = format_ident!("{}", name);
  quote! { #ident }
}

fn tuple_type_to_rs_type(ty: &TupleType) -> TokenStream {
  match ty {
    TupleType::Tuple(t) if t.len() == 0 => quote! { () },
    TupleType::Tuple(t) => {
      let elems = t.iter().map(|vt| tuple_type_to_rs_type(vt)).collect::<Vec<_>>();
      quote! { (#(#elems),*,) }
    }
    TupleType::Value(v) => value_type_to_rs_type(v),
  }
}

fn value_type_to_rs_type(ty: &ValueType) -> TokenStream {
  match ty {
    ValueType::I8 => quote! { i8 },
    ValueType::I16 => quote! { i16 },
    ValueType::I32 => quote! { i32 },
    ValueType::I64 => quote! { i64 },
    ValueType::I128 => quote! { i128 },
    ValueType::ISize => quote! { isize },
    ValueType::U8 => quote! { u8 },
    ValueType::U16 => quote! { u16 },
    ValueType::U32 => quote! { u32 },
    ValueType::U64 => quote! { u64 },
    ValueType::U128 => quote! { u128 },
    ValueType::USize => quote! { usize },
    ValueType::F32 => quote! { f32 },
    ValueType::F64 => quote! { f64 },
    ValueType::Bool => quote! { bool },
    ValueType::Char => quote! { char },
    ValueType::Str => quote! { &'static str },
    ValueType::String => quote! { String },
    ValueType::Symbol => unimplemented!(),
    ValueType::DateTime => quote! { DateTime<Utc> },
    ValueType::Duration => quote! { Duration },
    ValueType::Entity => quote! { u64 },
    ValueType::Tensor => unimplemented!(),
  }
}

fn expr_to_rs_expr(expr: &Expr) -> TokenStream {
  match expr {
    Expr::Tuple(t) => {
      if t.is_empty() {
        quote! { () }
      } else {
        let elems = t.iter().map(|a| expr_to_rs_expr(a)).collect::<Vec<_>>();
        quote! { (#(#elems),*,) }
      }
    }
    Expr::Access(a) => {
      if a.len() == 0 {
        quote! { t }
      } else {
        let indices = a.iter().map(|id| syn::Index::from(id));
        quote! { t.#(#indices).* }
      }
    }
    Expr::Constant(c) => value_to_rs_value(c),
    Expr::Binary(b) => {
      let op1 = expr_to_rs_expr(&b.op1);
      let op2 = expr_to_rs_expr(&b.op2);
      let op = binary_op_to_rs(&b.op);
      quote! { (#op1 #op #op2) }
    }
    Expr::Unary(u) => {
      let op1 = expr_to_rs_expr(&u.op1);
      match &u.op {
        UnaryOp::TypeCast(target_ty) => {
          let rs_ty = value_type_to_rs_type(target_ty);
          quote! { (#op1 as #rs_ty) }
        }
        UnaryOp::Not => quote! { (!#op1) },
        UnaryOp::Pos => quote! { (+#op1) },
        UnaryOp::Neg => quote! { (-#op1) },
      }
    }
    Expr::IfThenElse(i) => {
      let cond = expr_to_rs_expr(&i.cond);
      let then_br = expr_to_rs_expr(&i.then_br);
      let else_br = expr_to_rs_expr(&i.else_br);
      quote! { if #cond { #then_br } else { #else_br } }
    }
    Expr::Call(_) => {
      unimplemented!()
    }
    Expr::New(_) => {
      unimplemented!()
    }
  }
}

fn input_tag_to_rs_input_tag(tag: &DynamicInputTag) -> TokenStream {
  match tag {
    DynamicInputTag::None => quote! { DynamicInputTag::None },
    DynamicInputTag::NewVariable => quote! { DynamicInputTag::NewVariable },
    DynamicInputTag::Exclusive(i) => quote! { DynamicInputTag::Exclusive(#i) },
    DynamicInputTag::Bool(b) => quote! { DynamicInputTag::Bool(#b) },
    DynamicInputTag::Natural(n) => quote! { DynamicInputTag::Natural(#n) },
    DynamicInputTag::Float(f) => quote! { DynamicInputTag::Float(#f) },
    DynamicInputTag::ExclusiveFloat(f, u) => quote! { DynamicInputTag::ExclusiveFloat(#f, #u) },
    DynamicInputTag::FloatWithID(id, f) => quote! { DynamicInputTag::Float(#id, #f) },
    DynamicInputTag::ExclusiveFloatWithID(id, f, u) => quote! { DynamicInputTag::ExclusiveFloat(#id, #f, #u) },
    DynamicInputTag::Tensor(_) => unimplemented!(),
  }
}

fn tuple_to_rs_tuple(tuple: &Tuple) -> TokenStream {
  match tuple {
    Tuple::Value(v) => value_to_rs_value(v),
    Tuple::Tuple(t) if t.is_empty() => quote! { () },
    Tuple::Tuple(t) => {
      let elems = t.iter().map(|a| tuple_to_rs_tuple(a)).collect::<Vec<_>>();
      quote! { (#(#elems),*,) }
    }
  }
}

fn binary_op_to_rs(bin_op: &BinaryOp) -> TokenStream {
  use BinaryOp::*;
  match bin_op {
    Add => quote! { + },
    Sub => quote! { - },
    Mul => quote! { * },
    Div => quote! { / },
    Mod => quote! { % },
    And => quote! { && },
    Or => quote! { || },
    Xor => quote! { ^ },
    Eq => quote! { == },
    Neq => quote! { != },
    Lt => quote! { < },
    Leq => quote! { <= },
    Gt => quote! { > },
    Geq => quote! { >= },
  }
}

fn value_to_rs_value(value: &Value) -> TokenStream {
  use Value::*;
  match value {
    I8(i) => quote! { #i },
    I16(i) => quote! { #i },
    I32(i) => quote! { #i },
    I64(i) => quote! { #i },
    I128(i) => quote! { #i },
    ISize(i) => quote! { #i },
    U8(u) => quote! { #u },
    U16(u) => quote! { #u },
    U32(u) => quote! { #u },
    U64(u) => quote! { #u },
    U128(u) => quote! { #u },
    USize(u) => quote! { #u },
    F32(f) => quote! { #f },
    F64(f) => quote! { #f },
    Char(c) => quote! { #c },
    Bool(b) => quote! { #b },
    Str(s) => quote! { #s },
    String(s) => quote! { String::from(#s) },
    Symbol(_) => unimplemented!(),
    SymbolString(_) => unimplemented!(),
    DateTime(_) => unimplemented!(),
    Duration(_) => unimplemented!(),
    Entity(e) => quote! { #e },
    EntityString(_) => panic!("[Internal Error] Should not have raw entity string during compilation"),
    Tensor(_) => panic!("[Internal Error] Should not have raw tensor during compilation"),
    TensorValue(_) => panic!("[Internal Error] Should not have tensor value during compilation"),
  }
}

type RelationToStratumMap = HashMap<String, usize>;

fn compute_relation_to_stratum_map(ast: &ast::Program) -> RelationToStratumMap {
  let mut map = RelationToStratumMap::new();
  for (i, stratum) in ast.strata.iter().enumerate() {
    for (predicate, _) in &stratum.relations {
      map.insert(predicate.clone(), i);
    }
  }
  map
}

type StratumDependency = HashMap<usize, BTreeSet<usize>>;

fn compute_stratum_dependency(ast: &ast::Program, rel_to_strat_map: &RelationToStratumMap) -> StratumDependency {
  let mut dep = StratumDependency::new();
  for (i, stratum) in ast.strata.iter().enumerate() {
    let dep_rels = stratum.dependency();
    let dep_strats = dep_rels.into_iter().map(|r| rel_to_strat_map[&r].clone()).collect();
    dep.insert(i, dep_strats);
  }
  dep
}

type InterStratumDependency = HashSet<String>;

fn compute_interstratum_dependency(ast: &ast::Program) -> InterStratumDependency {
  let mut dep = HashSet::new();
  for stratum in &ast.strata {
    let dep_rels = stratum.dependency();
    dep.extend(dep_rels.into_iter().filter(|r| !stratum.relations.contains_key(r)));
  }
  dep
}
