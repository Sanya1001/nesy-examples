use std::fmt::{Display, Formatter, Result};

use super::*;

impl Display for Program {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for (i, stratum) in self.strata.iter().enumerate() {
      f.write_fmt(format_args!("Stratum #{}:\n", i))?;
      Display::fmt(stratum, f)?;
    }
    Ok(())
  }
}

impl Display for Stratum {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    if !self.relations.is_empty() {
      f.write_str("  Relations:\n")?;
      for (pred, relation) in &self.relations {
        f.write_fmt(format_args!("    {}::{:?}", pred, relation.tuple_type))?;
        if let Some(input_file) = &relation.input_file {
          f.write_fmt(format_args!(", input: {:?}", input_file))?;
        }
        if relation.output.is_not_hidden() {
          f.write_fmt(format_args!(", output: {}", relation.output))?;
        }
        if !relation.facts.is_empty() {
          f.write_fmt(format_args!(
            ", facts: {{{}}}",
            relation
              .facts
              .iter()
              .map(|f| format!("{}", f))
              .collect::<Vec<_>>()
              .join(", ")
          ))?;
        }
        f.write_str("\n")?;
      }
    }
    if !self.updates.is_empty() {
      f.write_str("  Updates:\n")?;
      for update in &self.updates {
        Display::fmt(update, f)?;
      }
    }
    Ok(())
  }
}

impl Display for Fact {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    if self.tag.is_some() {
      f.write_fmt(format_args!("{}::{}", self.tag, self.tuple))
    } else {
      self.tuple.fmt(f)
    }
  }
}

impl Display for Update {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!("    {} <- ", self.target))?;
    self.dataflow.pretty_print(f, 4, 2)?;
    f.write_str("\n")
  }
}

impl Dataflow {
  pub fn pretty_print(&self, f: &mut Formatter<'_>, base_indent: usize, indent_size: usize) -> Result {
    let next_indent = base_indent + indent_size;
    let padding = vec![' '; next_indent].into_iter().collect::<String>();
    match self {
      // Base relations
      Self::Unit(t) => f.write_fmt(format_args!("Unit({})", t)),
      Self::UntaggedVec(v) => f.write_fmt(format_args!(
        "Vec([{}])",
        v.iter().map(|t| format!("{}", t)).collect::<Vec<_>>().join(", ")
      )),
      Self::Relation(r) => f.write_fmt(format_args!("Relation {}", r)),

      // Unary operations
      Self::Project(d, project) => {
        f.write_fmt(format_args!("Project[{:?}]\n{}", project, padding))?;
        d.pretty_print(f, next_indent, indent_size)
      }
      Self::Filter(d, filter) => {
        f.write_fmt(format_args!("Filter[{:?}]\n{}", filter, padding))?;
        d.pretty_print(f, next_indent, indent_size)
      }
      Self::Find(d, tuple) => {
        f.write_fmt(format_args!("Find[{}]\n{}", tuple, padding))?;
        d.pretty_print(f, next_indent, indent_size)
      }
      Self::Sorted(d) => {
        f.write_fmt(format_args!("Sorted\n{}", padding))?;
        d.pretty_print(f, next_indent, indent_size)
      }

      // Binary operations
      Self::Union(d1, d2) => {
        f.write_fmt(format_args!("Union\n{}", padding))?;
        d1.pretty_print(f, next_indent, indent_size)?;
        f.write_fmt(format_args!("\n{}", padding))?;
        d2.pretty_print(f, next_indent, indent_size)
      }
      Self::Join(d1, d2) => {
        f.write_fmt(format_args!("Join\n{}", padding))?;
        d1.pretty_print(f, next_indent, indent_size)?;
        f.write_fmt(format_args!("\n{}", padding))?;
        d2.pretty_print(f, next_indent, indent_size)
      }
      Self::Intersect(d1, d2) => {
        f.write_fmt(format_args!("Intersect\n{}", padding))?;
        d1.pretty_print(f, next_indent, indent_size)?;
        f.write_fmt(format_args!("\n{}", padding))?;
        d2.pretty_print(f, next_indent, indent_size)
      }
      Self::Product(d1, d2) => {
        f.write_fmt(format_args!("Product\n{}", padding))?;
        d1.pretty_print(f, next_indent, indent_size)?;
        f.write_fmt(format_args!("\n{}", padding))?;
        d2.pretty_print(f, next_indent, indent_size)
      }
      Self::Antijoin(d1, d2) => {
        f.write_fmt(format_args!("Antijoin\n{}", padding))?;
        d1.pretty_print(f, next_indent, indent_size)?;
        f.write_fmt(format_args!("\n{}", padding))?;
        d2.pretty_print(f, next_indent, indent_size)
      }
      Self::Difference(d1, d2) => {
        f.write_fmt(format_args!("Difference\n{}", padding))?;
        d1.pretty_print(f, next_indent, indent_size)?;
        f.write_fmt(format_args!("\n{}", padding))?;
        d2.pretty_print(f, next_indent, indent_size)
      }

      // Specialized binary operations
      Self::JoinIndexedVec(d1, s) => {
        f.write_fmt(format_args!("JoinIndexedVec\n{}", padding))?;
        d1.pretty_print(f, next_indent, indent_size)?;
        f.write_fmt(format_args!("\n{}{}", padding, s))
      }

      // Aggregation
      Self::Reduce(r) => {
        let group_by_predicate = match &r.group_by {
          ReduceGroupByType::Join(group_by_predicate) => format!(" where {}", group_by_predicate),
          ReduceGroupByType::Implicit => format!(" implicit group"),
          _ => format!(""),
        };
        let info = &r.aggregate_info;
        let pos_params = if !info.pos_params.is_empty() {
          info
            .pos_params
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ")
        } else {
          format!("")
        };
        let named_params = if !info.named_params.is_empty() {
          let comma = if info.pos_params.is_empty() { "" } else { ", " };
          let s = info
            .named_params
            .iter()
            .map(|(n, p)| format!("{} = {}", n, p))
            .collect::<Vec<_>>()
            .join(", ");
          format!("{comma}{s}")
        } else {
          format!("")
        };
        let params = if info.pos_params.is_empty() && info.named_params.is_empty() {
          format!("")
        } else {
          format!("<{}{}>", pos_params, named_params)
        };
        let exclamation_mark = if info.has_exclamation_mark {
          format!("!")
        } else {
          format!("")
        };
        f.write_fmt(format_args!(
          "Aggregation {}{}{}({}{})",
          r.aggregator, params, exclamation_mark, r.predicate, group_by_predicate
        ))
      }

      Self::OverwriteOne(d) => {
        f.write_fmt(format_args!("OverwriteOne\n{}", padding))?;
        d.pretty_print(f, next_indent, indent_size)
      }
      Self::Exclusion(d1, d2) => {
        f.write_fmt(format_args!("Exclusion\n{}", padding))?;
        d1.pretty_print(f, next_indent, indent_size)?;
        f.write_fmt(format_args!("\n{}", padding))?;
        d2.pretty_print(f, next_indent, indent_size)
      }

      Self::ForeignPredicateGround(pred, args) => {
        let args = args.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>();
        f.write_fmt(format_args!("ForeignPredicateGround[{}({})]", pred, args.join(", ")))
      }
      Self::ForeignPredicateConstraint(d, pred, args) => {
        let args = args.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>();
        f.write_fmt(format_args!(
          "ForeignPredicateConstraint[{}({})]\n{}",
          pred,
          args.join(", "),
          padding
        ))?;
        d.pretty_print(f, next_indent, indent_size)
      }
      Self::ForeignPredicateJoin(d, pred, args) => {
        let args = args.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>();
        f.write_fmt(format_args!(
          "ForeignPredicateJoin[{}({})]\n{}",
          pred,
          args.join(", "),
          padding
        ))?;
        d.pretty_print(f, next_indent, indent_size)
      }
    }
  }
}
