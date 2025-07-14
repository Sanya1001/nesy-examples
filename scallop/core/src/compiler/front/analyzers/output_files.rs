use std::collections::*;
use std::path::PathBuf;

use crate::common::output_option::{OutputCSVFile, OutputFile};
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct OutputFilesAnalysis {
  pub output_files: HashMap<String, OutputFile>,
  pub errors: Vec<OutputFilesError>,
}

impl OutputFilesAnalysis {
  pub fn new() -> Self {
    Self {
      output_files: HashMap::new(),
      errors: Vec::new(),
    }
  }

  pub fn output_file(&self, relation: &String) -> Option<&OutputFile> {
    self.output_files.get(relation)
  }

  pub fn process_deliminator(&self, attr_arg: Option<&AttributeValue>) -> Result<Option<u8>, OutputFilesError> {
    match attr_arg {
      Some(v) => match v.as_string() {
        Some(s) => {
          if s.len() == 1 {
            let c = s.chars().next().unwrap();
            if c.is_ascii() {
              Ok(Some(c as u8))
            } else {
              Err(OutputFilesError::DeliminatorNotASCII {
                loc: v.location().clone(),
              })
            }
          } else {
            Err(OutputFilesError::DeliminatorNotSingleCharacter {
              loc: v.location().clone(),
            })
          }
        }
        _ => Err(OutputFilesError::DeliminatorNotString {
          loc: v.location().clone(),
        }),
      },
      None => Ok(None),
    }
  }

  pub fn process_attribute(&self, attr: &Attribute) -> Result<OutputFile, OutputFilesError> {
    if let Some(arg) = attr.pos_arg(0) {
      match arg.as_string() {
        Some(s) => {
          let path = PathBuf::from(s);
          match path.extension() {
            Some(s) if s == "csv" => {
              let deliminator = self.process_deliminator(attr.kw_arg("deliminator"))?;
              let output_file = OutputFile::CSV(OutputCSVFile::new_with_options(path, deliminator));
              Ok(output_file)
            }
            Some(s) => Err(OutputFilesError::UnknownExtension {
              ext: String::from(s.to_str().unwrap()),
              attr_arg_loc: arg.location().clone(),
            }),
            None => Err(OutputFilesError::NoExtension {
              attr_arg_loc: arg.location().clone(),
            }),
          }
        }
        _ => Err(OutputFilesError::InvalidArgument {
          attr_arg_loc: arg.location().clone(),
        }),
      }
    } else {
      Err(OutputFilesError::InvalidNumAttrArgument {
        actual_num_args: attr.num_pos_args(),
        attr_loc: attr.location().clone(),
      })
    }
  }

  pub fn process_attributes(&mut self, rela: String, attrs: &Attributes) {
    if let Some(attr) = attrs.find("file") {
      match self.process_attribute(attr) {
        Ok(output_file) => {
          self.output_files.insert(rela, output_file);
        }
        Err(err) => {
          self.errors.push(err);
        }
      }
    }
  }
}

impl NodeVisitor<QueryDecl> for OutputFilesAnalysis {
  fn visit(&mut self, qd: &QueryDecl) {
    self.process_attributes(qd.query().create_relation_name(), qd.attrs());
  }
}

#[derive(Clone, Debug)]
pub enum OutputFilesError {
  InvalidNumAttrArgument {
    actual_num_args: usize,
    attr_loc: NodeLocation,
  },
  InvalidArgument {
    attr_arg_loc: NodeLocation,
  },
  NoExtension {
    attr_arg_loc: NodeLocation,
  },
  UnknownExtension {
    ext: String,
    attr_arg_loc: NodeLocation,
  },
  DeliminatorNotString {
    loc: NodeLocation,
  },
  DeliminatorNotSingleCharacter {
    loc: NodeLocation,
  },
  DeliminatorNotASCII {
    loc: NodeLocation,
  },
}

impl FrontCompileErrorTrait for OutputFilesError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::InvalidNumAttrArgument {
        actual_num_args,
        attr_loc,
      } => {
        format!(
          "Invalid number attributes of @file attribute. Expected 1, Found {}\n{}",
          actual_num_args,
          attr_loc.report(src)
        )
      }
      Self::InvalidArgument { attr_arg_loc } => {
        format!(
          "Invalid argument of @file attribute. Expected String, found\n{}",
          attr_arg_loc.report(src)
        )
      }
      Self::NoExtension { attr_arg_loc } => {
        format!(
          "Input file name does not have an extension\n{}",
          attr_arg_loc.report(src)
        )
      }
      Self::UnknownExtension { ext, attr_arg_loc } => {
        format!(
          "Unknown input file extension `.{}`. Expected one from [`.csv`, `.txt`]\n{}",
          ext,
          attr_arg_loc.report(src)
        )
      }
      Self::DeliminatorNotString { loc } => {
        format!("`deliminator` attribute is not a string\n{}", loc.report(src))
      }
      Self::DeliminatorNotSingleCharacter { loc } => {
        format!(
          "`deliminator` attribute is not a single character string\n{}",
          loc.report(src)
        )
      }
      Self::DeliminatorNotASCII { loc } => {
        format!("`deliminator` attribute is not an ASCII character\n{}", loc.report(src))
      }
    }
  }
}
