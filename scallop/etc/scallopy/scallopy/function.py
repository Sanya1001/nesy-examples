from typing import *
import inspect

from . import torch_importer
from . import syntax
from . import utils

# Generic type parameters
class GenericTypeParameter:
  """
  A generic type parameter used for Scallop foreign function and predicate
  """

  COUNTER = 0

  def __init__(self, type_family = "Any"):
    self.id = GenericTypeParameter.COUNTER
    if type_family == any or type_family == None:
      self.type_family = "Any"
    elif type_family == int:
      self.type_family = "Integer"
    elif type_family == float:
      self.type_family = "Float"
    elif isinstance(type_family, TypeVar):
      self.type_family = GenericTypeParameter.sanitize_type_family(type_family.__name__)
    elif isinstance(type_family, str):
      self.type_family = GenericTypeParameter.sanitize_type_family(type_family)
    else:
      raise Exception(f"Unknown type family {type_family}")
    GenericTypeParameter.COUNTER += 1

  def __repr__(self):
    return f"T{self.id}({self.type_family})"

  @staticmethod
  def sanitize_type_family(tf: str) -> str:
    if tf == "Any":
      return "Any"
    elif tf == "Number":
      return "Number"
    elif tf == "Integer":
      return "Integer"
    elif tf == "SignedInteger":
      return "SignedInteger"
    elif tf == "UnsignedInteger":
      return "UnsignedInteger"
    elif tf == "Float":
      return "Float"
    else:
      raise Exception(f"Unknown type family {tf}")


# Scallop Data Type
class Type:
  def __init__(self, value):
    if type(value) == GenericTypeParameter:
      self.kind = "generic"
      self.id = value.id
      self.type_family = value.type_family
    elif isinstance(value, syntax.AstTypeNode):
      self.kind = "base"
      self.type = value.name()
    elif isinstance(value, TypeVar):
      (self.kind, self.type) = Type.sanitize_type_str(value.__name__)
    elif value == float:
      self.kind = "family"
      self.type_family = "Float"
    elif value == int:
      self.kind = "family"
      self.type_family = "Integer"
    elif value == any or value == None:
      self.kind = "family"
      self.type_family = "Any"
    elif value == bool:
      self.kind = "base"
      self.type = "bool"
    elif value == str:
      self.kind = "base"
      self.type = "String"
    elif value == torch_importer.Tensor:
      self.kind = "base"
      self.type = "Tensor"
    else:
      raise Exception(f"Unknown scallop function type annotation {value}")

  def __repr__(self):
    if self.kind == "base":
      return f"BaseType({self.type})"
    elif self.kind == "family":
      return f"TypeFamily({self.type_family})"
    elif self.kind == "generic":
      return f"Generic({self.id}, {self.type_family})"
    else:
      raise Exception(f"Unknown parameter kind {self.kind}")

  @staticmethod
  def sanitize_type_str(value: str) -> Tuple[str, str]:
    if value == "Float":
      return ("family", "Float")
    elif value == "Integer":
      return ("family", "Integer")
    elif value == "UnsignedInteger":
      return ("family", "UnsignedInteger")
    elif value == "SignedInteger":
      return ("family", "SignedInteger")
    elif value == "Number":
      return ("family", "Number")
    elif value == "Any":
      return ("family", "Any")
    elif value == "String":
      return ("base", "String")
    elif value == "Tensor":
      return ("base", "Tensor")
    elif value == "i8" or value == "i16" or value == "i32" or value == "i64" or value == "i128" or value == "isize" or \
         value == "u8" or value == "u16" or value == "u32" or value == "u64" or value == "u128" or value == "usize" or \
         value == "f32" or value == "f64" or value == "bool" or value == "char" or value == "String" or value == "Tensor" or \
         value == "Symbol" or value == "Entity":
      return ("base", value)

  def is_base(self):
    return self.kind == "base"

  def is_generic(self):
    return self.kind == "generic"

  def is_type_family(self):
    return self.kind == "family"


class ForeignFunction:
  """
  A Scallop Foreign Function
  """
  def __init__(
    self,
    func: Callable,
    name: str,
    generic_type_params: List[str],
    static_arg_types: List[Type],
    optional_arg_types: List[Type],
    var_arg_types: Optional[Type],
    return_type: Type,
    suppress_warning: bool = False
  ):
    self.func = func
    self.name = name
    self.generic_type_params = generic_type_params
    self.static_arg_types = static_arg_types
    self.optional_arg_types = optional_arg_types
    self.var_arg_types = var_arg_types
    self.return_type = return_type
    self.suppress_warning = suppress_warning

  def __call__(self, *args):
    return self.func(*args)

  def arg_type_repr(self, arg):
    if arg.is_generic():
      return f"T{arg.id}"
    elif arg.is_type_family():
      return arg.type_family
    else:
      return arg.type

  def __repr__(self):
    r = f"extern fn ${self.name}"

    # Generic Type Parameters
    if len(self.generic_type_params) > 0:
      r += "<"
      for (i, param) in enumerate(self.generic_type_params):
        if i > 0:
          r += ", "
        r += f"T{i}: {param}"
      r += ">"

    # Start
    r += "("

    # Static arguments
    for (i, arg) in enumerate(self.static_arg_types):
      if i > 0:
        r += ", "
      r += self.arg_type_repr(arg)

    # Optional arguments
    if len(self.static_arg_types) > 0 and len(self.optional_arg_types) > 0:
      r += ", "
    for (i, arg) in enumerate(self.optional_arg_types):
      if i > 0:
        r += ", "
      r += f"{self.arg_type_repr(arg)}?"

    # Variable arguments
    if self.var_arg_types is not None:
      if len(self.static_arg_types) + len(self.optional_arg_types) > 0:
        r += ", "
      r += f"{self.arg_type_repr(self.var_arg_types)}..."

    # Return type
    r += f") -> {self.arg_type_repr(self.return_type)}"

    return r


@utils.doublewrap
def foreign_function(
  func,
  name: Optional[str] = None,
  arg_types: Optional[List] = None,
  opt_arg_types: Optional[List] = None,
  var_arg_type: Optional[Any] = None,
  ret_type: Optional[Any] = None,
  suppress_warning: bool = False,
):
  """
  A decorator to create a Scallop foreign function, for example

  ``` python
  @scallopy.foreign_function
  def string_index_of(s1: str, s2: str) -> usize:
    return s1.index(s2)
  ```

  This foreign function can be then registered into Scallop for invokation

  ``` python
  ctx.register_foreign_function(string_index_of)
  ```
  """

  # Get the function name
  func_name = func.__name__ if not name else name

  # Get the function signature
  signature = inspect.signature(func)

  # Store all the argument types
  static_argument_types = [] if not arg_types else arg_types
  optional_argument_types = [] if not opt_arg_types else opt_arg_types
  variable_argument_type = None if not var_arg_type else var_arg_type

  # Find argument types
  for (arg_name, item) in signature.parameters.items():
    optional = item.default != inspect.Parameter.empty
    if item.kind == inspect.Parameter.VAR_POSITIONAL:
      if var_arg_type is None:
        if item.annotation is None:
          raise Exception(f"Variable argument {arg_name} type annotation not provided")
        variable_argument_type = Type(item.annotation)
    elif not optional:
      if arg_types is None:
        static_argument_types.append(Type(item.annotation))
    else:
      if opt_arg_types is None:
        if item.default != None:
          raise Exception("Optional arguments need to have default `None`")
        optional_argument_types.append(Type(item.annotation))

  # Get all argument types
  all_arg_types = static_argument_types + \
                  optional_argument_types + \
                  ([variable_argument_type] if variable_argument_type is not None else [])

  # Find return types
  if ret_type is None:
    if signature.return_annotation is None:
      raise Exception(f"Return type annotation not provided")
    return_type = Type(signature.return_annotation)
  else:
    return_type = Type(ret_type)

  # If the return type is generic, at least one of its argument also needs to have the same type
  if return_type.is_generic():
    is_return_generic_type_ok = False
    for arg_type in all_arg_types:
      if arg_type.is_generic() and arg_type.id == return_type.id:
        is_return_generic_type_ok = True
    if not is_return_generic_type_ok:
      raise Exception(f"Return generic type not bounded by any input argument")
  elif return_type.is_type_family():
    raise Exception(f"Return type cannot be a type family ({return_type})")

  # Put all types together and find generic type
  generic_types_map = {}
  generic_type_params = []
  all_types = all_arg_types + [return_type]
  for param in all_types:
    if param.is_generic():
      if param.id not in generic_types_map:
        generic_types_map[param.id] = []
      generic_types_map[param.id].append(param)
  for (i, (_, params)) in enumerate(generic_types_map.items()):
    assert len(params) > 0, "Should not happen; there has to be at least one type using generic type parameter"
    for param in params:
      param.id = i
    generic_type_params.append(params[0].type_family)

  # Return a Scallop Foreign Function class
  return ForeignFunction(
    func,
    func_name,
    generic_type_params,
    static_argument_types,
    optional_argument_types,
    variable_argument_type,
    return_type,
    suppress_warning,
  )
