// RUN: byteir-opt %s

module {
  pdl.pattern : benefit(0) {
    %0 = types
    %1 = operands
    %2 = attribute
    %3 = operation "test.test_op_A"(%1 : !pdl.range<value>) {"__rewrite__" = %2} -> (%0 : !pdl.range<type>)
    rewrite %3 {
      %4 = operation "test.test_op_B"(%1 : !pdl.range<value>) -> (%0 : !pdl.range<type>)
      replace %3 with %4
    }
  }
}
