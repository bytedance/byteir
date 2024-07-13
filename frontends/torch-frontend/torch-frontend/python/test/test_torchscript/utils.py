import torch

def numerical_test_helper(module, inputs, torch_output, atol=1e-5, rtol=1e-5):
    if isinstance(torch_output, torch.Tensor):
        torch_output = [torch_output]
    np_inputs = [t.numpy() for t in inputs]

    from mhlo_tools.ir_executor import Interpreter
    func_name = module.body.operations[0].name.value
    interp = Interpreter.load_from_string(module.operation.get_asm(), is_stablehlo=True)
    mhlo_outputs = interp.call_function(func_name, np_inputs)
    mhlo_outputs = [torch.tensor(t) for t in mhlo_outputs]

    assert len(torch_output) == len(mhlo_outputs)
    for t, m in zip(torch_output, mhlo_outputs):
        torch.testing.assert_close(m, t, rtol=rtol, atol=atol, equal_nan=True)
