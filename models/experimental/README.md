To unit test the input configurations of PETR VovnetCP submodule, run the following commands.
Conv2D:
- All 24 convs passed. 3 convs passed by running conv with split.
- Run the command to unit test the convs: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_petr_vovnetcp`
- Run the command to unit test the convs with split: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_petr_vovnetcp_split_conv`
ReLU6:
- All 4 configurations passed.
- Run the command to unit test the relu6: `pytest tests/ttnn/unit_tests/operations/eltwise/test_activation.py::test_petr_vovnetcp_relu6`
ReLU:
- All 9 configurations passed.
- Run the command to unit test the relu: `pytest tests/ttnn/unit_tests/operations/eltwise/test_activation.py::test_petr_vovnetcp_relu`
MaxPool2D:
- All 3 configurations skipped with the information "kernel size and padding combination not supported".
- Run the command to unit test the MaxPool2d: `pytest tests/ttnn/unit_tests/operations/test_maxpool2d.py::test_petr_vovnetcp`
AdaptiveAvgPool2D:
- All 4 configurations passed.
- Run the command to unit test the AdaptiveAvgPool2D: `pytest tests/ttnn/unit_tests/operations/test_global_avg_pool2d.py::test_petr_vovnetcp_avgpool`
