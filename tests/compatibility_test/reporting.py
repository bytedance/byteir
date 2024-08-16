# Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import NamedTuple, Optional, List


class TestResult(NamedTuple):
    unique_name: str
    execution_error: Optional[str]
    numerical_error: Optional[str]


def report_results(results: List[TestResult]):
    fail_case = []
    pass_case = []
    for result in results:
        if result.execution_error is not None:
            fail_case.append([
                result.unique_name, "execution failed: " + result.unique_name +
                "\n" + result.execution_error
            ])
        elif result.numerical_error is not None:
            fail_case.append([
                result.unique_name, "numerical failed: " + result.unique_name +
                "\n" + result.numerical_error
            ])
        else:
            pass_case.append(result)
    pass_case.sort(key=lambda x: x.unique_name)
    fail_case.sort(key=lambda x: x[0])

    print(f"\n****** PASS tests - {len(pass_case)} tests")
    for test in pass_case:
        print(test.unique_name, " --- PASS")
    for test in fail_case:
        print(test[1])
    print(f"\n****** FAILED tests - {len(fail_case)} tests")
    for test in fail_case:
        print(test[0])
    return len(fail_case) > 0
