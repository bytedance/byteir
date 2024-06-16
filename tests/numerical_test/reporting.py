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
    compilation_error: Optional[str]
    runtime_error: Optional[str]
    numerical_error: Optional[str]
    performance_result: Optional[int]


def report_results(results: List[TestResult]):
    fail_set = []
    pass_set = []
    for result in results:
        if result.compilation_error is not None:
            fail_set.append('compilation failed: ' +
                            result.unique_name + "\n" + result.compilation_error)
        elif result.runtime_error is not None:
            fail_set.append('runtime failed: ' +
                            result.unique_name + "\n" + result.runtime_error)
        elif result.numerical_error is not None:
            fail_set.append('numerical failed: ' +
                            result.unique_name + "\n" + result.numerical_error)
        else:
            pass_set.append(result)
    print(f"\n****** PASS tests - {len(pass_set)} tests")
    for test in pass_set:
        if test.performance_result is not None:
            print(test.unique_name, f" {test.performance_result} ms")
        else:
            print(test.unique_name, " --- PASS")
    print(f"\n****** FAILED tests - {len(fail_set)} tests")
    for reason in fail_set:
        print(reason)
    return len(fail_set) > 0
