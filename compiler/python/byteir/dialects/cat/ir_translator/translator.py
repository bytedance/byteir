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

from collections import namedtuple

class DispatchableIRTranslatorMeta(type):
    def __new__(cls, name, bases, attrs):
        attrs["_dispatchers"] = []
        return super().__new__(cls, name, bases, attrs)


class DispatchableIRTranslatorBase(metaclass=DispatchableIRTranslatorMeta):
    DispatchTableEntry = namedtuple("DispatchTableEntry", ["matcher", "fn", "priority"])

    @classmethod
    def register(cls, matcher, fn, priority=0):
        cls._dispatchers.append(
            DispatchableIRTranslatorBase.DispatchTableEntry(matcher, fn, priority)
        )
        cls._dispatchers.sort(reverse=True, key=lambda entry: entry.priority)

    @classmethod
    def dispatch(cls, op, inputs):
        for entry in cls._dispatchers:
            if entry.matcher(op):
                return entry.fn(op, inputs)

        raise NotImplementedError("unsupported operation {}".format(op))


class IRTranslator(DispatchableIRTranslatorBase):
    @classmethod
    def register(cls, dialect):
        def matcher(op):
            return str(op.operation.name).startswith(dialect)

        def impl(fn):
            super(IRTranslator, cls).register(matcher, fn, len(dialect))
            return fn

        return impl

    @classmethod
    def _check_io(cls, op, ir_types, tensors):
        assert isinstance(
            tensors, (tuple, list)
        ), "expect tuple or list but got {}, associated op is {}".format(tensors, op)
        assert len(ir_types) == len(
            tensors
        ), "the number of ir types and computing tensors mismatch, {} vs {}".format(
            ir_types, tensors
        )


    @classmethod
    def translate(cls, op, inputs):
        cls._check_io(op, [i.type for i in op.operands], inputs)
        outputs = cls.dispatch(op, inputs)
        cls._check_io(op, [i.type for i in op.results], outputs)
        return outputs

