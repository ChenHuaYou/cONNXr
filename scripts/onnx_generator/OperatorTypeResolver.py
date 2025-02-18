from .Template import Template

class CaseSkip(Template):
    _template = '''
/* skip non tensor constraint '{constraint}' ('{original}') */
'''
    def __init__(self, constraint, original):
        self.constraint = constraint
        self.original = original

class CaseSwitch(Template):
    _template = '''
case {case}: {{ {switch} break; }}
'''
    def __init__(self, schema, case, permutationsMap):
        self.case = case
        self.switch = Switch(schema, permutationsMap)

class CaseExecuter(Template):
    _template = '''
case {case}: {{ executer = (operator_executer) &execute_{operator_name}__{typePermutationText}; break; }}
'''
    def __init__(self, case, operator_name, typePermutationText):
        self.case = case
        self.operator_name = operator_name
        self.typePermutationText = typePermutationText

class Type(Template):
    _template = '''
uint32_t {constraint} = 0;
if (ctx->{inOrOutput}[{name}]) {{
    {constraint} = ctx->{inOrOutput}[{name}]->data_type;
}}
'''
    def __init__(self, constraint, inOrOutput, name):
        self.constraint = constraint
        self.inOrOutput = inOrOutput
        self.name = name

class Switch(Template):
    _template = '''
switch ( {constraint} ) {{
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    {cases}
    default: {{
        fprintf(stderr, "no matching type for {schema.operator_name} and constraint '{constraint}' with type '%s' found!\\n",operator_info_tensorType2str({constraint}));
        break;
    }}
}}
'''
    def __init__(self, schema, permutationMap):
        self.schema = schema
        self.constraint = list(permutationMap.keys())[0][-1][0]
        cases = []
        for k,v in permutationMap.items():
            case = k[-1][1].onnxTensorDataTypes()
            if not case:
                cases.append(CaseSkip(k[-1][0],k[-1][1].original))
                continue
            if not v:
                operator_name = self.schema.operator_name
                typePermutationText = self.schema.constraints.typePermutationText(k)
                cases.append(CaseExecuter(case[0],operator_name,typePermutationText))
            else:
                cases.append(CaseSwitch( schema, case[0],v))
        self.cases = "\n".join(map(str,cases))

class Resolve(Template):
    _template = '''
{{
    {types}
    {switch}
}}
'''
    def __init__(self, schema):
        self.schema = schema

        resolveTypes = []

        for constraint in filter(lambda x: x.input, self.schema.constraints.values()):
            inOrOutput = None
            name = None
            for idx, input in enumerate(self.schema.inputs):
                if constraint.name != input.constraint:
                    continue
                inOrOutput = "inputs"
                name = idx
                if input.optional:
                    continue
                break
            else:
                for idx, output in enumerate(self.schema.outputs):
                    if constraint.name != output.constraint:
                        continue
                    inOrOutput = "outputs"
                    name = idx
                    if output.optional:
                        continue
                    break
            inOrOutput = "outputs"
            resolveTypes.append(Type(constraint.name,inOrOutput,name))
        permutationsMap = schema.constraints.typePermutationsMap(filterInput=True)
        self.types = "\n".join([ str(t) for t in resolveTypes ])
        if permutationsMap:
            self.switch = Switch(schema, permutationsMap)
        else:
            self.types = "/* skipping constraint check, because no constraint exist */"
            self.switch = f"executer = NULL;//&{schema.operator_name};"



class Source(Template):
    _basepath = "{path}"
    _filepath = "{schema.domain}/{schema.name}/{schema.version}/resolve_{schema.operator_name}.c"
    _template = '''
//this file was generated by {scriptpath}
#include "{header_path}"
#include "operators/operator_stub.h"
#include <inttypes.h>
#include <stdio.h>

operator_executer
resolve_{schema.operator_name}(
    Onnx__NodeProto *ctx
){{
    operator_executer executer = NULL;
    {switch}
    if (!executer) {{
        executer = &operator_stub;
    }}
    return executer;
}}
'''

    def __init__(self, header, path):
        self.schema = header.schema
        self.path = path
        self.header = header
        self.header_path = header.filepath(False, False).parts[-1]
        self.switch = Resolve(self.schema)
