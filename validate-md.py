"""Validate Python sessions in a Markdown documents.

Python sessions are comment blocks that start with a line containing

  ```python

(triple backquotes followed by `python`) and end with a line containing

  ```

(triple backquotes). The Python sessions typically contain lines starting with

  >>> 

(triple greater-than symbols and one space) that may be extended with
lines starting with

  ... 

(triple dots and one space). Everything following these lines are the
outputs of the Python sessions. Also Python sessions in comment blocks
`<!-- ... -->` are processed.

Usage:

  To validate the md files, run:

    python validate-md.py <paths to md files>

  that will output the Python statements from Python sessions and stop
  immidiately if the output in the file difference from the actual
  output.

  To update the md files in-place, run

    python validate-md.py --inplace <paths to md files>

Each md file is considered as an independent Python script.

Note 1
------

Some Python sessions may require to be run from specialized conda
environments.  If the current process is run from a different
environment, the corresponding Python sessions must be skipped (the
subsequent Python sessions should not use variables from such Python
sessions). To mark certain regions in the Markdown document to have
constraints on installed packages, use the following specialized comments:

  <!--REQUIRE(torch): tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 9) -->
  <!--REQUIRE(numpy) -->

meaning that all subsequent code blocks require torch.__version__>=1.9
and numpy installed. The argument to `REQUIRE` is a subpackage name
that must be importable and anything following `:` must be a
predicated expression that will be evaluated using Python eval function.

The following comment line means that the pytorch requirement
constraint is removed for subsequent Python sessions:

  <!--UNREQUIRE(torch)-->

"""
# Author: Pearu Peterson
# Created: March 2021

import ast
import difflib
import os
import re
import sys
import argparse


class Requirements:

    def __init__(self):
        self.globals = {}
        self.state = {}

    def __str__(self):
        s = []
        for n, v in self.state.items():
            if v:
                s.append(n)
            else:
                s.append(n + '!')
        return 'REQUIRE[' + ', '.join(s) + ']'

    def __bool__(self):
        return all(self.state.values()) if self.state else True

    def add(self, packagename, predicate='True'):
        try:
            exec(f'import {packagename}', self.globals, self.globals)
        except ImportError:
            self.state[packagename] = False
        else:
            self.state[packagename] = eval(predicate, self.globals, self.globals)

    def drop(self, packagename):
        self.state.pop(packagename, None)


def unified_diff(a, b, fromfile='', tofile='', fromfiledate='',
                 tofiledate='', n=1, lineterm='\n', isjunk=None):
    """Modified version of difflib.unified_diff with isjunk argument support.
    """
    started = False
    matcher = difflib.SequenceMatcher(isjunk, a, b, autojunk=False)
    for group in matcher.get_grouped_opcodes(n):
        if not started:
            started = True
            fromdate = '\t{}'.format(fromfiledate) if fromfiledate else ''
            todate = '\t{}'.format(tofiledate) if tofiledate else ''
            yield '--- {}{}{}'.format(fromfile, fromdate, lineterm)
            yield '+++ {}{}{}'.format(tofile, todate, lineterm)

        first, last = group[0], group[-1]
        file1_range = difflib._format_range_unified(first[1], last[2])
        file2_range = difflib._format_range_unified(first[3], last[4])

        group_lines = []
        group_lines.append('@@ -{} +{} @@{}'.format(
            file1_range, file2_range, lineterm))

        keep_group = False
        for tag, i1, i2, j1, j2 in group:
            lines = []
            if tag == 'equal':
                for line in a[i1:i2]:
                    lines.append(' ' + line)
                group_lines.extend(lines)
                continue
            if tag in {'replace', 'delete'}:
                for line in a[i1:i2]:
                    lines.append('-' + line)
            if tag in {'replace', 'insert'}:
                for line in b[j1:j2]:
                    if line in matcher.bjunk:
                        lines = None
                        break
                    keep_group = True
                    lines.append('+' + line)
            if lines is not None:
                group_lines.extend(lines)

        if keep_group:
            yield from group_lines


def isjunk(s):
    if re.match(r'\A[<]\w[\w\s\d]+(at\s+0x[\dabcdef]+)[>]\s*\Z', s):
        return True
    if re.match(r"\A[{].*'data': [(]\d+,\s*(True|False)[)].*[}]\s*\Z", s):
        return True
    return False


def apply_indent(code, indent):
    lines = []
    for line in code.splitlines(True):
        if indent < 0:
            tab = line[:-indent]
            if tab.replace(' ', ''):
                raise RuntimeError(
                    f'cannot unindent `{line}` by {-indent} spaces ({tab=})')
            lines.append(line[-indent:])
        else:
            tab = ' ' * indent
            lines.append(tab + line)
    return ''.join(lines)


def process_code(code, lineno, doc_globals, doc_locals):
    """Execute Python sessions in code.
    """
    plan = []
    stmt = ''
    output = ''
    mode = 0
    for i, line in enumerate(code.splitlines(True)):
        if mode == 1 and line.startswith('>>> '):  # no output
            mode = 2
        if mode == 2:  # output continued or new input
            if line.startswith('>>> '):
                plan.append((stmt, output))
                stmt = ''
                output = ''
                mode = 0
            else:
                output += line
                continue
        if mode == 0:  # look for input starting with '>>> '
            if line.startswith('>>> '):
                stmt += line[4:]
                mode = 1
            else:
                raise RuntimeError(
                    f'expected line starting with `>>> ` but got'
                    f' `{line!r}` at line #{lineno+i}')
            continue
        if mode == 1:  # look for '... ' or start of output
            if line.startswith('... '):
                stmt += line[4:]
                continue
            output += line
            mode = 2
    else:
        if stmt:
            plan.append((stmt, output))

    actual = []
    for i, (stmt, output) in enumerate(plan):
        t = ast.parse(stmt)
        if stmt.endswith('\n\n'):
            stmt = stmt.rstrip() + '\n'
        if stmt.count('\n') <= 1:
            s = stmt.rstrip()
        else:
            s = stmt.replace('\n', '\n... ')
        if len(t.body) == 1 and isinstance(t.body[0], ast.Expr):
            sys.stdout.write('.')
            actual.append(f'>>> {s}')
            try:
                r = eval(stmt, doc_globals, doc_locals)
            except Exception as msg:
                actual.append(f'{type(msg).__name__}: {msg}')
            else:
                actual.append(f'{r!r}')
        else:
            sys.stdout.write('*')
            actual.append(f'>>> {s}')
            try:
                exec(stmt, doc_globals, doc_globals)
            except Exception as msg:
                actual.append(f'{type(msg).__name__}: {msg}')
        sys.stdout.flush()

    return ('\n'.join(actual) + '\n')


def process_document(document):
    """Read the Python sessions from filename and execute statements.

    Return new Markdown document with all Python sessions evaluated.
    """
    requirements = Requirements()
    content = ['']
    flag = False
    code = ''

    doc_globals = {}
    doc_locals = {}
    collector = []
    for lineno, line in enumerate(document.splitlines(True)):
        if line.rstrip().endswith('-->'):
            if line.lstrip().startswith('<!--REQUIRE(') or line.lstrip().startswith('<!--UNREQUIRE('):
                i0 = line.find('(')
                i1 = line.find(')')
                predicate = line[i1+1].rstrip()[:-3].strip() or 'True'
                for packagename in line[i0+1:i1].split(','):
                    if 'UNREQUIRE' in line:
                        requirements.drop(packagename.strip())
                    else:
                        requirements.add(packagename.strip(), predicate)
        if not flag:
            if line.strip() == '```python':
                flag = True
                indent = line.index('`')
            content[-1] += line
        else:
            if line.strip() == '```':
                flag = False
                if requirements:
                    code = apply_indent(code, -indent)
                    code = process_code(code, lineno, doc_globals, doc_locals)
                    code = apply_indent(code, indent)
                else:
                    startlineno = lineno - code.count('\n')
                    endlineno = lineno
                    print(f'{requirements} not satisfied. Skip processing lines {startlineno}:{endlineno}')
                content.append(code)
                content.append(line)
                code = ''
            else:
                code += line
    sys.stdout.write('\n')
    sys.stdout.flush()
    return ''.join(content)


def main():
    parser = argparse.ArgumentParser(
        description='Validate Python sessions in Markdown documents.')

    parser.add_argument(
        'paths', metavar='paths', type=str, nargs='*',
        default=['.'],
        help=('Paths to the location of .md files.'
              ' Default is current working director.'))

    parser.add_argument(
        '--inplace', dest='inplace',
        action='store_const', const=True,
        default=False,
        help='Replace the outputs of Python sessions in md files.')

    parser.add_argument(
        '--verbose', dest='verbose',
        action='store_const', const=True,
        default=False,
        help='Be verbose, useful for debugging. Default is False.')

    args = parser.parse_args()
    if args.verbose:
        print(f'{args}')

    files = []

    for path in args.paths:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            for root_, dirs_, files_ in os.walk(path):
                for fn in files_:
                    if fn.endswith('.md'):
                        files.append(os.path.join(root_, fn))
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f'{path} does not exist. Skiping')
            continue

    if args.verbose:
        print(f'{files=}')

    for fn in files:
        print(f'Evalute Python sessions in {fn}')
        document = open(fn).read()
        new_document = process_document(document)
        if document != new_document:
            print('Detected differences:')
            for line in unified_diff(
                    document.splitlines(keepends=True),
                    new_document.splitlines(keepends=True),
                    fromfile=fn,
                    tofile='evaluated',
                    isjunk=isjunk):
                print(line.rstrip())
            if args.inplace:
                print(f'Updating {fn}')
                f = open(fn, 'w')
                f.write(new_document)
                f.close()


if __name__ == '__main__':
    main()
