"""
Utilities to turn SymPy objects into C strings.
"""

import numpy as np
import sympy

from mpmath.libmp import prec_to_dps, to_str
from packaging.version import Version
from numbers import Real

from sympy.core import S
from sympy.core.numbers import equal_valued, Float
from sympy.logic.boolalg import BooleanFunction
from sympy.printing.precedence import PRECEDENCE_VALUES, precedence
from sympy.printing.c import C99CodePrinter

from devito.arch.compiler import AOMPCompiler
from devito.symbolics.inspection import has_integer_args, sympy_dtype
from devito.types.basic import AbstractFunction

__all__ = ['ccode']


class CodePrinter(C99CodePrinter):

    """
    Decorator for sympy.printing.ccode.CCodePrinter.

    Parameters
    ----------
    settings : dict
        Options for code printing.
    """
    _default_settings = {'compiler': None, 'dtype': np.float32,
                         **C99CodePrinter._default_settings}

    @property
    def dtype(self):
        return self._settings['dtype']

    @property
    def compiler(self):
        return self._settings['compiler']

    def single_prec(self, expr=None):
        dtype = sympy_dtype(expr) if expr is not None else self.dtype
        return dtype in [np.float32, np.float16]

    def parenthesize(self, item, level, strict=False):
        if isinstance(item, BooleanFunction):
            return "(%s)" % self._print(item)
        return super().parenthesize(item, level, strict=strict)

    def _print_Function(self, expr):
        if isinstance(expr, AbstractFunction):
            return str(expr)
        else:
            if expr.func.__name__ not in self.known_functions:
                self.known_functions[expr.func.__name__] = expr.func.__name__
            return super()._print_Function(expr)

    def _print_CondEq(self, expr):
        return "%s == %s" % (self._print(expr.lhs), self._print(expr.rhs))

    def _print_Indexed(self, expr):
        """
        Print an Indexed as a C-like multidimensional array.

        Examples
        --------
        U[t,x,y,z] -> U[t][x][y][z]
        """
        inds = ''.join(['[' + self._print(x) + ']' for x in expr.indices])
        return '%s%s' % (self._print(expr.base.label), inds)

    def _print_FIndexed(self, expr):
        """
        Print an FIndexed, that is a special Indexed, as a C-like multiarguments function.

        Examples
        --------
        U[t,x,y,z] -> U(t,x,y,z)
        """
        inds = ', '.join(self._print(x) for x in expr.indices)
        try:
            label = expr.accessor.label
        except AttributeError:
            label = expr.base.label
        return '%s(%s)' % (self._print(label), inds)

    def _print_Rational(self, expr):
        """Print a Rational as a C-like float/float division."""
        # This method and _print_Float below forcefully add a F to any
        # literals generated in code. This forces all float literals
        # to be 32-bit floats.
        # http://en.cppreference.com/w/cpp/language/floating_literal
        p, q = int(expr.p), int(expr.q)
        if self.dtype == np.float64:
            return '%d.0/%d.0' % (p, q)
        else:
            return '%d.0F/%d.0F' % (p, q)

    def _print_math_func(self, expr, nest=False, known=None):
        cls = type(expr)
        name = cls.__name__

        try:
            cname = self.known_functions[name]
        except KeyError:
            return super()._print_math_func(expr, nest=nest, known=known)

        if cname not in self._prec_funcs:
            return super()._print_math_func(expr, nest=nest, known=known)

        if self.single_prec(expr):
            cname = '%sf' % cname

        if nest and len(expr.args) > 2:
            args = ', '.join([self._print(expr.args[0]),
                              self._print_math_func(cls(*expr.args[1:]))])
        else:
            args = ', '.join([self._print(arg) for arg in expr.args])

        return f'{cname}({args})'

    def _print_Pow(self, expr):
        # Completely reimplement `_print_Pow` from sympy, since it doesn't
        # correctly handle precision
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        suffix = 'f' if self.single_prec(expr) else ''
        if equal_valued(expr.exp, -1):
            return self._print_Float(Float(1.0)) + '/' + \
                self.parenthesize(expr.base, PREC)
        elif equal_valued(expr.exp, 0.5):
            return f'sqrt{suffix}({self._print(expr.base)})'
        elif expr.exp == S.One/3 and self.standard != 'C89':
            return f'cbrt{suffix}({self._print(expr.base)})'
        else:
            return f'pow{suffix}({self._print(expr.base)}, {self._print(expr.exp)})'

    def _print_SafeInv(self, expr):
        """Print a SafeInv as a C-like division with a check for zero."""
        base = self._print(expr.base)
        val = self._print(expr.val)
        return f'SAFEINV({val}, {base})'

    def _print_Mod(self, expr):
        """Print a Mod as a C-like %-based operation."""
        args = ['(%s)' % self._print(a) for a in expr.args]
        return '%'.join(args)

    def _print_Mul(self, expr):
        term = super()._print_Mul(expr)
        return term.replace("(-1)*", "-")

    def _print_Min(self, expr):
        if has_integer_args(*expr.args) and len(expr.args) == 2:
            return "MIN(%s)" % self._print(expr.args)[1:-1]
        else:
            return super()._print_Min(expr)

    def _print_Max(self, expr):
        if has_integer_args(*expr.args) and len(expr.args) == 2:
            return "MAX(%s)" % self._print(expr.args)[1:-1]
        else:
            return super()._print_Max(expr)

    def _print_Abs(self, expr):
        """Print an absolute value. Use `abs` if can infer it is an Integer"""
        # AOMPCC errors with abs, always use fabs
        if isinstance(self.compiler, AOMPCompiler):
            return "fabs(%s)" % self._print(expr.args[0])
        # Check if argument is an integer
        if has_integer_args(*expr.args[0].args):
            func = "abs"
        elif self.single_prec(expr):
            func = "fabsf"
        elif any([isinstance(a, Real) for a in expr.args[0].args]):
            # The previous condition isn't sufficient to detect case with
            # Python `float`s in that case, fall back to the "default"
            func = "fabsf" if self.single_prec() else "fabs"
        else:
            func = "fabs"
        return f"{func}({self._print(expr.args[0])})"

    def _print_Add(self, expr, order=None):
        """"
        Print an addition.
        """
        terms = self._as_ordered_terms(expr, order=order)

        PREC = precedence(expr)
        l = []
        for term in terms:
            t = self._print(term)
            if precedence(term) < PREC:
                l.extend(["+", "(%s)" % t])
            elif t.startswith('-'):
                l.extend(["-", t[1:]])
            else:
                l.extend(["+", t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)

    def _print_Float(self, expr):
        """Print a Float in C-like scientific notation."""
        prec = expr._prec

        if prec < 5:
            dps = 0
        else:
            dps = prec_to_dps(expr._prec)

        if self._settings["full_prec"] is True:
            strip = False
        elif self._settings["full_prec"] is False:
            strip = True
        elif self._settings["full_prec"] == "auto":
            strip = self._print_level > 1

        rv = to_str(expr._mpf_, dps, strip_zeros=strip, max_fixed=-2, min_fixed=2)

        if rv.startswith('-.0'):
            rv = "-0." + rv[3:]
        elif rv.startswith('.0'):
            rv = "0." + rv[2:]

        # Remove trailing zero except first one to avoid 1. instead of 1.0
        if 'e' not in rv:
            rv = rv.rstrip('0') + "0"

        if self.single_prec():
            rv = '%sF' % rv

        return rv

    def _print_Differentiable(self, expr):
        return "(%s)" % self._print(expr._expr)

    _print_EvalDerivative = C99CodePrinter._print_Add

    def _print_CallFromPointer(self, expr):
        indices = [self._print(i) for i in expr.params]
        return "%s->%s(%s)" % (expr.pointer, expr.call, ', '.join(indices))

    def _print_CallFromComposite(self, expr):
        indices = [self._print(i) for i in expr.params]
        return "%s.%s(%s)" % (expr.pointer, expr.call, ', '.join(indices))

    def _print_FieldFromPointer(self, expr):
        return "%s->%s" % (expr.pointer, expr.field)

    def _print_FieldFromComposite(self, expr):
        return "%s.%s" % (expr.pointer, expr.field)

    def _print_ListInitializer(self, expr):
        return "{%s}" % ', '.join([self._print(i) for i in expr.params])

    def _print_IndexedPointer(self, expr):
        return "%s%s" % (expr.base, ''.join('[%s]' % self._print(i) for i in expr.index))

    def _print_IntDiv(self, expr):
        lhs = self._print(expr.lhs)
        if not expr.lhs.is_Atom:
            lhs = '(%s)' % (lhs)
        rhs = self._print(expr.rhs)
        PREC = precedence(expr)
        return self.parenthesize("%s / %s" % (lhs, rhs), PREC)

    def _print_InlineIf(self, expr):
        cond = self._print(expr.cond)
        true_expr = self._print(expr.true_expr)
        false_expr = self._print(expr.false_expr)
        PREC = precedence(expr)
        return self.parenthesize("(%s) ? %s : %s" % (cond, true_expr, false_expr), PREC)

    def _print_UnaryOp(self, expr):
        if expr.base.is_Symbol:
            return "%s%s" % (expr._op, self._print(expr.base))
        else:
            return "%s(%s)" % (expr._op, self._print(expr.base))

    def _print_ComponentAccess(self, expr):
        return "%s.%s" % (self._print(expr.base), expr.sindex)

    def _print_TrigonometricFunction(self, expr):
        func_name = str(expr.func)
        if self.single_prec():
            func_name = '%sf' % func_name
        return '%s(%s)' % (func_name, self._print(*expr.args))

    def _print_DefFunction(self, expr):
        arguments = [self._print(i) for i in expr.arguments]
        if expr.template:
            template = '<%s>' % ','.join([str(i) for i in expr.template])
        else:
            template = ''
        return "%s%s(%s)" % (expr.name, template, ','.join(arguments))

    _print_MathFunction = _print_DefFunction

    def _print_Fallback(self, expr):
        return expr.__str__()

    _print_Namespace = _print_Fallback
    _print_Rvalue = _print_Fallback
    _print_MacroArgument = _print_Fallback
    _print_IndexedBase = _print_Fallback
    _print_IndexSum = _print_Fallback
    _print_ReservedWord = _print_Fallback
    _print_Basic = _print_Fallback


# Lifted from SymPy so that we go through our own `_print_math_func`
for k in ('exp log sin cos tan ceiling floor').split():
    setattr(CodePrinter, '_print_%s' % k, CodePrinter._print_math_func)


# Always parenthesize IntDiv and InlineIf within expressions
PRECEDENCE_VALUES['IntDiv'] = 1
PRECEDENCE_VALUES['InlineIf'] = 1


def ccode(expr, **settings):
    """Generate C++ code from an expression.

    Parameters
    ----------
    expr : expr-like
        The expression to be printed.
    settings : dict
        Options for code printing.

    Returns
    -------
    str
        The resulting code as a C++ string. If something went south, returns
        the input ``expr`` itself.
    """
    return CodePrinter(settings=settings).doprint(expr, None)


# Sympy 1.11 has introduced a bug in `_print_Add`, so we enforce here
# to always use the correct one from our printer
if Version(sympy.__version__) >= Version("1.11"):
    setattr(sympy.printing.str.StrPrinter, '_print_Add', CodePrinter._print_Add)
