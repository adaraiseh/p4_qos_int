# p4_rules/__init__.py
"""
P4 rule generation package.

Provides automatic generation of P4 table rules based on topology configuration.

Note: Import P4RuleGenerator and generate_rules directly from p4_rules.generator
to avoid circular import warnings when running as module.
"""

__all__ = ['P4RuleGenerator', 'generate_rules']

def __getattr__(name):
    """Lazy import to avoid circular import warnings."""
    if name in ('P4RuleGenerator', 'generate_rules'):
        from .generator import P4RuleGenerator, generate_rules
        return P4RuleGenerator if name == 'P4RuleGenerator' else generate_rules
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
