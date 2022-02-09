# Copyright 2014-2020 Carnegie Mellon University

"""PyIBL is an implementation of a subset of Instance Based Learn Theory (IBLT).
The principle class is Agent, an instance of which is a cognitive entity learning and
making decisions based on its experience from prior decisions, primarily by calls to its
:meth:`Agent.choose` and :meth:`Agent.respond` methods. The decisions an agent is choosing
between can be further decorated with information about their current state. There are
facilities for inspecting details of the IBL decision making process programmatically
facilitating debugging, logging and fine grained control of complex models.
"""

import collections
import collections.abc as abc
import csv
import io
import numbers
import os
import pyactup
import random
import re
import sys

from itertools import count
from keyword import iskeyword
from ordered_set import OrderedSet
from pprint import pprint
from prettytable import PrettyTable
from warnings import warn

__version__ = "4.1.2"

class Agent:
    """A cognitive entity learning and making decisions based on its experience from prior decisions.
    The main entry point to PyIBL. An Agent has a *name*, a string, which can be retrieved
    with the :attr:`name` property. The name cannot be changed after an agent is created.
    If, when creating an agent, the *name* argument is not supplied or is ``None``, a name
    will be created of the form ``'Anonymous-Agent-n'``, where *n* is a unique integer.

    An :class:`Agent` also has zero or more *attributes*, named by strings. The attribute
    names can be retrieved with the :attr:`attributes` property, and also cannot be
    changed after an agent is created. Attribute names must be formed from letters, digits
    and underscore, must begin with a letter, and may not be Python keywords.

    The agent properties :attr:`noise`, :attr:`decay`, :attr:`temperature`,
    :attr:`mismatch_penalty`, :attr:`optimized_learning` and :attr:`default_utility` can
    be initialized when creating an Agent.
    """

    _agent_number = 0

    def __init__(self,
                 name=None,
                 attributes=[],
                 noise=pyactup.DEFAULT_NOISE,
                 decay=pyactup.DEFAULT_DECAY,
                 temperature=None,
                 mismatch_penalty=None,
                 optimized_learning=False,
                 default_utility=None):
        self._attributes = Agent._ensure_attribute_names(list(attributes))
        if name is None:
            Agent._agent_number += 1
            name = f"agent-{Agent._agent_number}"
        elif not (isinstance(name, str) and len(name) > 0):
            raise TypeError(f"Agent name {name} is not a non-empty string")
        self._name = name
        self._memory = pyactup.Memory(learning_time_increment=0,
                                      optimized_learning=optimized_learning)
        self.temperature = temperature # set temperature BEFORE noise
        self.noise = noise
        self.decay = decay
        self.mismatch_penalty = mismatch_penalty
        self.default_utility = default_utility
        self.default_utility_populates = True
        self._attribute_similarities = [None] * len(self._attributes)
        self._details = None
        self._trace = False
        self.reset()
        self._test_default_utility()

    @staticmethod
    def _ensure_attribute_names(attributes):
        result = OrderedSet()
        for a in attributes:
            if not (isinstance(a, str) and
                    re.fullmatch(r"\w(?<![\d_])\w*", a) and
                    not iskeyword(a)):
                raise ValueError(f"'{a}' cannot be used as an attribute name")
            if a in result:
                raise ValueError(f"duplicate attribute {a}")
            result.add(a)
        return result

    def __repr__(self):
        return f"<Agent {str(self)} {id(self)}>"

    def __str__(self):
        return str(self._name)

    @property
    def name(self):
        """The name of this Agent.
        It is a string, provided when the agent was created, and cannot be changed
        thereafter.
        """
        return self._name

    @property
    def attributes(self):
        """A tuple of the names of the attributes included in all situations associated with decisions this agent will be asked to make.
        These names are assigned when the agent is created and cannot be
        changed, and are strings. The order of them in the returned
        tuple is the same as that in which they were given when the
        agent was created.
        """
        return tuple(self._attributes)

    def reset(self, preserve_prepopulated=False, optimized_learning=None):
        """Erases this agent's memory and resets its time to zero.
        If *preserve_prepopulated* is false it delets all the instances from this agent;
        if it is true it deletes all those not created at time zero. IBLT parameters such
        as :attr:`noise` and :attr:`decay` are not affected. Any prepopulated instances,
        including those created automatically if a :attr:`defaultUtility` is provided and
        :attr:`defaultUtilityPopulates` is true are removed, but the settings of those
        properties are not altered.

        If *optimized_learning* is supplied and is ``True`` or ``False`` it sets the
        value of :attr:`optimized_learning` for this :class:`Agent`. If it is not supplied
        or is ``None`` the current value of :attr:`optimized_learning` is not changed.
        """
        self._memory.reset(preserve_prepopulated=preserve_prepopulated,
                           optimized_learning=optimized_learning)
        self._last_learn_time = 0
        self._previous_choices = None
        self._pending_decision = None

    @property
    def time(self):
        """This agent's current time.
        Time in PyIBL is a dimensionless quantity, simply counting the number of
        choose/respond cycles that have occurred since the Memory was last :meth:`reset`.
        """
        return self._memory.time

    @property
    def noise(self):
        """The amount of noise to add during instance activation computation.
        This is typically a positive, possibly floating point, number between about 0.1 and 1.5.
        It defaults to 0.25.
        If explicitly zero, no noise is added during activation computation.
        If set to ``None`` it reverts the value to its default, 0.25.
        If an explicit :attr:`temperature` is not set, the value of noise is also used
        to compute a default temperature for the blending operation on result utilities.
        Attempting to set :attr:`noise` to a negative number raises a :exc:`ValueError`.
        """
        return self._memory.noise

    @noise.setter
    def noise(self, value):
        if value is None or value is False:
            value = pyactup.DEFAULT_NOISE
        if value != getattr(self._memory, "noise", None):
            self._memory.noise = float(value)

    @property
    def temperature(self):
        """The temperature parameter used for blending values.
        If ``None``, the default, the square root of 2 times the value of
        :attr:`noise` will be used. If the temperature is too close to zero, which
        can also happen if it is ``None`` and the :attr:`noise` is too low, or negative, a
        :exc:`ValueError` is raised.
        """
        return self._memory.temperature

    @temperature.setter
    def temperature(self, value):
        self._memory.temperature = value

    @property
    def decay(self):
        """Controls the rate at which activation for previously experienced instances in memory decay with the passage of time.
        Time in this sense is dimensionless, and simply the number of choose/respond cycles that have occurred since the
        agent was created or last :meth:`reset`.
        The :attr:`decay` is typically between about 0.1 to about 10.
        The default value is 0.5. If zero memory does not decay.
        If set to ``None`` it reverts the value to its default, 0.5.
        Attempting to set it to a negative number raises a :exc:`ValueError`.
        It must be less one 1 if this agent's :attr:`optimized_learning` parameter is set.
        """
        return self._memory.decay

    @decay.setter
    def decay(self, value):
        if value is None or value is False:
            value = pyactup.DEFAULT_DECAY
        if value != getattr(self._memory, "decay", None):
            self._memory.decay = float(value)

    @property
    def mismatch_penalty(self):
        """The mismatch penalty applied to partially matching values when computing activations.
        If ``None`` no partial matching is done.
        Otherwise any defined similarity functions (see :func:`similarity`) are called as necessary, and
        the resulting values are multiplied by the mismatch penalty and subtracted
        from the activation. For any attributes and decisions for which similarity
        functions are not defined exact matches are viewed as maximally similar (1) and
        any non-exact matches as maximally dissimilar (0).

        Attempting to set this parameter to a value other than ``None`` or a real number
        raises a :exc:`ValueError`.
        """
        return self._memory.mismatch

    @mismatch_penalty.setter
    def mismatch_penalty(self, value):
        if value is False:
            value = None
        self._memory.mismatch = value
        self._test_default_utility()

    @property
    def optimized_learning(self):
        """Whether or not this :class:"`Agent` uses the optimized_learning approximation when computing instance activations.
        This can only be changed for an :class:`Agent` by calling :meth:`reset`.
        """
        return self._memory.optimized_learning

    @property
    def details(self):
        """A :class:`MutableSequence` into which details of this Agent's internal computations will be added.
        If ``None``, the default, such details are not accumulated. It can be explicitly
        set to a :class:`MutableSequence` of the modeler's choice, typically a list, into
        which details are accumulated. Setting it to ``True`` sets the value to a fresh,
        empty list, whose value can be ascertained by consulting the value of
        :attr:`details`.

        A :exc:`ValueError` is raised if an attempt is made to set its value to anything
        other than ``None``, ``True`` or a :class:`MutableSequence`.

        >>> a = Agent(default_utility=10)
        >>> a.choose("a", "b", "c")
        'c'
        >>> a.respond(5)
        >>> a.details = True
        >>> a.choose()
        'a'
        >>> pprint(a.details)
        [[OrderedDict([('decision', 'a'),
                       ('activations',
                        [OrderedDict([('name', '0000'),
                                      ('creation_time', 1),
                                      ('attributes',
                                       (('_utility', 10), ('_decision', 'a'))),
                                      ('references', (1,)),
                                      ('base_activation', 0.0),
                                      ('activation_noise', -0.1115130828909049),
                                      ('activation', -0.1115130828909049),
                                      ('retrieval_probability', 1.0)])]),
                       ('blended', 10.0)]),
          OrderedDict([('decision', 'b'),
                       ('activations',
                        [OrderedDict([('name', '0001'),
                                      ('creation_time', 1),
                                      ('attributes',
                                       (('_utility', 10), ('_decision', 'b'))),
                                      ('references', (1,)),
                                      ('base_activation', 0.0),
                                      ('activation_noise', 0.49069430928516194),
                                      ('activation', 0.49069430928516194),
                                      ('retrieval_probability', 1.0)])]),
                       ('blended', 10.0)]),
          OrderedDict([('decision', 'c'),
                       ('activations',
                        [OrderedDict([('name', '0002'),
                                      ('creation_time', 1),
                                      ('attributes',
                                       (('_utility', 10), ('_decision', 'c'))),
                                      ('references', (1,)),
                                      ('base_activation', 0.0),
                                      ('activation_noise', 0.0870818061822312),
                                      ('activation', 0.0870818061822312),
                                      ('retrieval_probability', 0.4420238074030374)]),
                         OrderedDict([('name', '0003'),
                                      ('creation_time', 1),
                                      ('attributes',
                                       (('_utility', 5), ('_decision', 'c'))),
                                      ('references', (1,)),
                                      ('base_activation', 0.0),
                                      ('activation_noise', 0.16944297091092395),
                                      ('activation', 0.16944297091092395),
                                      ('retrieval_probability', 0.5579761925969626)])]),
                       ('blended', 7.210119037015187)])]]
        """
        return self._details

    @details.setter
    def details(self, value):
        if value == 0:
            value = None
        elif value == True:
            value = []
        if not (value is None or isinstance(value, abc.MutableSequence)):
            raise ValueError("the value of details must be None or a list or other MutableSequence")
        self._details = value

    @property
    def trace(self):
        """A boolean which, if ``True``, causes the :class:`Agent` to print details of its computations to standard output.
        Intended for use as a tool for debugging models. By default it is ``False``.

        The output is divided into the blocks, the first line of which describes the
        choice being described and the blended value of its outcome. This is followed by
        a tabular description of various intermediate values used to arrive at this
        blended value.

        ::

         >>> a = Agent(default_utility=10)
         >>> a.choose("a", "b", "c")
         'a'
         >>> a.respond(5)
         >>> a.choose("a", "b", "c")
         'c'
         >>> a.respond(7.2)
         >>> a.choose("a", "b", "c")
         'b'
         >>> a.respond(2.3)
         >>> a.choose()
         'a'
         >>> a.respond(5)
         >>> a.trace = True
         >>> a.choose()

         a → 5.7482098963642425
         +------+----------+---------+-------------+---------+---------------------+---------------------+----------------------+-----------------------+
         |  id  | decision | created | occurrences | outcome |   base activation   |   activation noise  |   total activation   | retrieval probability |
         +------+----------+---------+-------------+---------+---------------------+---------------------+----------------------+-----------------------+
         | 0022 |    a     |    0    |     [0]     |    10   | -0.6931471805599453 |  0.2696498251765441 | -0.42349735538340116 |  0.14964197927284847  |
         | 0025 |    a     |    1    |    [0, 4]   |    5    |  0.4054651081081644 | -0.2146946217750441 |  0.1907704863331203  |   0.8503580207271516  |
         +------+----------+---------+-------------+---------+---------------------+---------------------+----------------------+-----------------------+

         b → 2.8892224885373707
         +------+----------+---------+-------------+---------+---------------------+---------------------+---------------------+-----------------------+
         |  id  | decision | created | occurrences | outcome |   base activation   |   activation noise  |   total activation  | retrieval probability |
         +------+----------+---------+-------------+---------+---------------------+---------------------+---------------------+-----------------------+
         | 0023 |    b     |    0    |     [0]     |    10   | -0.6931471805599453 | 0.01639160687781119 | -0.6767555736821341 |  0.07652240110874947  |
         | 0027 |    b     |    3    |     [3]     |   2.3   | -0.3465735902799726 |  0.5503650166906361 |  0.2037914264106635 |   0.9234775988912505  |
         +------+----------+---------+-------------+---------+---------------------+---------------------+---------------------+-----------------------+

         c → 7.442068460676917
         +------+----------+---------+-------------+---------+---------------------+----------------------+---------------------+-----------------------+
         |  id  | decision | created | occurrences | outcome |   base activation   |   activation noise   |   total activation  | retrieval probability |
         +------+----------+---------+-------------+---------+---------------------+----------------------+---------------------+-----------------------+
         | 0024 |    c     |    0    |     [0]     |    10   | -0.6931471805599453 |  -0.787690810308673  | -1.4808379908686184 |  0.08645302167032752  |
         | 0026 |    c     |    2    |     [2]     |   7.2   | -0.5493061443340549 | -0.09794712508874652 | -0.6472532694228014 |   0.9135469783296726  |
         +------+----------+---------+-------------+---------+---------------------+----------------------+---------------------+-----------------------+

         'c'
        """
        return self._trace

    @trace.setter
    def trace(self, value):
        self._trace = bool(value)

    @property
    def default_utility(self):
        """The utility, or a function to compute the utility, if there is no matching instance.
        If when :meth:`choose` is called, for some choice passed to it there is no
        existing instance that matches the choice the value of this property is consulted.
        Note that an instance added with :meth:`populate` counts as matching, and will
        prevent the interrogation of this property. If partial matching
        (:attr:`mismatch_penalty`) is enabled, any instance that even partially matches a
        choice will prevent the iterrogation of this property.

        The value of this property may be a :class:`Real`, in which case when needed it is
        simply used as the default utility. If it is not a Real, it is assumed to be a
        function that takes one argument, one of the choices passed to :meth:`choose`.
        When a default utility is needed that function will be called, passing the choice
        in question to it, and the value returned, which should be a Real, will be used.
        If at that time the value is not a function of one argument, or it does not return
        a Real, an :exc:`RuntimeError` is raised.

        The :attr:`default_utility_populates` property, which is ``True`` by default,
        controls whether or not an instance is added for each interrogation of
        the attr:`default_utility` property. If an instance is added, it is added
        as by :meth:`populate_at` with a first argument of zero.

        Setting :attr:`default_utility` to ``None`` or ``False`` (the initial default)
        causes no default utility to be used. In this case, if :meth:`choose` is called
        for a decision in a situation for which there is no instance available, an
        :exc:`RuntimeError` will be raised.

        """
        return self._default_utility

    @default_utility.setter
    def default_utility(self, value):
        if value is False:
            value = None
        self._callable_default_utility =  not (value is None or isinstance(value, numbers.Real))
        self._default_utility = value
        self._test_default_utility()

    def _test_default_utility(self):
        try:
            if self._default_utility and self._memory.mismatch:
                warn("Setting both a default_utility when partial matching is in use is usually ill-advised")
        except AttributeError:
            # We were called before self was completely initialized.
            pass

    @property
    def default_utility_populates(self):
        """Whether or not a default utility provided by the :attr:`default_utility` property is also entered as an instance in memory.
        This property has no effect if default_utility has been set to ``None`` or ``False``.
        """
        return self._default_utility_populates

    @default_utility_populates.setter
    def default_utility_populates(self, value):
        self._default_utility_populates = bool(value)

    def populate(self, outcome, *choices):
        """Adds instances to memory, one for each of the *choices*, with the given outcome, at the current time, without advancing that time.
        The *outcome* should be a :class:`Real` number.
        The *choices* are as described in :meth:`choose`.
        Time is a dimensionless quantity, simply a count of the number of choose/respond
        cycles that have occurred since the agent was created or last :meth:`reset`.

        This is typically used to enable startup of a model by adding instances before the
        first call to :meth:`choose`. When used in this way the timestamp associated with
        this occurrence of the instance will be zero. Subsequent occurrences are possible
        if :meth:`respond` is called with the same outcome after :meth:`choose` has
        returned the same decision in the same situation, in which case those reinforcing
        occurrences will have later timestamps. An alternative mechanism to facilitate
        sartup of a model is setting the :attr:`defaultUtility` property of the agent.
        While rarely done, a modeler can even combine the two mechanisms, if desired.

        It is also possible to call prepopulate after choose/respond cycles have occurred.
        In this case the instances are added with the current time as the timestamp. This
        is one less than the timestamp that would be used were an instance to be added by
        being experienced as part of a choose/respond cycle instead. Each agent keeps
        internally a clock, the number of choose/respond cycles that have occurred since
        it was created or last :meth:`reset`. When :meth:`choose` is called it advances
        that clock by one *before* computing the activations of the existing instances, as
        it must since the activation computation depends upon all experiences having been in
        the past. That advanced clock is the timestamp used when an instance is added or
        reinforced by :meth:`respond`. See also :meth:`populate_at`.

        .. warning::
            In normal use you should only call :meth:`populate` before any choose/respond
            cycles. If, for exotic purposes, you do wish to call it after, caution should
            be exercised to avoid biologically implausible models.

        Raises a :exc:`ValueError` if *outcome* is not a :class:`Real` number, or if any
        of the *choices* are malformed or duplicates.
        """
        Agent._outcome_value(outcome)
        for choice in self._make_queries(choices):
            self._memory.learn(_utility=outcome, **choice)
            self._last_learn_time = self._memory.time

    @staticmethod
    def _attribute_value(value, attribute):
        if not isinstance(value, abc.Hashable):
            raise ValueError(
                f"{value} is not hashable and cannot be used as the value of attribute {attribute}")
        return value

    def _canonicalize_choice(self, choice):
        if self.attributes:
            if isinstance(choice, abc.Mapping):
                return { a: Agent._attribute_value(choice.get(a), a)
                         for a in self._attributes }
            elif isinstance(choice, abc.Sequence):
                return { a: Agent._attribute_value(c, a)
                         for c, a in zip(choice, self._attributes) }
            else:
                raise ValueError(f"{choice} cannot be used as a choice")
        elif isinstance(choice, abc.Hashable):
            return { "_decision": choice }
        else:
            raise ValueError(f"{choice} is not hashable and cannot be used as a choice")

    def _make_queries(self, choices):
        result = [ self._canonicalize_choice(c) for c in choices ]
        if len(set(tuple(d.items()) for d in result)) != len(result):
            raise ValueError("duplicate choices")
        return result

    @staticmethod
    def _outcome_value(value):
        if not isinstance(value, numbers.Real):
            raise ValueError(
                f"outcome {outcome} does not have a (non-complex) numeric value")
        return value

    def _at_time(self, when, callback):
        if not isinstance(when, int):
            raise ValueError(f"Time {when} is not an integer")
        if when > self.time:
            raise ValueError(f"Time {when} cannot be greater than the current time, {self.time}")
        saved = self._memory.time
        try:
            self._memory._time = when
            callback()
        finally:
            self._memory._time = saved

    def populate_at(self, outcome, when, *choices):
        """Adds instances to memory, one for each of the *choices*, with the given outcome, at the stipulated time.
        The *outcome* should be a :class:`Real` number.
        The *choices* are as described in :meth:`choose`.
        The time at which the instances are added is given by *when*, an integer denoting
        the time, a dimensionless quantity advanced by one for each
        :meth:`choose`/:meth:`respond` cycle.

        .. warning::
            In normal use :meth:`populate_at` should not be needed. If, for exotic
            purposes, you do wish to use it, caution should be exercised to avoid
            biologically implausible models.

        Raises a :exc:`ValueError` if *outcome* is not a number; if *when* is not an
        integer, or is greater than the current time; or if any of the *choices* are
        malformed or duplicates.

        """
        self._at_time(when, lambda: self.populate(outcome, *choices))

    def choose(self, *choices):
        """Selects which of the *choices* is expected to result in the largest payoff, and returns it.
        The expected form of the *choices* depends upon whether or not this :class:`Agent`
        has any attributes or not. If it does not, each of the *choices* should be a
        :class:`Hashable`, representing an atomic choice; if any of the *choices* are not
        hashable a :exc:`ValueError` is raised.

        If this :class:`Agent` does have attributes (that is, the *attributes* argument
        was supplied and non-empty when it was created, or, equivalently, the
        :meth:`attributes` method returns a non-empty tuple), then each of the *choices*
        can be either a :class:`Mapping`, typically a :class:`dict`, mapping attribute
        names to their values, or a :class:`Sequence`, typically a :class:`list` or
        :class:`tuple`, containing attribute values in the order they were declared when
        this :class:`Agent` was created and would be returned by calling
        :meth:`attributes`. Attributes not present (either no key in the :class:`Mapping`,
        or a :class:`Sequence` shorter than the number of attributes) have a value of
        `None`, while values not corresponding to attributes of the :class:`Agent` (either
        a key in the :class:`Mapping` that does not match an attribute name, or a
        :class:`Sequence` longer than the number of attributes) are ignored. Whether a
        :class:`Mapping` or a :class:`Sequence`, all the attribute values must be
        :class:`Hashable`, and are typically strings or numbers. If any of the *choices*
        do not have one of these forms a :exc:`ValueError` is raised.

        In either case, if any pair of the *choices* duplicate each other, even if of
        different forms (e.g. dictionary versus list), and after adding default ``None``
        values and removing ignored values, a :exc:`ValueError` is raised.

        It is also possible to supply no *choices*, in which case those used
        in the most recent previous call to this method are reused. If there was no
        previous call to :meth:`choose` since the last time this :class:`Agent` was
        :meth:`reset` a :exc:`ValueError` is raised.

        For each of the *choices* this method finds all instances in memory that match,
        and computes their activations at the current time based upon when in the past
        they have been seen, modified by the value of the :attr:`decay` property, and with
        noise added as controlled by the :attr:`noise` property. If partial matching has
        been enabled with :attr:`mismatch-penalty` such matching instances need not match
        exactly, and the similarities modified by the mismatch penalty are subtracted from
        the activations. If partial matching is not enabled only those instances that
        match exactly are consulted. "Exact" matches are based on Python's ``==``
        operator, not ``is``. Thus, for example ``0``, ``0.0`` and ``False`` all match on
        another, as do ``1``, ``1.0`` and ``True``.

        Looking at the activations of the whole ensemble of instances matching a choice a
        retrieval probability is computed for each possible outcome, and these are
        combined to arrive at a blended value expected for the choice. This blending
        operation depends upon the value of the :attr:`temperature` property; if none is
        supplied a default is computed based on the value of the :attr:`noise` parameter.
        The value chosen and returned is that element of *choices* with the highest
        blended value. In case of a tie one will be chosen at random.

        After a call to :attr:`choose` a corresponding call must be made to
        :meth:`respond` before calling :attr:`choose` again, or a :exc:`RuntimeError`
        will be raised.

        Because of noise the results returned by :attr:`choose` are stochastic the results
        of running the following examples may differ in their details from those shown.

        >>> a = Agent("Button Pusher", default_utility=10)
        >>> a.choose("left", "right")
        'right'
        >>> a.respond(5)
        >>> a.choose()
        'left'
        >>> a = Agent("Pet Purchaser", ["species", "state"])
        >>> a.populate(0, ["parrot", "dead"])
        >>> a.populate(10, ["parrot", "squawking"])
        >>> a.choose(["parrot", "dead"], ["parrot", "squawking"])
        ['parrot', 'squawking']

        """
        if self._pending_decision:
            raise RuntimeError("choice requested before previous outcome was supplied")
        choices = list(choices)
        if not choices:
            if self._previous_choices:
                choices = self._previous_choices
            else:
                raise ValueError(
                    "choose() called with no choices without having been called since reset")
        queries = self._make_queries(choices)
        self._previous_choices = choices
        details = [] if self._details is not None else None
        if details is not None or self._trace:
            history = []
            self._memory.activation_history = history
        else:
            history = None
        if self._last_learn_time >= self._memory.time:
            self._memory.advance(self._last_learn_time - self._memory.time + 1)
        utilities = []
        for c, q, i in zip(choices, queries, count()):
            u = (self._memory.blend("_utility", **q), i)
            if u[0] is None:
                if self._default_utility:
                    if self._callable_default_utility:
                        u = (self._default_utility(c), i)
                    else:
                        u = (self._default_utility, i)
                    if self._default_utility_populates:
                        self._at_time(0, lambda: self._memory.learn(_utility=u[0], **q))
                else:
                    raise RuntimeError(f"No experience available for choice {c}")
            utilities.append(u)
            if details is not None:
                d = collections.OrderedDict(q if self.attributes
                                            else (("decision", q["_decision"]),))
                d["activations"] = history
                d["blended"] = u[0]
                details.append(d)
            if history is not None:
                if self._trace:
                    self._print_trace(q, u[0], history)
                history = []
                self._memory.activation_history = history
        best_utility = max(utilities, key=lambda x: x[0])[0]
        best = random.choice(list(filter(lambda x: x[0] == best_utility, utilities)))[1]
        self._pending_decision = (queries[best], best_utility)
        if self._details is not None:
            self._details.append(details)
        if self._trace:
            print(f"\n   {'='*140}")
        return choices[best]

    def _print_trace(self, query, utility, history):
        print()
        if self.attributes:
            print(", ".join(list(f"{k}: {v}" for k, v in query.items())), end="")
        else:
            print(query["_decision"], end="")
        print(f" → {utility}")
        tab = PrettyTable()
        fields = (["id"] + (list(self.attributes) or ["decision"]) +
                  ["created", "occurrences", "outcome", "base activation", "activation noise"])
        if self._memory.mismatch:
            fields.append("mismatch adjustment")
        fields.extend(["total activation", "retrieval probability"])
        tab.field_names = fields
        for h in history:
            attrs = dict(h["attributes"])
            row = [h["name"]]
            if self.attributes:
                for a in self.attributes:
                    row.append(attrs.get(a, ""))
            else:
                row.append(attrs["_decision"])
            row.append(h["creation_time"])
            row.append(h["references"] if self._memory.optimized_learning else list(h["references"]))
            row.append(attrs["_utility"])
            row.append(h["base_activation"])
            row.append(h["activation_noise"])
            if self._memory.mismatch:
                row.append(h["mismatch"])
            row.append(h["activation"])
            row.append(h["retrieval_probability"])
            tab.add_row(row)
        print(tab, flush=True)

    def respond(self, outcome=None):
        """Provide the *outcome* resulting from the most recent decision returned by :meth:`choose`.
        The *outcome* should be a real number, where larger numbers are considered "better."
        This results in the creation or reinforcemnt of an instance in memory for the
        decision with the given outcome, and is the fundamental way in which the PyIBL
        model "learns from experience."

        It is also possible to delay feedback, by calling :meth:`respond` without an
        argument. This tells the :class:`Agent` to assume it has received feedback equal
        to that it expected, that is, the blended value resulting from past experiences.
        In this case :meth:`respond` returns a value, a :class:`DelayedRespone` object,
        which can be used subsequently to update the response.

        .. warning::
            Delayed feedback is an experimental feature and care should be exercised in
            its use to avoid biologically implausible models.

        If there has not been a call to :meth:`choose` since the last time respond was
        called a :exc:`RuntimeError` is raised. If *outcome* is neither ``None`` nor a
        real number a :exc:`ValueError` is raised.

        """
        if not self._pending_decision:
            raise RuntimeError(
                f"outcome {outcome} supplied when no decision requiring an outcome is pending")
        if outcome is not None:
            self._memory.learn(_utility=Agent._outcome_value(outcome), **(self._pending_decision[0]))
            self._last_learn_time = self._memory.time
            self._pending_decision = None
        else:
            self._memory.learn(_utility=self._pending_decision[1], **(self._pending_decision[0]))
            self._last_learn_time = self._memory.time
            result = DelayedResponse(self, *self._pending_decision)
            self._pending_decision = None
            return result

    def instances(self, file=sys.stdout, pretty=True):
        """Prints or returns all the instances currently stored in this :class:`Agent`.
        If *file* is ``None`` a list of dictionaries is returned, each corresponding
        to an instance. If *file* is a string it is taken as a file name, opened for
        writing, and the results printed thereto; otherwise *file* is assumed to be
        an open, writable ``file``.

        When printing to a file if *pretty* is true, the default, a format intended for
        reading by humans is used. Otherwise comma separated values (CSV) format, more
        suitable for importing into spreadsheets, numpy, and the like, is used.
        """
        attrs = [ (a, a) for a in self.attributes ]
        if not attrs:
            attrs = [ ("decision", "_decision") ]
        result = []
        for c in self._memory.values():
            d = collections.OrderedDict((name, c[a]) for name, a in attrs)
            d["outcome"] = c["_utility"]
            d["created"] = c._creation
            d["occurrences"] = c._references
            result.append(d)
        if file is None:
            return result
        if isinstance(file, io.TextIOBase):
            Agent._print_instance_data(result, pretty, file)
        else:
            with open(file, "w+", newline=(None if pretty else "")) as f:
                Agent._print_instance_data(result, pretty, f)

    @staticmethod
    def _print_instance_data(data, pretty, file):
        if not data:
            return
        if pretty:
            tab = PrettyTable()
            tab.field_names = data[0].keys()
            for d in data:
                tab.add_row(d.values())
            print(tab, file=file, flush=True)
        else:
            w = csv.DictWriter(file, data[0].keys())
            w.writeheader()
            for d in data:
                w.writerow(d)


class DelayedResponse:
    """A representation of an intermediate state of the computation of a decision, as returned from :meth:`respond` called with no arguments.
    """

    def __init__(self, agent, attributes, expectation):
        self._agent = agent
        self._time = agent.time
        self._attributes = attributes
        self._resolved = False
        self._expectation = expectation
        self._outcome = expectation

    @property
    def is_resolved(self):
        """Whether or not ground truth feedback to the :class:`Agent` regarding this decision has yet been delivered by the user.
        """
        return self._resolved

    @property
    def expectation(self):
        """ The expected value learned when this :class:`DelayedReponse` was created.
        """
        return self._expectation

    @property
    def outcome(self):
        """The most recent response learned by the :class:`Agent` for this decision.
        When :attr:`is_resolved` is ``False`` this will be the reward expected by the
        :class:`Agent` when the decision was made. After it has been resolved by calling
        :meth:`update`, delivering the ground truth reward, this will be that real value.
        """
        return self._outcome

    def update(self, outcome):
        """Replaces current reward learned, either expected or ground truth, by a new ground truth value.

        The *outcome* is a real number. Typically this value replaces that learned when
        :meth:`respond` was called, though it
        might instead replace the value supplied by an earlier call to :meth:`update`.
        It is always learned at the time of the original call to :meth:`respond`.

        The most recent previous value of the learned reward, either the expected value, 
        or that set by a previous call of :metho:`update`, is returned.

        Raises a :exc:`ValueError` if *outcome* is not a real number.

        Because of noise the results returned by :attr:`choose` are stochastic the results
        of running the following examples will differ in their details from those shown.

        >>> a = Agent(default_utility=10)
        >>> a.choose("a", "b")
        'b'
        >>> a.respond(2)
        >>> a.choose("a", "b")
        'a'
        >>> a.respond(3)
        >>> a.choose("a", "b")
        'a'
        >>> r = a.respond()
        >>> a.choose("a", "b")
        'a'
        >>> a.respond(7)
        >>> a.instances()
        +----------+-------------------+---------+-------------+
        | decision |      outcome      | created | occurrences |
        +----------+-------------------+---------+-------------+
        |    a     |         10        |    0    |     [0]     |
        |    b     |         10        |    0    |     [0]     |
        |    b     |         2         |    1    |     [1]     |
        |    a     |         3         |    2    |     [2]     |
        |    a     | 8.440186635799552 |    3    |     [3]     |
        |    a     |         7         |    4    |     [4]     |
        +----------+-------------------+---------+-------------+
        >>> r.update(1)
        8.440186635799552
        >>> a.instances()
        +----------+---------+---------+-------------+
        | decision | outcome | created | occurrences |
        +----------+---------+---------+-------------+
        |    a     |    10   |    0    |     [0]     |
        |    b     |    10   |    0    |     [0]     |
        |    b     |    2    |    1    |     [1]     |
        |    a     |    3    |    2    |     [2]     |
        |    a     |    1    |    3    |     [3]     |
        |    a     |    7    |    4    |     [4]     |
        +----------+---------+---------+-------------+
        """
        outcome = Agent._outcome_value(outcome)
        old = self._outcome
        self._agent._memory.forget(self._time, _utility=self._outcome, **self._attributes)
        self._agent._at_time(self._time,
                             lambda: self._agent._memory.learn(_utility=outcome,
                                                               **self._attributes))
        self._resolved = True
        self._outcome = outcome
        return old


def similarity(function, *attributes):
    """Add a function to compute the similarity of attribute values that are not equal.
    The *attributes* are names of attributes of any :class:`Agent`. If called with no
    *attributes* the *function* will be applied to the choices of any :class:`Agent` that
    has no attributes. If *attributes* contains names not acceptable as attribute names
    a :exc:`ValueError` is raised.

    The similarity value returned should be a real number between zero and one,
    inclusive. If, when called, the function returns a number outside that range a
    warning will be printed and the value will be modified to be zero (if negative) or
    one (if greater than one). If, when the similarity function is called, the return
    value is not a real number a :exc:`ValueError` is raised.

    Similarity functions are only called when the `Agent` has a :attr:`mismatch_penalty`
    specified.
    When a similarity function is called it is passed two arguments, attribute
    values to compare.
    The function should be commutative; that is, if called with the same arguments
    in the reverse order, it should return the same value.
    It should also be stateless, always returning the same values if passed
    the same arguments.
    If either of these constraints is violated no error is raised, but the results
    will, in most cases, be meaningless.

    If ``None`` is passed as the value of *function* the similarity
    function(s) for the specified attributes are cleared.

    In the following examples the height and width are assumed to range from zero to
    ten, and similarity of either is computed linearly, as the difference between
    them normalized by the maximum length of ten. The colors pink and red are considered
    50% similar, and all other color pairs are similar only if identical.

    >>> similarity(lambda v1, v2: 1 - abs((v1 - v2) / 10), "height", "width")
    >>> def color_similarity(c1, c2):
    ...     if c1 == c2:
    ...         return 1
    ...     elif c1 in ("red", "pink") and c2 in ("red", "pink"):
    ...         return 0.5
    ...     else:
    ...         return 0
    ...
    >>> similarity(color_similarity, "color")
    """
    pyactup.set_similarity_function(function or None,
                                    *(Agent._ensure_attribute_names(attributes)
                                      or [ "_decision" ]))

def identity_similarity(x, y):
    """Returns one if x and y are identical (Python's ``is``), and zero otherwise.
Sometimes useful as an argument to similarity if you wish distinct attribute values
not to prohibit consideration of a choice, but merely to reduce it's likelihood depending
upon the :attr:`mismatch_penalty`.

>>> identity_similarity(1, 1)
1
>>> identity_similarity(1, 1.0)
0

>>> identity_similarity(None, False)
0
>>> identity_similarity(None, None)
1
>>> identity_similarity(False, 0)
0
"""
    return int(x is y)

def equality_similarity(x, y):
    """Returns one if x and y are equal (Python's ``==``), and zero otherwise.
Sometimes useful as an argument to similarity if you wish unequal attribute values
not to prohibit consideration of a choice, but merely to reduce it's likelihood depending
upon the :attr:`mismatch_penalty`.

>>> equality_similarity(0, 0.0)
1
>>> sys.float_info.epsilon
2.220446049250313e-16
>>> equality_similarity(_, 0)
0
>>> equality_similarity(0, False)
1
>>> equality_similarity(0, None)
0
>>> equality_similarity(None, None)
1
"""
    return int(x == y)

def positive_linear_similarity(x, y):
    """Returns a similarity value of two positive :class:`Real` numbers, scaled linearly by the larger of them.
If *x* and *y* are equal the value is one, and otherwise a positive float less than one
the gets smaller the greater the difference between *x* and *y*.

If either *x* or *y* is not positive a :exc:`ValueError` is raised.

>>> positive_linear_similarity(1, 2)
0.5
>>> positive_linear_similarity(2, 1)
0.5
>>> positive_linear_similarity(1, 10)
0.09999999999999998
>>> positive_linear_similarity(10, 100)
0.09999999999999998
>>> positive_linear_similarity(1, 2000)
0.0004999999999999449
>>> positive_linear_similarity(1999, 2000)
0.9995
>>> positive_linear_similarity(1, 1)
1
>>> positive_linear_similarity(0.001, 0.002)
0.5
>>> positive_linear_similarity(10.001, 10.002)
0.9999000199960006
"""
    if x <= 0 or y <= 0:
        raise ValueError(f"the arguments, {x} and {y}, are not both positive")
    if x == y:
        return 1
    if x > y:
        x, y = y, x
    return 1 - (y - x) / y

def positive_quadratic_similarity(x, y):
    """Returns a similarity value of two positive :class:`Real` numbers, scaled quadratically by the larger of them.
If *x* and *y* are equal the value is one, and otherwise a positive float less than one
the gets smaller the greater the difference between *x* and *y*.

If either *x* or *y* is not positive a :exc:`ValueError` is raised.

>>> positive_quadratic_similarity(1, 2)
0.25
>>> positive_quadratic_similarity(2, 1)
0.25
>>> positive_quadratic_similarity(1, 10)
0.009999999999999995
>>> positive_quadratic_similarity(10, 100)
0.009999999999999995
>>> positive_quadratic_similarity(1, 2000)
2.4999999999994493e-07
>>> positive_quadratic_similarity(1999, 2000)
0.9990002500000001
>>> positive_quadratic_similarity(1, 1)
1
>>> positive_quadratic_similarity(0.001, 0.002)
0.25
>>> positive_quadratic_similarity(10.001, 10.002)
0.9998000499880025
"""
    return positive_linear_similarity(x, y)**2

def bounded_linear_similarity(minimum, maximum):
    """Returns a function of two arguments that returns a similarity value reflecting a linear scale between *minimum* and *maximum*.
The two arguments to the function returned should be :class:`Real` numbers between
*minimum* and *maximum*, inclusive. If the two arguments to the function returned are
equal they are maximally similar, and one is returned. If the absolute value of their
difference is as large as possible, they are maximally different, and zero is returned.
Otherwise a scaled value on a linear scale between these two extrema, measuring the
magnitude of the difference between the arguments two the returned function is used, a
value between zero and one being returned.

Raises a :exc:`ValueError` if either *minimum* or *maximum* is not a Real number, or if
*minimum* is not less than *maximum*.

When the returned function is called if either of its arguments is not a Real number a
:exc:`ValueError` is then raised. If either of those arguments is less than *minimum*,
or greater than *maximum*, a warning is issued, and either *minimum* or *maximum*,
respectively, is instead used as the argument's value.

>>> f = bounded_linear_similarity(-1, 1)
>>> f(0, 1)
0.5
>>> f(-0.1, 0.1)
0.9
>>> f(-1, 1)
0.0
>>> f(0, 0)
1.0
>>> sys.float_info.epsilon
2.220446049250313e-16
>>> f(0, _)
0.9999999999999999

    """
    if minimum >= maximum:
        raise ValueError(f"minimum, {minimum}, is not less than maximum, {maximum}")
    def _similarity(x, y):
        if x < minimum:
            warn(f"{x} is less than {minimum}, so {minimum} is instead being used in computing similarity")
            x = minimum
        elif x > maximum:
            warn(f"{x} is greater than {maximum}, so {maximum} is instead being used in computing similarity")
            x = maximum
        if y < minimum:
            warn(f"{y} is less than {minimum}, so {minimum} is instead being used in computing similarity")
            y = minimum
        elif y > maximum:
            warn(f"{y} is greater than {maximum}, so {maximum} is instead being used in computing similarity")
            y = maximum
        return 1 - abs(x - y) / abs(maximum - minimum)
    return _similarity

def bounded_quadratic_similarity(minimum, maximum):
    """Returns a function of two arguments that returns a similarity value reflecting a quadratic scale between *minimum* and *maximum*.
Both arguments to the function returned should be :class:`Real` numbers between *minimum*
and *maximum*, inclusive. If the two arguments to the function returned are equal they are
maximally similar, and one is returned. If the absolute value of their difference is as
large as possible, they are maximally different, and zero is returned. Otherwise a scaled
value on a quadratic scale between these two extrema, measuring the magnitude of the
difference between the arguments two the returned function is used, a value between zero
and one being returned.

Raises a :exc:`ValueError` if either *minimum* or *maximum* is not a Real number, or if
*minimum* is not less than *maximum*.

When the returned function is called if either of its arguments is not a Real number a
:exc:`ValueError` is then raised. If either of those arguments is less than *minimum*,
or greater than *maximum*, a warning is issued, and either *minimum* or *maximum*,
respectively, is instead used as the argument's value.

>>> f = bounded_quadratic_similarity(-1, 1)
>>> f(0, 1)
0.25
>>> f(-0.1, 0.1)
0.81
>>> f(-1, 1)
0.0
>>> f(0, 0)
1.0
>>> sys.float_info.epsilon
2.220446049250313e-16
>>> f(0, _)
0.9999999999999998

    """
    f = bounded_linear_similarity(minimum, maximum)
    return lambda x, y: f(x, y)**2


# Local variables:
# fill-column: 90
# End:
