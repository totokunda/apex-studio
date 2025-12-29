from typing import TypeVar, overload, Callable, Any
import functools

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T", bound=type)


class FunctionRegister:
    """
    An instantiable registry/decorator.

    >>> api = register()            # first registry
    >>> math_ops = register()       # second registry
    >>>
    >>> @api("healthcheck")
    ... def ping():
    ...     return {"status": "ok"}
    >>>
    >>> @math_ops("double")
    ... def double(x):
    ...     return 2 * x
    >>>
    >>> api.call("healthcheck")     # {'status': 'ok'}
    >>> math_ops.call("double", 3)  # 6
    """

    # ------------------------------------------------------------------ #
    def __init__(self, *, allow_overwrite: bool = False):
        """
        Parameters
        ----------
        allow_overwrite : bool
            If False (default), re-using a key raises KeyError.
            If True, newly-decorated callables replace existing ones.
        """
        self._store: dict[str, dict] = {}
        self._allow_overwrite = allow_overwrite

    # --------------- decorator interface ------------------------------ #
    @overload
    def __call__(
        self, key_or_func: F, *, overwrite: bool | None = None, available: bool = True
    ) -> F: ...

    @overload
    def __call__(
        self,
        key_or_func: str | None = None,
        *,
        overwrite: bool | None = None,
        available: bool = True,
    ) -> Callable[[F], F]: ...

    def __call__(
        self, key_or_func=None, *, overwrite: bool | None = None, available: bool = True
    ):
        """
        Acts as either:
            @registry                 – uses the function's __name__ as key
            @registry("explicit_key") – uses the supplied key

        `overwrite` overrides the instance-wide `allow_overwrite` flag.
        `available` determines if the function is available for use.
        """
        # Case 1: used bare -- @registry
        if callable(key_or_func) and overwrite is None:
            func = key_or_func
            key = func.__name__
            self._add(key, func, self._allow_overwrite, available)
            return func

        # Case 2: used with explicit key -- @registry("name")
        key = key_or_func

        def decorator(func: F) -> F:
            self._add(
                key,
                func,
                overwrite if overwrite is not None else self._allow_overwrite,
                available,
            )
            return func

        return decorator

    # --------------- CRUD helpers ------------------------------------- #
    def _add(self, key: str, func: callable, allow: bool, available: bool):
        if not allow and key in self._store:
            raise KeyError(
                f"Key '{key}' already registered. Use overwrite=True to replace."
            )
        self._store[key] = {"func": func, "available": available}

    def get(self, key: str) -> callable:
        """Return the callable registered under *key* (raises KeyError if missing)."""
        if key not in self._store:
            raise KeyError(f"Key '{key}' not found in registry.")
        return self._store[key]["func"]

    def is_available(self, key: str) -> bool:
        """Check if a registered function is available for use."""
        if key not in self._store:
            return False
        return self._store[key]["available"]

    def set_availability(self, key: str, available: bool):
        """Update the availability status of a registered function."""
        if key not in self._store:
            raise KeyError(f"Key '{key}' not found in registry.")
        self._store[key]["available"] = available

    def call(self, *args, key: str | None = None, **kwargs):
        """Directly invoke the registered function."""
        if key is None and hasattr(self, "_default"):
            key = self._default

        if not self.is_available(key):
            raise RuntimeError(f"Function '{key}' is not available.")

        return self.get(key)(*args, **kwargs)

    def all(self) -> dict[str, callable]:
        """Return a shallow copy of the internal mapping."""
        return {key: entry["func"] for key, entry in self._store.items()}

    def all_available(self) -> dict[str, callable]:
        """Return a mapping of only available functions."""
        return {
            key: entry["func"]
            for key, entry in self._store.items()
            if entry["available"]
        }

    def set_default(self, key: str):
        self._default = key

    def get_default(self) -> str:
        return self._default

    # --------------- syntactic sugar ---------------------------------- #
    __getitem__ = get  # registry["key"]
    __iter__ = lambda self: iter(self._store)  # iterate over keys
    __len__ = lambda self: len(self._store)


class ClassRegister:
    """
    An instantiable registry/decorator for **classes**.

    >>> models = ClassRegister()          # first registry
    >>> handlers = ClassRegister()        # second registry
    >>>
    >>> @models("user")
    ... class User:
    ...     def __init__(self, name):
    ...         self.name = name
    ...     def greet(self):
    ...         return f"Hi, I'm {self.name}"
    >>>
    >>> @handlers                       # key defaults to class name → "Logger"
    ... class Logger:
    ...     def __call__(self, msg):    # behaves like a callable handler
    ...         print(f"[log] {msg}")
    >>>
    >>> u = models.call("user", "Alice")  # creates a User instance
    >>> u.greet()                         # "Hi, I'm Alice"
    >>> log = handlers.call()             # default call (set below)
    >>> log("Started!")                   # prints: [log] Started!
    """

    # ------------------------------------------------------------------ #
    def __init__(self, *, allow_overwrite: bool = False):
        self._store: dict[str, dict] = {}
        self._allow_overwrite = allow_overwrite

    # ---------------- decorator interface ----------------------------- #
    @overload
    def __call__(
        self, key_or_cls: T, *, overwrite: bool | None = None, available: bool = True
    ) -> T: ...

    @overload
    def __call__(
        self,
        key_or_cls: str | None = None,
        *,
        overwrite: bool | None = None,
        available: bool = True,
    ) -> Callable[[T], T]: ...

    def __call__(
        self, key_or_cls=None, *, overwrite: bool | None = None, available: bool = True
    ):
        """
        Acts as either:
            @registry                – uses the class' __name__ as key
            @registry("explicit")    – uses the supplied key

        `overwrite` overrides the instance-wide `allow_overwrite` flag.
        `available` determines if the class is available for use.
        """
        # Case 1 – bare decorator: @registry
        if isinstance(key_or_cls, type) and overwrite is None:
            cls = key_or_cls
            key = cls.__name__
            self._add(key, cls, self._allow_overwrite, available)
            return cls

        # Case 2 – explicit key: @registry("name")
        key = key_or_cls

        def decorator(cls: T) -> T:
            if not isinstance(cls, type):
                raise TypeError("Only classes can be registered in ClassRegister.")
            self._add(
                key,
                cls,
                overwrite if overwrite is not None else self._allow_overwrite,
                available,
            )
            return cls

        return decorator

    def register(self, key: str, cls: type):
        self._add(key, cls, self._allow_overwrite, True)

    # ---------------- CRUD helpers ------------------------------------ #
    def _add(self, key: str, cls: type, allow: bool, available: bool):
        if not allow and key in self._store:
            raise KeyError(
                f"Key '{key}' already registered. Use overwrite=True to replace."
            )
        self._store[key] = {"cls": cls, "available": available}

    def get(self, key: str) -> type:
        """Return the class registered under *key* (raises KeyError if missing)."""
        if key not in self._store:
            raise KeyError(f"Key '{key}' not found in registry.")
        return self._store[key]["cls"]

    def is_available(self, key: str) -> bool:
        """Check if a registered class is available for use."""
        if key not in self._store:
            return False
        return self._store[key]["available"]

    def set_availability(self, key: str, available: bool):
        """Update the availability status of a registered class."""
        if key not in self._store:
            raise KeyError(f"Key '{key}' not found in registry.")
        self._store[key]["available"] = available

    def call(self, key: str | None = None, *args, **kwargs):
        """
        Instantiate and return an object of the registered class.

        If *key* is omitted and a default has been set via ``set_default``,
        that default is used.
        """
        if key is None and hasattr(self, "_default"):
            key = self._default

        if not self.is_available(key):
            raise RuntimeError(f"Class '{key}' is not available.")

        return self.get(key)(*args, **kwargs)

    def all(self) -> dict[str, type]:
        """Return a shallow copy of the internal mapping."""
        return {key: entry["cls"] for key, entry in self._store.items()}

    def all_available(self) -> dict[str, type]:
        """Return a mapping of only available classes."""
        return {
            key: entry["cls"]
            for key, entry in self._store.items()
            if entry["available"]
        }

    def set_default(self, key: str):
        """Mark *key* as the default used when ``call()`` receives no key."""
        if key not in self._store:
            raise KeyError(f"No class registered under key '{key}'.")
        self._default = key

    # ---------------- syntactic sugar --------------------------------- #
    __getitem__ = get  # registry["key"]
    __iter__ = lambda self: iter(self._store)  # iterate over keys
    __len__ = lambda self: len(self._store)
