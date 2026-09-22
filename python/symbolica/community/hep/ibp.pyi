"""Native RustRed IBP solving in the host Symbolica kernel."""

from typing import overload

from symbolica import Expression

from . import IntegralFamily

class IBPFamily:
    def __init__(self, family: IntegralFamily, *, name: str = "F") -> None: ...
    @property
    def index_symbols(self) -> list[Expression]: ...
    @property
    def denominator_count(self) -> int: ...
    def ibp_identities(self) -> list[list[tuple[list[Expression], Expression]]]: ...
    def solve_parametric(
        self,
        sector: list[bool],
        *,
        fixed: list[int | None] | None = None,
        max_depth: int = 2,
        include_lorentz: bool = False,
    ) -> IBPSolution: ...
    def reduce_laporta(
        self,
        targets: list[list[int]],
        *,
        max_depth: int = 2,
        include_lorentz: bool = False,
        max_targets: int = 1024,
    ) -> IBPSolution: ...

class IBPRule:
    @property
    def target(self) -> list[Expression]: ...
    @property
    def terms(self) -> list[tuple[list[Expression], Expression]]: ...
    @property
    def nonzero_conditions(self) -> list[Expression]: ...
    @property
    def exceptions(self) -> list[list[Expression]]: ...
    @property
    def sector(self) -> list[bool]: ...
    @overload
    def apply(
        self, powers: list[int], *, integral: None = None
    ) -> list[tuple[list[int], Expression]]: ...
    @overload
    def apply(self, powers: list[int], *, integral: Expression) -> Expression: ...

class IBPSolution:
    @property
    def rules(self) -> list[IBPRule]: ...
    @property
    def residuals(self) -> list[list[int]]: ...
    @property
    def stats(self) -> dict[str, int]: ...
    @overload
    def reduce(
        self, powers: list[int], *, integral: None = None
    ) -> list[tuple[list[int], Expression]]: ...
    @overload
    def reduce(self, powers: list[int], *, integral: Expression) -> Expression: ...
