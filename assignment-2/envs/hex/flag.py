from enum import Flag, auto


class HexFlag(Flag):
    """Flag stating whether a cell is connected to a given edge."""
    ISOLATED = 0

    # Wall connections
    P1_LOWER = auto()
    P1_UPPER = auto()
    P2_LOWER = auto()
    P2_UPPER = auto()

    # Victory flags (inferred)
    P1_WIN = P1_LOWER | P1_UPPER
    P2_WIN = P2_LOWER | P2_UPPER
    WIN = P1_WIN | P2_WIN

    @staticmethod
    def is_win(flag):
        return HexFlag.P1_WIN in flag or HexFlag.P2_WIN in flag
