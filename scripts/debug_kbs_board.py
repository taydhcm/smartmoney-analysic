"""Debug: check KBS price_board columns and foreign_ownership_ratio value."""
import sys
sys.path.insert(0, ".")
import pandas as pd
from vnstock.api.trading import Trading

board = Trading(source="KBS").price_board(symbols_list=["ACB"])
print("=== Columns ===")
print(list(board.columns))
print()
print("=== ACB row ===")
row = board.iloc[0]
for col, val in row.items():
    print(f"  {col}: {val!r}")
