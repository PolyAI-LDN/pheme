import unittest

from utils.symbol_table import SymbolTable


class SymbolTableEqualityTests(unittest.TestCase):
    def test_different_symbols_compare_false(self):
        a = SymbolTable.from_str("<eps> 0\na 1")
        b = SymbolTable.from_str("<eps> 0\nb 1")
        self.assertFalse(a == b)
        self.assertFalse(b == a)

    def test_equal_tables_compare_true(self):
        a = SymbolTable.from_str("<eps> 0\na 1\nb 2")
        b = SymbolTable.from_str("b 2\n<eps> 0\na 1")
        self.assertEqual(a, b)

    def test_different_ids_compare_false(self):
        a = SymbolTable.from_str("<eps> 0\na 1")
        b = SymbolTable.from_str("<eps> 0\na 2")
        self.assertNotEqual(a, b)

    def test_unrelated_object_compares_false(self):
        self.assertFalse(SymbolTable() == object())
