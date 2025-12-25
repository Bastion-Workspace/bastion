"""
Formula Evaluator - Parse and evaluate Excel-style cell formulas
Supports arithmetic operations, cell references, and basic functions
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class FormulaEvaluator:
    """Evaluate Excel-style formulas with cell references and functions"""
    
    def __init__(self):
        self.error_codes = {
            'REF': '#REF!',
            'VALUE': '#VALUE!',
            'DIV0': '#DIV/0!',
            'NAME': '#NAME?',
            'NA': '#N/A'
        }
    
    def is_formula(self, value: Any) -> bool:
        """Check if value is a formula (starts with =)"""
        return isinstance(value, str) and value.strip().startswith('=')
    
    def parse_formula(self, formula: str) -> Dict[str, Any]:
        """
        Parse formula string into components
        Returns dict with type, expression, function_name, args, etc.
        """
        if not self.is_formula(formula):
            return {'type': 'value', 'formula': formula}
        
        # Remove leading =
        expression = formula.strip()[1:].strip()
        
        # Check for function call (e.g., SUM(A1:A10))
        func_match = re.match(r'^([A-Z]+)\s*\((.*)\)$', expression, re.IGNORECASE)
        if func_match:
            func_name = func_match.group(1).upper()
            args_str = func_match.group(2)
            return {
                'type': 'function',
                'function_name': func_name,
                'args': args_str,
                'formula': formula
            }
        
        # Arithmetic expression (e.g., A1+B1, A1*B1)
        return {
            'type': 'expression',
            'expression': expression,
            'formula': formula
        }
    
    def resolve_reference(self, ref: str, current_row_index: int, schema: Dict[str, Any]) -> Optional[str]:
        """
        Convert Excel-style cell reference (A1, B2) to column name
        A = first column, B = second column, etc.
        current_row_index is 0-based
        """
        try:
            # Parse reference (e.g., "A1" -> column A, row 1)
            match = re.match(r'^([A-Z]+)(\d+)$', ref.upper())
            if not match:
                return None
            
            col_letter = match.group(1)
            ref_row = int(match.group(2)) - 1  # Convert to 0-based
            
            # Convert column letter to index (A=0, B=1, C=2, ...)
            col_index = 0
            for char in col_letter:
                col_index = col_index * 26 + (ord(char) - ord('A') + 1)
            col_index -= 1  # A should be 0
            
            # Get column name from schema
            columns = schema.get('columns', [])
            if col_index < 0 or col_index >= len(columns):
                return None
            
            return columns[col_index]['name']
        
        except Exception as e:
            logger.warning(f"Failed to resolve reference {ref}: {e}")
            return None
    
    def resolve_range(self, range_str: str, current_row_index: int, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve range reference (e.g., A1:A10) to column name and row indices
        Returns dict with column_name, start_row, end_row
        """
        try:
            # Parse range (e.g., "A1:A10")
            match = re.match(r'^([A-Z]+\d+):([A-Z]+\d+)$', range_str.upper())
            if not match:
                return None
            
            start_ref = match.group(1)
            end_ref = match.group(2)
            
            # Resolve start and end references
            start_col_match = re.match(r'^([A-Z]+)(\d+)$', start_ref)
            end_col_match = re.match(r'^([A-Z]+)(\d+)$', end_ref)
            
            if not start_col_match or not end_col_match:
                return None
            
            # Check columns match
            if start_col_match.group(1) != end_col_match.group(1):
                return None  # Range must be in same column
            
            col_letter = start_col_match.group(1)
            start_row = int(start_col_match.group(2)) - 1
            end_row = int(end_col_match.group(2)) - 1
            
            # Convert column letter to index
            col_index = 0
            for char in col_letter:
                col_index = col_index * 26 + (ord(char) - ord('A') + 1)
            col_index -= 1
            
            # Get column name from schema
            columns = schema.get('columns', [])
            if col_index < 0 or col_index >= len(columns):
                return None
            
            return {
                'column_name': columns[col_index]['name'],
                'start_row': start_row,
                'end_row': end_row
            }
        
        except Exception as e:
            logger.warning(f"Failed to resolve range {range_str}: {e}")
            return None
    
    def get_cell_value(self, row_data: Dict[str, Any], column_name: str, all_rows: List[Dict[str, Any]], row_index: int) -> Any:
        """Get cell value, handling formulas recursively"""
        value = row_data.get(column_name)
        
        # If value is a number or non-formula string, return it
        if not isinstance(value, str) or not value.strip().startswith('='):
            return value
        
        # If it's a formula, we'd need to evaluate it, but for now return None
        # This prevents infinite recursion - formulas should be evaluated separately
        return None
    
    def evaluate_function(self, func_name: str, args_str: str, row_data: Dict[str, Any], 
                         all_rows: List[Dict[str, Any]], current_row_index: int, 
                         schema: Dict[str, Any]) -> Union[float, str]:
        """Evaluate function call (SUM, AVERAGE, COUNT)"""
        try:
            # Parse arguments
            if ':' in args_str:
                # Range reference (e.g., A1:A10)
                range_info = self.resolve_range(args_str.strip(), current_row_index, schema)
                if not range_info:
                    return self.error_codes['REF']
                
                column_name = range_info['column_name']
                start_row = range_info['start_row']
                end_row = range_info['end_row']
                
                # Collect values from range
                values = []
                for row in all_rows:
                    row_idx = row.get('row_index', 0)
                    if start_row <= row_idx <= end_row:
                        cell_value = row.get('row_data', {}).get(column_name)
                        if cell_value is not None and cell_value != '':
                            try:
                                num_value = float(cell_value)
                                values.append(num_value)
                            except (ValueError, TypeError):
                                pass  # Skip non-numeric values
                
                # Apply function
                if func_name == 'SUM':
                    return sum(values) if values else 0
                elif func_name == 'AVERAGE':
                    return sum(values) / len(values) if values else self.error_codes['DIV0']
                elif func_name == 'COUNT':
                    return len(values)
                else:
                    return self.error_codes['NAME']
            
            else:
                # Single cell reference or expression
                # For now, treat as single cell reference
                ref = args_str.strip()
                column_name = self.resolve_reference(ref, current_row_index, schema)
                if not column_name:
                    return self.error_codes['REF']
                
                value = row_data.get(column_name)
                try:
                    return float(value) if value is not None else 0
                except (ValueError, TypeError):
                    return self.error_codes['VALUE']
        
        except Exception as e:
            logger.error(f"Error evaluating function {func_name}: {e}")
            return self.error_codes['VALUE']
    
    def evaluate_expression(self, expression: str, row_data: Dict[str, Any], 
                           all_rows: List[Dict[str, Any]], current_row_index: int, 
                           schema: Dict[str, Any]) -> Union[float, str]:
        """Evaluate arithmetic expression with cell references"""
        try:
            # Replace cell references with values
            # Find all cell references (e.g., A1, B2)
            ref_pattern = r'([A-Z]+\d+)'
            refs = re.findall(ref_pattern, expression.upper())
            
            # Replace each reference with its value
            for ref in refs:
                column_name = self.resolve_reference(ref, current_row_index, schema)
                if not column_name:
                    expression = expression.replace(ref, self.error_codes['REF'])
                    continue
                
                # Get value from current row or referenced row
                # For relative references, calculate row offset
                ref_match = re.match(r'^([A-Z]+)(\d+)$', ref.upper())
                if ref_match:
                    ref_row = int(ref_match.group(2)) - 1
                    row_offset = ref_row - current_row_index
                    
                    # Get value from appropriate row
                    if row_offset == 0:
                        # Same row
                        value = row_data.get(column_name)
                    else:
                        # Different row - find it in all_rows
                        target_row_index = current_row_index + row_offset
                        value = None
                        for row in all_rows:
                            if row.get('row_index') == target_row_index:
                                value = row.get('row_data', {}).get(column_name)
                                break
                    
                    # Convert value to number or error
                    if value is None or value == '':
                        expression = expression.replace(ref, '0')
                    else:
                        try:
                            num_value = float(value)
                            expression = expression.replace(ref, str(num_value))
                        except (ValueError, TypeError):
                            expression = expression.replace(ref, self.error_codes['VALUE'])
            
            # Evaluate the expression safely using ast.literal_eval for numbers
            # For arithmetic, we'll parse and evaluate manually
            try:
                # Replace operators with Python equivalents and evaluate
                # This is safer than eval() but still handles basic arithmetic
                import ast
                import operator
                
                # Create a safe evaluator for arithmetic expressions
                ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,
                    ast.USub: operator.neg,
                }
                
                def safe_eval(node):
                    if isinstance(node, ast.Num):
                        return node.n
                    elif isinstance(node, ast.Constant):
                        return node.value
                    elif isinstance(node, ast.BinOp):
                        return ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
                    elif isinstance(node, ast.UnaryOp):
                        return ops[type(node.op)](safe_eval(node.operand))
                    else:
                        raise ValueError(f"Unsupported AST node: {type(node)}")
                
                tree = ast.parse(expression, mode='eval')
                result = safe_eval(tree.body)
                return float(result) if isinstance(result, (int, float)) else result
            except ZeroDivisionError:
                return self.error_codes['DIV0']
            except Exception as e:
                logger.warning(f"Expression evaluation failed: {e}")
                return self.error_codes['VALUE']
        
        except Exception as e:
            logger.error(f"Error evaluating expression: {e}")
            return self.error_codes['VALUE']
    
    def evaluate_formula(self, formula: str, row_data: Dict[str, Any], 
                        all_rows: List[Dict[str, Any]], current_row_index: int, 
                        schema: Dict[str, Any]) -> Union[float, str, None]:
        """
        Main evaluation function
        Returns computed value or error code string
        """
        if not self.is_formula(formula):
            return None
        
        try:
            parsed = self.parse_formula(formula)
            
            if parsed['type'] == 'function':
                return self.evaluate_function(
                    parsed['function_name'],
                    parsed['args'],
                    row_data,
                    all_rows,
                    current_row_index,
                    schema
                )
            
            elif parsed['type'] == 'expression':
                return self.evaluate_expression(
                    parsed['expression'],
                    row_data,
                    all_rows,
                    current_row_index,
                    schema
                )
            
            else:
                return self.error_codes['VALUE']
        
        except Exception as e:
            logger.error(f"Formula evaluation error: {e}")
            return self.error_codes['VALUE']

