import unittest
import nbformat
import os
import numpy as np
import pandas as pd

class TestAssignmentNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load and execute the notebook before running tests"""
        notebook_path = "assignment.ipynb"
        assert os.path.exists(notebook_path), f"Notebook {notebook_path} not found"
        
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        
        cls.global_env = {}
        
        for cell in nb.cells:
            if cell.cell_type == "code":
                exec(cell.source, cls.global_env)
    
    def test_numpy_array_creation(self):
        """Test if a NumPy array is created correctly"""
        self.assertIn("my_array", self.global_env, "my_array not found in notebook")
        my_array = self.global_env["my_array"]
        self.assertIsInstance(my_array, np.ndarray, "my_array is not a NumPy array")
    
    def test_dataframe_creation(self):
        """Test if a Pandas DataFrame is created correctly"""
        self.assertIn("df", self.global_env, "df not found in notebook")
        df = self.global_env["df"]
        self.assertIsInstance(df, pd.DataFrame, "df is not a Pandas DataFrame")
    
    def test_dataframe_columns(self):
        """Test if DataFrame contains required columns"""
        df = self.global_env.get("df", None)
        required_columns = {"Name", "Age", "Salary"}
        self.assertIsNotNone(df, "df not found in notebook")
        self.assertTrue(required_columns.issubset(df.columns), "Missing required columns in df")

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
