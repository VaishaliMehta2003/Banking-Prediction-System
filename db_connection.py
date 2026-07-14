import pyodbc
import pandas as pd

SERVER = r"DESKTOP-L14VCCS\SQLEXPRESS"
DATABASE = "Banking_Analysis"

conn = pyodbc.connect(
    f"""
    DRIVER={{ODBC Driver 17 for SQL Server}};
    SERVER={SERVER};
    DATABASE={DATABASE};
    Trusted_Connection=yes;
    """
)

def load_table(table):
    return pd.read_sql(f"SELECT * FROM {table}", conn)