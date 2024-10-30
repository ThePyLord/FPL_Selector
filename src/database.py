import sqlite3
import yaml
from sqlite3 import Error

config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

def create_connection(db_file: str = config['db_file']):
	""" create a database connection to a SQLite database """
	conn = None
	try:
		conn = sqlite3.connect(db_file)
		print(sqlite3.version)
	except Error as e:
		print(e)
	finally:
		if conn:
			conn.close()


def create_table(db_file: str = config['db_file'], sql_statement: str = None):
	""" create a table from the create_table_sql statement """
	conn = sqlite3.connect(db_file)
	c = conn.cursor()
	c.execute(sql_statement)
	conn.commit()
	conn.close()