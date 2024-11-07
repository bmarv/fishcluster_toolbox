import pandas as pd
import subprocess
import config
from mysql.connector import connect, Error
from typing import List, Dict


class ClusterOccupancyDBInterface:

    def __init__(
        self,
        database_name: str,
        cluster_occupancy_dir: str,
        password: str,
        mysql_path: str,
        db_created: bool = True,
    ):
        self.cluster_occupancy_dir = cluster_occupancy_dir
        self.password = password
        self.database_name = database_name
        self.mysql_path = mysql_path

        if not db_created:
            print("=======\nCreating database...")
            self.create_and_load_db()
        try:
            self.conn = connect(
                user="root",
                password=self.password,
                host="localhost",
                database=self.database_name,
            )
            print("=======\nConnected to database.")
        except Error as err:
            print("Error message: " + err.msg)

    def create_and_load_db(self, verbose: bool = True) -> None:
        result = subprocess.run(
            [
                "utils/db_management/create_load_cluster_occupancy_db.command",
                self.database_name,
                self.cluster_occupancy_dir,
                self.password,
                self.mysql_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if verbose:
            print(result.stdout)
            print(result.stderr)

    def select_from_db(self,
                       columns: List[str] = None,
                       matches: Dict[str, List[str]] = None,
                       filters: List[str] = None,
                       sample: int = None,
                       group_by: List[str] = None,
                       query: str = None
    ) -> pd.DataFrame:
        try:
            cursor = self.conn.cursor()
            if query is None:
                query = f"SELECT {', '.join(columns)} FROM cluster_occupancies"
                first = True

                if matches is not None:
                    for column, values in matches.items():
                        to_match = "', '".join(values)
                        if first:
                            query += f" WHERE {column} IN ('{to_match}')"
                            first = False
                        else:
                            query += f" AND {column} IN ('{to_match}')"

                if filters is not None:
                    for filter in filters:
                        if first:
                            query += f" WHERE {filter}"
                            first = False
                        else:
                            query += f" AND {filter}"

                if sample is not None:
                    query += f" ORDER BY RAND( ) LIMIT {sample}"

                if group_by is not None:
                    query += f" GROUP BY {', '.join(group_by)}"

            print("=======\nExecuting query: \n" + query)
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return pd.DataFrame(result, columns=columns)

        except Error as err:
            print("Error message: " + err.msg)

    def close_connection(self) -> None:
        try:
            self.conn.close()
            print("=======\nConnection to database closed.")
        except Error as err:
            print("Error message: " + err.msg)


if __name__ == "__main__":
    cluster_occupancy_db = ClusterOccupancyDBInterface(
        config.DATABASE_NAME,
        config.PROJ_PATH + "/cluster_occupancy/",
        config.PASSWORD,
        config.MYSQL_PATH,
        db_created=False,
    )
    # Example Usage
    # columns = ["row_id", "individual", "day", "positions_x", "treatment"]
    # sample_df = cluster_occupancy_db.select_from_db(
    #     columns=columns, conditions={"individual": ["block3_23520289_front"]}
    # )
    cluster_occupancy_db.close_connection()

    # print(sample_df)
