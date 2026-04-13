from numbers import Number

import numpy as np

import polars as pl

from ...db import DBHandler
from .seeker_base import Seeker


class Correlation(Seeker):
    def __init__(
        self,
        source_values: list[str],
        target_values: list[Number],
        k: int = 10,
        hash_size: int = 256,
    ) -> None:
        super().__init__(k)

        grouped = (
            pl.DataFrame({"source": source_values, "target": target_values})
            .drop_nulls()
            .group_by("source")
            .mean()
        )

        # the input source save the index values, which,
        # after grouping, become the source values
        self.input_source = grouped.get_column("source")  #  .index.values

        # Here we keep the numerical values (i.e. those
        # on which we actually compute the correlation)
        self.input_target = grouped.get_column("target")  # ["target"].values

        # BLEND makes more flexible the hash size
        self.hash_size = hash_size

        # in the SQL query we do not return QCR score, we
        # are only interested in the order of the resutls
        self.base_sql = f"""
            SELECT 
                table_id, 
                catcol, 
                numcol,
                qcr
            FROM (
                SELECT table_id, catcol, numcol, 
                        (2 * SUM((cell_value IN ($TRUETOKENS$) = quadrant)::INT) - COUNT(*)) AS score,
                        (2 * SUM((cell_value IN ($TRUETOKENS$) = quadrant)::INT) - COUNT(*)) / COUNT(*) AS qcr,
                FROM (
                    SELECT
                        categorical.cell_value,
                        categorical.table_id,
                        categorical.column_id catcol,
                        numerical.column_id numcol,
                        SUM(numerical.quadrant::INT) / COUNT(*) > 0.5 AS quadrant,
                        COUNT(DISTINCT numerical.cell_value) AS num_unique,
                        MIN(numerical.cell_value) AS any_cellvalue
                    FROM (
                        SELECT * 
                        FROM all_tables 
                        WHERE 
                            row_id < {self.hash_size}
                        AND (
                                cell_value IN ($FALSETOKENS$)
                            OR 
                                cell_value IN ($TRUETOKENS$)
                            ) $ADDITIONALS$
                        ) categorical
                    JOIN (
                        SELECT * 
                        FROM all_tables 
                        WHERE 
                            row_id < {self.hash_size}
                        AND 
                            quadrant IS NOT NULL $ADDITIONALS$
                        ) numerical
                    ON categorical.table_id = numerical.table_id 
                    AND categorical.row_id = numerical.row_id
                    GROUP BY categorical.table_id, categorical.column_id, numerical.column_id, categorical.cell_value
                ) grouped_cellvalues
                GROUP BY table_id, catcol, numcol
                HAVING 
                    COUNT(*) > 1 
                AND (
                        COUNT(DISTINCT any_cellvalue) > 1 
                    OR 
                        SUM(num_unique) > COUNT(*)
                    )
            ) scores
            WHERE catcol != numcol
            ORDER BY ABS(score) DESC
            LIMIT $TOPK$
        """

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        # create an array of 0/1 values, where 1 if the relative target
        # value is above the average of the target numbers, 0 otherwise
        # the 0/1 value, is basically the quadrant indicator
        target_average = self.input_target.mean()
        target_int = np.where(self.input_target >= target_average, 1, 0).astype(int)

        # clean the string values (remove 'nan', etc)
        self.input_source = db.clean_value_collection(self.input_source)

        # take the negative target values
        source_0 = db.create_sql_list_str(
            [key for key, qdr in zip(self.input_source, target_int) if qdr == 0]
        )

        # take the positive target values
        source_1 = db.create_sql_list_str(
            [key for key, qdr in zip(self.input_source, target_int) if qdr == 1]
        )

        sql = (
            self.base_sql.replace("$TOPK$", str(self.k))
            .replace("$ADDITIONALS$", additionals)
            .replace("$FALSETOKENS$", source_0)
            .replace("$TRUETOKENS$", source_1)
        )

        return sql

    def cost(self) -> int:
        return 6

    def ml_cost(self, db: DBHandler) -> float:
        return self._predict_runtime([[token for token in self.input_source]], db)
