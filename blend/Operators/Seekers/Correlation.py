from numbers import Number
from typing import List, Literal

import numpy as np
import pandas as pd

# Typing imports
from blend.DBHandler import DBHandler
from blend.Operators.Seekers.SeekerBase import Seeker


class Correlation(Seeker):
    def __init__(
        self,
        source_values: List[str],
        target_values: List[Number],
        k: int = 10,
        granularity: str = "base",
        hash_size: Literal[256, 512, 1024] = 256,
        verbosity: int = 1,
    ) -> None:
        super().__init__(k, granularity, verbosity)

        grouped = (
            pd.DataFrame({"source": source_values, "target": target_values})
            .dropna()
            .groupby("source")
            .mean()
        )
        self.input_source = grouped.index.values
        self.input_target = grouped["target"].values
        self.hash_size = hash_size

        # the top level SELECT FROM is really necessary?
        self.base_sql = f"""
            SELECT $VERBOSITY$
            FROM (
                SELECT TableId, catcol, numcol, 
                        2 * SUM(((CellValue IN ($FALSETOKENS$) AND Quadrant = 0) OR (CellValue IN ($TRUETOKENS$) AND Quadrant = 1))::INT) - COUNT(*) AS score,
                        (2 * SUM(((CellValue IN ($FALSETOKENS$) AND Quadrant = 0) OR (CellValue IN ($TRUETOKENS$) AND Quadrant = 1))::INT) - COUNT(*)) / COUNT(*) AS score_float
                        -- 2 * SUM((CellValue IN ($TRUETOKENS$) = Quadrant)::INT) - COUNT(*) AS score,
                        -- (2 * SUM((CellValue IN ($TRUETOKENS$) = Quadrant)::INT) - COUNT(*)) / COUNT(*) AS score_float
                FROM (
                    SELECT
                        categorical.CellValue,
                        categorical.TableId,
                        categorical.ColumnId catcol,
                        numerical.ColumnId numcol,
                        -- this operation maybe is done because the after-join calculation
                        -- may lead to a different mean and thus to a different quadrant (?)
                        SUM(numerical.Quadrant::INT) / COUNT(*) > 0.5 AS Quadrant,
                        COUNT(DISTINCT numerical.CellValue) AS num_unique,
                        MIN(numerical.CellValue) AS any_cellvalue
                    FROM (
                        SELECT * 
                        FROM AllTables 
                        WHERE RowId < {self.hash_size} 
                        AND Granularity = '$GRANULARITY$'
                        AND (CellValue IN ($FALSETOKENS$)
                        OR CellValue IN ($TRUETOKENS$)) $ADDITIONALS$
                        ) categorical
                    JOIN (
                        SELECT * 
                        FROM AllTables 
                        WHERE RowId < {self.hash_size}
                        -- Here a filter on the granularity is 
                        -- not needed since it is used only 
                        -- on the key values
                        AND Quadrant IS NOT NULL $ADDITIONALS$
                        ) numerical
                    ON categorical.TableId = numerical.TableId 
                    AND categorical.RowId = numerical.RowId
                    GROUP BY categorical.TableId, categorical.ColumnId, numerical.ColumnId, categorical.CellValue
                ) grouped_cellvalues
                GROUP BY TableId, catcol, numcol
                HAVING COUNT(*) > 1 AND (COUNT(DISTINCT any_cellvalue) > 1 OR SUM(num_unique) > COUNT(*))
            ) scores
            ORDER BY ABS(score) DESC
            LIMIT $TOPK$
        """

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        self.input_target = self.input_target.astype(float)
        target_average = np.mean(self.input_target)
        target_int = np.where(self.input_target >= target_average, 1, 0)
        target_int = target_int.astype(int)
        self.input_source = db.clean_value_collection(self.input_source)
        source_0 = db.create_sql_list_str(
            [key for key, qdr in zip(self.input_source, target_int) if qdr == 0]
        )
        source_1 = db.create_sql_list_str(
            [key for key, qdr in zip(self.input_source, target_int) if qdr == 1]
        )

        v = ["TableId", "catcol", "numcol", "score", "score_float"]
        match self.verbosity:
            case 1:
                v = v[:1]
            case 2:
                v = v[:3]
            case 3:
                v = v[:4]
            case 4:
                v = v[:]
        v = ", ".join(v)

        sql = self.base_sql.replace("$VERBOSITY$", v)
        sql = sql.replace("$TOPK$", str(self.k))
        sql = sql.replace("$ADDITIONALS$", additionals)
        sql = sql.replace("$FALSETOKENS$", source_0)
        sql = sql.replace("$TRUETOKENS$", source_1)
        sql = sql.replace(
            "$GRANULARITY$",
            self.granularity,
        )

        return sql

    def cost(self) -> int:
        return 6

    def ml_cost(self, db: DBHandler) -> float:
        return self._predict_runtime([[token for token in self.input_source]], db)
