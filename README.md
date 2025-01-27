# TATA

TATA contains two parts: self-supervision and data enrichment. Follow step 1 for self-supervision. Step 2 and 3 are preparation for enrichment. Step 4 uses enrichment in transfer learning.

1. run pretrain script
2. run fit pseudo labeler to fit the formula for each operator
3. create a psycopg connector to call `explain`
4. run the transfer script