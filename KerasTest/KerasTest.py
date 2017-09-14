import DBAccessor as dbac
import csv
import datetime
import numpy as np

rows = dbac.DBAccessor.ExecuteQuery("""SELECT * FROM CORRECTED_GPS_modified

""")

array = np.array(rows)

print(array)
