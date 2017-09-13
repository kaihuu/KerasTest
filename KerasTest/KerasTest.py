import DBAccessor as dbac
import csv
import datetime

rows = dbac.DBAccessor.ExecuteQuery("SELECT * FROM CORRECTED_GPS_modified")
#print(rows)

f = open('data.csv' + str(datetime.date.today()), 'ab')

csvWriter = csv.writer(f)

for row in rows:
    csvWriter.writerow(row)

f.close()