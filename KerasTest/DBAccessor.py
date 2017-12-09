import pyodbc

class DBAccessor:
    """DB Access"""

    config = "DRIVER={SQL Server};SERVER=ECOLOGDB2016;DATABASE=ECOLOGDBver3"

    @classmethod
    def ExecuteQuery(self, query):
        cnn = pyodbc.connect(self.config)
        cur = cnn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        cnn.close()
        return rows

    def SemanticLinksInputQueryString():
        query = """
        SELECT 
        --TRIP_ID, SEMANTIC_LINK_ID,
        MIN(time), MIN(avgspeed), MIN(stdspeed), MIN(maxspeed)
        , MIN(minspeed),
        MIN(avgspeed2), MIN(stdspeed2), MIN(maxspeed2), MIN(minspeed2),
        MIN(fspeed), MIN(lspeed), MIN(hspeed), MIN(sfspeed), MIN(tfspeed),
        MIN(avgbearing),
        MIN(stdbearing), MIN(maxbearing), MIN(minbearing), 
        MIN(fbearing),
        MIN(lbearing), MIN(hbearing), MIN(sfbearing), MIN(tfbearing),
        MIN(avgacc), MIN(avgstd), MIN(maxacc), MIN(minacc),
        MIN(avgaccuracy),
        MIN(stdaccuracy), MIN(maxaccuracy), min(minaccuracy),
        min(avgtheta), MIN(stdtheta)
        , MIN(maxtheta)
        , MIN(mintheta), 
        min(avgsin), MIN(stdsin), MIN(maxsin), MIN(minsin), 
        min(avgcos), MIN(stdcos), 
        MIN(maxcos),
        MIN(mincos), 
        AVG(BAROMETRIC_VALUE), AVG(ATMOSPHERIC_PRESSURE), 
        AVG(PRECIPITATION),
        AVG(TEMPERATURE), AVG(CONVERT(float, HUMIDITY)), AVG(WIND_SPEED), AVG(POWER(WIND_SPEED,2)), AVG(DEGREE), 
        AVG(POWER(MAX_WIND_SPEED,2)), AVG(MAXDEGREE), 
        MIN(FIRSTGIDS)
        --,MIN(START_TIME)
        FROM(

        SELECT DISTINCT ALLDATA.*, FIRST_VALUE(GIDS) OVER(PARTITION BY LEAFSPY_RAW2.TRIP_ID, SEMANTIC_LINK_ID ORDER BY LEAFSPY_RAW2.DATETIME) AS FIRSTGIDS
        FROM
        (
        SELECT 
        GPS_DATA.TRIP_ID, SEMANTIC_LINK_ID,MIN(GPS_DATA.CAR_ID) as CAR_ID, MIN(GPS_DATA.DRIVER_ID) as DRIVER_ID,
        MIN(JST) as minjst, MAX(JST) as maxjst,
        COUNT(*) as count, DATEDIFF(second, MIN(JST), MAX(JST)) as time,
        AVG(GPS_DATA.SPEED) as avgspeed, STDEV(GPS_DATA.SPEED) as stdspeed, MAX(GPS_DATA.SPEED) as maxspeed, MIN(GPS_DATA.SPEED) as minspeed,
        AVG(POWER(GPS_DATA.SPEED, 2)) as avgspeed2,STDEV(POWER(GPS_DATA.SPEED, 2)) as stdspeed2, MAX(POWER(GPS_DATA.SPEED, 2)) as maxspeed2, MIN(POWER(GPS_DATA.SPEED, 2)) as minspeed2,
        AVG(FIRSTSPEED) as fspeed,AVG(LASTSPEED) as lspeed, AVG(halfspeed) as hspeed, AVG(sfspeed) as sfspeed, AVG(tfspeed) as tfspeed,
        AVG(BEARING) as avgbearing, STDEV(BEARING) as stdbearing, MAX(BEARING) as maxbearing, MIN(BEARING) as minbearing, AVG(FIRSTBEARING) as fbearing,
        AVG(LASTBEARING) as lbearing, AVG(halfbearing) as hbearing, AVG(sfbearing) as sfbearing, AVG(tfbearing) as tfbearing,
        AVG(ACC) as avgacc, STDEV(ACC) as avgstd, MAX(ACC) as maxacc, MIN(ACC) as minacc,
        AVG(CONVERT(float,ACCURACY)) as avgaccuracy, STDEV(CONVERT(float,ACCURACY)) as stdaccuracy, MAX(CONVERT(float,ACCURACY)) AS maxaccuracy, MIN(CONVERT(float,ACCURACY)) as minaccuracy,
        AVG(RoadGrade) as avgtheta, STDEV(RoadGrade) as stdtheta, MAX(RoadGrade) as maxtheta, MIN(RoadGrade) as mintheta,
        AVG(SIN(RoadGrade)) as avgsin, STDEV(SIN(RoadGrade)) as stdsin, MAX(SIN(RoadGrade)) as maxsin, MIN(SIN(RoadGrade)) as minsin,
        AVG(COS(RoadGrade)) as avgcos, STDEV(COS(RoadGrade)) as stdcos, MAX(COS(RoadGrade)) as maxcos, MIN(COS(RoadGrade)) as mincos,
        MIN(START_TIME) as START_TIME, MIN(END_TIME) as END_TIME
        FROM
            (
            --GPSデータを求めるクエリの始まり
            SELECT GPS_DATA.*,
            FIRST_VALUE(SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID ORDER BY GPS_DATA.JST) AS FIRSTSPEED,
            LAST_VALUE(SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID ORDER BY GPS_DATA.JST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS LASTSPEED,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) AS halfspeed,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as sfspeed,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as tfspeed,
            FIRST_VALUE(BEARING) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST) AS FIRSTBEARING,
            LAST_VALUE(BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID ORDER BY GPS_DATA.JST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS LASTBEARING,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) AS halfbearing,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as sfbearing,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as tfbearing, SEMANTIC_LINK_ID
            FROM(	
            SELECT GPS_DATA.DRIVER_ID, GPS_DATA.CAR_ID,GPS_DATA.SENSOR_ID, 
                GPS_DATA.JST ,GPS_DATA.LATITUDE,GPS_DATA.LONGITUDE,GPS_DATA.ALTITUDE,
                GPS_DATA.HEADING,
                ACCURACY, LINK_ID, ROAD_THETA, TRIP_ID, (LEAD(CASE WHEN SPEED IS NULL THEN MOVING_SPEED * 0.514 ELSE SPEED END, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST) -
                LAG(CASE WHEN SPEED IS NULL THEN MOVING_SPEED * 0.514 ELSE SPEED END, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST))
                / (DATEDIFF(second, LAG(GPS_DATA.JST, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST), 
                LEAD(GPS_DATA.JST, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST))) as ACC
                , CASE WHEN SPEED IS NULL THEN MOVING_SPEED * 0.514 ELSE SPEED END AS SPEED,
                    CASE  WHEN BEARING IS NULL THEN MOVING_AZIMUTH ELSE BEARING END AS BEARING,
                    GPS_DATA.START_TIME, GPS_DATA.END_TIME, GPS_DATA.TRIP_DIRECTION,
                    CASE WHEN DISTANCE_DIFFERENCE != 0 THEN ATAN((TERRAIN_ALTITUDE - LAG(TERRAIN_ALTITUDE) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST)) / DISTANCE_DIFFERENCE)
                    ELSE 0 END AS RoadGrade 
                    FROM (
                SELECT CORRECTED_GPS_modified.DRIVER_ID, CORRECTED_GPS_modified.CAR_ID,CORRECTED_GPS_modified.SENSOR_ID, 
                CORRECTED_GPS_modified.JST,CORRECTED_GPS_modified.LATITUDE,CORRECTED_GPS_modified.LONGITUDE,CORRECTED_GPS_modified.ALTITUDE,
                CORRECTED_GPS_modified.HEADING,
                ACCURACY, LINK_ID, ROAD_THETA, TRIP_ID, TRIP_DIRECTION, DISTANCE_DIFFERENCE, TERRAIN_ALTITUDE,
                SPEED, BEARING, TRIPS_modified.START_TIME, TRIPS_modified.END_TIME
                FROM CORRECTED_GPS_modified
                INNER JOIN (SELECT DISTINCT TRIPS_modified.*
                FROM TRIPS_modified, WEATHER
                WHERE WEATHER.LAST_10MIN_DATETIME >= TRIPS_modified.START_TIME AND WEATHER.DATETIME <= TRIPS_modified.END_TIME
                --AND WEATHER.TEMPERATURE > 15
                ) AS TRIPS_modified
                ON CORRECTED_GPS_modified.SENSOR_ID = TRIPS_modified.SENSOR_ID AND CORRECTED_GPS_modified.JST >= TRIPS_modified.START_TIME
                AND CORRECTED_GPS_modified.JST <= TRIPS_modified.END_TIME
                
                AND TRIP_ID NOT IN (1034,1264,1308)
                ) AS GPS_DATA
                LEFT OUTER JOIN GPRMC_RAW
                ON GPS_DATA.SENSOR_ID = GPRMC_RAW.SENSOR_ID AND GPS_DATA.JST = GPRMC_RAW.JST
                ) AS GPS_DATA
            INNER JOIN SEMANTIC_LINKS
            ON SEMANTIC_LINKS.LINK_ID = GPS_DATA.LINK_ID AND
            ((SEMANTIC_LINK_ID IN (2) AND TRIP_DIRECTION = 'homeward' 
            --AND TRIP_ID NOT IN (788,636,563,790,39,638,31,626,683,553,27,645,551,647,557, 787,561, 640)
            ))
            WHERE GPS_DATA.ACC IS NOT NULL AND GPS_DATA.SPEED IS NOT NULL AND GPS_DATA.BEARING IS NOT NULL
            --GPSデータを求めるクエリの終わり
            )  as GPS_DATA 

        GROUP BY GPS_DATA.TRIP_ID, GPS_DATA.SEMANTIC_LINK_ID
        HAVING STDEV(GPS_DATA.SPEED) IS NOT NULL
        ) AS ALLDATA
        INNER JOIN LEAFSPY_RAW2
        ON LEAFSPY_RAW2.CAR_ID = ALLDATA.CAR_ID AND LEAFSPY_RAW2.DRIVER_ID = ALLDATA.DRIVER_ID
        AND LEAFSPY_RAW2.DATETIME >= ALLDATA.minjst AND LEAFSPY_RAW2.DATETIME <= ALLDATA.maxjst

        ) AS ALLDATA
        --LEAFSPYデータがあるか確認用クエリ
        INNER JOIN (SELECT DATETIME, LAST_10MIN_DATETIME, BAROMETRIC_VALUE, ATMOSPHERIC_PRESSURE, PRECIPITATION,
        TEMPERATURE, HUMIDITY, WIND_SPEED, B.DEGREE, MAX_WIND_SPEED, A.DEGREE AS MAXDEGREE
        FROM WEATHER, DIRECTION_TO_DEGREE as A, DIRECTION_TO_DEGREE as B
        WHERE WEATHER.MAX_WIND_DIRECTION = A.DIRECTION AND WEATHER.WIND_DIRECTION = B.DIRECTION) as WEATHER
        ON WEATHER.DATETIME >= minjst AND WEATHER.LAST_10MIN_DATETIME <= maxjst
        GROUP BY TRIP_ID, SEMANTIC_LINK_ID
        ORDER BY TRIP_ID, SEMANTIC_LINK_ID
        """
        return query

    def SemanticLinksTrueQueryString():
        query = """
        SELECT 
        --gps.TRIP_ID, gps.SEMANTIC_LINK_ID,
        SUM(CONVERT(float,GIDS_DIFFERENCE) * 0.0775)
        --,DATEDIFF(second, MIN(minjst) , MIN(maxjst)) 
        --,MIN(minjst)
        FROM
        (
        SELECT LEAFSPY_TIME_INTERVAL_View.*, LEAFSPY_RAW2.GIDS
        FROM LEAFSPY_TIME_INTERVAL_View,LEAFSPY_RAW2
        WHERE LEAFSPY_TIME_INTERVAL_View.TRIP_ID = LEAFSPY_RAW2.TRIP_ID AND LEAFSPY_TIME_INTERVAL_View.START_TIME = LEAFSPY_RAW2.DATETIME
        ) AS LEAFSPY,
        (
        SELECT TRIP_ID, MIN(JST) as minjst, MAX(JST) as maxjst, SEMANTIC_LINK_ID
        FROM
        (
        SELECT A.*
        FROM(
        SELECT CORRECTED_GPS_modified.*, SEMANTIC_LINK_ID
        FROM CORRECTED_GPS_modified
        INNER JOIN SEMANTIC_LINKS
        ON CORRECTED_GPS_modified.LINK_ID = SEMANTIC_LINKS.LINK_ID 
        ) AS A
        LEFT OUTER JOIN GPRMC_RAW
        ON GPRMC_RAW.SENSOR_ID = A.SENSOR_ID AND GPRMC_RAW.JST = A.JST
        WHERE (GPRMC_RAW.MOVING_SPEED IS NOT NULL OR A.SPEED IS NOT NULL)
        ) AS GPS
        INNER JOIN 
        (SELECT DISTINCT TRIPS_modified.*
        FROM TRIPS_modified, WEATHER
        WHERE WEATHER.LAST_10MIN_DATETIME >= TRIPS_modified.START_TIME AND WEATHER.DATETIME <= TRIPS_modified.END_TIME
        --AND WEATHER.TEMPERATURE > 15
        ) AS TRIPS_modified
        ON TRIPS_modified.SENSOR_ID = GPS.SENSOR_ID AND GPS.JST >= TRIPS_modified.START_TIME AND GPS.JST <= TRIPS_modified.END_TIME 
        AND TRIP_ID NOT IN (1034,1264,1308)
        AND SEMANTIC_LINK_ID IN (2) AND TRIP_DIRECTION = 'homeward'
        GROUP BY TRIP_ID, SEMANTIC_LINK_ID
        ) AS gps
        WHERE LEAFSPY.START_TIME >= gps.minjst AND LEAFSPY.START_TIME <= gps.maxjst
        GROUP BY gps.TRIP_ID, SEMANTIC_LINK_ID
        --HAVING DATEDIFF(second, MIN(minjst) , MIN(maxjst)) < 200
        ORDER BY gps.TRIP_ID, SEMANTIC_LINK_ID 
        """

        return query

    def SemanticLinksInputQueryStringTest():

        query = """
        SELECT 
        --TRIP_ID, SEMANTIC_LINK_ID,
        MIN(time), 
        MIN(avgspeed), 
        --MIN(stdspeed), MIN(maxspeed)
        --, MIN(minspeed),
        MIN(avgspeed2),
        --MIN(stdspeed2), MIN(maxspeed2), MIN(minspeed2),
        --MIN(fspeed), MIN(lspeed), MIN(hspeed), MIN(sfspeed), MIN(tfspeed),
        --MIN(avgbearing),
        --MIN(stdbearing), MIN(maxbearing), MIN(minbearing), 
        --MIN(fbearing),
        --MIN(lbearing), MIN(hbearing), MIN(sfbearing), MIN(tfbearing),
        MIN(avgacc),
        -- MIN(avgstd), MIN(maxacc), MIN(minacc),
        MIN(avgaccuracy),
        --MIN(stdaccuracy), MIN(maxaccuracy), min(minaccuracy),
        min(avgtheta),
        -- MIN(stdtheta)
        --, MIN(maxtheta)
        --, MIN(mintheta), 
        --min(avgsin), MIN(stdsin), MIN(maxsin), MIN(minsin), 
        --min(avgcos), MIN(stdcos), 
        --MIN(maxcos),
        --MIN(mincos), 
        --AVG(BAROMETRIC_VALUE), AVG(ATMOSPHERIC_PRESSURE), 
        --AVG(PRECIPITATION),
        AVG(TEMPERATURE),
        -- AVG(CONVERT(float, HUMIDITY)), AVG(WIND_SPEED), AVG(POWER(WIND_SPEED,2)), AVG(DEGREE), 
        --AVG(POWER(MAX_WIND_SPEED,2)), AVG(MAXDEGREE), 
        MIN(FIRSTGIDS)
        --,MIN(START_TIME)
        FROM(

        SELECT DISTINCT ALLDATA.*, FIRST_VALUE(GIDS) OVER(PARTITION BY LEAFSPY_RAW2.TRIP_ID, SEMANTIC_LINK_ID ORDER BY LEAFSPY_RAW2.DATETIME) AS FIRSTGIDS
        FROM
        (
        SELECT 
        GPS_DATA.TRIP_ID, SEMANTIC_LINK_ID,MIN(GPS_DATA.CAR_ID) as CAR_ID, MIN(GPS_DATA.DRIVER_ID) as DRIVER_ID,
        MIN(JST) as minjst, MAX(JST) as maxjst,
        COUNT(*) as count, DATEDIFF(second, MIN(JST), MAX(JST)) as time,
        AVG(GPS_DATA.SPEED) as avgspeed, STDEV(GPS_DATA.SPEED) as stdspeed, MAX(GPS_DATA.SPEED) as maxspeed, MIN(GPS_DATA.SPEED) as minspeed,
        AVG(POWER(GPS_DATA.SPEED, 2)) as avgspeed2,STDEV(POWER(GPS_DATA.SPEED, 2)) as stdspeed2, MAX(POWER(GPS_DATA.SPEED, 2)) as maxspeed2, MIN(POWER(GPS_DATA.SPEED, 2)) as minspeed2,
        AVG(FIRSTSPEED) as fspeed,AVG(LASTSPEED) as lspeed, AVG(halfspeed) as hspeed, AVG(sfspeed) as sfspeed, AVG(tfspeed) as tfspeed,
        AVG(BEARING) as avgbearing, STDEV(BEARING) as stdbearing, MAX(BEARING) as maxbearing, MIN(BEARING) as minbearing, AVG(FIRSTBEARING) as fbearing,
        AVG(LASTBEARING) as lbearing, AVG(halfbearing) as hbearing, AVG(sfbearing) as sfbearing, AVG(tfbearing) as tfbearing,
        AVG(ACC) as avgacc, STDEV(ACC) as avgstd, MAX(ACC) as maxacc, MIN(ACC) as minacc,
        AVG(CONVERT(float,ACCURACY)) as avgaccuracy, STDEV(CONVERT(float,ACCURACY)) as stdaccuracy, MAX(CONVERT(float,ACCURACY)) AS maxaccuracy, MIN(CONVERT(float,ACCURACY)) as minaccuracy,
        AVG(RoadGrade) as avgtheta, STDEV(RoadGrade) as stdtheta, MAX(RoadGrade) as maxtheta, MIN(RoadGrade) as mintheta,
        AVG(SIN(RoadGrade)) as avgsin, STDEV(SIN(RoadGrade)) as stdsin, MAX(SIN(RoadGrade)) as maxsin, MIN(SIN(RoadGrade)) as minsin,
        AVG(COS(RoadGrade)) as avgcos, STDEV(COS(RoadGrade)) as stdcos, MAX(COS(RoadGrade)) as maxcos, MIN(COS(RoadGrade)) as mincos,
        MIN(START_TIME) as START_TIME, MIN(END_TIME) as END_TIME
        FROM
            (
            --GPSデータを求めるクエリの始まり
            SELECT GPS_DATA.*,
            FIRST_VALUE(SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID ORDER BY GPS_DATA.JST) AS FIRSTSPEED,
            LAST_VALUE(SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID ORDER BY GPS_DATA.JST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS LASTSPEED,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) AS halfspeed,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as sfspeed,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as tfspeed,
            FIRST_VALUE(BEARING) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST) AS FIRSTBEARING,
            LAST_VALUE(BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID ORDER BY GPS_DATA.JST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS LASTBEARING,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) AS halfbearing,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as sfbearing,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as tfbearing, SEMANTIC_LINK_ID
            FROM(	
            SELECT GPS_DATA.DRIVER_ID, GPS_DATA.CAR_ID,GPS_DATA.SENSOR_ID, 
                GPS_DATA.JST ,GPS_DATA.LATITUDE,GPS_DATA.LONGITUDE,GPS_DATA.ALTITUDE,
                GPS_DATA.HEADING,
                ACCURACY, LINK_ID, ROAD_THETA, TRIP_ID, (LEAD(CASE WHEN SPEED IS NULL THEN MOVING_SPEED * 0.514 ELSE SPEED END, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST) -
                LAG(CASE WHEN SPEED IS NULL THEN MOVING_SPEED * 0.514 ELSE SPEED END, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST))
                / (DATEDIFF(second, LAG(GPS_DATA.JST, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST), 
                LEAD(GPS_DATA.JST, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST))) as ACC
                , CASE WHEN SPEED IS NULL THEN MOVING_SPEED * 0.514 ELSE SPEED END AS SPEED,
                    CASE  WHEN BEARING IS NULL THEN MOVING_AZIMUTH ELSE BEARING END AS BEARING,
                    GPS_DATA.START_TIME, GPS_DATA.END_TIME, GPS_DATA.TRIP_DIRECTION,
                    CASE WHEN DISTANCE_DIFFERENCE != 0 THEN ATAN((TERRAIN_ALTITUDE - LAG(TERRAIN_ALTITUDE) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST)) / DISTANCE_DIFFERENCE)
                    ELSE 0 END AS RoadGrade 
                    FROM (
                SELECT CORRECTED_GPS_modified.DRIVER_ID, CORRECTED_GPS_modified.CAR_ID,CORRECTED_GPS_modified.SENSOR_ID, 
                CORRECTED_GPS_modified.JST,CORRECTED_GPS_modified.LATITUDE,CORRECTED_GPS_modified.LONGITUDE,CORRECTED_GPS_modified.ALTITUDE,
                CORRECTED_GPS_modified.HEADING,
                ACCURACY, LINK_ID, ROAD_THETA, TRIP_ID, TRIP_DIRECTION, DISTANCE_DIFFERENCE, TERRAIN_ALTITUDE,
                SPEED, BEARING, TRIPS_modified.START_TIME, TRIPS_modified.END_TIME
                FROM CORRECTED_GPS_modified
                INNER JOIN (SELECT DISTINCT TRIPS_modified.*
                FROM TRIPS_modified, WEATHER
                WHERE WEATHER.LAST_10MIN_DATETIME >= TRIPS_modified.START_TIME AND WEATHER.DATETIME <= TRIPS_modified.END_TIME
                --AND WEATHER.TEMPERATURE > 15
                ) AS TRIPS_modified
                ON CORRECTED_GPS_modified.SENSOR_ID = TRIPS_modified.SENSOR_ID AND CORRECTED_GPS_modified.JST >= TRIPS_modified.START_TIME
                AND CORRECTED_GPS_modified.JST <= TRIPS_modified.END_TIME
                
                AND TRIP_ID NOT IN (1034,1264,1308)
                ) AS GPS_DATA
                LEFT OUTER JOIN GPRMC_RAW
                ON GPS_DATA.SENSOR_ID = GPRMC_RAW.SENSOR_ID AND GPS_DATA.JST = GPRMC_RAW.JST
                ) AS GPS_DATA
            INNER JOIN SEMANTIC_LINKS
            ON SEMANTIC_LINKS.LINK_ID = GPS_DATA.LINK_ID AND
            ((SEMANTIC_LINK_ID IN (2) AND TRIP_DIRECTION = 'homeward' 
            --AND TRIP_ID NOT IN (788,636,563,790,39,638,31,626,683,553,27,645,551,647,557, 787,561, 640)
            ))
            WHERE GPS_DATA.ACC IS NOT NULL AND GPS_DATA.SPEED IS NOT NULL AND GPS_DATA.BEARING IS NOT NULL
            --GPSデータを求めるクエリの終わり
            )  as GPS_DATA 

        GROUP BY GPS_DATA.TRIP_ID, GPS_DATA.SEMANTIC_LINK_ID
        HAVING STDEV(GPS_DATA.SPEED) IS NOT NULL
        ) AS ALLDATA
        INNER JOIN LEAFSPY_RAW2
        ON LEAFSPY_RAW2.CAR_ID = ALLDATA.CAR_ID AND LEAFSPY_RAW2.DRIVER_ID = ALLDATA.DRIVER_ID
        AND LEAFSPY_RAW2.DATETIME >= ALLDATA.minjst AND LEAFSPY_RAW2.DATETIME <= ALLDATA.maxjst

        ) AS ALLDATA
        --LEAFSPYデータがあるか確認用クエリ
        INNER JOIN (SELECT DATETIME, LAST_10MIN_DATETIME, BAROMETRIC_VALUE, ATMOSPHERIC_PRESSURE, PRECIPITATION,
        TEMPERATURE, HUMIDITY, WIND_SPEED, B.DEGREE, MAX_WIND_SPEED, A.DEGREE AS MAXDEGREE
        FROM WEATHER, DIRECTION_TO_DEGREE as A, DIRECTION_TO_DEGREE as B
        WHERE WEATHER.MAX_WIND_DIRECTION = A.DIRECTION AND WEATHER.WIND_DIRECTION = B.DIRECTION) as WEATHER
        ON WEATHER.DATETIME >= minjst AND WEATHER.LAST_10MIN_DATETIME <= maxjst
        GROUP BY TRIP_ID, SEMANTIC_LINK_ID
        ORDER BY TRIP_ID, SEMANTIC_LINK_ID
        """
        return query


    def SemanticLinksInputQueryStringV2():
        query = """
        SELECT 
        --TRIP_ID, SEMANTIC_LINK_ID,
        MIN(time), MIN(avgspeed), MIN(stdspeed), MIN(maxspeed)
        , MIN(minspeed),
        MIN(avgspeed2), MIN(stdspeed2), MIN(maxspeed2), MIN(minspeed2),
        MIN(fspeed), MIN(lspeed), MIN(hspeed), MIN(sfspeed), MIN(tfspeed),
        MIN(avgbearing),
        MIN(stdbearing), MIN(maxbearing), MIN(minbearing), 
        MIN(fbearing),
        MIN(lbearing), MIN(hbearing), MIN(sfbearing), MIN(tfbearing),
        MIN(avgacc), MIN(stdacc), MIN(maxacc), MIN(minacc),
        MIN(avgpacc), MIN(stdpacc), MIN(maxpacc), 
        --MIN(minpacc),
        MIN(avgmacc), MIN(stdmacc), 
        --MIN(maxmacc),
        MIN(minmacc),
        MIN(avgjerk), MIN(stdjerk), MIN(maxjerk), MIN(minjerk),
        MIN(avgaccuracy),
        MIN(stdaccuracy), MIN(maxaccuracy), min(minaccuracy),
        min(avgtheta), MIN(stdtheta)
        , MIN(maxtheta)
        , MIN(mintheta), 
        min(avgsin), MIN(stdsin), MIN(maxsin), MIN(minsin), 
        min(avgcos), MIN(stdcos), 
        --MIN(maxcos),
        MIN(mincos), 
        AVG(BAROMETRIC_VALUE), AVG(ATMOSPHERIC_PRESSURE), 
        --AVG(PRECIPITATION),
        AVG(TEMPERATURE), AVG(CONVERT(float, HUMIDITY)), AVG(WIND_SPEED), AVG(POWER(WIND_SPEED,2)), AVG(DEGREE), 
        AVG(POWER(MAX_WIND_SPEED,2)), AVG(MAXDEGREE), 
        MIN(FIRSTGIDS)
        --,MIN(START_TIME)
        FROM(

        SELECT DISTINCT ALLDATA.*, FIRST_VALUE(GIDS) OVER(PARTITION BY LEAFSPY_RAW2.TRIP_ID, SEMANTIC_LINK_ID ORDER BY LEAFSPY_RAW2.DATETIME) AS FIRSTGIDS
        FROM
        (
        SELECT 
        GPS_DATA.TRIP_ID, SEMANTIC_LINK_ID,MIN(GPS_DATA.CAR_ID) as CAR_ID, MIN(GPS_DATA.DRIVER_ID) as DRIVER_ID,
        MIN(JST) as minjst, MAX(JST) as maxjst,
        COUNT(*) as count, DATEDIFF(second, MIN(JST), MAX(JST)) as time,
        AVG(GPS_DATA.SPEED) as avgspeed, STDEV(GPS_DATA.SPEED) as stdspeed, MAX(GPS_DATA.SPEED) as maxspeed, MIN(GPS_DATA.SPEED) as minspeed,
        AVG(POWER(GPS_DATA.SPEED, 2)) as avgspeed2,STDEV(POWER(GPS_DATA.SPEED, 2)) as stdspeed2, MAX(POWER(GPS_DATA.SPEED, 2)) as maxspeed2, MIN(POWER(GPS_DATA.SPEED, 2)) as minspeed2,
        AVG(FIRSTSPEED) as fspeed,AVG(LASTSPEED) as lspeed, AVG(halfspeed) as hspeed, AVG(sfspeed) as sfspeed, AVG(tfspeed) as tfspeed,
        AVG(BEARING) as avgbearing, STDEV(BEARING) as stdbearing, MAX(BEARING) as maxbearing, MIN(BEARING) as minbearing, AVG(FIRSTBEARING) as fbearing,
        AVG(LASTBEARING) as lbearing, AVG(halfbearing) as hbearing, AVG(sfbearing) as sfbearing, AVG(tfbearing) as tfbearing,
        AVG(ACC) as avgacc, STDEV(ACC) as stdacc, MAX(ACC) as maxacc, MIN(ACC) as minacc,
        AVG(PLUSACC) as avgpacc, STDEV(PLUSACC) as stdpacc, MAX(PLUSACC) as maxpacc, MIN(PLUSACC) as minpacc,
        AVG(MINUSACC) as avgmacc, STDEV(MINUSACC) as stdmacc, MAX(MINUSACC) as maxmacc, MIN(MINUSACC) as minmacc,
        AVG(JERK) as avgjerk, STDEV(JERK) as stdjerk, MAX(JERK) as maxjerk, MIN(JERK) as minjerk,
        AVG(CONVERT(float,ACCURACY)) as avgaccuracy, STDEV(CONVERT(float,ACCURACY)) as stdaccuracy, MAX(CONVERT(float,ACCURACY)) AS maxaccuracy, MIN(CONVERT(float,ACCURACY)) as minaccuracy,
        AVG(RoadGrade) as avgtheta, STDEV(RoadGrade) as stdtheta, MAX(RoadGrade) as maxtheta, MIN(RoadGrade) as mintheta,
        AVG(SIN(RoadGrade)) as avgsin, STDEV(SIN(RoadGrade)) as stdsin, MAX(SIN(RoadGrade)) as maxsin, MIN(SIN(RoadGrade)) as minsin,
        AVG(COS(RoadGrade)) as avgcos, STDEV(COS(RoadGrade)) as stdcos, MAX(COS(RoadGrade)) as maxcos, MIN(COS(RoadGrade)) as mincos,
        MIN(START_TIME) as START_TIME, MIN(END_TIME) as END_TIME
        FROM
            (
            --GPSデータを求めるクエリの始まり
            SELECT GPS_DATA.*,CASE WHEN ACC > 0 THEN ACC ELSE 0 END AS PLUSACC,CASE WHEN ACC < 0 THEN ACC ELSE 0 END AS MINUSACC,
            (LEAD(ACC, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST) -
                LAG(ACC, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST))
                / (DATEDIFF(second, LAG(GPS_DATA.JST, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST), 
                LEAD(GPS_DATA.JST, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST))) as JERK,
            FIRST_VALUE(SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID ORDER BY GPS_DATA.JST) AS FIRSTSPEED,
            LAST_VALUE(SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID ORDER BY GPS_DATA.JST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS LASTSPEED,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) AS halfspeed,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as sfspeed,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY SPEED) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as tfspeed,
            FIRST_VALUE(BEARING) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST) AS FIRSTBEARING,
            LAST_VALUE(BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID ORDER BY GPS_DATA.JST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS LASTBEARING,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) AS halfbearing,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as sfbearing,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY BEARING) OVER(PARTITION BY TRIP_ID, SEMANTIC_LINK_ID) as tfbearing, SEMANTIC_LINK_ID
            FROM(	
            
            SELECT GPS_DATA.DRIVER_ID, GPS_DATA.CAR_ID,GPS_DATA.SENSOR_ID, 
                GPS_DATA.JST ,GPS_DATA.LATITUDE,GPS_DATA.LONGITUDE,GPS_DATA.ALTITUDE,
                GPS_DATA.HEADING,
                ACCURACY, LINK_ID, ROAD_THETA, TRIP_ID, (LEAD(CASE WHEN SPEED IS NULL THEN MOVING_SPEED * 0.514 ELSE SPEED END, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST) -
                LAG(CASE WHEN SPEED IS NULL THEN MOVING_SPEED * 0.514 ELSE SPEED END, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST))
                / (DATEDIFF(second, LAG(GPS_DATA.JST, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST), 
                LEAD(GPS_DATA.JST, 1) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST))) as ACC
                , CASE WHEN SPEED IS NULL THEN MOVING_SPEED * 0.514 ELSE SPEED END AS SPEED,
                    CASE  WHEN BEARING IS NULL THEN MOVING_AZIMUTH ELSE BEARING END AS BEARING,
                    GPS_DATA.START_TIME, GPS_DATA.END_TIME, GPS_DATA.TRIP_DIRECTION,
                    CASE WHEN DISTANCE_DIFFERENCE != 0 THEN ATAN((TERRAIN_ALTITUDE - LAG(TERRAIN_ALTITUDE) OVER(PARTITION BY TRIP_ID ORDER BY GPS_DATA.JST)) / DISTANCE_DIFFERENCE)
                    ELSE 0 END AS RoadGrade 
                    FROM (
                SELECT CORRECTED_GPS_modified.DRIVER_ID, CORRECTED_GPS_modified.CAR_ID,CORRECTED_GPS_modified.SENSOR_ID, 
                CORRECTED_GPS_modified.JST,CORRECTED_GPS_modified.LATITUDE,CORRECTED_GPS_modified.LONGITUDE,CORRECTED_GPS_modified.ALTITUDE,
                CORRECTED_GPS_modified.HEADING,
                ACCURACY, LINK_ID, ROAD_THETA, TRIP_ID, TRIP_DIRECTION, DISTANCE_DIFFERENCE, TERRAIN_ALTITUDE,
                SPEED, BEARING, TRIPS_modified.START_TIME, TRIPS_modified.END_TIME
                FROM CORRECTED_GPS_modified
                INNER JOIN (SELECT DISTINCT TRIPS_modified.*
                FROM TRIPS_modified, WEATHER
                WHERE WEATHER.LAST_10MIN_DATETIME >= TRIPS_modified.START_TIME AND WEATHER.DATETIME <= TRIPS_modified.END_TIME
                --AND WEATHER.TEMPERATURE > 15
                ) AS TRIPS_modified
                ON CORRECTED_GPS_modified.SENSOR_ID = TRIPS_modified.SENSOR_ID AND CORRECTED_GPS_modified.JST >= TRIPS_modified.START_TIME
                AND CORRECTED_GPS_modified.JST <= TRIPS_modified.END_TIME
                
                AND TRIP_ID NOT IN (1034,1264,1308)
                ) AS GPS_DATA
                LEFT OUTER JOIN GPRMC_RAW
                ON GPS_DATA.SENSOR_ID = GPRMC_RAW.SENSOR_ID AND GPS_DATA.JST = GPRMC_RAW.JST
                
                ) AS GPS_DATA
            INNER JOIN SEMANTIC_LINKS
            ON SEMANTIC_LINKS.LINK_ID = GPS_DATA.LINK_ID AND
            ((SEMANTIC_LINK_ID IN (2) AND TRIP_DIRECTION = 'homeward' 
            --AND TRIP_ID NOT IN (788,636,563,790,39,638,31,626,683,553,27,645,551,647,557, 787,561, 640)
            ))
            WHERE GPS_DATA.ACC IS NOT NULL AND GPS_DATA.SPEED IS NOT NULL AND GPS_DATA.BEARING IS NOT NULL
            --GPSデータを求めるクエリの終わり
            )  as GPS_DATA 

        GROUP BY GPS_DATA.TRIP_ID, GPS_DATA.SEMANTIC_LINK_ID
        HAVING STDEV(GPS_DATA.SPEED) IS NOT NULL
        ) AS ALLDATA
        INNER JOIN LEAFSPY_RAW2
        ON LEAFSPY_RAW2.CAR_ID = ALLDATA.CAR_ID AND LEAFSPY_RAW2.DRIVER_ID = ALLDATA.DRIVER_ID
        AND LEAFSPY_RAW2.DATETIME >= ALLDATA.minjst AND LEAFSPY_RAW2.DATETIME <= ALLDATA.maxjst

        ) AS ALLDATA
        --LEAFSPYデータがあるか確認用クエリ
        INNER JOIN (SELECT DATETIME, LAST_10MIN_DATETIME, BAROMETRIC_VALUE, ATMOSPHERIC_PRESSURE, PRECIPITATION,
        TEMPERATURE, HUMIDITY, WIND_SPEED, B.DEGREE, MAX_WIND_SPEED, A.DEGREE AS MAXDEGREE
        FROM WEATHER, DIRECTION_TO_DEGREE as A, DIRECTION_TO_DEGREE as B
        WHERE A.DIRECTION = CONVERT(nvarchar(10),WEATHER.MAX_WIND_DIRECTION) AND WEATHER.WIND_DIRECTION = B.DIRECTION) as WEATHER
        ON WEATHER.DATETIME >= minjst AND WEATHER.LAST_10MIN_DATETIME <= maxjst
        GROUP BY TRIP_ID, SEMANTIC_LINK_ID
        ORDER BY TRIP_ID, SEMANTIC_LINK_ID
        """
        return query        