import mysql.connector as con
import time
from datetime import datetime

# expected usage of this file:
# 

# don't alter, must not be used/ accessed directly
timeFormat = '%Y-%m-%d %H:%M:%S'
cur = None
db = None
tableName = "reservoirtabletest"

# needs error handling code, ATM none is present
# needs better encapsulation
# can use better cursor methods for commands like insert
# better contracts must be used - date not str of date in mysql friendly 

# db related functions, don't use directly
def initDBcon(host_inp=IP, user_inp=USERNAME, passwd_inp=PASSWORD, db_inp="reservoirdb"):
    global db
    global cur
    db = con.connect(host=host_inp, user=user_inp, passwd=passwd_inp, db=db_inp)
    cur = db.cursor()

def closeDBcon():
    global db
    db.commit()
    db.close()

def executeCommand(command):
    global cur
    cur.execute(command)
    all_rows = cur.fetchall()
    if cur.rowcount > 0:
        list_rows = []
        for row in all_rows:
            list_rows.append(row)
        return list_rows


def queryDB(cmd):
    initDBcon()
    oup = executeCommand(cmd)
    closeDBcon()
    if oup:
        return oup

def getDateStr(inp_date):
    # date must be in mm/dd/yy format
    return datetime.strptime(inp_date, '%m/%d/%y').strftime(timeFormat)
    
# function to insert raw data, by default everything except predicted value. Speicfy list of columns for all other scenarios. Notice it is a string, use similar format
def insertRawData(list_rows, columns="FLOW_DATE, PRESENT_STORAGE_TMC, INFLOW_CUSECS, OUTFLOW_CUECS, tempC, windspeedKmph, precipMM, humidity, pressure, cloudcover, HeatIndexC, DewPointC, WindChillC, WindGustKmph, RES_LEVEL_FT, city"):
    cmd = "INSERT INTO " + tableName +" (" + columns + ") VALUES "
    for ind, val in enumerate(list_rows):
        # each row is a sting of , separated values, ready to be appended
        cmd += "(" + val + ")"
        if ind != len(list_rows) - 1:
            cmd += ", "
    print(cmd)
    queryDB(cmd)

# function to insert prediction for a list of dates, date and city must be present
def insertPredictedValues(list_rows):
    # row= ["FLOW_DATE", "city", "predictedVal"]
    initDBcon()
    for row in list_rows:
        cmd = "UPDATE " + tableName + " SET predictedVal=" + row[2] + " WHERE FLOW_DATE='" + row[0] + "' AND city=" + row[1]
        print(cmd)
        executeCommand(cmd)
    closeDBcon()

# function to read data from the table, startDate and endDate None would give unbounded ie all rows
def getData(startDate = None, endDate = None):
    cmd = None
    if startDate is None and endDate is None:
        cmd = "SELECT * FROM " + tableName
    elif startDate is None:
        cmd = "SELECT * FROM " + tableName + " WHERE FLOW_DATE <='" + endDate +"'"
    elif endDate is None:
        cmd = "SELECT * FROM " + tableName + " WHERE FLOW_DATE >='" + startDate +"'"
    else:
        cmd = "SELECT * FROM " + tableName + " WHERE FLOW_DATE >='" + startDate +"' AND FLOW_DATE <='" + endDate +"'"
    oup = queryDB(cmd)
    return oup    

# can use getDateStr for converting the date
"""
The following demonstrates the usage:
oup = getData(getDateStr('12/20/20'), getDateStr('12/21/20'))
print(len(oup))


row1 = "'" + datetime.strptime("12/20/20", '%m/%d/%y').strftime(timeFormat) + "',42.83,2416,3243,25,15,0,79,1015,42,22,17,21,20,119.83,'Agra'"
row2 = "'" + datetime.strptime("12/21/20", '%m/%d/%y').strftime(timeFormat) + "',42.83,2416,3243,25,15,0,79,1015,42,22,17,21,20,119.83,'Agra'"
insertRawData([row1, row2])


row1 = [datetime.strptime("12/20/20", '%m/%d/%y').strftime(timeFormat), "'Agra'", "600"]
row2 = [datetime.strptime("12/21/20", '%m/%d/%y').strftime(timeFormat), "'Agra'", "600"]
insertPredictedValues([row1, row2])
"""