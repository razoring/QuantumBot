import json
import os
from time import time
from fastapi import FastAPI, Request
import psycopg2
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
app = FastAPI() # uvicorn main:app --reload | must cd api first

load_dotenv()
try:
    connection = psycopg2.connect(dbname="QuantumBot",user=os.getenv("PG_USERNAME"),password=os.getenv("PG_PASSWORD"),host="localhost")
    if connection is not None:
        def getTicker(cursor, ticker):
            cursor.execute(f"select * from Ticker where ticker = '{str(ticker).upper()}';")
            row  = cursor.fetchall()
            return row
        
        class Ticker(BaseModel):
            ticker:str
            sector:str
            industry:str
            active:bool
            accuracy:float
            weight:list[float]
            datapoints:dict[str, any]
            updated:str

        @app.post("/tickers/")
        def createTicker(ticker:Ticker):
            cursor = connection.cursor()

            print(ticker)
            updated = int(time())

            return {"status":200}

except Exception as e:
    print(e.with_traceback)

if __name__ == "__main__":
    uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)

"""
insert into ticker(
	ticker,
	sector,
	industry,
	active,
	accuracy,
	weight,
	datapoints,
	updated
) values (
	'NVDA',
	'semiconductors',
	'technology',
	true,
	0.87235,
	'[0.17936215747100248, 0.20798101331510577, 0.45905395176356334, 0.10320645249543611, 0.050396425957627985]',
	'{"dates": ["01-01-2025","01-02-2025"], "prices": [179.23, 184.55]}',
	'1769455179'
)
"""