import os
from fastapi import FastAPI
import psycopg2
import uvicorn
from dotenv import load_dotenv
app = FastAPI() # uvicorn main:app --reload | must cd api first

load_dotenv()
try:
    connection = psycopg2.connect(dbname="QuantumBot",user=os.getenv("PG_USERNAME"),password=os.getenv("PG_PASSWORD"),host="localhost")
    if connection is not None:
        with connection.cursor() as cursor:
            @app.get("/get/tickers/{ticker}")
            def locateTicker(ticker):
                result = cursor.execute(f"select id from Ticker where ticker = '{str(ticker).upper()}';")
                print(str(ticker).upper(), result)
                return {result}
except Exception as e:
    print(e.with_traceback())

if __name__ == "__main__":
    uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)