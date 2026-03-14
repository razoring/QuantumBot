import sys
import os
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Add the project root and bot directory to sys.path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("bot"))
load_dotenv(os.path.abspath(".env"))

import functions

def simulate_user(user_id, ticker):
    print(f"User {user_id} starting prediction for {ticker}...")
    start_time = time.time()
    try:
        charts = functions.Charts()
        # model=2 is the full ensemble model
        result = charts.project(
            ticker=ticker,
            model=2,
            serverName="StressTestServer",
            serverInvite="https://discord.gg/stresstest",
            serverIcon="bot/assets/placeholderIcon.jpg",
            userID=user_id
        )
        duration = time.time() - start_time
        if result:
            print(f"User {user_id} SUCCESS for {ticker} in {duration:.2f}s")
            return True, duration
        else:
            print(f"User {user_id} FAILED (No Result) for {ticker} in {duration:.2f}s")
            return False, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"User {user_id} ERROR for {ticker}: {e} in {duration:.2f}s")
        import traceback
        traceback.print_exc()
        return False, duration

def run_stress_test(num_users=20):
    tickers = ["BAC", "WFC", "CVX", "BP", "T", "VZ", "PFE", "MRK", "LLY", "NKE", 
               "SBUX", "MCD", "BA", "GE", "F", "GM", "MS", "BLK", "AXP", "CAT"]
    
    print(f"Starting Stress Test with {num_users} concurrent users...")
    overall_start = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = {executor.submit(simulate_user, i+1, tickers[i]): i for i in range(num_users)}
        
        for future in as_completed(futures):
            results.append(future.result())
            
    overall_duration = time.time() - overall_start
    
    successes = [r for r in results if r[0]]
    durations = [r[1] for r in results]
    
    print("\n" + "="*30)
    print("STRESS TEST RESULTS")
    print("="*30)
    print(f"Total Users: {num_users}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {num_users - len(successes)}")
    print(f"Total Time: {overall_duration:.2f}s")
    if durations:
        print(f"Average Response Time: {sum(durations)/len(durations):.2f}s")
        print(f"Min Response Time: {min(durations):.2f}s")
        print(f"Max Response Time: {max(durations):.2f}s")
    print("="*30)

if __name__ == "__main__":
    run_stress_test(20)
