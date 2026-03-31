import sys
import os
import asyncio
import json
from datetime import datetime

# Add bot directory to path
sys.path.append(os.path.join(os.getcwd(), "bot"))

import functions

async def test_macro():
    print("Testing Macro Indicators Integration...")
    charts = functions.Charts()
    
    # We need a dummy stock and history
    # For now, let's just trigger project() with a ticker
    # project(self, ticker, model, serverName, serverInvite, serverIcon, userID)
    
    try:
        # Mocking the required arguments
        # serverIcon can be a URL or local path
        icon = "https://cdn.discordapp.com/embed/avatars/0.png"
        
        # This will actually fetch data via yfinance
        print("Running prediction for AAPL...")
        result, price = charts.project("AAPL", 1, "Test Server", "https://discord.gg/test", icon, None)
        
        if result:
            print(f"Prediction successful. Price: {price}")
            # The Stamp object's image method returns a buffer
            # We can't easily see the image, but we can check if it ran without error
            print("Factors gathered successfully.")
        else:
            print("Prediction failed (returned None).")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_macro())
