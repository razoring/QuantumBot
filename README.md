![Banner](/bot/assets/marketing/CenteredBanner.png)

[![Join](https://camo.githubusercontent.com/d98378c65f343e4caef2b96cbe18bdc5bda7f5858f14b2691157ae3dffcc1b2e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6a6f696e5f75732d6f6e5f646973636f72642d3538363546323f6c6f676f3d646973636f7264266c6f676f436f6c6f723d7768697465)](https://discord.gg/CEtxwANbAy)
[![Install](https://img.shields.io/badge/on_discord-5865F2?logo=discord&logoColor=white&label=install
)](https://discord.com/oauth2/authorize?client_id=1447285084402094212)

<details>
<summary>Tell me how the code works!</summary>

## Architecture

### Infrastructure
*   **Deployment**: Docker (via Oracle Cloud)
*   **Database**: PostgreSQL 18 (hosted on Azure Cloud) | [Database Schema](https://drawsql.app/teams/razoringg/diagrams/quantum-bot)
*   **Backend**: Python
*   **Core Libraries**: Facebook Prophet, SciPy, NumPy, Pandas, Matplotlib, Pillow/PIL
*   **API Access**: yfinance, psycopg2, discord, FastAPI

**Concurrency**: The system uses a hybrid concurrency model: N-1 processes (where N is the CPU count) for parallel computation and a pool of 20 threads for I/O operations. This approach dedicates processes to CPU-intensive tasks like model fitting, while threads handle I/O-bound operations like data fetching, maximizing throughput.

**Database Connection/Registry**: A global, thread-safe database connection pool with locking mechanisms manages access to persistent model data. A status registry tracks ongoing operations in real-time, enabling asynchronous progress updates to Discord users.

### Facebook Prophet
#### Why Prophet?
Prophet was chosen for its speed, lightweight nature, extensive documentation, and active maintenance by Meta. It generates predictions in seconds on a CPU, offering a significant efficiency advantage over GPU-dependent models like LSTMs.

#### Shortfalls of Prophet
With default configurations, Prophet can be inaccurate. According to [Sumedh Kaninde et al.](https://www.itm-conferences.org/articles/itmconf/pdf/2022/04/itmconf_icacc2022_03060.pdf), it can yield a high RMSE. As a univariate model, it analyzes only historical price data, lacking awareness of broader market regimes. It also misinterprets events like earnings announcements, dividend payouts, and stock splits as data anomalies or noise, rather than predictable market events. Finally, it assumes that historical trends will repeat indefinitely.

#### Mitigations
To address these shortcomings, the system trains a separate Prophet model for each lookback period (90, 180, 365, 730, and 1825 days). This ensemble approach allows shorter-term models to capture recent sentiment while longer-term models identify structural trends.

Each Prophet instance is configured with:
-   **Logistic Growth**: Constrains predictions within historical price bounds.
-   **Multiplicative Seasonality**: Models yearly (e.g., earnings cycles) and monthly (e.g., rebalancing) patterns.
-   **Changepoint Detection**: Automatically adapts to new market conditions.
-   **Custom Holidays**: Defines earnings, dividends, and ex-dividend dates to prevent the model from treating them as anomalies.

## The Prediction Pipeline

### Request Flow
When a prediction is requested via Discord, the system executes the following steps:

**1. Historical Data Acquisition**
Fetches up to 10 years of historical daily price data (defaulting to 5). The data is resampled to a consistent daily interval using linear interpolation to handle weekends and trading halts.

**2. Model Weight Retrieval**
Queries the database for pre-trained ensemble weights for the ticker. If the weights are older than 24 hours, a new training cycle is triggered.

**3. Parallel Forecasting**
Five forecasting tasks are launched in parallel (one for each lookback period), reducing a process that would take minutes sequentially to mere seconds. Each task:
-   Extracts its historical window (e.g., last 90 days).
-   Prepares data with price caps and floors derived from the current price ±30%.
-   Fits a Prophet model.
-   Returns a 90-day forward prediction, uncertainty bands, and detected changepoints.

**4. Ensemble Blending**
The five prediction curves are combined into a single forecast using the retrieved ensemble weights. If the weights favor the 90-day model, its curve will have a greater influence on the final blended prediction.

**5. Multi-Factor Adjustment**
The blended forecast is adjusted using several contextual factors:
-   **Sector Trends (10% weight)**: The system forecasts the trend of the ticker's corresponding sector ETF and blends it into the prediction.
-   **Macro Trends (10% weight)**: The S&P 500 index is similarly forecasted and blended.
-   **Earnings Surprises (5% weight)**: An adjustment multiplier is applied based on historical earnings beat/miss rates for the ticker and its industry peers.
-   **Short Float (5% weight)**: High short interest acts as a headwind, dampening the prediction.

The final prediction weighs the ticker's Prophet forecast at 70%, with the remaining 30% distributed across these contextual factors.

**6. Volatility Surface Construction**
A probabilistic price surface is constructed using implied volatility from the options chain. Geometric Brownian Motion is used to project price quantiles (5th to 95th percentile) forward 90 days. If options data is unavailable, the system uses Prophet's built-in uncertainty bands.

## The Training and Optimization System

### The Challenge
Model accuracy depends on two key elements: Prophet's hyperparameters and the ensemble weights for blending lookback periods. A one-size-fits-all approach is ineffective; a volatile growth stock requires different parameters than a stable blue-chip stock.

### The Training Approach: Dynamic Parameter Search

**Phase 1: Hyperparameter Search**
The system generates N candidate parameter sets, where N is the number of available CPU cores. It creates a spectrum of configurations, from low-flexibility/high-seasonality to high-flexibility/low-seasonality. Each candidate is evaluated by:

1.  Splitting recent history into 90 days of training data and 90 days of test data.
2.  Fitting Prophet with the candidate parameter set.
3.  Forecasting the test period.
4.  Calculating two metrics:
    -   **SMAPE (Symmetric Mean Absolute Percentage Error)**: Measures the accuracy of price predictions. It penalizes over- and under-prediction equally.
    -   **Shape Score**: Measures the percentage of days the model correctly predicted the price direction (up or down).

The best parameter set is chosen based on the optimal balance between low error (SMAPE) and high directional accuracy (Shape Score).

**Phase 2: Weight Optimization**
Using the selected hyperparameters, all five models are trained. A numerical optimization algorithm (SciPy's SLSQP) then determines the optimal weight vector that minimizes prediction error over the test period. The optimization is constrained so that weights must sum to 1.0 and each weight must be between 0% and 85% to prevent overreliance on any single model.

The resulting weight vector is stored in the database and used for future predictions until it expires.

### Batch Evaluation
All parameter candidates are submitted to a process pool and evaluated in parallel. A system with 8 cores can evaluate 8 parameter sets simultaneously, reducing the search time from minutes to seconds.

## Caching Strategy

### Addressing Redundancy
Fitting a Prophet model is computationally expensive, taking 5-20 seconds per instance. Re-fitting for recent, identical requests is inefficient.

### Solution Architecture
An in-memory, 64-entry LRU (Least Recently Used) cache stores model-fitting results. Each entry is keyed by a unique combination of:
-   Last date of historical data
-   Lookback period (90d, 180d, etc.)
-   Last 5 closing prices
-   The current weight vector
-   A cache version identifier

If a request matches an existing cache key within a 24-hour TTL, the cached result is returned instantly. Otherwise, the model is fitted, and the new result is cached for future use. This ensures that if multiple users request the same stock, only the first request triggers a computation.

### Memory Efficiency
Cache capacity is capped to prevent unbounded memory growth. The LRU eviction policy removes the least-accessed entry when the cache is full. Historical data is downsampled to daily intervals, and prices are stored as floats to balance accuracy with memory footprint.

## The Evaluation and Backtesting System

### The Evaluation Method
The system uses a 90-day rolling backtest to measure performance:
1.  Takes all available historical data.
2.  Reserves the most recent 90 days as a test set.
3.  Trains the model on the data preceding the test set.
4.  Generates a 90-day forward prediction.
5.  Compares the prediction to the actual prices in the test set, calculating:
    -   Average daily error ($)
    -   SMAPE (%)
    -   Price difference range (min/max error)

## The Feature Engineering Pipeline

### Market Factor Analysis
To provide interpretable insights, the system generates "market factors" for each forecast:
1.  Extracts changepoints (dates of structural shifts) from each Prophet model.
2.  Weights each changepoint by its model's contribution to the ensemble.
3.  Filters for significant changepoints (impact > 2.2 standard deviations).
4.  Ranks the remaining changepoints by magnitude.

This process surfaces insights like "Similar pattern on 2025-11-03 suggested upward movement," making the model's reasoning transparent.

### Fundamental Data Integration
For historical chart requests, the system integrates fundamental data from Yahoo Finance:
-   52-week high/low, volume, market cap
-   P/E ratio, EPS, Beta
-   Dividend yield, payout dates, payment history
-   Analyst rating consensus

Data is fetched concurrently using a thread pool executor to minimize latency.

## Visualization and Output

### Chart Generation
Generates high-quality PNG charts using Matplotlib with several features:
1.  **Multi-Layered Plotting**: Displays historical price, sector trend, macro trend (S&P 500), the forward prediction, and confidence bands.
2.  **Event Markers**: Marks past and projected earnings dates and dividend ex-dates on the chart.
3.  **Adaptive Axis Formatting**: Adjusts X and Y-axis tick spacing and labels for readability based on the price and time scale.
4.  **Branded Output**: Embeds the chart into a template with the server icon and name, and overlays market factors and the price target.

### Historical Chart Generation
For analysis requests, the system generates candlestick charts with a corresponding volume subplot, with color-coding and adaptive date formatting.

## Performance Optimization Techniques

### Parallel Processing Hierarchy
The system employs a three-tier parallelization strategy:
1.  **Process-Level**: The process executor handles Prophet model fitting for true parallelism, bypassing Python's GIL.
2.  **Thread-Level**: The thread executor handles I/O-bound tasks like data fetching and API calls.
3.  **Batch-Level**: Parameter candidates for training are evaluated as a batch to maximize executor utilization.

### I/O Optimization
Database access is optimized with connection pooling and prepared statements to reduce query overhead and prevent contention. API calls and database queries are batched where possible.

</details>

## Quantum 

- Cross Platform (Available on iOS, Android, Desktop, Web via Discord)
- Advertise your server with shareable graphics!
- Quote: Returns just the stock quote with an update button to fetch the latest quote fast. Update button to retrieve latest quote.
- Alerts: **set**, **delete**, **clear all**, or **list all** alerts. Alerts can be set at a specific price or can send daily volatility at the end of trading hours.
- Prediction:
    - **Implied Volatility (Most Reliable):** Uses a modified Black-scholes implied volatility formula that extracts options data (Calls/Puts) and uses a modified probability density function to display as a fan chart. However, data is less precise if the stock is less popular and final projection is less detailed.
    - **Extrapolation (Fallback):** Uses a modified times-series forecasting model with a custom fourier order and custom, individual weights. Predicts based on past data; does not understand why those patterns appear. Model takes into consideration other economic factors such that 70% of the final project is based on the model's prediction, 10% is based on the expected trend of the S&P 500, 10% is the expected trend of its current sector, 5% is based on the expected earnings, and 5% is based on the short sentiment (short float ratio).
    - **Grounded-Extrapolation (Default):** The best of both worlds. It takes both the implied volatility and the extrapolation results and averages the two to get both foresight and hindsight. (Not available if implied volatility data does not exist)
- Candlestick Charts: Returns candlestick charts with the latest stock quote. Update button to retrieve latest quote.
    - Past 24 Hours (1d)
    - Past Week (5d)
    - Past Month (1mo)
    - Past 3 Months (3mo)
    - Past 6 Months (6mo)
    - Past Year (1y)
    - Past Year from Today (ytd)
    - Past 2 Years (2y)
    - Past 5 Years (5y)
    - Past 10 Years (10y)
    - Maximum Displayable (all)

## Installation
[![Install](https://img.shields.io/badge/on_discord-5865F2?logo=discord&logoColor=white&label=install
)](https://discord.com/oauth2/authorize?client_id=1447285084402094212)
## Usage

#### Prediction

```
/predict <ticker> <model>
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `ticker` | `string` | **Required**. The symbol to fetch data of. Use ```\tickers``` to validate if ticker exists.|
| `model` | `string` | *Optional*. The model to use. **Default: Aggregate-Extrapolation**|
| `lookback` | `string` | *Optional*. The range of the past prices to display. **Default: 90d**|

#### History Charts

```
/chart <ticker> <duration>
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `ticker` | `string` | **Required**. The symbol to fetch data of. Use ```\tickers``` to validate if ticker exists.|
| `duration` | `string` | **Required**. The cutoff of the data. |

#### Live Quote

```
/quote <ticker>
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `ticker` | `string` | **Required**. The symbol to fetch data of. Use ```\tickers``` to validate if ticker exists.|

#### Alerts

```
/alerts
```
| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `none` | - | - |

#### Tickers

```
/tickers <ticker>
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `query` | `string` | **Required**. The query to search for ticker.|

#### Help

```
/help
```
| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `none` | - | - |

## Support

For support, join the [discord server](https://discord.gg/CEtxwANbAy).
