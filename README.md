![Banner](/bot/assets/marketing/CenteredBanner.png)

[![Join](https://camo.githubusercontent.com/d98378c65f343e4caef2b96cbe18bdc5bda7f5858f14b2691157ae3dffcc1b2e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6a6f696e5f75732d6f6e5f646973636f72642d3538363546323f6c6f676f3d646973636f7264266c6f676f436f6c6f723d7768697465)](https://discord.gg/CEtxwANbAy)
[![Install](https://img.shields.io/badge/on_discord-5865F2?logo=discord&logoColor=white&label=install
)](https://discord.com/oauth2/authorize?client_id=1447285084402094212)

Select what best describes you:
<details>
<summary>I'm a professional/researcher!</summary>
</details>
<details>
<summary>I'm a user!</summary>
## Quantum 

- Cross Platform (Available on IOS, Android, Desktop Web)
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
</details>
