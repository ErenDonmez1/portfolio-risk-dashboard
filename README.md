# Portfolio Risk Dashboard

A Streamlit dashboard for exploring portfolio risk from price data. The project focuses on clear, testable Python calculations for returns, volatility, drawdowns, Value at Risk, stress testing and Monte Carlo simulation.

I built this as a student portfolio project to practise applying data analysis and engineering-style modelling to finance and risk analytics.

## Example Screenshots

**Dashboard overview**
![Dashboard overview](assets/dashboard-overview.png)

**Asset risk tab**
![Asset risk tab](assets/asset-risk-tab.png)

**Portfolio analysis tab**
![Portfolio analysis tab](assets/portfolio-analysis-tab.png)

**Correlated Monte Carlo simulation**
![Correlated Monte Carlo simulation](assets/correlated-monte-carlo.png)

The Data Quality Lab reviews the selected dataset before risk results are
interpreted. It summarises missing or suspicious observations, highlights
assets that need review, compares return distributions and exports a CSV
validation report.

## Features

- Load demo data, upload a CSV, or fetch optional yfinance data
- Validate market datasets in the Data Quality Lab
- Detect missing, duplicate, non-numeric, stale and irregular observations
- Flag extreme returns using configurable z-score and IQR rules
- Compare return distributions with descriptive statistics and a two-sample KS test
- Export a CSV validation report with warnings and recommended actions
- Calculate daily and cumulative returns
- Measure annualised volatility
- Analyse drawdowns and maximum drawdown
- View return correlations between assets
- Set custom portfolio weights
- Calculate historical 95% one-day Value at Risk
- Run scenario stress tests
- Run a basic portfolio-level Monte Carlo simulation
- Run a correlated asset-level Monte Carlo simulation using historical covariance
- Summarise simulated VaR, Expected Shortfall / CVaR and probability of loss

## Tech Stack

- Python
- pandas
- NumPy
- matplotlib
- Plotly
- SciPy
- Streamlit
- pytest

## Project Structure

```text
app.py                  Streamlit dashboard UI
src/data_quality.py     Data validation and analysis orchestration
src/data_quality_stats.py   Return and statistical diagnostics
src/data_quality_report.py  Quality scoring, warnings and report tables
src/data_quality_ui.py      Streamlit rendering for the Data Quality Lab
src/                    Calculation, chart and UI helper modules
tests/                  pytest tests
data/sample_prices.csv  Synthetic demo price data
assets/                 README screenshots
```

The calculation logic is kept separate from the Streamlit app so the finance functions can be tested directly.

## Data Quality Lab

The Data Quality Lab adds a research-focused validation stage before portfolio
metrics are interpreted. Its purpose is to make common market-data problems
visible, quantify their potential impact on an analysis, and record sensible
follow-up actions. The output is an automated screening report, not a conclusion
that data is corrupt.

### Methodology

- Missing values are counted by column.
- Duplicate dates are checked as duplicate `Ticker` and `Date` pairs because
  different assets are expected to share trading dates.
- Non-numeric closing prices are reported separately from missing values.
- Stale prices are flagged when a price remains unchanged for a configurable
  number of consecutive observations.
- Daily percentage returns are calculated independently for each ticker.
- Extreme returns are screened using configurable absolute z-scores and IQR
  fences. A flagged return may still be a valid market move or corporate action.
- Observation counts, asset-specific date gaps and rolling volatility are
  calculated for each ticker.
- Each return history is split chronologically into two halves. Their means and
  standard deviations are compared, followed by a two-sample
  Kolmogorov-Smirnov test.
- The 0-100 quality score applies capped, weighted deductions across the checks.
  It is a screening aid, not a certification of correctness. A high score also
  does not guarantee enough history for reliable modelling.

### Statistical Assumptions

Returns are simple percentage returns calculated from ordered closing prices.
Rolling daily volatility is annualised using the square-root-of-252 convention.
Z-scores describe distance from the sample mean in sample standard deviations;
they are sensitive to the outliers they are intended to flag. IQR fences are
non-parametric but can still over-flag small samples.

The KS test compares empirical return distributions without identifying the
cause of a difference. Financial returns are time ordered and may not satisfy
the independence assumptions used in simple two-sample inference. A low p-value
is therefore a diagnostic warning for further investigation, not proof of a
market regime change.

## Risk Metrics

**Returns** measure price changes over time. Cumulative returns are compounded
rather than summed. Missing prices are not forward-filled; the dashboard uses
dates with complete return observations across the displayed assets.

**Volatility** measures how much returns vary. The dashboard annualises daily volatility using 252 trading days.

**Drawdown** measures the fall from a previous peak. Maximum drawdown is the worst peak-to-trough decline in the selected period.

**Correlation** is calculated on returns, not prices, to show how assets have moved together historically.

**Historical VaR** uses the lower tail of historical portfolio returns and reports the result as a positive loss.

**Stress testing** applies user-defined or preset shocks to each holding. Shocks are handled as decimal returns in the calculation layer.

**Monte Carlo simulation** is included as a simplified risk illustration, not a forecast. The correlated version simulates assets jointly using their historical covariance matrix.

## Run Locally

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## CSV Format

Custom CSV files should contain:

```text
Date,Ticker,Close
```

Example:

```text
Date,Ticker,Close
2026-01-02,AAPL,185.64
2026-01-02,MSFT,412.21
```

## Tests

Run the test suite with:

Windows PowerShell:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

macOS/Linux:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests include hand-calculated finance examples and synthetic data-quality
datasets containing deliberate missing values, duplicates, stale prices,
irregular gaps and outliers.

Run only the Data Quality Lab tests with:

```powershell
python -m pytest tests/test_data_quality.py -q
```

## Limitations

- The demo data is synthetic.
- yfinance data is optional, may be delayed or unavailable, and remains subject
  to Yahoo's terms of use.
- The risk models are simplified for learning and portfolio-project purposes.
- Data-quality thresholds are configurable screening rules, not universal
  definitions of bad data.
- Stale prices, gaps and extreme returns may have legitimate market explanations.
- The distribution-shift comparison does not control for dependence, seasonality,
  corporate actions or multiple statistical tests.
- Historical data and simulations do not predict future returns.
- This project is not investment advice.

## Future Improvements

- Add printable HTML or PDF reporting
- Add separate integrity and research-readiness scores
- Add factor exposure analysis
- Deploy the dashboard online
